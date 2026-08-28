#include "Engine.h"
#include "Autograd.h"
#include "AnomalyMode.h"
#include "ManualNodes.h"
#include "Exception.h"
#include "LinearAlgebraNames.h"
#include "Profiler.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <limits>
#include <algorithm>
#include <utility>
#include <optional>

namespace tensorplay {
namespace tpx {

thread_local int Engine::nested_depth_ = 0;

namespace {
// Sets the thread-local GradMode to `enabled` for the duration of a scope.
struct GradModeGuard {
    explicit GradModeGuard(bool enabled) : prev_(GradMode::is_enabled()) {
        GradMode::set_enabled(enabled);
    }
    ~GradModeGuard() { GradMode::set_enabled(prev_); }
    bool prev_;
};

inline bool engine_trace_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("TP_ENGINE_TRACE");
        return e && e[0] == '1';
    }();
    return on;
}
#define TP_ENGINE_TRACE(msg) do { if (engine_trace_enabled()) fprintf(stderr, "[tp-engine] %s\n", (msg)); } while (0)

// ---------------------------------------------------------------------------
// Structured backward-graph tracing (TP_ENGINE_TRACE).
//
// A lightweight debugging surface the upstream engine does not have:
//   TP_ENGINE_TRACE=0/unset  off (single static check on the hot path)
//   TP_ENGINE_TRACE=1        lifecycle events only
//   TP_ENGINE_TRACE=2        + per-node apply (shapes) and every delivery
//                            decision (dependency counter, buffer/enqueue)
//   TP_ENGINE_TRACE=3        + gradient VALUES (truncated) instead of shapes
//   TP_ENGINE_TRACE_FILE=... redirect the stream (default stderr)
//
// Every line carries the GraphTask id so nested/concurrent backwards are
// separable, and nodes are labeled name#seq@ptr-suffix which stays stable
// across a single graph.
struct EngineTrace {
    static int level() {
        static const int lvl = [] {
            const char* e = std::getenv("TP_ENGINE_TRACE");
            if (!e || !e[0]) return 0;
            return e[0] >= '1' && e[0] <= '3' ? e[0] - '0' : 1;
        }();
        return lvl;
    }
    static FILE* sink() {
        static FILE* f = [] {
            const char* p = std::getenv("TP_ENGINE_TRACE_FILE");
            return (p && p[0]) ? fopen(p, "w") : stderr;
        }();
        return f;
    }
    static uint64_t next_graph_id() {
        static std::atomic<uint64_t> counter{0};
        return counter.fetch_add(1) + 1;
    }

    // Stable short label for a node: name#seq@suffix-of-pointer.
    static void node_label(char* buf, size_t n, const Node* fn) {
        if (!fn) { snprintf(buf, n, "<null>"); return; }
        snprintf(buf, n, "%s#%llu@%zx", fn->name().c_str(),
                 static_cast<unsigned long long>(fn->sequence_nr()),
                 reinterpret_cast<size_t>(fn) % 0x10000);
    }

    // Level 2 renders shape/dtype; level 3 appends up to 8 elements.
    static void tensor_desc(char* buf, size_t n, const Tensor& t) {
        if (!t.defined()) { snprintf(buf, n, "<undef>"); return; }
        std::string shape = "(";
        const auto sizes = static_cast<std::vector<int64_t>>(t.shape());
        for (size_t i = 0; i < sizes.size(); ++i) {
            shape += std::to_string(sizes[i]);
            if (i + 1 < sizes.size()) shape += ",";
        }
        shape += ")";
        char base[48];
        snprintf(base, sizeof(base), "%s%s rg=%d",
                 c10_style_dtype_name(t.dtype()), shape.c_str(),
                 t.requires_grad() ? 1 : 0);
        if (level() < 3) { snprintf(buf, n, "%s", base); return; }
        std::string vals = "{";
        Tensor flat = t.reshape({-1});
        const int64_t total = flat.numel();
        const int64_t show = std::min<int64_t>(total, 8);
        for (int64_t i = 0; i < show; ++i) {
            vals += std::to_string(flat.select(0, i).item().toDouble());
            if (i + 1 < show) vals += ",";
        }
        if (total > show) vals += ",...";
        vals += "}";
        snprintf(buf, n, "%s%s", base, vals.c_str());
    }

    template <typename... Args>
    static void emit(uint64_t graph_id, const char* fmt, Args&&... args) {
        FILE* out = sink();
        fprintf(out, "[tp-engine g%llu] ",
                static_cast<unsigned long long>(graph_id));
        fprintf(out, fmt, std::forward<Args>(args)...);
        fputc('\n', out);
    }
};


// Restores a thread-local counter on scope exit, including via exceptions.
struct DepthGuard {
    explicit DepthGuard(int& depth) : depth_(depth) { ++depth_; }
    ~DepthGuard() { --depth_; }
    int& depth_;
};

uint64_t compute_min_topological_nr(const edge_list& outputs) {
    uint64_t min_topo_nr = std::numeric_limits<uint64_t>::max();
    for (const auto& edge : outputs) {
        if (edge.function) {
            min_topo_nr = std::min(min_topo_nr, edge.function->topological_nr());
        }
    }
    return min_topo_nr == std::numeric_limits<uint64_t>::max() ? 0 : min_topo_nr;
}

// Anomaly-mode NaN probe: relies on IEEE `NaN != NaN`; runs through the
// dispatcher so it works on both CPU and CUDA tensors.
bool tensor_has_nan(const Tensor& t) {
    if (!t.defined() || !isFloatingOrComplexType(t.dtype())) return false;
    return t.ne(t).any().item<bool>();
}
} // namespace

Engine& Engine::get_default_engine() {
    // Deliberately leaked: worker threads and queued tasks may outlive static
    // destruction order (same rationale as torch::autograd::Engine).
    static Engine* engine = new Engine();
    return *engine;
}

ReadyQueue* Engine::queue_for_device(int device_index) {
    std::lock_guard<std::mutex> lock(queues_mutex_);
    auto it = ready_queues_.find(device_index);
    if (it != ready_queues_.end()) return it->second;

    auto* queue = new ReadyQueue();
    ready_queues_[device_index] = queue;
    if (device_index >= 0) {
        // Spawn one persistent worker per CUDA device on first use, mirroring
        // torch::autograd::Engine::initialize_device_threads_pool().
        device_threads_.emplace(
            device_index, std::thread([this, queue] { worker_main(*queue); }));
    }
    return queue;
}

void Engine::worker_main(ReadyQueue& queue) {
    TP_ENGINE_TRACE("worker started");
    for (;;) {
        auto task = queue.pop_until([] { return false; });
        if (!task) continue; // spurious wakeup; keep waiting
        execute_task(std::move(*task), queue, nullptr);
    }
}

void Engine::execute_task(ReadyQueue::NodeTask&& task, ReadyQueue& cpu_queue,
                          ReadyQueue* local_queue) {
    GraphTask& graph = *task.graph_;
    if (engine_trace_enabled()) fprintf(stderr, "[tp-engine] exec node %s\n", task.fn_->name().c_str());
    try {
        evaluate_function(graph, task.fn_.get(), task.input_buffer_, cpu_queue, local_queue);
    } catch (...) {
        // A failing node must not hang the whole backward: record the error
        // and account for this task so the graph still drains naturally; the
        // initiator rethrows once every queued task has been evaluated.
        graph.record_exception(std::current_exception());
        if (graph.task_completed()) {
            queue_for_device(-1)->notify();
        }
    }
}

void Engine::compute_dependencies(Node* root, GraphTask& task, uint64_t min_topo_nr) {
    // Computes the number of dependencies for each function which requires grad
    std::vector<Node*> queue{root};
    auto& dependencies = task.dependencies_;
    auto& nodes_in_graph = task.nodes_in_graph_;
    while (!queue.empty()) {
        auto fn = queue.back();
        queue.pop_back();
        // Nodes created before the first output cannot have an edge to it.
        if (fn->topological_nr() < min_topo_nr) {
            continue;
        }
        for (const auto& edge : fn->next_edges()) {
            if (auto next_ptr = edge.function.get()) {
                dependencies[next_ptr] += 1;
                if (EngineTrace::level() >= 2) {
                    char from[128], to[128];
                    EngineTrace::node_label(from, sizeof(from), fn);
                    EngineTrace::node_label(to, sizeof(to), next_ptr);
                    EngineTrace::emit(
                        task.trace_id_, "dep  %s -> %s (count=%d)",
                        from, to, dependencies[next_ptr]);
                }
                const bool was_inserted = nodes_in_graph.insert(next_ptr).second;
                if (was_inserted) {
                    queue.push_back(next_ptr);
                }
            }
        }
    }
}

void GraphTask::init_to_execute(Node& graph_root, const edge_list& outputs,
                                bool accumulate_grad, uint64_t min_topo_nr) {
    int output_idx = 0;
    for (auto& output_edge : outputs) {
        Node* output = output_edge.function.get();
        auto& info = exec_info_[output];
        if (accumulate_grad) {
            info.needed_ = true;
        } else {
            if (!info.captures_) {
                info.captures_ = std::make_unique<std::vector<ExecInfo::Capture>>();
            }
            info.captures_->emplace_back(output_edge.input_nr, output_idx++);
        }
    }
    captured_vars_.resize(output_idx);

    struct Frame {
        Frame(Node* fn) : fn_(fn) {}
        Node* fn_{};
        size_t next_next_fn_{};

        Node* get_next_fn() {
            const auto& next = fn_->next_edges();
            auto num_next = next.size();
            while (next_next_fn_ < num_next) {
                auto fn = next[next_next_fn_++].function.get();
                if (fn) return fn;
            }
            return nullptr;
        }
    };

    auto nodeShouldExecute = [this](Node* fn) {
        auto it = exec_info_.find(fn);
        return it != exec_info_.end() && it->second.should_execute();
    };

    std::vector<Frame> stack;
    std::unordered_set<Node*> seen;
    stack.emplace_back(&graph_root);
    exec_info_.emplace(stack.back().fn_, ExecInfo());

    while (!stack.empty()) {
        auto& frame = stack.back();
        const auto fn = frame.fn_;

        Node* child_fn = nullptr;
        while ((child_fn = frame.get_next_fn()) && !seen.emplace(child_fn).second) {
            // Child already seen: if it should execute, so must we.
            if (nodeShouldExecute(child_fn)) {
                exec_info_[fn].needed_ = true;
            }
        }

        if (child_fn) {
            // Child created before the first output cannot have an edge to it.
            if (child_fn->topological_nr() < min_topo_nr) {
                continue;
            }
            stack.emplace_back(child_fn);
        } else {
            // No next child: `fn`'s needed is finalized. Pop and update parent.
            stack.pop_back();
            if (nodeShouldExecute(fn) && !stack.empty()) {
                exec_info_[stack.back().fn_].needed_ = true;
            }
        }
    }
}

// Mirrors torch's validate_outputs/at::sum_to: reduce a gradient whose shape
// doesn't match the recorded forward-input shape of its destination slot.
// Without this, gradients of broadcast operands keep their broadcast-inflated
// shape mid-graph and break consumers expecting the operand's true shape.
static Tensor sum_to_shape(const Tensor& grad, const std::vector<int64_t>& target) {
    // Mirrors ATen ExpandUtils.h::_sum_to: sum the extra leading dims and any
    // broadcast-inflated (target==1) dims with keepdim=true, then view down
    // to the exact target rank.
    if (target.empty()) {
        return ops::sum(grad);
    }
    const int64_t leading =
        static_cast<int64_t>(grad.dim()) - static_cast<int64_t>(target.size());
    if (leading < 0) {
        // Gradient rank below the forward-input rank (e.g. a scalar seed
        // feeding a (1,) leaf): reshape up when element counts line up --
        // broadcast-compatible -- otherwise hand the grad through untouched.
        int64_t target_numel = 1;
        for (const auto d : target) target_numel *= d;
        if (grad.numel() == target_numel) {
            return ops::reshape(grad, target);
        }
        return grad;
    }
    std::vector<int64_t> reduce_dims;
    for (int64_t i = 0; i < leading; ++i) reduce_dims.push_back(i);
    for (int64_t i = leading; i < static_cast<int64_t>(grad.dim()); ++i) {
        if (target[static_cast<size_t>(i - leading)] == 1 &&
            grad.size(i) != 1) {
            reduce_dims.push_back(i);
        }
    }
    Tensor cur = reduce_dims.empty()
        ? grad : ops::sum(grad, reduce_dims, /*keepdim=*/true);
    if (leading > 0) {
        cur = ops::reshape(cur, target);
    }
    return cur;
}

void Engine::evaluate_function(GraphTask& task, Node* func, InputBuffer& inputs,
                               ReadyQueue& cpu_queue, ReadyQueue* local_queue) {
    auto& exec_info_ = task.exec_info_;

    // Worker threads must honor this graph's create_graph decision regardless
    // of their thread-local GradMode.
    GradModeGuard grad_guard(task.grad_mode_);

    if (!exec_info_.empty()) {
        GraphTask::ExecInfo& fn_info = exec_info_.at(func);
        variable_list new_inputs = inputs.buffer;
        if (auto* capture_vec = fn_info.captures_.get()) {
            std::lock_guard<std::mutex> lock(task.mutex_);
            for (const auto& capture : *capture_vec) {
                task.captured_vars_[capture.output_idx_] = new_inputs[capture.input_idx_];
            }
        }
        if (!fn_info.needed_) {
            // Skip execution if we don't need to execute the function.
            if (task.task_completed()) {
                queue_for_device(-1)->notify();
            }
            return;
        }
    }

    variable_list outputs;
    {
        // Per-node backward event ("backward::MulBackward0" style, matching
        // upstream's profiler surface).  Inactive cost: one atomic load;
        // the virtual demangle runs only when a session is live, and names
        // are interned so long training loops don't grow any arena.
        const bool __tp_prof_on =
            tensorplay::prof::g_active.load(std::memory_order_acquire);
        const char* __tp_node_nm = "";
        if (__tp_prof_on) {
            __tp_node_nm = tensorplay::prof::intern_name(
                "backward::" + func->name());
        }
        tensorplay::prof::OpRecord __tp_node_rec(__tp_node_nm);
        if (EngineTrace::level() >= 2) {
            char label[128], buf[256];
            EngineTrace::node_label(label, sizeof(label), func);
            std::string in = "apply " + std::string(label) + " inputs=[";
            for (size_t i = 0; i < inputs.buffer.size(); ++i) {
                EngineTrace::tensor_desc(buf, sizeof(buf), inputs.buffer[i]);
                in += (i ? ", " : "");
                in += buf;
            }
            in += "]";
            EngineTrace::emit(task.trace_id_, "%s", in.c_str());
        }
        variable_list vars = InputBuffer::variables(std::move(inputs));
        // torch parity (InputMetadata + materialize_grads): zero-fill any
        // undefined input gradient so user backward functions never see None
        // unless they opted out via set_materialize_grads(false).
        //
        // Metadata source differs by node kind: custom-function nodes
        // (PyNode) record per-OUTPUT-slot metadata at attach time (their
        // backward inputs ARE the forward outputs); generated derivative
        // nodes fall back to the metadata recorded on their next_edges.
        if (func->materialize_grads()) {
            const auto& out_metas = func->output_metas();
            const auto& in_edges = func->next_edges();
            const size_t n = std::min(vars.size(),
                out_metas.empty() ? in_edges.size() : out_metas.size());
            for (size_t i = 0; i < n; ++i) {
                if (vars[i].defined()) continue;
                std::vector<int64_t> shape;
                DType dt = DType::Undefined;
                DeviceType dev_type = DeviceType::CPU;
                int64_t dev_idx = -1;
                bool have = false;
                if (!out_metas.empty() && i < out_metas.size()
                    && out_metas[i].valid) {
                    shape = out_metas[i].shape;
                    dt = out_metas[i].dtype;
                    dev_idx = out_metas[i].device_index;
                    have = true;
                } else if (i < in_edges.size() && in_edges[i].has_shape_hint
                           && in_edges[i].grad_dtype.has_value()
                           && in_edges[i].device_type_hint.has_value()
                           && in_edges[i].device_index_hint.has_value()) {
                    shape = in_edges[i].shape_hint;
                    dt = *in_edges[i].grad_dtype;
                    dev_type = *in_edges[i].device_type_hint;
                    dev_idx = *in_edges[i].device_index_hint;
                    have = true;
                }
                if (!have) continue;
                Device dev(dev_type,
                           dev_type == DeviceType::CPU ? -1 : dev_idx);
                vars[i] = ops::zeros(shape, dt, dev);
            }
        }
        for (const auto& hook : func->pre_hooks()) {
            vars = hook(std::move(vars));
        }
        // Track the node under evaluation so nodes created during a
        // create_graph backward record it as their anomaly-mode parent, and
        // surface the forward traceback when the backward itself fails.
        CurrentEvalNodeGuard eval_node_guard(
            AnomalyMode::is_enabled() ? func->shared_from_this() : nullptr);
        if (AnomalyMode::is_enabled()) {
            try {
                outputs = func->apply(std::move(vars));
            } catch (...) {
                if (auto* metadata = func->anomaly_metadata()) {
                    metadata->print_stack(func->name());
                }
                throw;
            }
        } else {
            outputs = func->apply(std::move(vars));
        }
        for (const auto& hook : func->post_hooks()) {
            outputs = hook(outputs, std::move(outputs));
        }
    }
    if (EngineTrace::level() >= 2) {
        char label[128], buf[256];
        EngineTrace::node_label(label, sizeof(label), func);
        std::string out = "emit  " + std::string(label) + " outputs=[";
        for (size_t i = 0; i < outputs.size(); ++i) {
            EngineTrace::tensor_desc(buf, sizeof(buf), outputs[i]);
            out += (i ? ", " : "");
            out += buf;
        }
        out += "]";
        EngineTrace::emit(task.trace_id_, "%s", out.c_str());
    }

    // Shape-validate gradients against the forward-input shapes recorded on
    // this node's edges (see Edge::shape_hint). Mirrors torch's
    // Engine::validate_outputs; under create_graph the reduction ops join the
    // second-order graph because GradMode is active here.
    {
        const auto& out_edges = func->next_edges();
        const size_t n = std::min(outputs.size(), out_edges.size());
        for (size_t i = 0; i < n; ++i) {
            if (!outputs[i].defined() || !out_edges[i].is_valid()) continue;
            const auto& hint = out_edges[i].shape_hint;
            if (out_edges[i].has_shape_hint) {
                bool shape_ok = true;
                if (static_cast<size_t>(outputs[i].dim()) == hint.size()) {
                    const auto out_sizes =
                        static_cast<std::vector<int64_t>>(outputs[i].shape());
                    shape_ok =
                        std::equal(hint.begin(), hint.end(), out_sizes.begin());
                } else {
                    shape_ok = false;
                }
                if (!shape_ok) {
                    outputs[i] = sum_to_shape(outputs[i], hint);
                }
            }
            // Dtype contract (torch InputMetadata::grad_dtype via
            // validate_outputs): a floating gradient crossing an edge must be
            // the forward input's dtype.  Autocast graphs depend on this --
            // unwrapped promote ops emit fp32 grads that must re-enter
            // backward nodes holding low-precision saved tensors.
            // Non-floating hints (bool masks / index tensors) never carry
            // gradients and are left alone.
            const auto& grad_dt = out_edges[i].grad_dtype;
            if (grad_dt.has_value() && isFloatingType(*grad_dt) &&
                isFloatingType(outputs[i].dtype()) &&
                outputs[i].dtype() != *grad_dt && !InferenceMode::is_enabled()) {
                outputs[i] = tpx::to(outputs[i], *grad_dt);
            }
        }
    }

    if (AnomalyMode::is_enabled() && AnomalyMode::should_check_nan()) {
        GradModeGuard grad_guard(false);
        for (size_t i = 0; i < outputs.size(); ++i) {
            TP_THROW_IF(tensor_has_nan(outputs[i]), RuntimeError,
                        "Function '", func->name(), "' returned nan values in its ",
                        i, "th output.");
        }
    }

    auto num_outputs = outputs.size();

    // Propagate BEFORE release_variables() (which clears next_edges_).
    const auto& edges = func->next_edges();
    for (size_t i = 0; i < num_outputs; ++i) {
        if (i >= edges.size()) break;
        auto& output = outputs[i];
        const auto& next = edges[i];

        if (!next.is_valid()) continue;

        // Check if the next function is ready to be computed. The dependency
        // counters and not_ready buffers are shared across workers, so all
        // bookkeeping happens under the task mutex.
        bool is_ready = false;
        bool enqueue_now = false;
        ReadyQueue::NodeTask pending(nullptr, InputBuffer(), &task);
        {
            std::lock_guard<std::mutex> lock(task.mutex_);
            auto& dependencies = task.dependencies_;
            auto it = dependencies.find(next.function.get());
            if (it == dependencies.end()) {
                TP_THROW(RuntimeError, "dependency not found for node ", func->sequence_nr());
            } else if (--it->second == 0) {
                dependencies.erase(it);
                is_ready = true;
            }

            auto& not_ready = task.not_ready_;
            auto not_ready_it = not_ready.find(next.function.get());
            if (not_ready_it == not_ready.end()) {
                // Skip functions that aren't supposed to be executed
                if (!exec_info_.empty()) {
                    auto it2 = exec_info_.find(next.function.get());
                    if (it2 == exec_info_.end() || !it2->second.should_execute()) {
                        continue;
                    }
                }
                // No buffers have been allocated for the function
                InputBuffer input_buffer(next.function->num_inputs());
                input_buffer.add(next.input_nr, std::move(output), task.grad_mode_);
                if (is_ready) {
                    pending = ReadyQueue::NodeTask(next.function, std::move(input_buffer), &task);
                    enqueue_now = true;
                } else {
                    not_ready.emplace(next.function.get(), std::move(input_buffer));
                }
            } else {
                // The function already has a buffer
                auto& input_buffer = not_ready_it->second;
                input_buffer.add(next.input_nr, std::move(output), task.grad_mode_);
                if (is_ready) {
                    pending = ReadyQueue::NodeTask(next.function, std::move(input_buffer), &task);
                    enqueue_now = true;
                    not_ready.erase(not_ready_it);
                }
            }
        }
        if (enqueue_now) {
            if (EngineTrace::level() >= 2) {
                char from[128], to[128];
                EngineTrace::node_label(from, sizeof(from), func);
                EngineTrace::node_label(to, sizeof(to), next.function.get());
                EngineTrace::emit(task.trace_id_,
                                  "send %s -> %s@%zu ready (dep exhausted)",
                                  from, to,
                                  static_cast<size_t>(next.input_nr));
            }
            enqueue_task(task, std::move(pending), cpu_queue, local_queue);
        } else if (EngineTrace::level() >= 2 && next.is_valid()) {
            char from[128], to[128];
            EngineTrace::node_label(from, sizeof(from), func);
            EngineTrace::node_label(to, sizeof(to), next.function.get());
            EngineTrace::emit(task.trace_id_,
                              "buf  %s -> %s@%zu (waiting on more inputs)",
                              from, to, static_cast<size_t>(next.input_nr));
        }
    }

    if (!task.keep_graph_) {
        func->release_variables();
    }

    if (task.task_completed()) {
        // The graph just finished on this (possibly device-worker) thread.
        // Wake the initiating thread, which blocks in pop_until() on the CPU
        // queue (queue_for_device(-1)). Completion only signalled task.cv_, so
        // without this the initiator would wait out a full poll interval. Note
        // the local `cpu_queue` param is the *device* queue when running on a
        // device worker (see worker_main), so notify the true CPU queue.
        // Mirrors torch's dummy-wakeup task (engine.cpp thread_main).
        queue_for_device(-1)->notify();
    }
}

void Engine::enqueue_task(GraphTask& task, ReadyQueue::NodeTask&& node_task,
                          ReadyQueue& cpu_queue, ReadyQueue* local_queue) {
    task.task_enqueued();
    if (local_queue != nullptr) {
        local_queue->push(std::move(node_task));
        return;
    }
    int dev = node_task.input_buffer_.device_index();
    ReadyQueue* target = (dev >= 0) ? queue_for_device(dev) : &cpu_queue;
    target->push(std::move(node_task));
}

variable_list Engine::execute(const edge_list& root_edges, const variable_list& inputs,
                              bool keep_graph, bool create_graph, bool accumulate_grad,
                              const edge_list& outputs) {
    // Backward-phase annotation for the profiler: one span per engine
    // execution, visible in chrome traces as the parent of every backward
    // node event (worker-thread op events overlap it by wall time).
    tensorplay::prof::OpRecord __tp_backward_span(
        "__backward__", tensorplay::prof::EventKind::kBackward);
    if (root_edges.size() != inputs.size()) {
        TP_THROW(RuntimeError, "Engine::execute: roots and inputs must have same size");
    }
    if (!accumulate_grad && outputs.empty()) {
        TP_THROW(RuntimeError, "grad requires non-empty inputs");
    }

    GraphTask graph_task(keep_graph, create_graph);
    graph_task.trace_id_ = EngineTrace::next_graph_id();
    if (EngineTrace::level() >= 1) {
        EngineTrace::emit(graph_task.trace_id_,
                          "execute create_graph=%d keep_graph=%d "
                          "accumulate_grad=%d roots=%zu",
                          create_graph ? 1 : 0, keep_graph ? 1 : 0,
                          accumulate_grad ? 1 : 0, root_edges.size());
    }

    // If we receive a single root, skip creating an extra root node.
    bool skip_dummy_node = root_edges.size() == 1;
    std::shared_ptr<Node> graph_root;
    if (skip_dummy_node) {
        graph_root = root_edges.at(0).function;
    } else {
        graph_root = std::make_shared<GraphRoot>(root_edges, inputs);
    }

    auto min_topo_nr = compute_min_topological_nr(outputs);
    compute_dependencies(graph_root.get(), graph_task, min_topo_nr);

    if (!outputs.empty()) {
        graph_task.init_to_execute(*graph_root, outputs, accumulate_grad, min_topo_nr);
    }

    const bool nested = nested_depth_ > 0;

    // Queue the root. In nested mode every task stays on a local queue that
    // only this thread drains, so reentrant backward can never deadlock
    // against workers busy with the outer graph.
    ReadyQueue local_queue;
    ReadyQueue& cpu_queue = *queue_for_device(-1);
    if (skip_dummy_node) {
        InputBuffer input_buffer(root_edges.at(0).function->num_inputs());
        input_buffer.add(root_edges.at(0).input_nr, Tensor(inputs.at(0)), create_graph);
        enqueue_task(graph_task,
                     ReadyQueue::NodeTask(root_edges.at(0).function, std::move(input_buffer), &graph_task),
                     cpu_queue, nested ? &local_queue : nullptr);
    } else {
        enqueue_task(graph_task,
                     ReadyQueue::NodeTask(graph_root, InputBuffer(), &graph_task),
                     cpu_queue, nested ? &local_queue : nullptr);
    }

    DepthGuard depth_guard(nested_depth_);
    TP_ENGINE_TRACE(nested ? "execute nested" : "execute top-level");
    if (nested) {
        while (!graph_task.is_completed()) {
            auto t = local_queue.pop_until([] { return false; });
            if (!t) break;
            execute_task(std::move(*t), cpu_queue, &local_queue);
        }
    } else {
        // The initiating thread participates by draining the CPU queue; CUDA
        // tasks are picked up by their dedicated workers.
        while (!graph_task.is_completed()) {
            auto t = cpu_queue.pop_until([&graph_task] { return graph_task.is_completed(); });
            if (!t) break; // graph completed while waiting
            execute_task(std::move(*t), cpu_queue, nullptr);
        }
        graph_task.wait_for_completion();
    }

    TP_ENGINE_TRACE("execute done");
    if (EngineTrace::level() >= 1) {
        EngineTrace::emit(graph_task.trace_id_, "done captured=%zu",
                          graph_task.captured_vars_.size());
    }
    if (graph_task.exception_) {
        std::rethrow_exception(graph_task.exception_);
    }

    return std::move(graph_task.captured_vars_);
}

} // namespace tpx
} // namespace tensorplay

#include "Engine.h"
#include "Autograd.h"
#include "AnomalyMode.h"
#include "ManualNodes.h"
#include "Exception.h"
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
        graph.task_completed();
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
            task.task_completed();
            return;
        }
    }

    variable_list outputs;
    {
        variable_list vars = InputBuffer::variables(std::move(inputs));
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
            enqueue_task(task, std::move(pending), cpu_queue, local_queue);
        }
    }

    if (!task.keep_graph_) {
        func->release_variables();
    }

    task.task_completed();
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
    if (root_edges.size() != inputs.size()) {
        TP_THROW(RuntimeError, "Engine::execute: roots and inputs must have same size");
    }
    if (!accumulate_grad && outputs.empty()) {
        TP_THROW(RuntimeError, "grad requires non-empty inputs");
    }

    GraphTask graph_task(keep_graph, create_graph);

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
    if (graph_task.exception_) {
        std::rethrow_exception(graph_task.exception_);
    }

    return std::move(graph_task.captured_vars_);
}

} // namespace tpx
} // namespace tensorplay

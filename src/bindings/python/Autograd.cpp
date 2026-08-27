#include "python_bindings.h"
#include "Node.h"
#include "AccumulateGrad.h"
#include "Autograd.h"
#include "AnomalyMode.h"
#include <typeinfo>
#include <string>
#include <pybind11/functional.h>

namespace {
// Cached tensor PyTypeObject for ns-scale type checks on the custom-function
// hot path (mirrors CPythonBridge's g_tensor_type trick).
PyTypeObject* g_fast_tensor_type = nullptr;
inline bool fast_is_tensor(PyObject* obj) {
    if (g_fast_tensor_type)
        return PyObject_TypeCheck(obj, g_fast_tensor_type) != 0;
    bool ok = py::isinstance<Tensor>(py::handle(obj));
    if (ok) g_fast_tensor_type = Py_TYPE(obj);
    return ok;
}
} // namespace

// Custom Node for Python-defined Autograd Functions
class PyNode : public tensorplay::tpx::Node {
public:
    PyNode(py::object py_ctx) : py_ctx_(std::move(py_ctx)) {}

    // Backward input slots correspond to forward OUTPUTS for custom
    // functions (torch CustomFunctionNode semantics), so the engine sizes
    // this node's incoming gradient buffer by the attached output count.
    size_t num_inputs() const override {
        return output_metas().empty() ? Node::num_inputs()
                                      : output_metas().size();
    }

    tensorplay::tpx::variable_list apply(tensorplay::tpx::variable_list&& inputs) override {
        if (std::getenv("TP_ENGINE_TRACE")) fprintf(stderr, "[tp-engine] PyNode: acquiring GIL\n");
        py::gil_scoped_acquire gil;
        if (std::getenv("TP_ENGINE_TRACE")) fprintf(stderr, "[tp-engine] PyNode: GIL acquired, calling backward\n");

        // Convert C++ grads to a positional args TUPLE directly (no
        // intermediate py::list): one allocation, PyTuple_SET_ITEM fills.
        size_t n_in = inputs.size();
        py::tuple py_inputs(static_cast<Py_ssize_t>(n_in));
        for (size_t i = 0; i < n_in; ++i) {
            if (inputs[i].defined()) {
                py_inputs[i] = py::cast(inputs[i]);
            } else {
                py_inputs[i] = py::none();
            }
        }
        inputs.clear();

        // Call backward on the context object
        if (!py::hasattr(py_ctx_, "backward")) {
             throw std::runtime_error("PyNode context object has no 'backward' method");
        }

        py::object result_obj = py_ctx_.attr("backward")(*py_inputs);
        if (std::getenv("TP_ENGINE_TRACE")) fprintf(stderr, "[tp-engine] PyNode: backward returned\n");

        tensorplay::tpx::variable_list results;

        if (result_obj.is_none()) {
            return results;
        } else if (py::isinstance<Tensor>(result_obj)) {
            results.push_back(py::cast<Tensor>(result_obj));
        } else if (py::isinstance<py::sequence>(result_obj)) {
            for (auto item : py::cast<py::sequence>(result_obj)) {
                if (item.is_none()) {
                    results.push_back(Tensor());
                } else {
                    results.push_back(py::cast<Tensor>(item));
                }
            }
        } else {
            throw std::runtime_error("backward must return a Tensor, a sequence of Tensors, or None");
        }

        return results;
    }

    py::object py_ctx_;
public:
    py::object ctx() const { return py_ctx_; }
};

void init_autograd(py::module_& m) {
    py::class_<tensorplay::tpx::Node, std::shared_ptr<tensorplay::tpx::Node>>(m, "Node")
        .def_property_readonly("name", [](const tensorplay::tpx::Node& self) {
            return std::string(typeid(self).name());
        })
        .def("_raw_ptr", [](const tensorplay::tpx::Node& self) -> int64_t {
            return reinterpret_cast<int64_t>(&self);
        })
        .def("add_pre_hook", [](tensorplay::tpx::Node& self,
                                std::function<std::vector<tensorplay::tpx::Tensor>(
                                    std::vector<tensorplay::tpx::Tensor>)> hook) {
            // Hooks may fire on engine worker threads; manage the GIL here so
            // the C++ hook invocation is always Python-safe.
            self.add_pre_hook([hook](std::vector<tensorplay::tpx::Tensor>&& grads) {
                py::gil_scoped_acquire gil;
                return hook(std::move(grads));
            });
        }, py::arg("hook"))
        .def("add_post_hook", [](tensorplay::tpx::Node& self,
                                 std::function<std::vector<tensorplay::tpx::Tensor>(
                                     const std::vector<tensorplay::tpx::Tensor>&,
                                     std::vector<tensorplay::tpx::Tensor>)> hook) {
            self.add_post_hook([hook](const std::vector<tensorplay::tpx::Tensor>& inputs,
                                      std::vector<tensorplay::tpx::Tensor>&& outputs) {
                py::gil_scoped_acquire gil;
                return hook(inputs, std::move(outputs));
            });
        }, py::arg("hook"))
        .def_property_readonly("next_functions", [](const tensorplay::tpx::Node& self) {
            std::vector<std::pair<std::shared_ptr<tensorplay::tpx::Node>, int>> result;
            for (const auto& edge : self.next_edges()) {
                result.push_back({edge.function, (int)edge.input_nr});
            }
            return result;
        })
        .def_property_readonly("variable", [](const tensorplay::tpx::Node& self) -> std::optional<tensorplay::tpx::Tensor> {
            auto* acc = dynamic_cast<const tensorplay::tpx::AccumulateGrad*>(&self);
            if (acc) {
                return acc->value_;
            }
            return std::nullopt;
        });

    py::module_ autograd = m.def_submodule("_autograd", "Autograd mechanism");

    py::class_<PyNode, tensorplay::tpx::Node, std::shared_ptr<PyNode>>(autograd, "PyNode")
        .def(py::init<py::object>())
        .def("add_next_edge", [](PyNode& self, std::shared_ptr<tensorplay::tpx::Node> next_node, int input_nr) {
            if (next_node) {
                self.add_next_edge(tensorplay::tpx::Edge(next_node, input_nr));
            } else {
                self.add_next_edge(tensorplay::tpx::Edge());
            }
        }, "next_node"_a.none(), "input_nr"_a = 0)
        .def("set_materialize_grads", &PyNode::set_materialize_grads,
             py::arg("value"))
        .def_property_readonly(
            "_py_ctx", [](PyNode& self) -> py::object { return self.py_ctx_; },
            "The Python context object this node wraps.")
        .def(
            "register_hook",
            [](PyNode& self, py::function hook) {
                self.ctx().attr("_hooks").cast<py::list>().append(hook);
            },
            py::keep_alive<1, 2>())
        .def(
            "register_prehook",
            [](PyNode& self, py::function hook) {
                self.ctx().attr("_prehooks").cast<py::list>().append(hook);
            },
            py::keep_alive<1, 2>())
        .def(
            "attach_outputs",
            [](PyNode& self, py::handle outputs) {
                // Single C++ crossing for graph attachment: marks every
                // tensor output as requiring grad and wires this node as
                // its grad_fn.  Non-tensor slots are skipped so multi-output
                // functions can return Nones mixed with Tensors.  Also
                // records torch-style per-output InputMetadata used by the
                // engine to zero-fill missing gradients.
                auto node = std::shared_ptr<tensorplay::tpx::Node>(
                    std::static_pointer_cast<tensorplay::tpx::Node>(
                        self.shared_from_this()));
                auto& metas = self.output_metas();
                metas.clear();
                int idx = 0;
                auto record_slot = [&](py::handle item) {
                    tensorplay::tpx::OutputSlotMeta m;
                    if (py::isinstance<Tensor>(item)) {
                        const Tensor& t = py::cast<const Tensor&>(item);
                        m.shape = static_cast<std::vector<int64_t>>(t.shape());
                        m.dtype = t.dtype();
                        m.device_index = t.device().index();
                        m.valid = true;
                    }
                    metas.push_back(std::move(m));
                };
                if (py::isinstance<Tensor>(outputs)) {
                    record_slot(outputs);
                    Tensor& t = py::cast<Tensor&>(outputs);
                    tensorplay::tpx::impl::set_requires_grad(t, true);
                    tensorplay::tpx::impl::set_grad_fn(t, node, 0);
                    return;
                }
                for (auto item : outputs.cast<py::sequence>()) {
                    record_slot(item);
                    if (py::isinstance<Tensor>(item)) {
                        Tensor& t = py::cast<Tensor&>(item);
                        tensorplay::tpx::impl::set_requires_grad(t, true);
                        tensorplay::tpx::impl::set_grad_fn(t, node, idx);
                    }
                    ++idx;
                }
            },
            "outputs"_a);
    
    autograd.def("collect_next_edges", [](const Tensor& t) {
        auto edges = tensorplay::tpx::collect_next_edges(t);
        std::vector<std::pair<std::shared_ptr<tensorplay::tpx::Node>, int>> result;
        for (const auto& edge : edges) {
            result.push_back({edge.function, (int)edge.input_nr});
        }
        return result;
    });

        // Mirror of torch's unpack_input(): one C++ pass over the flat argument
    // tuple producing needs_input_grad bits AND wiring this node's
    // next_edges.  Returns (needs_list, any_requires_grad) so the Python
    // layer avoids N per-input pybind round-trips.
    autograd.def("setup_custom_function_graph",
        [](py::object node_obj, py::sequence args) {            auto node = node_obj.cast<std::shared_ptr<tensorplay::tpx::Node>>();
            Py_ssize_t n = PyTuple_GET_SIZE(args.ptr());
            py::list needs(n);
            bool any_rg = false;
            std::vector<tensorplay::tpx::Edge> edges;
            edges.reserve((size_t)n);
            for (Py_ssize_t i = 0; i < n; ++i) {
                PyObject* item = PyTuple_GET_ITEM(args.ptr(), i);
                if (py::isinstance<Tensor>(item)) {
                    const Tensor& t = py::cast<const Tensor&>(item);
                    bool rg = t.requires_grad();
                    any_rg |= rg;
                    needs[i] = py::bool_(rg);
                    if (rg) {
                        for (auto& e : tensorplay::tpx::collect_next_edges(t)) {
                            edges.push_back(std::move(e));
                        }
                    } else {
                        edges.emplace_back();
                    }
                } else {
                    needs[i] = py::bool_(false);
                    edges.emplace_back();
                }
            }
            if (any_rg) {
                node->add_next_edge_list(std::move(edges));
            }
            return py::make_tuple(needs, any_rg);
        },
        "node"_a, "args"_a);

    // Mirror of THPFunction_apply's inner block: toggles grad off, calls
    // the user forward, then setup_context -- all inside ONE crossing so
    // the Python layer pays no per-step pybind/GIL-mode round-trips.
    autograd.def("run_custom_function_forward",
        [](py::object ctx, py::object forward_fn,
           std::optional<py::object> setup_ctx_fn, py::sequence args) {
            const bool prev = tensorplay::tpx::GradMode::is_enabled();
            tensorplay::tpx::GradMode::set_enabled(false);
            py::object output;
            try {
                if (setup_ctx_fn.has_value()) {
                    // new style: forward(*args); setup_context(ctx, args, out)
                    output = (*forward_fn)(*(args.cast<py::tuple>()));
                    if (output) {
                        (*setup_ctx_fn)(
                            ctx, args, py::object(output));
                    }
                } else {
                    // legacy style: forward(ctx, *args)
                    py::tuple full(args.size() + 1);
                    full[0] = ctx;
                    for (Py_ssize_t i = 0; i < args.size(); ++i) {
                        full[i + 1] = args[i];
                    }
                    output = (*forward_fn)(*full);
                }
            } catch (...) {
                tensorplay::tpx::GradMode::set_enabled(prev);
                throw;
            }
            tensorplay::tpx::GradMode::set_enabled(prev);
            return output;
        },
        "ctx"_a, "forward_fn"_a, "setup_ctx_fn"_a.none(), "args"_a);

    // THE single-entry hot path -- a wholesale mirror of THPFunction_apply
    // (python_function.cpp:1699): node creation, unpack_input, the
    // AutoGradMode(false) forward block, setup_context and _wrap_outputs
    // all happen inside ONE pybind crossing.  Returns (output, ctx, needs,
    // executable); Python only builds the backward closure afterwards.
    autograd.def("custom_function_apply",
        [](py::object ctx_factory, py::object node_factory,
           py::object forward_fn, std::optional<py::object> setup_ctx_fn,
           py::sequence args) {
            auto ctx = ctx_factory();
            auto node = node_factory(ctx);
            auto* py_node = node.cast<PyNode*>();

            // ---- unpack_input mirror ----
            Py_ssize_t n_args = PyTuple_GET_SIZE(args.ptr());
            py::tuple needs(n_args);
            bool any_rg = false;
            for (Py_ssize_t i = 0; i < n_args; ++i) {
                PyObject* item = PyTuple_GET_ITEM(args.ptr(), i);
                if (fast_is_tensor(item)) {
                    bool rg = py::cast<const Tensor&>(item).requires_grad();
                    any_rg |= rg;
                    needs[i] = py::bool_(rg);
                } else if (py::isinstance<py::sequence>(item)) {
                    // Nested containers are rare; mark conservatively and
                    // let the Python fallback re-wire if needed.
                    bool nested_rg = false;
                    for (auto inner : py::reinterpret_borrow<py::sequence>(item)) {
                        if (py::isinstance<Tensor>(inner)
                            && py::cast<const Tensor&>(inner).requires_grad()) {
                            nested_rg = true;
                            break;
                        }
                    }
                    any_rg |= nested_rg;
                    needs[i] = py::bool_(nested_rg);
                } else {
                    needs[i] = py::bool_(false);
                }
            }

            const bool prev_grad = tensorplay::tpx::GradMode::is_enabled();
            const bool executable = prev_grad && any_rg;
            if (executable) {
                // next_edges from every tensor arg (single pass)
                std::vector<tensorplay::tpx::Edge> edges;
                edges.reserve((size_t)n_args);
                for (Py_ssize_t i = 0; i < n_args; ++i) {
                    PyObject* item = PyTuple_GET_ITEM(args.ptr(), i);
                    if (fast_is_tensor(item)
                        && py::cast<const Tensor&>(item).requires_grad()) {
                        for (auto& e : tensorplay::tpx::collect_next_edges(
                                 py::cast<const Tensor&>(item))) {
                            edges.push_back(std::move(e));
                        }
                    } else {
                        edges.emplace_back();
                    }
                }
                py_node->add_next_edge_list(std::move(edges));
                py_node->set_materialize_grads(true);
            }

            // ---- forward block under AutoGradMode(false) ----
            py::object output;
            tensorplay::tpx::GradMode::set_enabled(false);
            try {
                if (setup_ctx_fn.has_value()) {
                    output = (*forward_fn)(*(args.cast<py::tuple>()));
                    if (!output) throw py::error_already_set();
                    (*setup_ctx_fn)(ctx, args, output);
                } else {
                    py::tuple full(n_args + 1);
                    full[0] = ctx;
                    for (Py_ssize_t i = 0; i < n_args; ++i) {
                        full[i + 1] = PyTuple_GET_ITEM(args.ptr(), i);
                    }
                    output = (*forward_fn)(*full);
                }
            } catch (...) {
                tensorplay::tpx::GradMode::set_enabled(prev_grad);
                throw;
            }
            tensorplay::tpx::GradMode::set_enabled(prev_grad);

            // ---- _wrap_outputs mirror (executable only) ----
            if (executable) {
                auto shared = std::shared_ptr<tensorplay::tpx::Node>(
                    std::static_pointer_cast<tensorplay::tpx::Node>(
                        py_node->shared_from_this()));
                auto& metas = py_node->output_metas();
                metas.clear();
                int idx = 0;
                auto mark = [&](py::handle item) {
                    tensorplay::tpx::OutputSlotMeta m;
                    if (fast_is_tensor(item.ptr())) {
                        Tensor& t = py::cast<Tensor&>(item);
                        tensorplay::tpx::impl::set_requires_grad(t, true);
                        tensorplay::tpx::impl::set_grad_fn(t, shared, idx);
                        m.shape =
                            static_cast<std::vector<int64_t>>(t.shape());
                        m.dtype = t.dtype();
                        m.device_index = t.device().index();
                        m.valid = true;
                    }
                    metas.push_back(std::move(m));
                    ++idx;
                };
                if (py::isinstance<Tensor>(output)) {
                    mark(output);
                } else if (py::isinstance<py::sequence>(output)) {
                    for (auto item : output.cast<py::sequence>()) mark(item);
                }
                ctx.attr("_outputs") = py::isinstance<py::tuple>(output)
                    ? output
                    : (py::isinstance<py::list>(output)
                           ? py::tuple(output.cast<py::sequence>())
                           : py::make_tuple(output));
                ctx.attr("requires_grad") = true;
                ctx.attr("backward_fn") = py::none();  // set by Python later
            }
            return py::make_tuple(output, ctx, needs, executable,
                                   node);
        },
        "ctx_factory"_a, "node_factory"_a, "forward_fn"_a,
        "setup_ctx_fn"_a.none(), "args"_a);

    autograd.def("backward", [](const std::vector<Tensor>& tensors, std::optional<std::vector<Tensor>> grad_tensors, std::optional<bool> retain_graph, bool create_graph) {
        bool keep_graph = retain_graph.value_or(create_graph);
        std::vector<Tensor> grads;
        if (grad_tensors) grads = *grad_tensors;
        // The engine may evaluate nodes on worker threads that need the GIL
        // for Python-backed autograd functions; the initiating thread must
        // not hold it while it waits for the graph to drain.
        py::gil_scoped_release release;
        tensorplay::tpx::backward(tensors, grads, keep_graph, create_graph);
    }, "tensors"_a, "grad_tensors"_a = py::none(), "retain_graph"_a = py::none(), "create_graph"_a = false);

    autograd.def("grad", [](const std::vector<Tensor>& outputs, const std::vector<Tensor>& inputs, std::optional<std::vector<Tensor>> grad_outputs, std::optional<bool> retain_graph, bool create_graph, bool allow_unused) {
        bool keep_graph = retain_graph.value_or(create_graph);
        std::vector<Tensor> grads;
        if (grad_outputs) grads = *grad_outputs;
        // Undefined gradients (unused inputs, or grads that arrive as
        // undefined through the graph) surface as None, matching torch.
        // torch.autograd.grad returns a tuple; functional.py's vjp/jvp rely
        // on it.
        std::vector<tensorplay::Tensor> captured;
        {
            py::gil_scoped_release release;
            captured = tensorplay::tpx::grad(outputs, inputs, grads,
                                             keep_graph, create_graph,
                                             allow_unused);
        }
        py::tuple result(captured.size());
        for (size_t i = 0; i < captured.size(); ++i) {
            if (captured[i].defined()) result[i] = py::cast(std::move(captured[i]));
            else result[i] = py::none();
        }
        return result;
    }, "outputs"_a, "inputs"_a, "grad_outputs"_a = py::none(), "retain_graph"_a = py::none(), "create_graph"_a = false, "allow_unused"_a = false);

    autograd.def("is_grad_enabled", &tensorplay::tpx::GradMode::is_enabled);
    autograd.def("set_grad_enabled", &tensorplay::tpx::GradMode::set_enabled);

    // Inference mode (mirrors torch._C._InferenceMode): a context object the
    // Python wrapper drives through __enter__/__exit__. Entering disables
    // autograd recording and freezes version counters; exit restores the
    // previous state so nested contexts behave like torch's guard stack.
    struct PyInferenceMode {
        bool prev_ = false;
        explicit PyInferenceMode(bool mode) {
            prev_ = tensorplay::tpx::InferenceMode::is_enabled();
            tensorplay::tpx::InferenceMode::set_enabled(mode);
        }
        void enter() {}
        void exit(const std::optional<py::object>&,
                  const std::optional<py::object>&,
                  const std::optional<py::object>&) {
            tensorplay::tpx::InferenceMode::set_enabled(prev_);
        }
    };

    py::class_<PyInferenceMode>(autograd, "_InferenceMode")
        .def(py::init<bool>(), py::arg("mode") = true)
        .def("__enter__", &PyInferenceMode::enter)
        .def("__exit__", &PyInferenceMode::exit);

    autograd.def("is_inference_mode_enabled", &tensorplay::tpx::InferenceMode::is_enabled);

    // Anomaly mode (mirrors torch._C._autograd anomaly bindings). Node
    // creation happens deep inside C++ op wrappers while the calling thread
    // holds the GIL, so capturing the Python traceback at that point records
    // the user-level call site of each forward op.
    autograd.def("is_anomaly_enabled", &tensorplay::tpx::AnomalyMode::is_enabled);
    autograd.def("is_anomaly_check_nan_enabled", &tensorplay::tpx::AnomalyMode::should_check_nan);
    autograd.def("set_anomaly_enabled",
                 [](bool enabled, bool check_nan) { tensorplay::tpx::AnomalyMode::set_enabled(enabled, check_nan); },
                 "enabled"_a, "check_nan"_a = true);

    // Profiler submodule
    py::module_ profiler = m.def_submodule("profiler", "Profiler");

    // Parallel submodule
    py::module_ parallel = m.def_submodule("parallel", "Parallel computing");

    // Install the anomaly-mode stack capturer: records the Python traceback
    // of the forward op call site (mirrors torch's PyAnomalyMetadata, which
    // overrides the C++ backtrace default for the Python engine).
    tensorplay::tpx::set_anomaly_stack_capture([]() -> std::string {
        if (!Py_IsInitialized()) return {};
        try {
            py::gil_scoped_acquire gil;
            if (!Py_IsInitialized()) return {};
            auto traceback = py::module_::import("traceback");
            auto stack = traceback.attr("format_stack")();
            std::string out = py::str(stack).cast<std::string>();
            return out;
        } catch (const std::exception&) {
            return {};
        }
    });
}

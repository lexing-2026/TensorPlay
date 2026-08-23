#include "python_bindings.h"
#include "Node.h"
#include "AccumulateGrad.h"
#include "Autograd.h"
#include "AnomalyMode.h"
#include <typeinfo>
#include <string>
#include <pybind11/functional.h>

// Custom Node for Python-defined Autograd Functions
class PyNode : public tensorplay::tpx::Node {
public:
    PyNode(py::object py_ctx) : py_ctx_(std::move(py_ctx)) {}

    tensorplay::tpx::variable_list apply(tensorplay::tpx::variable_list&& inputs) override {
        if (std::getenv("TP_ENGINE_TRACE")) fprintf(stderr, "[tp-engine] PyNode: acquiring GIL\n");
        py::gil_scoped_acquire gil;
        if (std::getenv("TP_ENGINE_TRACE")) fprintf(stderr, "[tp-engine] PyNode: GIL acquired, calling backward\n");

        // Convert C++ inputs (grads) to Python
        py::list py_inputs;
        for (const auto& input : inputs) {
            if (input.defined()) {
                py_inputs.append(py::cast(input));
            } else {
                py_inputs.append(py::none());
            }
        }

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

private:
    py::object py_ctx_;
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
        }, "next_node"_a.none(), "input_nr"_a = 0);
    
    autograd.def("collect_next_edges", [](const Tensor& t) {
        auto edges = tensorplay::tpx::collect_next_edges(t);
        std::vector<std::pair<std::shared_ptr<tensorplay::tpx::Node>, int>> result;
        for (const auto& edge : edges) {
            result.push_back({edge.function, (int)edge.input_nr});
        }
        return result;
    });

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
        py::gil_scoped_release release;
        auto captured = tensorplay::tpx::grad(outputs, inputs, grads, keep_graph, create_graph, allow_unused);
        // Undefined gradients (unused inputs, or grads that arrive as
        // undefined through the graph) surface as None, matching torch.
        std::vector<std::optional<Tensor>> result;
        result.reserve(captured.size());
        for (auto& t : captured) {
            if (t.defined()) result.emplace_back(std::move(t));
            else result.emplace_back(std::nullopt);
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

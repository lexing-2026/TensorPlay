#include "python_bindings.h"
#include "Node.h"
#include "AccumulateGrad.h"
#include "TPXTensor.h"
#include <typeinfo>
#include <string>

// Custom Node for Python-defined Autograd Functions
class PyNode : public tensorplay::tpx::Node {
public:
    PyNode(nb::object py_ctx) : py_ctx_(std::move(py_ctx)) {}

    tensorplay::tpx::variable_list apply(tensorplay::tpx::variable_list&& inputs) override {
        nb::gil_scoped_acquire gil;

        // Convert C++ inputs (grads) to Python
        nb::list py_inputs;
        for (const auto& input : inputs) {
            if (input.defined()) {
                py_inputs.append(nb::cast(input));
            } else {
                py_inputs.append(nb::none());
            }
        }

        // Call backward on the context object
        if (!nb::hasattr(py_ctx_, "backward")) {
             throw std::runtime_error("PyNode context object has no 'backward' method");
        }
        
        nb::object result_obj = py_ctx_.attr("backward")(*py_inputs);

        tensorplay::tpx::variable_list results;

        if (result_obj.is_none()) {
            return results;
        } else if (nb::isinstance<Tensor>(result_obj)) {
            results.push_back(nb::cast<Tensor>(result_obj));
        } else if (nb::isinstance<nb::sequence>(result_obj)) {
            for (auto item : nb::cast<nb::sequence>(result_obj)) {
                if (item.is_none()) {
                    results.push_back(Tensor());
                } else {
                    results.push_back(nb::cast<Tensor>(item));
                }
            }
        } else {
            throw std::runtime_error("backward must return a Tensor, a sequence of Tensors, or None");
        }

        return results;
    }

private:
    nb::object py_ctx_;
};

void init_autograd(nb::module_& m) {
    nb::class_<tensorplay::tpx::Node>(m, "Node")
        .def_prop_ro("name", [](const tensorplay::tpx::Node& self) {
            return std::string(typeid(self).name());
        })
        .def_prop_ro("next_functions", [](const tensorplay::tpx::Node& self) {
            std::vector<std::pair<std::shared_ptr<tensorplay::tpx::Node>, int>> result;
            for (const auto& edge : self.next_edges()) {
                result.push_back({edge.function, (int)edge.input_nr});
            }
            return result;
        })
        .def_prop_ro("variable", [](const tensorplay::tpx::Node& self) -> std::optional<tensorplay::tpx::Tensor> {
            auto* acc = dynamic_cast<const tensorplay::tpx::AccumulateGrad*>(&self);
            if (acc) {
                if (auto impl = acc->impl_.lock()) {
                    if (auto meta = acc->meta_.lock()) {
                         tensorplay::Tensor p10_t(impl);
                         return tensorplay::tpx::Tensor(p10_t, meta);
                    }
                }
            }
            return std::nullopt;
        });

    nb::module_ autograd = m.def_submodule("_autograd", "Autograd mechanism");

    nb::class_<PyNode, tensorplay::tpx::Node>(autograd, "PyNode")
        .def(nb::init<nb::object>())
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
        tensorplay::tpx::backward(tensors, grads, keep_graph, create_graph);
    }, "tensors"_a, "grad_tensors"_a = nb::none(), "retain_graph"_a = nb::none(), "create_graph"_a = false);

    autograd.def("grad", [](const std::vector<Tensor>& outputs, const std::vector<Tensor>& inputs, std::optional<std::vector<Tensor>> grad_outputs, std::optional<bool> retain_graph, bool create_graph, bool allow_unused) {
        bool keep_graph = retain_graph.value_or(create_graph);
        std::vector<Tensor> grads;
        if (grad_outputs) grads = *grad_outputs;
        return tensorplay::tpx::grad(outputs, inputs, grads, keep_graph, create_graph, allow_unused);
    }, "outputs"_a, "inputs"_a, "grad_outputs"_a = nb::none(), "retain_graph"_a = nb::none(), "create_graph"_a = false, "allow_unused"_a = false);

    autograd.def("is_grad_enabled", &tensorplay::tpx::GradMode::is_enabled);
    autograd.def("set_grad_enabled", &tensorplay::tpx::GradMode::set_enabled);

    // Profiler submodule
    nb::module_ profiler = m.def_submodule("profiler", "Profiler");
    
    // Parallel submodule
    nb::module_ parallel = m.def_submodule("parallel", "Parallel computing");
}
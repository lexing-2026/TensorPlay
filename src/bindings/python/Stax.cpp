#include "python_bindings.h"
#include "Graph.h"
#include "Fusion.h"
#include "StaxPointwise.h"
#include <sstream>

using namespace tensorplay::stax;

void init_stax(py::module_& m) {
    py::module_ stax_m = m.def_submodule("_stax", "Stax Static Graph Optimization");

    py::class_<Graph>(stax_m, "Graph")
        .def(py::init<>())
        .def("print", &Graph::print)
        .def("create_node", &Graph::createNode, py::return_value_policy::reference, py::arg("op_type"), py::arg("name") = "")
        .def("add_input", &Graph::addInput, py::return_value_policy::reference)
        .def("register_output", &Graph::registerOutput)
        .def("execute", &Graph::execute, py::arg("inputs"))
        .def("fuse", [](Graph& self) {
            fuseGraph(self);
        })
        .def_property_readonly("nodes", [](const Graph& g) {
            std::vector<OpNode*> nodes;
            for(auto& n : g.nodes) nodes.push_back(n.get());
            return nodes;
        }, py::return_value_policy::reference)
        .def_property_readonly("inputs", [](const Graph& g) { return g.inputs; }, py::return_value_policy::reference)
        .def_property_readonly("outputs", [](const Graph& g) { return g.outputs; }, py::return_value_policy::reference);

    py::class_<OpNode>(stax_m, "OpNode")
        .def_property("op_type", [](const OpNode& n) { return n.op_type; }, [](OpNode& n, const std::string& k) { n.op_type = k; })
        .def_property_readonly("name", [](const OpNode& n) { return n.name; })
        .def_property_readonly("input_count", [](const OpNode& n) { return n.inputs.size(); })
        .def("add_input", &OpNode::addInput)
        .def("add_output", &OpNode::addOutput, py::return_value_policy::reference)
        .def_property_readonly("inputs", [](const OpNode& n) { return n.inputs; }, py::return_value_policy::reference)
        .def_property_readonly("outputs", [](const OpNode& n) { return n.outputs; }, py::return_value_policy::reference)
        .def("set_int_attr", [](OpNode& n, const std::string& key, int64_t val) { n.setAttr(key, val); })
        .def("set_float_attr", [](OpNode& n, const std::string& key, double val) { n.setAttr(key, val); })
        .def("set_str_attr", [](OpNode& n, const std::string& key, const std::string& val) { n.setAttr(key, val); })
        .def("set_ints_attr", [](OpNode& n, const std::string& key, const std::vector<int64_t>& val) { n.setAttr(key, val); })
        .def("set_floats_attr", [](OpNode& n, const std::string& key, const std::vector<double>& val) { n.setAttr(key, val); })
        .def("get_int_attr", [](OpNode& n, const std::string& key) { 
            return n.getAttr<int64_t>(key);
        })
        .def("get_float_attr", [](OpNode& n, const std::string& key) {
            return n.getAttr<double>(key);
        })
        .def("has_attr", [](OpNode& n, const std::string& key) { return n.attrs.count(key) > 0; });
    
    py::class_<ValueNode>(stax_m, "ValueNode")
        .def_readonly("id", &ValueNode::id)
        .def_property("shape", 
            [](const ValueNode& v) { return v.shape; },
            [](ValueNode& v, const std::vector<int64_t>& s) { v.shape = s; })
        .def_property("dtype", 
            [](const ValueNode& v) { return v.dtype; },
            [](ValueNode& v, const std::string& d) { v.dtype = d; })
        .def_property_readonly("use_count", [](const ValueNode& v) {
            return v.uses.size();
        });

            
    py::class_<IRBuilder>(stax_m, "IRBuilder")
        .def(py::init<Graph&>())
        .def("create_input", &IRBuilder::createInput, py::return_value_policy::reference, py::arg("shape"), py::arg("dtype")="float32")
        .def("create_op", &IRBuilder::createOp, py::return_value_policy::reference, 
             py::arg("op_type"), py::arg("inputs"), py::arg("out_shape")=std::vector<int64_t>{}, py::arg("name")="")
        .def("mark_output", &IRBuilder::markOutput);

    stax_m.def(
        "execute_fused_pointwise_multi",
        [](const std::vector<tensorplay::Tensor>& inputs,
           const std::vector<int64_t>& program,
           const std::vector<double>& constants,
           const std::vector<int64_t>& output_refs) {
            return tensorplay::cpu::stax_fused_pointwise_cpu_multi(
                inputs,
                program,
                constants,
                output_refs);
        },
        py::arg("inputs"),
        py::arg("program"),
        py::arg("constants"),
        py::arg("output_refs"));
}

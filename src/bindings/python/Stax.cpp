#include "python_bindings.h"
#include "Graph.h"
#include "Fusion.h"
#include <sstream>

using namespace tensorplay::stax;

void init_stax(nb::module_& m) {
    nb::module_ stax_m = m.def_submodule("_stax", "Stax Static Graph Optimization");

    nb::class_<Graph>(stax_m, "Graph")
        .def(nb::init<>())
        .def("print", &Graph::print)
        .def("create_node", &Graph::createNode, nb::rv_policy::reference, nb::arg("op_type"), nb::arg("name") = "")
        .def("add_input", &Graph::addInput, nb::rv_policy::reference)
        .def("register_output", &Graph::registerOutput)
        .def("fuse", [](Graph& self) {
            fuseGraph(self);
        })
        .def_prop_ro("nodes", [](const Graph& g) {
            std::vector<OpNode*> nodes;
            for(auto& n : g.nodes) nodes.push_back(n.get());
            return nodes;
        }, nb::rv_policy::reference)
        .def_prop_ro("inputs", [](const Graph& g) { return g.inputs; }, nb::rv_policy::reference)
        .def_prop_ro("outputs", [](const Graph& g) { return g.outputs; }, nb::rv_policy::reference);

    nb::class_<OpNode>(stax_m, "OpNode")
        .def_prop_rw("op_type", [](const OpNode& n) { return n.op_type; }, [](OpNode& n, const std::string& k) { n.op_type = k; })
        .def_prop_ro("name", [](const OpNode& n) { return n.name; })
        .def_prop_ro("input_count", [](const OpNode& n) { return n.inputs.size(); })
        .def("add_input", &OpNode::addInput)
        .def("add_output", &OpNode::addOutput, nb::rv_policy::reference)
        .def_prop_ro("inputs", [](const OpNode& n) { return n.inputs; }, nb::rv_policy::reference)
        .def_prop_ro("outputs", [](const OpNode& n) { return n.outputs; }, nb::rv_policy::reference)
        .def("set_int_attr", [](OpNode& n, const std::string& key, int64_t val) { n.setAttr(key, val); })
        .def("set_float_attr", [](OpNode& n, const std::string& key, double val) { n.setAttr(key, val); })
        .def("get_int_attr", [](OpNode& n, const std::string& key) { 
            return n.getAttr<int64_t>(key);
        })
        .def("get_float_attr", [](OpNode& n, const std::string& key) {
            return n.getAttr<double>(key);
        })
        .def("has_attr", [](OpNode& n, const std::string& key) { return n.attrs.count(key) > 0; });
    
    nb::class_<ValueNode>(stax_m, "ValueNode")
        .def_ro("id", &ValueNode::id)
        .def_prop_rw("shape", 
            [](const ValueNode& v) { return v.shape; },
            [](ValueNode& v, const std::vector<int64_t>& s) { v.shape = s; })
        .def_prop_rw("dtype", 
            [](const ValueNode& v) { return v.dtype; },
            [](ValueNode& v, const std::string& d) { v.dtype = d; });
            
    nb::class_<IRBuilder>(stax_m, "IRBuilder")
        .def(nb::init<Graph&>())
        .def("create_input", &IRBuilder::createInput, nb::rv_policy::reference, nb::arg("shape"), nb::arg("dtype")="float32")
        .def("create_op", &IRBuilder::createOp, nb::rv_policy::reference, 
             nb::arg("op_type"), nb::arg("inputs"), nb::arg("out_shape")=std::vector<int64_t>{}, nb::arg("name")="")
        .def("mark_output", &IRBuilder::markOutput);
}

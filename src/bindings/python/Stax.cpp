#include "python_bindings.h"
#include "Graph.h"
#include "Fusion.h"
#include "StaxPointwise.h"
#include <sstream>
#include <vector>

using namespace tensorplay::stax;


// ---------------------------------------------------------------------------
// Compiled-call trampoline (steady-state METH_FASTCALL entry).
//
// Uses the same capsule-carried
// PyCFunction that absorbs the per-call Python dispatch layers
// (api.optimized key memo -> lowering.__call__ route/bind memos -> execute)
// for the exact steady state -- same tensor objects, unchanged versions.
//
// Fast path:  arity ok, all args are the installed tensor exact-type, every
//   (id, _version) matches the stored fingerprint, and no input requires grad
//   when this lowering owns an autograd plan.  Then the cached input vector
//   (args + attribute targets + constants) is refreshed in place and handed
//   straight to the native Graph.execute callable -- zero Python frames.
// Divert:     anything else vectorcalls the Python lowering, which re-resolves
//   binding/route through its own memos; the stored fingerprint is refreshed
//   first so the next identical call re-enters the fast path.
//
// In-place mutation bumps _version (ConvKernels guard contract), so a stale
// fast hit is impossible by construction; requires_grad rides the fingerprint
// and is additionally checked whenever a gradient plan exists.
// ---------------------------------------------------------------------------

namespace {

struct CallTrampolineState {
    PyObject* lowering;      // strong: divert target (Python __call__)
    PyObject* exec_fn;       // strong: native Graph.execute bound method
    PyObject* tail;          // strong list: attribute targets + constants
    PyObject* tensor_type;   // strong: tensorplay.Tensor (exact-type check)
    Py_ssize_t nargs_expected;
    int output_count;
    bool gradient_plan;
    // Fingerprint: per-arg (id, version, requires_grad-int).
    std::vector<PyObject*> fp_ids;    // borrowed refs into args on refresh
    std::vector<long long> fp_versions;
    std::vector<int> fp_rg;
};

PyObject* call_trampoline(PyObject* self, PyObject* const* args,
                          Py_ssize_t nargs, PyObject* kwnames) {
    auto* st = static_cast<CallTrampolineState*>(PyCapsule_GetPointer(self, nullptr));
    if (!st) return nullptr;

    bool can_fast = (kwnames == nullptr && nargs == st->nargs_expected);
    if (can_fast) {
        for (Py_ssize_t i = 0; i < nargs; ++i) {
            PyObject* v = args[i];
            if (Py_TYPE(v) != reinterpret_cast<PyTypeObject*>(st->tensor_type)) {
                can_fast = false;
                break;
            }
            PyObject* ver = PyObject_GetAttrString(v, "_version");
            if (!ver) { PyErr_Clear(); can_fast = false; break; }
            long long vv = PyLong_AsLongLong(ver);
            Py_DECREF(ver);
            if (vv == -1 && PyErr_Occurred()) { PyErr_Clear(); can_fast = false; break; }
            PyObject* rg = PyObject_GetAttrString(v, "requires_grad");
            int rgi = 0;
            if (!rg) { PyErr_Clear(); can_fast = false; break; }
            rgi = PyObject_IsTrue(rg);
            Py_DECREF(rg);
            if (rgi < 0) { PyErr_Clear(); can_fast = false; break; }
            if (static_cast<size_t>(i) >= st->fp_ids.size() ||
                st->fp_ids[i] != v || st->fp_versions[i] != vv ||
                st->fp_rg[i] != rgi) {
                // Refresh so the next identical call re-enters fast path,
                // then divert once for full Python-side re-resolution.
                if (static_cast<Py_ssize_t>(st->fp_ids.size()) != nargs) {
                    st->fp_ids.assign(nargs, nullptr);
                    st->fp_versions.assign(nargs, 0);
                    st->fp_rg.assign(nargs, 0);
                }
                st->fp_ids[i] = v;
                st->fp_versions[i] = vv;
                st->fp_rg[i] = rgi;
                can_fast = false;
                break;
            }
            if (rgi && st->gradient_plan) {
                // Autograd route decided by the Python lowering.
                can_fast = false;
                break;
            }
        }
    }

    if (!can_fast) {
        return PyObject_Vectorcall(st->lowering, args, nargs, kwnames);
    }

    Py_ssize_t total = nargs + PyList_GET_SIZE(st->tail);
    PyObject* inputs = PyList_New(total);
    if (!inputs) return nullptr;
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        Py_INCREF(args[i]);
        PyList_SET_ITEM(inputs, i, args[i]);
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(st->tail); ++i) {
        PyObject* item = PyList_GET_ITEM(st->tail, i);  // borrowed
        Py_INCREF(item);
        PyList_SET_ITEM(inputs, nargs + i, item);
    }
    PyObject* out = PyObject_CallOneArg(st->exec_fn, inputs);
    Py_DECREF(inputs);
    if (!out) return nullptr;
    if (st->output_count == 1 &&
        (PyList_CheckExact(out) || PyTuple_CheckExact(out)) &&
        PySequence_Fast_GET_SIZE(out) == 1) {
        // Graph.execute returns py::list; Python lowering unwraps single
        // outputs -- mirror that contract exactly.
        PyObject* single = PySequence_Fast_GET_ITEM(out, 0);  // borrowed
        Py_INCREF(single);
        Py_DECREF(out);
        return single;
    }
    if (PyList_CheckExact(out)) {
        PyObject* as_tuple = PyList_AsTuple(out);
        Py_DECREF(out);
        return as_tuple;  // multi-output: match Python's tuple contract
    }
    return out;
}

void call_trampoline_free(PyObject* self) {
    auto* st = static_cast<CallTrampolineState*>(
        PyCapsule_GetPointer(self, nullptr));
    if (!st) return;
    Py_XDECREF(st->lowering);
    Py_XDECREF(st->exec_fn);
    Py_XDECREF(st->tail);
    Py_XDECREF(st->tensor_type);
    delete st;
}

PyMethodDef trampoline_def = {
    "call_trampoline", (PyCFunction)(void(*)(void))call_trampoline,
    METH_FASTCALL | METH_KEYWORDS, nullptr,
};

}  // namespace

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
    // Compiled-call trampoline installer: see the CallTrampolineState block
    // above for the fast-path/divert contract.
    stax_m.def(
        "install_call_trampoline",
        [](py::object lowering, py::object exec_fn, py::list tail,
           py::object tensor_type, Py_ssize_t nargs_expected,
           int output_count, bool gradient_plan) -> py::object {
            auto* st = new CallTrampolineState();
            st->lowering = lowering.ptr();
            st->exec_fn = exec_fn.ptr();
            st->tail = tail.ptr();
            st->tensor_type = tensor_type.ptr();
            st->nargs_expected = nargs_expected;
            st->output_count = output_count;
            st->gradient_plan = gradient_plan;
            Py_INCREF(st->lowering);
            Py_INCREF(st->exec_fn);
            Py_INCREF(st->tail);
            Py_INCREF(st->tensor_type);
            PyObject* cap = PyCapsule_New((void*)st, nullptr,
                                          [](PyObject* c) {
                auto* hh = (CallTrampolineState*)PyCapsule_GetPointer(c, nullptr);
                if (!hh) return;
                Py_XDECREF(hh->lowering);
                Py_XDECREF(hh->exec_fn);
                Py_XDECREF(hh->tail);
                Py_XDECREF(hh->tensor_type);
                delete hh;
            });
            if (!cap) { PyErr_Clear(); delete st; throw py::error_already_set(); }
            PyObject* fn = PyCFunction_New(&trampoline_def, cap);
            Py_DECREF(cap);  // fn holds the self reference
            if (!fn) throw py::error_already_set();
            return py::reinterpret_steal<py::object>(fn);
        },
        py::arg("lowering"), py::arg("exec_fn"), py::arg("tail"),
        py::arg("tensor_type"), py::arg("nargs_expected"),
        py::arg("output_count"), py::arg("gradient_plan"));

    stax_m.def(
        "capture_state_enter",
        [](bool compiling, bool exporting, bool disabled) {
            enterCaptureState(compiling, exporting, disabled);
        },
        py::arg("compiling") = false,
        py::arg("exporting") = false,
        py::arg("disabled") = false);
    stax_m.def(
        "capture_state_exit",
        [](bool compiling, bool exporting, bool disabled) {
            exitCaptureState(compiling, exporting, disabled);
        },
        py::arg("compiling") = false,
        py::arg("exporting") = false,
        py::arg("disabled") = false);
    stax_m.def("capture_state", []() {
        CaptureState state = currentCaptureState();
        return py::make_tuple(
            state.compile_depth,
            state.disabled_depth,
            state.exporting_depth);
    });

}

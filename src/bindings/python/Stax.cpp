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
//   When the installed kernel exposes a direct entry (``direct``), the
//   steady state skips the graph callable entirely: pointers are read from
//   the C++ value holders, the kernel allocates its pinned output in C, and
//   the result comes back pre-wrapped.
// Divert:     anything else vectorcalls the Python lowering, which re-resolves
//   binding/route through its own memos; the stored fingerprint is refreshed
//   first so the next identical call re-enters the fast path.
//
// In-place mutation bumps _version (ConvKernels guard contract), so a stale
// fast hit is impossible by construction; requires_grad rides the fingerprint
// and is additionally checked whenever a gradient plan exists.  Guard reads
// go through non-throwing C++ helpers (tpx_tensor_version /
// tpx_tensor_requires_grad) instead of Python attribute lookups.
// ---------------------------------------------------------------------------

#include "CPythonBridge.h"

using namespace tensorplay::python_c;

namespace {

struct CallTrampolineState {
    PyObject* lowering;      // borrowed: owned by the compiled wrapper
    PyObject* fallback;      // borrowed: outer optimized wrapper, if present
    PyObject* exec_fn;       // strong: native Graph.execute bound method
    PyObject* tail;          // strong list: attribute targets + constants
    PyObject* tensor_type;   // strong: tensorplay.Tensor (exact-type check)
    Py_ssize_t nargs_expected;
    int output_count;
    bool gradient_plan;
    // Direct kernel entry (pinned fused kernels): returns the wrapped
    // output PyObject* for a C array of data pointers.  Null keeps the
    // exec_fn path.
    PyObject* (*direct)(const void* const* ins);
    // Fingerprint: per-arg (id, version, requires_grad-int).
    std::vector<PyObject*> fp_ids;    // strong refs pin the identity
    std::vector<long long> fp_versions;
    std::vector<int> fp_rg;
};

bool capture_busy_probe();

CallTrampolineState* trampoline_state_from_callable(PyObject* callable) {
    if (callable == nullptr || !PyCFunction_Check(callable)) return nullptr;
    PyObject* capsule = PyCFunction_GET_SELF(callable);
    if (capsule == nullptr || !PyCapsule_CheckExact(capsule)) return nullptr;
    void* pointer = PyCapsule_GetPointer(capsule, nullptr);
    if (pointer == nullptr) {
        PyErr_Clear();
        return nullptr;
    }
    return static_cast<CallTrampolineState*>(pointer);
}

void destroy_trampoline_state(CallTrampolineState* st) {
    Py_XDECREF(st->exec_fn);
    Py_XDECREF(st->tail);
    Py_XDECREF(st->tensor_type);
    for (PyObject* id : st->fp_ids) Py_XDECREF(id);
    delete st;
}

PyObject* call_trampoline(PyObject* self, PyObject* const* args,
                          Py_ssize_t nargs, PyObject* kwnames) {
    auto* st = static_cast<CallTrampolineState*>(PyCapsule_GetPointer(self, nullptr));
    if (!st) return nullptr;

    if (capture_busy_probe()) {
        PyObject* fallback = st->fallback != nullptr ? st->fallback : st->lowering;
        return PyObject_Vectorcall(fallback, args, nargs, kwnames);
    }

    bool can_fast = (kwnames == nullptr && nargs == st->nargs_expected);
    if (can_fast) {
        for (Py_ssize_t i = 0; i < nargs; ++i) {
            PyObject* v = args[i];
            if (Py_TYPE(v) != reinterpret_cast<PyTypeObject*>(st->tensor_type)) {
                can_fast = false;
                break;
            }
            long long vv = tpx_tensor_version(v);
            if (vv < 0) { can_fast = false; break; }
            int rgi = 0;
            if (st->gradient_plan) {
                // Autograd route decided by the Python lowering: any input
                // demanding grad diverts.  Only read when a plan exists so
                // inference-only kernels skip one guard probe per arg.
                rgi = tpx_tensor_requires_grad(v);
                if (rgi < 0) { can_fast = false; break; }
                if (rgi) { can_fast = false; break; }
            }
            if (static_cast<size_t>(i) >= st->fp_ids.size() ||
                st->fp_ids[i] != v || st->fp_versions[i] != vv ||
                st->fp_rg[i] != rgi) {
                // Refresh so the next identical call re-enters fast path,
                // then divert once for full Python-side re-resolution.
                if (static_cast<Py_ssize_t>(st->fp_ids.size()) != nargs) {
                    for (PyObject* old : st->fp_ids) Py_XDECREF(old);
                    st->fp_ids.assign(nargs, nullptr);
                    st->fp_versions.assign(nargs, -1);
                    st->fp_rg.assign(nargs, 0);
                }
                Py_INCREF(v);
                Py_XDECREF(st->fp_ids[i]);
                st->fp_ids[i] = v;
                st->fp_versions[i] = vv;
                st->fp_rg[i] = rgi;
                can_fast = false;
                break;
            }
        }
    }

    if (!can_fast) {
        PyObject* fallback = st->fallback != nullptr ? st->fallback : st->lowering;
        return PyObject_Vectorcall(fallback, args, nargs, kwnames);
    }

    if (st->direct != nullptr) {
        // Steady state with a pinned kernel: pointers straight from the
        // C++ value holders, kernel allocates and wraps its own output.
        const void* inline_ins[16];
        std::vector<const void*> dynamic_ins;
        const void** ins = inline_ins;
        if (nargs > static_cast<Py_ssize_t>(
                         sizeof(inline_ins) / sizeof(inline_ins[0]))) {
            dynamic_ins.resize(static_cast<size_t>(nargs));
            ins = dynamic_ins.data();
        }
        for (Py_ssize_t i = 0; i < nargs; ++i) {
            ins[i] = tpx_py_tensor_cref(args[i]).unsafeGetTensorImpl()->data();
        }
        return st->direct(ins);
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
        // outputs -- preserve that contract exactly.
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

PyMethodDef trampoline_def = {
    "call_trampoline", (PyCFunction)(void(*)(void))call_trampoline,
    METH_FASTCALL | METH_KEYWORDS, nullptr,
};

// ---------------------------------------------------------------------------
// Compiled wrapper dispatcher: the outermost entry a user actually calls.
//
// The Python wrapper (api.compile's ``optimized``) owns capture checks, the
// specialization cache and guard chains, but its steady state is: same
// objects, unchanged versions, no capture, no gates -- work the trampoline
// already certifies.  This type short-circuits that Python frame: one C
// vectorcall hands the call to the trampoline, which owns the fingerprint
// check and the slow-path diversion.
// Anything else (new tensors, version bumps, kwargs, capture active)
// vectorcalls the Python wrapper, which resolves everything and rebinds
// ``fast`` through the ``tpx_set_fast`` method below.
// ---------------------------------------------------------------------------

struct CallDispatcher {
    PyObject_HEAD
    vectorcallfunc vectorcall = nullptr;
    // The GC allocator hands out raw memory; pointer fields are initialized
    // by the installer and released by tp_clear/dealloc.
    PyObject* python_entry;   // strong: api.compile's optimized wrapper
    PyObject* fast;           // strong: trampoline PyCFunction (or null)
    PyObject* dict;           // optional instance attributes
    Py_ssize_t nargs;
};
PyObject* call_dispatcher_call(PyObject* self, PyObject* const* args,
                               size_t nargsf, PyObject* kwnames) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    if (d->fast != nullptr) {
        return PyObject_Vectorcall(d->fast, args, nargsf, kwnames);
    }
    return PyObject_Vectorcall(d->python_entry, args, nargsf, kwnames);
}

PyObject* call_dispatcher_getattro(PyObject* self, PyObject* name) {
    // Type-level surface (tpx_set_fast) resolves normally; everything else
    // forwards to the Python wrapper -- attribute traffic is not
    // steady-state.
    if (PyObject* res = PyObject_GenericGetAttr(self, name)) {
        return res;
    }
    if (!PyErr_ExceptionMatches(PyExc_AttributeError)) {
        return nullptr;
    }
    PyErr_Clear();
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    return PyObject_GetAttr(d->python_entry, name);
}

PyObject* call_dispatcher_set_fast(PyObject* self, PyObject* const* args,
                                   Py_ssize_t nargs) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "tpx_set_fast(fast, nargs)");
        return nullptr;
    }
    PyObject* old_fast = d->fast;
    d->fast = nullptr;
    if (old_fast != nullptr) {
        if (auto* old_state = trampoline_state_from_callable(old_fast)) {
            old_state->fallback = nullptr;
        }
        Py_DECREF(old_fast);
    }
    if (args[0] != Py_None) {
        d->fast = args[0];
        Py_INCREF(d->fast);
        d->nargs = PyNumber_AsSsize_t(args[1], PyExc_OverflowError);
        if (d->nargs == -1 && PyErr_Occurred()) {
            Py_CLEAR(d->fast);
            return nullptr;
        }
        if (auto* state = trampoline_state_from_callable(d->fast)) {
            state->fallback = d->python_entry;
        }
    }
    Py_RETURN_NONE;
}

int call_dispatcher_traverse(PyObject* self, visitproc visit, void* arg) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    Py_VISIT(d->python_entry);
    Py_VISIT(d->fast);
    Py_VISIT(d->dict);
    return 0;
}

int call_dispatcher_clear(PyObject* self) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    if (d->fast != nullptr) {
        if (auto* state = trampoline_state_from_callable(d->fast)) {
            state->fallback = nullptr;
        }
    }
    Py_CLEAR(d->python_entry);
    Py_CLEAR(d->fast);
    Py_CLEAR(d->dict);
    return 0;
}

void call_dispatcher_dealloc(PyObject* self) {
    auto* d = reinterpret_cast<CallDispatcher*>(self);
    PyObject_GC_UnTrack(self);
    call_dispatcher_clear(self);
    Py_TYPE(self)->tp_free(self);
}

// Exposes the capture-busy probe used by the dispatcher
// to decide between the C steady state and the Python wrapper.
bool capture_busy_probe() {
    CaptureState state = currentCaptureState();
    return state.disabled_depth > 0 || state.compile_depth > 0 ||
           state.exporting_depth > 0;
}

PyMethodDef dispatcher_methods[] = {
    {"tpx_set_fast", (PyCFunction)(void(*)(void))call_dispatcher_set_fast,
     METH_FASTCALL, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyTypeObject CallDispatcherType = []() {
    PyTypeObject t = {PyVarObject_HEAD_INIT(nullptr, 0)};
    t.tp_name = "tensorplay._C._stax.CallDispatcher";
    t.tp_basicsize = sizeof(CallDispatcher);
    t.tp_dealloc = call_dispatcher_dealloc;
    t.tp_call = PyVectorcall_Call;
    t.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
                 Py_TPFLAGS_HAVE_VECTORCALL;
    t.tp_doc = "C dispatch entry for a compiled TensorPlay wrapper";
    t.tp_traverse = call_dispatcher_traverse;
    t.tp_clear = call_dispatcher_clear;
    t.tp_getattro = call_dispatcher_getattro;
    t.tp_setattro = PyObject_GenericSetAttr;
    t.tp_methods = dispatcher_methods;
    t.tp_vectorcall_offset = offsetof(CallDispatcher, vectorcall);
    t.tp_dictoffset = offsetof(CallDispatcher, dict);
    return t;
}();

}  // namespace

void init_stax(py::module_& m) {
    py::module_ stax_m = m.def_submodule("_stax", "Stax Static Graph Optimization");

    if (PyType_Ready(&CallDispatcherType) < 0) {
        throw py::error_already_set();
    }

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
    // above for the fast-path/divert contract.  ``direct_addr`` is the
    // address of a kernel module's ``tp_direct`` entry (ctypes round-trips
    // it as an integer); zero keeps the Graph.execute path.
    stax_m.def(
        "install_call_trampoline",
        [](py::object lowering, py::object exec_fn, py::list tail,
           py::object tensor_type, Py_ssize_t nargs_expected,
           int output_count, bool gradient_plan,
           unsigned long long direct_addr) -> py::object {
            auto* st = new CallTrampolineState();
            st->lowering = lowering.ptr();
            st->fallback = nullptr;
            st->exec_fn = exec_fn.ptr();
            st->tail = tail.ptr();
            st->tensor_type = tensor_type.ptr();
            st->nargs_expected = nargs_expected;
            st->output_count = output_count;
            st->gradient_plan = gradient_plan;
            st->direct = reinterpret_cast<PyObject* (*)(const void* const*)>(
                direct_addr);
            Py_INCREF(st->exec_fn);
            Py_INCREF(st->tail);
            Py_INCREF(st->tensor_type);
            PyObject* cap = PyCapsule_New((void*)st, nullptr,
                                          [](PyObject* c) {
                auto* hh = (CallTrampolineState*)PyCapsule_GetPointer(c, nullptr);
                if (!hh) return;
                destroy_trampoline_state(hh);
            });
            if (!cap) {
                destroy_trampoline_state(st);
                throw py::error_already_set();
            }
            PyObject* fn = PyCFunction_New(&trampoline_def, cap);
            Py_DECREF(cap);  // fn holds the self reference
            if (!fn) throw py::error_already_set();
            return py::reinterpret_steal<py::object>(fn);
        },
        py::arg("lowering"), py::arg("exec_fn"), py::arg("tail"),
        py::arg("tensor_type"), py::arg("nargs_expected"),
        py::arg("output_count"), py::arg("gradient_plan"),
        py::arg("direct_addr") = 0ULL);

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
    // Boolean probe for the compiled wrapper dispatch fast path: nonzero
    // when capture is disabled or a compile/export pass is running, i.e.
    // when the wrapper must fall back to the eager function instead of the
    // C steady-state launch.  ContextVar reads stay in the slow path.
    stax_m.def("capture_busy", []() {
        CaptureState state = currentCaptureState();
        return state.disabled_depth > 0 || state.compile_depth > 0 ||
               state.exporting_depth > 0;
    });
    // Compiled wrapper dispatcher installer: see the CallDispatcher block.
    // The dispatcher forwards attribute traffic to ``python_entry`` and
    // rebinds the steady-state target through ``tpx_set_fast``.
    stax_m.def(
        "make_call_dispatcher",
        [](py::object python_entry) -> py::object {
            auto* d = PyObject_GC_New(CallDispatcher, &CallDispatcherType);
            if (!d) throw py::error_already_set();
            d->vectorcall = call_dispatcher_call;
            d->python_entry = python_entry.ptr();
            Py_INCREF(d->python_entry);
            d->fast = nullptr;
            d->dict = nullptr;
            d->nargs = 0;
            PyObject_GC_Track(d);
            return py::reinterpret_steal<py::object>(
                reinterpret_cast<PyObject*>(d));
        },
        py::arg("python_entry"));

}

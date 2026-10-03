#pragma once

#include <Python.h>

#include <cstdlib>
#include <vector>

#include "TensorImpl.h"
#include "CPythonBridge.h"
#include "python_bindings.h"

namespace tensorplay { namespace python_c {

// The layout queries, hand-written as CPython method descriptors.
//
// A method reached through the type should be the operation itself: the same
// object however many times it is fetched, and hashable so it can be compared,
// put in a table and looked up.  The interpreter's method descriptors are that
// -- they are what a type whose methods are named in a method table hands out,
// and calling one passes the value as the first argument.
//
// A binding that is not a method descriptor does not give that.  What it leaves
// in the type's dictionary is a wrapper that knows the method belongs to the
// type, and a wrapper is a promise to bind an operation rather than the
// operation: two fetches are not the same object and it is not hashable at all.
// A graph records which operation a value went through, which means asking a
// recorded name whether it is the operation another node recorded and whether
// two nodes did the same thing -- questions a promise cannot be asked, because
// it is not the thing being asked about.
//
// These are therefore named in a method table.  What they answer is the layout,
// which is a fact already carried by the value rather than a computation over
// it: reading it is a couple of loads, so these release no lock and consult no
// hook layer, and the whole cost of asking is the call itself.  That matters
// here more than anywhere else, because a program asks these constantly -- once
// per shape it reasons about -- and a query that spent more time resolving the
// name of the operation that would answer it than answering would be paid for on
// every one of those questions.

// ``size()`` is the extent of every dimension, ``size(dim)`` one dimension's.
//
// The whole body is guarded: a nested tensor refuses to report a single dense
// size, and that refusal arrives as a C++ exception.  These hand-written slots
// bypass the shared pybind translator, so the exception has to be turned into
// a Python exception here or it would escape into the interpreter.
inline PyObject* tpx_size_call(PyObject* self_obj, PyObject* const* args,
                               Py_ssize_t nargs, PyObject* kwnames) {
    try {
        if (kwnames != nullptr && PyTuple_GET_SIZE(kwnames) != 0) {
            PyErr_SetString(PyExc_TypeError,
                            "size() got an unexpected keyword argument");
            return nullptr;
        }
        if (nargs > 1) {
            PyErr_SetString(PyExc_TypeError, "size() takes at most 1 argument");
            return nullptr;
        }
        const Tensor& self = tpx_py_tensor_cref(self_obj);
        if (nargs == 0) return Size_New(self.shape());
        if (PyIndex_Check(args[0]) == 0) {
            PyErr_SetString(PyExc_TypeError, "size(): dim must be an integer");
            return nullptr;
        }
        const int64_t dim = PyLong_AsLongLong(args[0]);
        if (dim == -1 && PyErr_Occurred() != nullptr) return nullptr;
        return PyLong_FromLongLong(self.size(dim));
    } catch (const std::exception& e) {
        tpx_py_set_error(e);
        return nullptr;
    }
}

// ``stride()`` is the step of every dimension, ``stride(dim)`` one dimension's.
inline PyObject* tpx_stride_call(PyObject* self_obj, PyObject* const* args,
                                 Py_ssize_t nargs, PyObject* kwnames) {
    try {
        if (kwnames != nullptr && PyTuple_GET_SIZE(kwnames) != 0) {
            PyErr_SetString(PyExc_TypeError,
                            "stride() got an unexpected keyword argument");
            return nullptr;
        }
        if (nargs > 1) {
            PyErr_SetString(PyExc_TypeError, "stride() takes at most 1 argument");
            return nullptr;
        }
        const Tensor& self = tpx_py_tensor_cref(self_obj);
        if (nargs == 1) {
            if (PyIndex_Check(args[0]) == 0) {
                PyErr_SetString(PyExc_TypeError,
                                "size(): dim must be an integer");
                return nullptr;
            }
            const int64_t dim = PyLong_AsLongLong(args[0]);

            if (dim == -1 && PyErr_Occurred() != nullptr) return nullptr;
            return PyLong_FromLongLong(self.stride(dim));
        }
        const auto strides = self.strides();
        PyObject* out = PyTuple_New(static_cast<Py_ssize_t>(strides.size()));
        if (out == nullptr) return nullptr;
        for (size_t i = 0; i < strides.size(); ++i) {
            PyObject* item = PyLong_FromLongLong(strides[i]);
            if (item == nullptr) {
                Py_DECREF(out);
                return nullptr;
            }
            PyTuple_SET_ITEM(out, static_cast<Py_ssize_t>(i), item);
        }
        return out;
    } catch (const std::exception& e) {
        tpx_py_set_error(e);
        return nullptr;
    }
}

// Restates the remaining hand-written methods as method descriptors.
//
// These already work, and reach their implementation by the shortest route a
// name can: a call goes straight to the code written for it.  What they lack is
// being the operation -- being the same object across fetches, and being
// hashable -- because what a reflected binding leaves in the type's dictionary
// is a wrapper that knows the method belongs to the type.
//
// So each name is restated as a method descriptor over the same implementation.
// The descriptor is what the name in the dictionary now is; calling it hands
// the value and the arguments to the implementation it was already reaching,
// so the work done and the way the arguments are read are unchanged.  Only the
// identity of the name changes, and that is what a recorded graph needs from
// it.


// ``numel()`` is how many elements there are.
inline PyObject* tpx_numel_call(PyObject* self_obj, PyObject* const*,
                                Py_ssize_t nargs, PyObject* kwnames) {
    if (nargs != 0 || (kwnames != nullptr && PyTuple_GET_SIZE(kwnames) != 0)) {
        PyErr_SetString(PyExc_TypeError, "numel() takes no arguments");
        return nullptr;
    }
    return PyLong_FromLongLong(tpx_py_tensor_cref(self_obj).numel());
}

// ``dim()`` is how many dimensions there are.
inline PyObject* tpx_dim_call(PyObject* self_obj, PyObject* const*,
                              Py_ssize_t nargs, PyObject* kwnames) {
    if (nargs != 0 || (kwnames != nullptr && PyTuple_GET_SIZE(kwnames) != 0)) {
        PyErr_SetString(PyExc_TypeError, "dim() takes no arguments");
        return nullptr;
    }
    return PyLong_FromLongLong(tpx_py_tensor_cref(self_obj).dim());
}

// ``storage_offset()`` is where the first element sits in the storage.
inline PyObject* tpx_storage_offset_call(PyObject* self_obj, PyObject* const*,
                                         Py_ssize_t nargs, PyObject* kwnames) {
    if (nargs != 0 || (kwnames != nullptr && PyTuple_GET_SIZE(kwnames) != 0)) {
        PyErr_SetString(PyExc_TypeError,
                        "storage_offset() takes no arguments");
        return nullptr;
    }
    const Tensor& self = tpx_py_tensor_cref(self_obj);
    if (!self.defined()) {
        // An undefined tensor has no storage to be offset into.
        PyErr_SetString(PyExc_RuntimeError,
                        "storage_offset() called on an undefined Tensor");
        return nullptr;
    }
    return PyLong_FromLongLong(static_cast<int64_t>(
        self.unsafeGetTensorImpl()->storage_offset()));
}

// ``get_device()`` is the ordinal of the device, or -1 for the host.
inline PyObject* tpx_get_device_call(PyObject* self_obj, PyObject* const*,
                                     Py_ssize_t nargs, PyObject* kwnames) {
    if (nargs != 0 || (kwnames != nullptr && PyTuple_GET_SIZE(kwnames) != 0)) {
        PyErr_SetString(PyExc_TypeError,
                        "get_device() takes no arguments");
        return nullptr;
    }
    const Device dev = tpx_py_tensor_cref(self_obj).device();
    return PyLong_FromLongLong(
        (dev.is_cuda() || dev.type() != DeviceType::CPU) ? dev.index() : -1);
}


// ``shape`` is the extent of every dimension, read as a value rather than
// asked for: a program reads it to decide what to do next, and a decision
// should not cost a call to ask.
inline PyObject* tpx_shape_get(PyObject* self_obj, void*) {
    try {
        return Size_New(tpx_py_tensor_cref(self_obj).shape());
    } catch (const std::exception& e) {
        tpx_py_set_error(e);
        return nullptr;
    }
}

// Installs the methods above under their names, replacing whatever the type
// dictionary held for them.  The definitions live in the method table, which
// outlives the type.
inline int install_layout_methods(PyObject* type_obj) {
    static PyMethodDef table[] = {
        {"size", reinterpret_cast<PyCFunction>(
                     reinterpret_cast<void (*)()>(tpx_size_call)),
         METH_FASTCALL | METH_KEYWORDS,
         "size() -> int[]\nsize(dim) -> int"},
        {"stride", reinterpret_cast<PyCFunction>(
                       reinterpret_cast<void (*)()>(tpx_stride_call)),
         METH_FASTCALL | METH_KEYWORDS,
         "stride() -> int[]\nstride(dim) -> int"},
        {"numel", reinterpret_cast<PyCFunction>(
                       reinterpret_cast<void (*)()>(tpx_numel_call)),
         METH_FASTCALL | METH_KEYWORDS, "numel() -> int"},
        {"dim", reinterpret_cast<PyCFunction>(
                     reinterpret_cast<void (*)()>(tpx_dim_call)),
         METH_FASTCALL | METH_KEYWORDS, "dim() -> int"},
        {"storage_offset", reinterpret_cast<PyCFunction>(
                               reinterpret_cast<void (*)()>(
                                   tpx_storage_offset_call)),
         METH_FASTCALL | METH_KEYWORDS, "storage_offset() -> int"},
        {"get_device", reinterpret_cast<PyCFunction>(
                           reinterpret_cast<void (*)()>(
                               tpx_get_device_call)),
         METH_FASTCALL | METH_KEYWORDS, "get_device() -> int"},
        {nullptr, nullptr, 0, nullptr},
    };
    static PyGetSetDef properties[] = {
        {const_cast<char*>("shape"), tpx_shape_get, nullptr, nullptr, nullptr},
        {nullptr, nullptr, nullptr, nullptr, nullptr},
    };
    auto* type = reinterpret_cast<PyTypeObject*>(type_obj);
    for (PyMethodDef* def = table; def->ml_name != nullptr; ++def) {
        PyObject* descr = PyDescr_NewMethod(type, def);
        if (descr == nullptr) return -1;
        int rc = PyObject_SetAttrString(type_obj, def->ml_name, descr);
        Py_DECREF(descr);
        if (rc != 0) return -1;
    }
    for (PyGetSetDef* def = properties; def->name != nullptr; ++def) {
        PyObject* descr = PyDescr_NewGetSet(type, def);
        if (descr == nullptr) return -1;
        int rc = PyObject_SetAttrString(type_obj, def->name, descr);
        Py_DECREF(descr);
        if (rc != 0) return -1;
    }
    return 0;
}



namespace detail {

// The implementation each name forwards to, indexed by the position the name
// is installed at.  A method descriptor is handed the value separately from
// the arguments, while what it forwards to takes the value as its first
// argument, so the value goes back at the front.
inline constexpr int kForwardedMethods = 21;
inline PyObject* g_forwarded[kForwardedMethods];

inline PyObject* forward(PyObject* impl, PyObject* self_obj,
                         PyObject* const* args, Py_ssize_t nargs,
                         PyObject* kwnames) {
    const Py_ssize_t nkw = kwnames == nullptr ? 0 : PyTuple_GET_SIZE(kwnames);
    const Py_ssize_t positional = nargs + 1;
    const Py_ssize_t total = positional + nkw;
    PyObject* stack[12];
    std::vector<PyObject*> heap;
    PyObject** call_args = stack;
    if (total > static_cast<Py_ssize_t>(sizeof(stack) / sizeof(stack[0]))) {
        heap.resize(static_cast<size_t>(total));
        call_args = heap.data();
    }
    call_args[0] = self_obj;
    for (Py_ssize_t i = 0; i < nargs; ++i) call_args[i + 1] = args[i];
    for (Py_ssize_t i = 0; i < nkw; ++i) {
        call_args[positional + i] = args[nargs + i];
    }
    return PyObject_Vectorcall(impl, call_args, positional, kwnames);
}

// One trampoline per name: a method descriptor is not told which name it was
// reached by, so each name gets its own function rather than one function that
// would have to look the name up again -- restating a name is precisely what
// stops it being looked up.
#define TPX_FORWARDER(index)                                                    \
    inline PyObject* forward_##index(PyObject* self_obj,                        \
                                     PyObject* const* args, Py_ssize_t nargs,   \
                                     PyObject* kwnames) {                      \
        return forward(g_forwarded[index], self_obj, args, nargs, kwnames);      \
    }

TPX_FORWARDER(0)  TPX_FORWARDER(1)  TPX_FORWARDER(2)  TPX_FORWARDER(3)
TPX_FORWARDER(4)  TPX_FORWARDER(5)  TPX_FORWARDER(6)  TPX_FORWARDER(7)
TPX_FORWARDER(8)  TPX_FORWARDER(9)  TPX_FORWARDER(10) TPX_FORWARDER(11)
TPX_FORWARDER(12) TPX_FORWARDER(13) TPX_FORWARDER(14) TPX_FORWARDER(15)
TPX_FORWARDER(16) TPX_FORWARDER(17) TPX_FORWARDER(18) TPX_FORWARDER(19)

TPX_FORWARDER(20)

#undef TPX_FORWARDER

}  // namespace detail

// Restates the remaining hand-written methods as method descriptors, in the
// order their names are listed.
inline int install_forwarded_methods(PyObject* type_obj) {
    static const char* const names[] = {
        "to", "detach", "is_contiguous", "type_as",
        "as_strided", "coalesce", "pin_memory", "requires_grad_", "retain_grad", "reshape_as",
        "detach_", "values", "dense_dim", "sparse_dim", "is_pinned",
        "is_floating_point", "is_coalesced", "col_indices", "crow_indices",
        "_indices", "_values",
    };
    using Fn = PyObject* (*)(PyObject*, PyObject* const*, Py_ssize_t,
                              PyObject*);
    static const Fn fns[] = {
        detail::forward_0,  detail::forward_1,  detail::forward_2,
        detail::forward_3,  detail::forward_4,  detail::forward_5,
        detail::forward_6,  detail::forward_7,  detail::forward_8,
        detail::forward_9,  detail::forward_10, detail::forward_11,
        detail::forward_12, detail::forward_13, detail::forward_14,
        detail::forward_15, detail::forward_16, detail::forward_17,
        detail::forward_18, detail::forward_19, detail::forward_20,
    };
    static PyMethodDef table[detail::kForwardedMethods + 1];
    static bool built = false;
    if (!built) {
        for (int i = 0; i < detail::kForwardedMethods; ++i) {
            table[i].ml_name = names[i];
            table[i].ml_meth = reinterpret_cast<PyCFunction>(
                reinterpret_cast<void (*)()>(fns[i]));
            table[i].ml_flags = METH_FASTCALL | METH_KEYWORDS;
            table[i].ml_doc = nullptr;
        }
        table[detail::kForwardedMethods].ml_name = nullptr;
        built = true;
    }

    auto* type = reinterpret_cast<PyTypeObject*>(type_obj);
    for (int i = 0; i < detail::kForwardedMethods; ++i) {
        // What the name resolves to now is the implementation to forward to.
        // It is taken before the name is replaced and kept for as long as the
        // type lives, because the descriptor needs it on every call.  It is
        PyObject* current = PyDict_GetItemString(type->tp_dict, names[i]);
        if (current == nullptr) continue;
        if (PyObject_TypeCheck(current, &PyInstanceMethod_Type) == 0) continue;
        // What is forwarded to is the operation behind the binding rather than
        // the binding's wrapper.  The wrapper knows it belongs to a type and
        // is reached through one, and is not itself something that can be
        // called; the operation is what takes the value as its first argument,
        // which is what the forwarder passes it.
        PyObject* operation = PyInstanceMethod_GET_FUNCTION(current);  // borrowed
        if (operation == nullptr) continue;
        Py_INCREF(operation);
        detail::g_forwarded[i] = operation;
        PyObject* descr = PyDescr_NewMethod(type, &table[i]);
        if (descr == nullptr) {
            Py_DECREF(operation);
            PyErr_Clear();
            continue;
        }
        int rc = PyObject_SetAttrString(type_obj, names[i], descr);
        Py_DECREF(descr);
        if (rc != 0) {
            if (std::getenv("TP_DEBUG_DESC") != nullptr) PyErr_Print();
            PyErr_Clear();
        }
    }
    return 0;
}

}}  // namespace tensorplay::python_c

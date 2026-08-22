// CPythonBridge.cpp -- implementation of the conversion surface declared in
// CPythonBridge.h.  Lives inside the _C extension target so pybind11's
// registered type casters for Tensor/Scalar/DType are available; the
// generated METH_FASTCALL layer (TensorCPythonGenerated.h) calls these
// helpers and therefore never touches pybind11 dispatch itself.
#include "CPythonBridge.h"

#include <pybind11/pybind11.h>

#include <cstring>
#include <stdexcept>
#include <tuple>

namespace py = ::pybind11;

namespace tensorplay {
namespace python_c {

namespace {

[[noreturn]] void type_error(const char* op, int index, const char* want) {
    std::string msg = std::string(op) + ": argument " + std::to_string(index)
                      + " must be " + want;
    throw std::invalid_argument(msg);
}

}  // namespace

// ---------------------------------------------------------------------------
// tpx_py_parse: merge positional args and keyword names into kwlist order.
// ---------------------------------------------------------------------------

ParsedArgs tpx_py_parse(PyObject* const* args, Py_ssize_t nargs,
                        PyObject* kwnames, const char* const* kwlist,
                        Py_ssize_t nkws, const char* op_name) {
    ParsedArgs out;
    out.owned.assign(static_cast<size_t>(nkws), nullptr);

    if (nargs > nkws) {
        std::string msg = std::string(op_name)
                          + ": too many positional arguments";
        throw std::invalid_argument(msg);
    }
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        out.owned[static_cast<size_t>(i)] = args[i];
    }

    if (kwnames != nullptr && !PyTuple_CheckExact(kwnames)) {
        throw std::invalid_argument("internal: kwnames is not a tuple");
    }
    Py_ssize_t nkw = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; ++i) {
        PyObject* key = PyTuple_GET_ITEM(kwnames, i);
        const char* name = PyUnicode_AsUTF8(key);
        if (!name) throw std::invalid_argument("internal: bad kwname");
        int slot = -1;
        for (Py_ssize_t k = 0; k < nkws; ++k) {
            if (kwlist[k] && std::strcmp(kwlist[k], name) == 0) { slot = static_cast<int>(k); break; }
        }
        if (slot < 0) {
            std::string msg = std::string(op_name)
                              + ": unexpected keyword argument '" + name + "'";
            throw std::invalid_argument(msg);
        }
        size_t u = static_cast<size_t>(slot);
        // METH_FASTCALL convention: keyword values follow the positional
        // ones inside args[], parallel to kwnames.
        if (out.owned[u] != nullptr) {
            std::string msg = std::string(op_name)
                              + ": got multiple values for argument '" + name + "'";
            throw std::invalid_argument(msg);
        }
        out.owned[u] = args[nargs + i];
    }
    return out;
}

// ---------------------------------------------------------------------------
// unpacking
// ---------------------------------------------------------------------------

namespace {

Tensor as_tensor(PyObject* obj, const char* op, int idx) {
    try {
        return py::reinterpret_borrow<py::object>(obj).cast<Tensor>();
    } catch (const py::cast_error&) {
        type_error(op, idx, "a Tensor");
    }
}

Scalar as_scalar(PyObject* obj, const char* op, int idx) {
    try {
        return py::reinterpret_borrow<py::object>(obj).cast<Scalar>();
    } catch (const py::cast_error&) {
        type_error(op, idx, "a Scalar");
    }
}

DType as_dtype(PyObject* obj, const char* op, int idx) {
    try {
        return py::reinterpret_borrow<py::object>(obj).cast<DType>();
    } catch (const py::cast_error&) {
        type_error(op, idx, "a DType");
    }
}

int64_t as_int(PyObject* obj, const char* op, int idx) {
    if (!PyIndex_Check(obj)) type_error(op, idx, "an integer");
    return static_cast<int64_t>(PyNumber_AsSsize_t(obj, nullptr));
}

double as_double(PyObject* obj, const char* op, int idx) {
    double v = PyFloat_AsDouble(obj);
    if (v == -1.0 && PyErr_Occurred()) {
        PyErr_Clear();
        type_error(op, idx, "a float");
    }
    return v;
}

}  // namespace

Tensor tpx_py_tensor(PyObject* obj) { return as_tensor(obj, "op", 0); }
Scalar tpx_py_scalar(PyObject* obj) { return as_scalar(obj, "op", 0); }
std::optional<Tensor> tpx_py_opt_tensor(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return as_tensor(obj, "op", 0);
}
int64_t tpx_py_int64(PyObject* obj) { return as_int(obj, "op", 0); }
double tpx_py_double(PyObject* obj) { return as_double(obj, "op", 0); }
bool tpx_py_bool(PyObject* obj) { return PyObject_IsTrue(obj) == 1; }
std::optional<int64_t> tpx_py_opt_int64(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return as_int(obj, "op", 0);
}
std::optional<double> tpx_py_opt_double(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return as_double(obj, "op", 0);
}
std::optional<bool> tpx_py_opt_bool(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return PyObject_IsTrue(obj) == 1;
}
std::optional<Scalar> tpx_py_opt_scalar(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return as_scalar(obj, "op", 0);
}
std::vector<int64_t> tpx_py_intlist(PyObject* obj) {
    std::vector<int64_t> r;
    if (PyLong_Check(obj)) { r.push_back(as_int(obj, "op", 0)); return r; }
    PyObject* seq = PySequence_Fast(obj, "expected a sequence of integers");
    if (!seq) { PyErr_Clear(); throw std::invalid_argument("expected a sequence of integers"); }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    r.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        r.push_back(as_int(PySequence_Fast_GET_ITEM(seq, i), "op", static_cast<int>(i)));
    }
    Py_DECREF(seq);
    return r;
}
std::string tpx_py_string(PyObject* obj) {
    const char* s = PyUnicode_AsUTF8(obj);
    if (!s) { PyErr_Clear(); throw std::invalid_argument("expected a string"); }
    return std::string(s);
}
DType tpx_py_dtype(PyObject* obj) { return as_dtype(obj, "op", 0); }
std::optional<DType> tpx_py_opt_dtype(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return as_dtype(obj, "op", 0);
}

// ---------------------------------------------------------------------------
// packing
// ---------------------------------------------------------------------------

PyObject* tpx_py_wrap(const Tensor& t) {
    return py::cast(t).release().ptr();
}
PyObject* tpx_py_wrap_tuple(const std::tuple<Tensor, Tensor>& t) {
    PyObject* a = tpx_py_wrap(std::get<0>(t));
    PyObject* b = tpx_py_wrap(std::get<1>(t));
    PyObject* tup = PyTuple_Pack(2, a, b);
    Py_XDECREF(a); Py_XDECREF(b);
    return tup;
}
PyObject* tpx_py_wrap_list(const std::vector<Tensor>& v) {
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(v.size()));
    for (size_t i = 0; i < v.size(); ++i) {
        // Steals a reference on success; wrap() gives us a new reference.
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), tpx_py_wrap(v[i]));
    }
    return list;
}

void tpx_py_keep_alive(PyObject*) {
    // No-op by construction: p10 views share their base's Storage
    // (shared_ptr) and VariableVersion (shared counter), so the returned
    // alias keeps both alive without an explicit keep-alive edge.  This is
    // where upstream installs view metadata once saved-view replay lands.
}

void tpx_py_set_error(const std::exception& e) {
    if (PyErr_Occurred()) return;  // already translated deeper down
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

}  // namespace python_c
}  // namespace tensorplay

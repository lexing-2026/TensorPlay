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
#include <unordered_map>

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
// tpx_py_parse[_into]: merge positional args and keyword names into kwlist
// order.  parse_into is the allocation-free core; parse wraps it for callers
// that want the ParsedArgs owner.
// ---------------------------------------------------------------------------

void tpx_py_parse_into(PyObject* const* args, Py_ssize_t nargs,
                       PyObject* kwnames, const char* const* kwlist,
                       Py_ssize_t nkws, const char* op_name,
                       PyObject** out) {
    for (Py_ssize_t i = 0; i < nkws; ++i) out[i] = nullptr;

    if (nargs > nkws) {
        std::string msg = std::string(op_name)
                          + ": too many positional arguments";
        throw std::invalid_argument(msg);
    }
    if (nargs > 0) {
        std::memcpy(out, args, static_cast<size_t>(nargs) * sizeof(PyObject*));
    }

    if (kwnames != nullptr && !PyTuple_CheckExact(kwnames)) {
        throw std::invalid_argument("internal: kwnames is not a tuple");
    }
    Py_ssize_t nkw = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    for (Py_ssize_t i = 0; i < nkw; ++i) {
        PyObject* key = PyTuple_GET_ITEM(kwnames, i);
        int slot = -1;
        for (Py_ssize_t k = 0; k < nkws; ++k) {
            // Identity-friendly ASCII compare; unlike PyUnicode_AsUTF8 this
            // never materializes a UTF-8 buffer for the key.
            if (kwlist[k]
                && PyUnicode_CompareWithASCIIString(key, kwlist[k]) == 0) {
                slot = static_cast<int>(k);
                break;
            }
        }
        if (slot < 0) {
            const char* name = PyUnicode_AsUTF8(key);
            std::string msg = std::string(op_name)
                              + ": unexpected keyword argument '"
                              + (name ? name : "?") + "'";
            throw std::invalid_argument(msg);
        }
        size_t u = static_cast<size_t>(slot);
        // METH_FASTCALL convention: keyword values follow the positional
        // ones inside args[], parallel to kwnames.
        if (out[u] != nullptr) {
            const char* name = PyUnicode_AsUTF8(key);
            std::string msg = std::string(op_name)
                              + ": got multiple values for argument '"
                              + (name ? name : "?") + "'";
            throw std::invalid_argument(msg);
        }
        out[u] = args[nargs + i];
    }
}

ParsedArgs tpx_py_parse(PyObject* const* args, Py_ssize_t nargs,
                        PyObject* kwnames, const char* const* kwlist,
                        Py_ssize_t nkws, const char* op_name) {
    ParsedArgs out;
    out.owned.resize(static_cast<size_t>(nkws));
    tpx_py_parse_into(args, nargs, kwnames, kwlist, nkws, op_name,
                      out.owned.data());
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
    // Fast paths for the overwhelmingly common number cases; complex and
    // exotic inputs fall through to the pybind caster.
    if (PyBool_Check(obj)) return Scalar(obj == Py_True);
    if (PyLong_Check(obj)) {
        long long v = PyLong_AsLongLong(obj);
        if (v != -1 || !PyErr_Occurred()) return Scalar(static_cast<int64_t>(v));
        PyErr_Clear();  // overflow: let the caster decide
        try {
            return py::reinterpret_borrow<py::object>(obj).cast<Scalar>();
        } catch (const py::cast_error&) {
            type_error(op, idx, "a Scalar");
        }
    }
    if (PyFloat_Check(obj)) return Scalar(PyFloat_AS_DOUBLE(obj));
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
    // AsSsize_t with an error-raising sentinel: without the check an
    // out-of-range value would silently saturate and the kernel would run
    // with garbage before anyone noticed the pending exception.
    int64_t v = PyNumber_AsSsize_t(obj, nullptr);
    if (v == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        std::string msg = std::string(op) + ": argument " + std::to_string(idx)
                          + " integer out of range";
        throw std::invalid_argument(msg);
    }
    return v;
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

namespace {

// PyTypeObject of the registered Tensor wrapper, populated on first
// slow-path cast.  With it, later unwraps take the direct-instance fast
// path below instead of paying a registry lookup per argument.
PyTypeObject* g_tensor_type = nullptr;

const Tensor& tensor_cref_slow(PyObject* obj) {
    try {
        const Tensor& t =
            py::cast<const Tensor&>(py::reinterpret_borrow<py::object>(obj));
        if (g_tensor_type == nullptr) g_tensor_type = Py_TYPE(obj);
        return t;
    } catch (const py::cast_error&) {
        type_error("op", 0, "a Tensor");
    }
}

Tensor& tensor_mref_slow(PyObject* obj) {
    try {
        Tensor& t = py::cast<Tensor&>(py::reinterpret_borrow<py::object>(obj));
        if (g_tensor_type == nullptr) g_tensor_type = Py_TYPE(obj);
        return t;
    } catch (const py::cast_error&) {
        type_error("op", 0, "a Tensor");
    }
}

}  // namespace

Tensor tpx_py_tensor(PyObject* obj) { return as_tensor(obj, "op", 0); }

const Tensor& tpx_py_tensor_cref(PyObject* obj) {
    if (g_tensor_type != nullptr && PyObject_TypeCheck(obj, g_tensor_type)) {
        // Registered Tensor wrappers are pybind11 `instance` objects; with
        // the simple layout the C++ value sits at value_holder[0].  This is
        // the THPVariable-style direct access, minus the registry hop.
        auto* inst = reinterpret_cast<py::detail::instance*>(obj);
        if (inst->simple_layout && inst->simple_value_holder[0] != nullptr) {
            return *static_cast<const Tensor*>(inst->simple_value_holder[0]);
        }
    }
    return tensor_cref_slow(obj);
}

Tensor& tpx_py_tensor_mref(PyObject* obj) {
    if (g_tensor_type != nullptr && PyObject_TypeCheck(obj, g_tensor_type)) {
        auto* inst = reinterpret_cast<py::detail::instance*>(obj);
        if (inst->simple_layout && inst->simple_value_holder[0] != nullptr) {
            return *static_cast<Tensor*>(inst->simple_value_holder[0]);
        }
    }
    return tensor_mref_slow(obj);
}

Scalar tpx_py_scalar(PyObject* obj) { return as_scalar(obj, "op", 0); }
std::optional<Tensor> tpx_py_opt_tensor(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return as_tensor(obj, "op", 0);
}
int64_t tpx_py_int64(PyObject* obj) { return as_int(obj, "op", 0); }
double tpx_py_double(PyObject* obj) { return as_double(obj, "op", 0); }
bool tpx_py_bool(PyObject* obj) {
    // Only real bools, matching the pybind11 bool caster the m.def surface
    // enforced (and upstream's PythonArgParser): truthiness of arbitrary
    // objects is a silent behavior change.
    if (PyBool_Check(obj)) return obj == Py_True;
    type_error("op", 0, "a bool");
}
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
    return tpx_py_bool(obj);
}
std::optional<Scalar> tpx_py_opt_scalar(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return as_scalar(obj, "op", 0);
}
std::optional<Device> tpx_py_opt_device(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    try {
        return py::cast<Device>(py::reinterpret_borrow<py::object>(obj));
    } catch (const py::cast_error&) {
        type_error("op", 0, "a Device");
    }
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
std::vector<double> tpx_py_doublelist(PyObject* obj) {
    std::vector<double> r;
    if (PyFloat_Check(obj)) { r.push_back(as_double(obj, "op", 0)); return r; }
    if (PyLong_Check(obj)) { r.push_back(as_double(obj, "op", 0)); return r; }
    PyObject* seq = PySequence_Fast(obj, "expected a sequence of numbers");
    if (!seq) { PyErr_Clear(); throw std::invalid_argument("expected a sequence of numbers"); }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    r.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        r.push_back(as_double(PySequence_Fast_GET_ITEM(seq, i), "op", static_cast<int>(i)));
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

namespace {

// Identity cache for returned wrappers, keyed by TensorImpl pointer: like
// upstream THPVariable_Wrap, re-wrapping the same impl yields the *same*
// Python object instead of a fresh copy each call.  Invalidation needs no
// p10 changes -- each cached object carries an attribute capsule whose
// destructor runs when the wrapper dies and erases the entry.  The map
// stores borrowed pointers only (the owning reference is the object's own);
// all access happens under the GIL.
std::unordered_map<const void*, PyObject*> g_wrap_cache;

void wrap_cache_capsule_destructor(PyObject* caps) {
    g_wrap_cache.erase(PyCapsule_GetPointer(caps, nullptr));
}

constexpr char kWrapCacheAttr[] = "__tp_impl_capsule__";

}  // namespace

PyObject* tpx_py_wrap(const Tensor& t) {
    const void* impl = t.unsafeGetTensorImpl().get();
    auto it = g_wrap_cache.find(impl);
    if (it != g_wrap_cache.end()) {
        Py_INCREF(it->second);
        return it->second;
    }
    PyObject* wrapped = py::cast(t).release().ptr();
    if (wrapped == nullptr) return nullptr;

    // dynamic_attr wrappers can carry the invalidation capsule; anything
    // else (should not happen for Tensor) skips caching gracefully.
    PyObject* caps = PyCapsule_New(const_cast<void*>(impl), nullptr,
                                   &wrap_cache_capsule_destructor);
    if (caps != nullptr) {
        if (PyObject_SetAttrString(wrapped, kWrapCacheAttr, caps) == 0) {
            g_wrap_cache.emplace(impl, wrapped);  // borrowed; owned by obj
        }
        Py_DECREF(caps);  // object holds its own reference via the attribute
    }
    return wrapped;
}
PyObject* tpx_py_wrap_scalar(const Scalar& s) {
    return py::cast(s).release().ptr();
}
PyObject* tpx_py_wrap_dtype(const DType& dt) {
    return py::cast(dt).release().ptr();
}
PyObject* tpx_py_wrap_device(const Device& d) {
    return py::cast(d).release().ptr();
}
PyObject* tpx_py_wrap_tuple(const std::tuple<Tensor, Tensor>& t) {
    PyObject* a = tpx_py_wrap(std::get<0>(t));
    PyObject* b = tpx_py_wrap(std::get<1>(t));
    PyObject* tup = PyTuple_Pack(2, a, b);
    Py_XDECREF(a); Py_XDECREF(b);
    return tup;
}
PyObject* tpx_py_wrap_tuple3(const std::tuple<Tensor, Tensor, Tensor>& t) {
    PyObject* a = tpx_py_wrap(std::get<0>(t));
    PyObject* b = tpx_py_wrap(std::get<1>(t));
    PyObject* c = tpx_py_wrap(std::get<2>(t));
    PyObject* tup = PyTuple_Pack(3, a, b, c);
    Py_XDECREF(a); Py_XDECREF(b); Py_XDECREF(c);
    return tup;
}
PyObject* tpx_py_wrap_tuple4(
    const std::tuple<Tensor, Tensor, Tensor, Tensor>& t) {
    PyObject* a = tpx_py_wrap(std::get<0>(t));
    PyObject* b = tpx_py_wrap(std::get<1>(t));
    PyObject* c = tpx_py_wrap(std::get<2>(t));
    PyObject* d = tpx_py_wrap(std::get<3>(t));
    PyObject* tup = PyTuple_Pack(4, a, b, c, d);
    Py_XDECREF(a); Py_XDECREF(b); Py_XDECREF(c); Py_XDECREF(d);
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
    // Mirror upstream HANDLE_TH_ERRORS: p10 exception kinds map onto their
    // matching Python builtins instead of flattening to RuntimeError.
    // Bridge argument-shape errors (std::invalid_argument) read as TypeError,
    // the builtin callers expect for bad arguments.
    if (dynamic_cast<const std::invalid_argument*>(&e)) {
        PyErr_SetString(PyExc_TypeError, e.what());
    } else if (dynamic_cast<const IndexError*>(&e)) {
        PyErr_SetString(PyExc_IndexError, e.what());
    } else if (dynamic_cast<const ValueError*>(&e)) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } else if (dynamic_cast<const TypeError*>(&e)) {
        PyErr_SetString(PyExc_TypeError, e.what());
    } else if (dynamic_cast<const NotImplementedError*>(&e)) {
        PyErr_SetString(PyExc_NotImplementedError, e.what());
    } else {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

}  // namespace python_c
}  // namespace tensorplay

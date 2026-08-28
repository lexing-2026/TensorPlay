// CPythonBridge.cpp -- implementation of the conversion surface declared in
// CPythonBridge.h.  Lives inside the _C extension target so pybind11's
// registered type casters for Tensor/Scalar/DType are available; the
// generated METH_FASTCALL layer (TensorCPythonGenerated.h) calls these
// helpers and therefore never touches pybind11 dispatch itself.
#include "CPythonBridge.h"

#include <pybind11/pybind11.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace py = ::pybind11;

// Shared exception translator (defined at the bottom of this file so it can
// serve both this fastcall bridge and the pybind11 translator in init.cpp).
namespace tensorplay {
class Exception;
PyObject* translate_exception(const Exception& e);
} // namespace tensorplay

namespace tensorplay {
namespace python_c {

namespace {

// torch python_arg_parser parity suffix: "... must be int, not tuple".
[[noreturn]] void type_error(PyObject* obj, const char* op, int index,
                             const char* want) {
    const char* got = obj ? Py_TYPE(obj)->tp_name : "None";
    std::string msg = std::string(op) + ": argument " + std::to_string(index)
                      + " must be " + want + ", not " + got;
    throw std::invalid_argument(msg);
}

}  // namespace
}  // namespace python_c

namespace {
// Borrowed pointer to the registered Python DeviceMismatchError type (set by
// init.cpp during module init). Null until then -> plain RuntimeError.
PyObject* g_device_mismatch_type = nullptr;
}  // namespace

// Single source of truth for C++ -> Python error mapping (see
// python_bindings.h).
PyObject* translate_exception(const Exception& e) {
    if (dynamic_cast<const IndexError*>(&e)) return PyExc_IndexError;
    if (dynamic_cast<const ValueError*>(&e)) return PyExc_ValueError;
    if (dynamic_cast<const TypeError*>(&e)) return PyExc_TypeError;
    if (dynamic_cast<const NotImplementedError*>(&e)) {
        return PyExc_NotImplementedError;
    }
    if (dynamic_cast<const DeviceMismatchError*>(&e)) {
        return g_device_mismatch_type ? g_device_mismatch_type : PyExc_RuntimeError;
    }
    return PyExc_RuntimeError;
}

void set_device_mismatch_error_type(PyObject* type) {
    g_device_mismatch_type = type;
}

} // namespace tensorplay

namespace tensorplay {

namespace python_c {

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
        type_error(obj, op, idx, "a Tensor");
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
            type_error(obj, op, idx, "a Scalar");
        }
    }
    if (PyFloat_Check(obj)) return Scalar(PyFloat_AS_DOUBLE(obj));
    if (PyComplex_Check(obj)) {
        // torch parity: a wrapped python complex becomes a complex128 scalar.
        return Scalar(std::complex<double>(PyComplex_RealAsDouble(obj),
                                           PyComplex_ImagAsDouble(obj)));
    }
    try {
        return py::reinterpret_borrow<py::object>(obj).cast<Scalar>();
    } catch (const py::cast_error&) {
        type_error(obj, op, idx, "a Scalar");
    }
}

DType as_dtype(PyObject* obj, const char* op, int idx) {
    try {
        return py::reinterpret_borrow<py::object>(obj).cast<DType>();
    } catch (const py::cast_error&) {
        type_error(obj, op, idx, "a DType");
    }
}

int64_t as_int(PyObject* obj, const char* op, int idx) {
    // Upstream PythonArgParser accepts integral-valued floats for int slots
    // (e.g. divisor_override=3.0); non-integral floats still raise.
    if (PyFloat_Check(obj)) {
        const double d = PyFloat_AS_DOUBLE(obj);
        if (static_cast<double>(static_cast<int64_t>(d)) == d) {
            return static_cast<int64_t>(d);
        }
        type_error(obj, op, idx, "an integer");
    }
    if (!PyIndex_Check(obj)) type_error(obj, op, idx, "an integer");
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
        type_error(obj, op, idx, "a float");
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
        type_error(obj, "op", 0, "a Tensor");
    }
}

Tensor& tensor_mref_slow(PyObject* obj) {
    try {
        Tensor& t = py::cast<Tensor&>(py::reinterpret_borrow<py::object>(obj));
        if (g_tensor_type == nullptr) g_tensor_type = Py_TYPE(obj);
        return t;
    } catch (const py::cast_error&) {
        type_error(obj, "op", 0, "a Tensor");
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
    type_error(obj, "op", 0, "a bool");
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
        type_error(obj, "op", 0, "a Device");
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
std::optional<std::string> tpx_py_opt_string(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return tpx_py_string(obj);
}
std::optional<std::vector<int64_t>> tpx_py_opt_intlist(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return tpx_py_intlist(obj);
}

// ---------------------------------------------------------------------------
// eager typed validation: torch python_arg_parser parity for error wording
// ---------------------------------------------------------------------------

namespace {

[[noreturn]] void arg_type_error(const char* op_name, const char* name,
                                 int index, const char* want, PyObject* obj) {
    std::string msg = std::string(op_name) + "(): argument '" + (name ? name : "?")
                      + "'";
    // Upstream annotates a position only for arguments that can be passed
    // positionally; kw-only args get the bare form.
    if (index >= 0) {
        msg += " (position " + std::to_string(index + 1) + ")";
    }
    msg += std::string(" must be ") + want + ", not "
           + (obj ? Py_TYPE(obj)->tp_name : "NoneType");
    throw std::invalid_argument(msg);
}

bool obj_is_tensor(PyObject* obj) {
    if (g_tensor_type && Py_TYPE(obj) == g_tensor_type) return true;
    try {
        return py::isinstance<tensorplay::Tensor>(py::handle(obj));
    } catch (...) {
        return false;
    }
}

bool seq_item_is_number(PyObject* o) {
    // Python numbers plus the registered tensorplay.Scalar wrapper, which
    // generated wrappers (e.g. addmm's beta/alpha) pass through directly.
    // Complex numbers count as Number for python_arg_parser parity.
    if (PyIndex_Check(o) || PyFloat_Check(o) || PyComplex_Check(o)) return true;
    try {
        return py::isinstance<tensorplay::Scalar>(py::handle(o));
    } catch (...) {
        return false;
    }
}

// INT_LIST / FLOAT_LIST upstream semantics: a bare scalar folds to a
// singleton list, otherwise only tuple/list qualify -- plus tensorplay._C.Size,
// a pybind sequence that mirrors torch.Size (a tuple subclass upstream).
bool check_list(PyObject* obj, bool integral) {
    bool single = integral ? PyIndex_Check(obj) != 0 : seq_item_is_number(obj);
    if (single) return true;
    if (!PySequence_Check(obj)) return false;
    PyObject* seq = PySequence_Fast(obj, nullptr);
    if (!seq) { PyErr_Clear(); return false; }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    bool ok = true;
    for (Py_ssize_t i = 0; i < n && ok; ++i) {
        PyObject* it = PySequence_Fast_GET_ITEM(seq, i);
        ok = integral ? PyIndex_Check(it) != 0 : seq_item_is_number(it);
    }
    Py_DECREF(seq);
    return ok;
}

const char* kind_want(unsigned char k) {
    switch (static_cast<tpx_py_type_kind>(k & ~TPK_OPTIONAL)) {
        case TPK_TENSOR:     return "Tensor";
        case TPK_NUMBER:     return "Number";
        case TPK_INT:        return "int";
        case TPK_FLOAT:      return "float";
        case TPK_BOOL:       return "bool";
        case TPK_STR:        return "str";
        case TPK_DTYPE:      return "dtype";
        case TPK_DEVICE:     return "Device";
        case TPK_INTLIST:    return "int[]";
        case TPK_FLOATLIST:  return "float[]";
        case TPK_TENSORLIST: return "Tensor[]";
        case TPK_SCALARLIST: return "Number[]";
    }
    return "?";
}

}  // namespace

// Non-throwing kind predicate shared by tpx_py_check_types and the generated
// multi-overload fast probe (tools/codegen/gen_python_c.py): picking the
// right overload by kind avoids throwing/catching std::invalid_argument on
// every scalar-argument call (mul_(1.0) etc.).
bool tpx_py_obj_matches_kind(PyObject* obj, unsigned char kind) {
    if (obj == nullptr) return false;
    if (obj == Py_None) return (kind & TPK_OPTIONAL) != 0;
    switch (static_cast<tpx_py_type_kind>(kind & ~TPK_OPTIONAL)) {
        case TPK_TENSOR:     return obj_is_tensor(obj);
        case TPK_NUMBER:     return seq_item_is_number(obj);
        case TPK_INT:
            // Upstream toInt() accepts floats with an exact integral value.
            if (PyIndex_Check(obj)) return true;
            return PyFloat_Check(obj) &&
                   std::fmod(PyFloat_AS_DOUBLE(obj), 1.0) == 0;
        case TPK_FLOAT:      return PyFloat_Check(obj) || PyIndex_Check(obj);
        case TPK_BOOL:       return PyBool_Check(obj);
        case TPK_STR:        return PyUnicode_Check(obj);
        case TPK_DTYPE:
        case TPK_DEVICE:     return true;  // enum casters own these
        case TPK_INTLIST:    return check_list(obj, true);
        case TPK_FLOATLIST:  return check_list(obj, false);
        case TPK_TENSORLIST:
            if (!PyTuple_Check(obj) && !PyList_Check(obj)) return false;
            {
                Py_ssize_t m = PyTuple_Check(obj) ? PyTuple_GET_SIZE(obj)
                                                  : PyList_GET_SIZE(obj);
                for (Py_ssize_t j = 0; j < m; ++j) {
                    PyObject* el = PyTuple_Check(obj)
                                       ? PyTuple_GET_ITEM(obj, j)
                                       : PyList_GET_ITEM(obj, j);
                    if (!obj_is_tensor(el)) return false;
                }
                return true;
            }
        case TPK_SCALARLIST: return check_list(obj, false);
    }
    return false;
}

void tpx_py_check_types(PyObject* const* slots, Py_ssize_t n,
                        const char* op_name, const char* const* names,
                        const unsigned char* kinds, int max_pos) {
    auto fail = [&](int i, PyObject* obj) {
        unsigned char k = kinds[i];
        int pos = (i < max_pos) ? i : -1;
        arg_type_error(op_name, names[i], pos, kind_want(k), obj);
    };
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* obj = slots[i];
        if (obj == nullptr || obj == Py_None) continue;  // absent/optional-None
        if (!tpx_py_obj_matches_kind(obj, kinds[i])) fail(static_cast<int>(i), obj);
    }
}


// Upstream parity (python_arg_parser.cpp is_tensor_list_and_append_overloaded):
// only tuple/list qualify; each element must be a Tensor wrapper.  Element
// errors stay std::invalid_argument so multi-overload dispatch can fall
// through to the next candidate signature.
std::vector<Tensor> tpx_py_tensorlist(PyObject* obj) {
    std::vector<Tensor> r;
    if (!PyTuple_Check(obj) && !PyList_Check(obj)) {
        throw std::invalid_argument("expected a sequence of tensors");
    }
    Py_ssize_t n = PyTuple_Check(obj) ? PyTuple_GET_SIZE(obj)
                                      : PyList_GET_SIZE(obj);
    r.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PyTuple_Check(obj) ? PyTuple_GET_ITEM(obj, i)
                                            : PyList_GET_ITEM(obj, i);
        try {
            r.push_back(tpx_py_tensor_cref(item));
        } catch (const std::invalid_argument&) {
            throw std::invalid_argument(std::string("expected Tensor as element ") +
                                        std::to_string(i) + ", but got " +
                                        Py_TYPE(item)->tp_name);
        }
    }
    return r;
}
std::vector<Scalar> tpx_py_scalarlist(PyObject* obj) {
    std::vector<Scalar> r;
    if (!PyTuple_Check(obj) && !PyList_Check(obj)) {
        throw std::invalid_argument("expected a sequence of scalars");
    }
    Py_ssize_t n = PyTuple_Check(obj) ? PyTuple_GET_SIZE(obj)
                                      : PyList_GET_SIZE(obj);
    r.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PyTuple_Check(obj) ? PyTuple_GET_ITEM(obj, i)
                                            : PyList_GET_ITEM(obj, i);
        r.push_back(tpx_py_scalar(item));
    }
    return r;
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

    // Undefined tensors carry a null impl; the invalidation capsule is just
    // a caching optimization, so skip it instead of tripping the
    // "PyCapsule_New called with null pointer" interpreter error.
    PyObject* caps = impl != nullptr
        ? PyCapsule_New(const_cast<void*>(impl), nullptr,
                        &wrap_cache_capsule_destructor)
        : nullptr;
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
    // Bridge argument-shape errors read as TypeError, the builtin callers
    // expect for bad arguments.
    if (dynamic_cast<const std::invalid_argument*>(&e)) {
        PyErr_SetString(PyExc_TypeError, e.what());
        return;
    }
    // p10 exception kinds map onto their matching Python types via the same
    // translator the pybind11 path uses (incl. DeviceMismatchError and the
    // TENSORPLAY_SHOW_CPP_STACKTRACES switch) instead of flattening
    // everything to RuntimeError.
    if (const tensorplay::Exception* tp = dynamic_cast<const tensorplay::Exception*>(&e)) {
        std::string msg = tp->msg();
        const char* env_val = std::getenv("TENSORPLAY_SHOW_CPP_STACKTRACES");
        if (env_val && std::string(env_val) == "1" && !tp->stacktrace().empty()) {
            msg += "\n\n" + tp->stacktrace();
        }
        PyErr_SetString(tensorplay::translate_exception(*tp), msg.c_str());
        return;
    }
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

}  // namespace python_c
}  // namespace tensorplay

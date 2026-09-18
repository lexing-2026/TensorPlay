#include "PythonDispatch.h"

#include <string>
#include <vector>

#include "Dispatcher.h"
#include "Exception.h"
#include "python_bindings.h"

namespace tensorplay {
namespace python_dispatch {
namespace {

void release_pyobject(void* object) {
    if (object == nullptr || !Py_IsInitialized()) return;
    PyGILState_STATE gil = PyGILState_Ensure();
    Py_DECREF(static_cast<PyObject*>(object));
    PyGILState_Release(gil);
}

PyObject* resolve_overload(const OpEntry& entry) {
    if (entry.overload != nullptr) return entry.overload;
    PyObject* module = PyImport_ImportModule("tensorplay._ops");
    if (module == nullptr) throw python_c::PythonError();
    PyObject* overload = PyObject_CallMethod(
        module, "_overload_for_dispatch", "s", entry.name);
    Py_DECREF(module);
    if (overload == nullptr) throw python_c::PythonError();
    // Kept for the process lifetime: entries are static and overloads are
    // interned per name, so a racing thread storing the same object is fine.
    entry.overload = overload;
    return overload;
}

bool is_dispatch_subclass(PyObject* value) {
    PyObject* type = reinterpret_cast<PyObject*>(Py_TYPE(value));
    const int has = PyObject_HasAttrString(type, "__tensorplay_dispatch__");
    return has > 0;
}

// Tensor subclass types that define __tensorplay_dispatch__, most-derived
// first; unrelated types keep first-seen order.
void collect_types(PyObject* value, std::vector<PyObject*>& types) {
    if (PyList_Check(value) || PyTuple_Check(value)) {
        PyObject* seq = PySequence_Fast(value, "argument");
        if (seq == nullptr) throw python_c::PythonError();
        const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
        PyObject** items = PySequence_Fast_ITEMS(seq);
        for (Py_ssize_t i = 0; i < n; ++i) collect_types(items[i], types);
        Py_DECREF(seq);
        return;
    }
    if (value == Py_None || PyLong_Check(value) || PyFloat_Check(value) ||
        PyBool_Check(value) || PyUnicode_Check(value)) {
        return;
    }
    if (!is_dispatch_subclass(value)) return;
    PyObject* type = reinterpret_cast<PyObject*>(Py_TYPE(value));
    for (PyObject* seen : types) {
        if (seen == type) return;
    }
    size_t index = types.size();
    for (size_t i = 0; i < types.size(); ++i) {
        const int derived = PyObject_IsSubclass(type, types[i]);
        if (derived < 0) throw python_c::PythonError();
        if (derived) {
            index = i;
            break;
        }
    }
    types.insert(types.begin() + static_cast<std::ptrdiff_t>(index), type);
}

}  // namespace

ModeCall::ModeCall(const OpEntry& entry)
    : entry_(entry), exclude_(DispatchKeySet::above(DispatchKey::Python)) {
    args_ = PyList_New(entry.num_args);
    if (args_ == nullptr) throw python_c::PythonError();
}

ModeCall::~ModeCall() {
    Py_XDECREF(args_);
}

void ModeCall::set_arg(int index, PyObject* value) {
    if (value == nullptr) throw python_c::PythonError();
    PyList_SET_ITEM(args_, index, value);
}

PyObject* ModeCall::invoke() {
    PyObject* func = resolve_overload(entry_);

    PyObject* args = PyTuple_New(entry_.num_positional);
    if (args == nullptr) throw python_c::PythonError();
    PyObject* kwargs = PyDict_New();
    if (kwargs == nullptr) {
        Py_DECREF(args);
        throw python_c::PythonError();
    }
    struct Owned {
        PyObject* args;
        PyObject* kwargs;
        PyObject* types = nullptr;
        ~Owned() {
            Py_XDECREF(args);
            Py_XDECREF(kwargs);
            Py_XDECREF(types);
        }
    } owned{args, kwargs};

    std::vector<PyObject*> type_list;
    for (int i = 0; i < entry_.num_args; ++i) {
        PyObject* value = PyList_GET_ITEM(args_, i);
        collect_types(value, type_list);
        if (i < entry_.num_positional) {
            PyTuple_SET_ITEM(args, i, Py_NewRef(value));
        } else if (PyDict_SetItemString(kwargs, entry_.arg_names[i], value) < 0) {
            throw python_c::PythonError();
        }
    }
    owned.types = PyTuple_New(static_cast<Py_ssize_t>(type_list.size()));
    if (owned.types == nullptr) throw python_c::PythonError();
    for (size_t i = 0; i < type_list.size(); ++i) {
        PyTuple_SET_ITEM(owned.types, static_cast<Py_ssize_t>(i),
                         Py_NewRef(type_list[i]));
    }

    PyObject* mode = static_cast<PyObject*>(mode_.mode->get());
    PyObject* result = PyObject_CallMethod(
        mode, "__tensorplay_dispatch__", "OOOO", func, owned.types, args, kwargs);
    if (result == nullptr) throw python_c::PythonError();
    if (result != Py_NotImplemented) return result;
    Py_DECREF(result);

    // The mode declined: tensor subclasses get their turn, most derived
    // first, before the call is reported as unhandled.
    for (PyObject* type : type_list) {
        PyObject* handled = PyObject_CallMethod(
            type, "__tensorplay_dispatch__", "OOOO", func, owned.types, args, kwargs);
        if (handled == nullptr) throw python_c::PythonError();
        if (handled != Py_NotImplemented) return handled;
        Py_DECREF(handled);
    }
    PyErr_Format(PyExc_TypeError,
                 "Multiple dispatch failed for '%s': the active dispatch mode "
                 "and every tensor subclass returned NotImplemented",
                 entry_.name);
    throw python_c::PythonError();
}

namespace {

PyObject* enum_from_int(const char* probe_attr, int64_t value) {
    // The enum classes are the types of tensorplay's public enum constants
    // (e.g. ``tensorplay.contiguous_format``).
    PyObject* module = PyImport_ImportModule("tensorplay");
    if (module == nullptr) return nullptr;
    PyObject* probe = PyObject_GetAttrString(module, probe_attr);
    Py_DECREF(module);
    if (probe == nullptr) return nullptr;
    PyObject* type = reinterpret_cast<PyObject*>(Py_TYPE(probe));
    PyObject* result = PyObject_CallFunction(type, "L", static_cast<long long>(value));
    Py_DECREF(probe);
    return result;
}

template <typename T, typename Wrap>
PyObject* wrap_sequence(const std::vector<T>& values, Wrap&& wrap) {
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(values.size()));
    if (list == nullptr) return nullptr;
    for (size_t i = 0; i < values.size(); ++i) {
        PyObject* item = wrap(values[i]);
        if (item == nullptr) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), item);
    }
    return list;
}

template <typename Parse>
auto parse_sequence(PyObject* obj, Parse&& parse)
    -> std::vector<decltype(parse(obj))> {
    PyObject* seq = PySequence_Fast(obj, "expected a sequence");
    if (seq == nullptr) throw python_c::PythonError();
    std::vector<decltype(parse(obj))> out;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    out.reserve(static_cast<size_t>(n));
    try {
        for (Py_ssize_t i = 0; i < n; ++i) {
            out.push_back(parse(PySequence_Fast_GET_ITEM(seq, i)));
        }
    } catch (...) {
        Py_DECREF(seq);
        throw;
    }
    Py_DECREF(seq);
    return out;
}

}  // namespace

PyObject* wrap_tensor_or_none(const Tensor& tensor) {
    if (!tensor.defined()) Py_RETURN_NONE;
    return python_c::tpx_py_wrap(tensor);
}

PyObject* wrap_optional_tensor_list(
    const std::optional<std::vector<Tensor>>& tensors) {
    if (!tensors.has_value()) Py_RETURN_NONE;
    return python_c::tpx_py_wrap_list(*tensors);
}

PyObject* wrap_memory_format(int64_t value) {
    return enum_from_int("contiguous_format", value);
}

PyObject* wrap_optional_memory_format(const std::optional<int64_t>& value) {
    if (!value.has_value()) Py_RETURN_NONE;
    return wrap_memory_format(*value);
}

PyObject* wrap_layout(int64_t value) {
    return enum_from_int("strided", value);
}

PyObject* wrap_optional_layout(const std::optional<int64_t>& value) {
    if (!value.has_value()) Py_RETURN_NONE;
    return wrap_layout(*value);
}

PyObject* wrap_string(const std::string& value) {
    return PyUnicode_FromStringAndSize(value.data(), static_cast<Py_ssize_t>(value.size()));
}

PyObject* wrap_optional_int64_list(const std::vector<std::optional<int64_t>>& values) {
    return wrap_sequence(values, [](const std::optional<int64_t>& v) {
        return python_c::tpx_py_wrap_optional_int64(v);
    });
}

PyObject* wrap_string_list(const std::vector<std::string>& values) {
    return wrap_sequence(values, [](const std::string& v) { return wrap_string(v); });
}

PyObject* wrap_optional_scalar_list(const std::optional<std::vector<Scalar>>& values) {
    if (!values.has_value()) Py_RETURN_NONE;
    return python_c::tpx_py_wrap_scalarlist(*values);
}

PyObject* wrap_optional_bool_list(const std::optional<std::vector<bool>>& values) {
    if (!values.has_value()) Py_RETURN_NONE;
    return python_c::tpx_py_wrap_boollist(*values);
}

SymInt parse_symint(PyObject* obj) {
    if (PyLong_Check(obj)) return SymInt(python_c::tpx_py_int64(obj));
    return py::handle(obj).cast<SymInt>();
}

SymBool parse_symbool(PyObject* obj) {
    if (PyBool_Check(obj)) return SymBool(obj == Py_True);
    return py::handle(obj).cast<SymBool>();
}

SymFloat parse_symfloat(PyObject* obj) {
    if (PyFloat_Check(obj) || PyLong_Check(obj)) {
        return SymFloat(python_c::tpx_py_double(obj));
    }
    return py::handle(obj).cast<SymFloat>();
}

std::vector<SymInt> parse_symint_list(PyObject* obj) {
    return parse_sequence(obj, [](PyObject* item) { return parse_symint(item); });
}

std::vector<int64_t> parse_int64_list(PyObject* obj) {
    return python_c::tpx_py_intlist(obj);
}

std::optional<Tensor> parse_optional_tensor(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return python_c::tpx_py_tensor(obj);
}

std::vector<std::optional<Tensor>> parse_optional_tensor_list(PyObject* obj) {
    return parse_sequence(obj, [](PyObject* item) { return parse_optional_tensor(item); });
}

PyObject* result_item(PyObject* result, Py_ssize_t size, Py_ssize_t index,
                      const char* op_name) {
    if (!PyTuple_Check(result) && !PyList_Check(result)) {
        PyErr_Format(PyExc_TypeError,
                     "dispatch mode returned %s for '%s'; expected a sequence of %zd values",
                     Py_TYPE(result)->tp_name, op_name, size);
        throw python_c::PythonError();
    }
    if (PySequence_Fast_GET_SIZE(result) != size) {
        PyErr_Format(PyExc_TypeError,
                     "dispatch mode returned %zd values for '%s'; expected %zd",
                     PySequence_Fast_GET_SIZE(result), op_name, size);
        throw python_c::PythonError();
    }
    return PySequence_Fast_GET_ITEM(result, index);
}

void initialize() {
    impl::set_pyobject_release_hook(&release_pyobject);
    register_python_kernels();
}

}  // namespace python_dispatch
}  // namespace tensorplay

void init_python_dispatch(py::module_& m) {
    using tensorplay::impl::DispatchModeTLS;
    using tensorplay::impl::SafePyObject;

    tensorplay::python_dispatch::initialize();

    m.def("_push_dispatch_mode", [](py::object mode) {
        DispatchModeTLS::push(
            std::make_shared<SafePyObject>(mode.release().ptr()));
    });
    m.def("_pop_dispatch_mode", []() -> py::object {
        auto mode = DispatchModeTLS::pop();
        return py::reinterpret_borrow<py::object>(
            static_cast<PyObject*>(mode->get()));
    });
    m.def("_len_dispatch_mode", []() { return DispatchModeTLS::stack_len(); });
    m.def("_get_dispatch_mode", [](int64_t index) -> py::object {
        auto mode = DispatchModeTLS::get_at(index);
        return py::reinterpret_borrow<py::object>(
            static_cast<PyObject*>(mode->get()));
    });
    m.def("_python_dispatch_entries", []() {
        int64_t count = 0;
        const auto* const* entries =
            tensorplay::python_dispatch::python_dispatch_entries(&count);
        py::list out;
        for (int64_t i = 0; i < count; ++i) {
            const auto* entry = entries[i];
            py::list names;
            for (int j = 0; j < entry->num_args; ++j) names.append(entry->arg_names[j]);
            out.append(py::make_tuple(entry->name, entry->schema, names,
                                      entry->num_positional, entry->tags));
        }
        return out;
    });
    m.def("_python_dispatch_key_included", []() {
        return tensorplay::impl::tls_local_dispatch_key_set().included.has(
            tensorplay::DispatchKey::Python);
    });
}

// CPythonBridge.cpp -- implementation of the conversion surface declared in
// CPythonBridge.h.  Lives inside the _C extension target so pybind11's
// registered type casters for Tensor/Scalar/DType are available; the
// generated METH_FASTCALL layer (TensorCPythonGenerated.h) calls these
// helpers and therefore never touches pybind11 dispatch itself.
#include "CPythonBridge.h"

#include <pybind11/pybind11.h>

#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

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

// Central C++ -> Python error mapping (see python_bindings.h).
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
// Python dispatch state and Tensor subclass dispatch
// ---------------------------------------------------------------------------

namespace {

struct PythonDispatchTLS {
    int function_state = TPX_FUNCTION_ENABLED;
    int dispatch_layer = 0;
    bool function_skip_next = false;
    bool subclass_skip_next = false;
    std::vector<PyObject*> function_modes;
    std::vector<std::pair<std::string, PyTypeObject*>> active_function_hooks;
    std::vector<std::pair<std::string, PyTypeObject*>> active_hooks;

    ~PythonDispatchTLS() {
        for (PyObject* mode : function_modes) {
            Py_XDECREF(mode);
        }
    }
};

thread_local PythonDispatchTLS g_python_dispatch_tls;

extern PyTypeObject* g_tensor_type;

struct DispatchLayerGuard {
    explicit DispatchLayerGuard(int layer)
        : previous(g_python_dispatch_tls.dispatch_layer) {
        g_python_dispatch_tls.dispatch_layer = layer;
    }
    ~DispatchLayerGuard() {
        g_python_dispatch_tls.dispatch_layer = previous;
    }
    int previous;
};

constexpr int kDispatchModeLayer = 1;
constexpr int kDispatchFunctionLayer = 2;
constexpr int kDispatchSubclassLayer = 3;

bool has_attribute(PyObject* value, const char* name) {
    const int result = PyObject_HasAttrString(value, name);
    if (result < 0) return false;
    return result != 0;
}

bool is_tensor_object(PyObject* value) {
    if (g_tensor_type == nullptr) {
        PyObject* module = PyImport_ImportModule("tensorplay._C");
        if (module != nullptr) {
            PyObject* type = PyObject_GetAttrString(module, "TensorBase");
            if (type != nullptr && PyType_Check(type)) {
                g_tensor_type = reinterpret_cast<PyTypeObject*>(type);
                Py_DECREF(type);
            } else {
                Py_XDECREF(type);
                PyErr_Clear();
            }
            Py_DECREF(module);
        } else {
            PyErr_Clear();
        }
    }
    return g_tensor_type != nullptr && PyObject_TypeCheck(value, g_tensor_type);
}

bool has_subclass_dispatch(PyObject* value) {
    PyObject* type = reinterpret_cast<PyObject*>(Py_TYPE(value));
    return has_attribute(type, "__tensorplay_dispatch__") ||
           has_attribute(type, "__torch_dispatch__");
}

bool has_function_dispatch(PyObject* value) {
    PyObject* type = reinterpret_cast<PyObject*>(Py_TYPE(value));
    return has_attribute(type, "__tensorplay_function__") ||
           has_attribute(type, "__torch_function__");
}

bool is_builtin_dispatch_free(PyObject* value) {
    PyTypeObject* type = Py_TYPE(value);
    return type == &PyBool_Type || type == &PyLong_Type ||
           type == &PyFloat_Type || type == &PyComplex_Type ||
           type == &PyUnicode_Type || type == &PyBytes_Type ||
           type == &PySet_Type || type == &PyFrozenSet_Type ||
           type == &PySlice_Type || type == Py_TYPE(Py_None) ||
           type == Py_TYPE(Py_Ellipsis) || type == Py_TYPE(Py_NotImplemented);
}

bool insert_candidate(
    PyObject* value, std::vector<PyObject*>& candidates,
    std::vector<PyTypeObject*>& candidate_types, bool require_subclass) {
    if (value == nullptr) return true;
    if (PyTuple_CheckExact(value) || PyList_CheckExact(value)) {
        const Py_ssize_t size = PySequence_Fast_GET_SIZE(value);
        PyObject** items = PySequence_Fast_ITEMS(value);
        for (Py_ssize_t i = 0; i < size; ++i) {
            if (!insert_candidate(items[i], candidates, candidate_types,
                                  require_subclass)) {
                return false;
            }
        }
        return true;
    }
    if (PyDict_CheckExact(value)) {
        PyObject* key = nullptr;
        PyObject* item = nullptr;
        Py_ssize_t position = 0;
        while (PyDict_Next(value, &position, &key, &item)) {
            if (!insert_candidate(item, candidates, candidate_types,
                                  require_subclass)) {
                return false;
            }
        }
        return true;
    }
    if (g_tensor_type != nullptr && Py_TYPE(value) == g_tensor_type) {
        return true;
    }
    if (is_builtin_dispatch_free(value)) return true;
    if (g_tensor_type == nullptr) {
        (void)is_tensor_object(value);
    }
    if (g_tensor_type != nullptr && Py_TYPE(value) == g_tensor_type) {
        return true;
    }
    if (require_subclass && !is_tensor_object(value)) return true;
    const bool has_hook = require_subclass ? has_subclass_dispatch(value)
                                           : has_function_dispatch(value);
    if (PyErr_Occurred()) return false;
    if (!has_hook) return true;

    auto* type = Py_TYPE(value);
    for (PyTypeObject* old_type : candidate_types) {
        if (old_type == type) return true;
    }

    size_t index = candidates.size();
    for (size_t i = 0; i < candidate_types.size(); ++i) {
        const int derived = PyObject_IsSubclass(
            reinterpret_cast<PyObject*>(type),
            reinterpret_cast<PyObject*>(candidate_types[i]));
        if (derived < 0) return false;
        if (derived != 0) {
            index = i;
            break;
        }
    }
    candidates.insert(candidates.begin() + static_cast<ptrdiff_t>(index), value);
    candidate_types.insert(candidate_types.begin() + static_cast<ptrdiff_t>(index),
                           type);
    return true;
}

PyObject* make_call_args(
    PyObject* receiver, bool is_method, PyObject* const* args,
    Py_ssize_t nargs) {
    const Py_ssize_t size = nargs + (is_method ? 1 : 0);
    PyObject* call_args = PyTuple_New(size);
    if (call_args == nullptr) return nullptr;
    Py_ssize_t out = 0;
    if (is_method) {
        Py_INCREF(receiver);
        PyTuple_SET_ITEM(call_args, out++, receiver);
    }
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        Py_INCREF(args[i]);
        PyTuple_SET_ITEM(call_args, out++, args[i]);
    }
    return call_args;
}

PyObject* make_call_kwargs(
    PyObject* const* args, Py_ssize_t nargs, PyObject* kwnames) {
    PyObject* call_kwargs = PyDict_New();
    if (call_kwargs == nullptr) return nullptr;
    const Py_ssize_t nkw = kwnames == nullptr ? 0 : PyTuple_GET_SIZE(kwnames);
    for (Py_ssize_t i = 0; i < nkw; ++i) {
        PyObject* key = PyTuple_GET_ITEM(kwnames, i);
        if (PyDict_SetItem(call_kwargs, key, args[nargs + i]) != 0) {
            Py_DECREF(call_kwargs);
            return nullptr;
        }
    }
    return call_kwargs;
}

PyObject* call_dispatch_hook(
    PyObject* hook, PyObject* func, PyObject* types, PyObject* call_args,
    PyObject* call_kwargs) {
    PyObject* hook_args[] = {func, types, call_args, call_kwargs};
    return PyObject_Vectorcall(hook, hook_args, 4, nullptr);
}

PyObject* make_public_api(const char* op_name, bool is_method) {
    PyObject* module = PyImport_ImportModule("tensorplay");
    if (module == nullptr) return nullptr;

    PyObject* api = nullptr;
    if (is_method) {
        PyObject* tensor_type = PyObject_GetAttrString(module, "Tensor");
        if (tensor_type != nullptr) {
            api = PyObject_GetAttrString(tensor_type, op_name);
            Py_DECREF(tensor_type);
            if (api != nullptr && PyCallable_Check(api)) {
                Py_DECREF(module);
                return api;
            }
            Py_XDECREF(api);
            api = nullptr;
            PyErr_Clear();
        } else {
            PyErr_Clear();
        }
    }

    api = PyObject_GetAttrString(module, op_name);
    Py_DECREF(module);
    if (api == nullptr) return nullptr;
    if (!PyCallable_Check(api)) {
        PyErr_Format(PyExc_TypeError,
                     "tensorplay.%s is not callable", op_name);
        Py_DECREF(api);
        return nullptr;
    }
    return api;
}

PyObject* get_hook(PyObject* value, bool function_hook) {
    const char* names[] = {
        function_hook ? "__tensorplay_function__" : "__tensorplay_dispatch__",
        function_hook ? "__torch_function__" : "__torch_dispatch__",
    };
    for (const char* name : names) {
        PyObject* hook = PyObject_GetAttrString(value, name);
        if (hook != nullptr) return hook;
        PyErr_Clear();
    }
    return nullptr;
}

bool active_hook(const char* op_name, PyTypeObject* type) {
    for (const auto& entry : g_python_dispatch_tls.active_hooks) {
        if (entry.first == op_name && entry.second == type) return true;
    }
    return false;
}

bool active_function_hook(const char* op_name, PyTypeObject* type) {
    for (const auto& entry : g_python_dispatch_tls.active_function_hooks) {
        if (entry.first == op_name && entry.second == type) return true;
    }
    return false;
}

void push_active_hook(const char* op_name, PyTypeObject* type) {
    g_python_dispatch_tls.active_hooks.emplace_back(op_name, type);
}

void push_active_function_hook(const char* op_name, PyTypeObject* type) {
    g_python_dispatch_tls.active_function_hooks.emplace_back(op_name, type);
}

void pop_active_hook(const char* op_name, PyTypeObject* type) {
    for (auto it = g_python_dispatch_tls.active_hooks.rbegin();
         it != g_python_dispatch_tls.active_hooks.rend(); ++it) {
        if (it->first == op_name && it->second == type) {
            g_python_dispatch_tls.active_hooks.erase(std::next(it).base());
            return;
        }
    }
}

void pop_active_function_hook(const char* op_name, PyTypeObject* type) {
    for (auto it = g_python_dispatch_tls.active_function_hooks.rbegin();
         it != g_python_dispatch_tls.active_function_hooks.rend(); ++it) {
        if (it->first == op_name && it->second == type) {
            g_python_dispatch_tls.active_function_hooks.erase(
                std::next(it).base());
            return;
        }
    }
}

bool collect_tensor_subclass_candidate(
    PyObject* value, std::vector<PyObject*>& candidates,
    std::vector<PyTypeObject*>& candidate_types) {
    return insert_candidate(value, candidates, candidate_types, true);
}

}  // namespace

int tpx_py_try_tensor_function_dispatch(
    const char* op_name, PyObject* receiver, bool is_method,
    PyObject* const* args, Py_ssize_t nargs, PyObject* kwnames,
    PyObject** result) {
    *result = nullptr;

    if (g_python_dispatch_tls.function_skip_next) {
        g_python_dispatch_tls.function_skip_next = false;
        return 0;
    }
    if (g_python_dispatch_tls.function_state == TPX_ALL_DISABLED) {
        return 0;
    }

    const Py_ssize_t nkw = kwnames == nullptr ? 0 : PyTuple_GET_SIZE(kwnames);

    std::vector<PyObject*> candidates;
    std::vector<PyTypeObject*> candidate_types;
    if (is_method && !insert_candidate(receiver, candidates, candidate_types,
                                       false)) {
        return -1;
    }
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        if (!insert_candidate(args[i], candidates, candidate_types, false)) {
            return -1;
        }
    }
    for (Py_ssize_t i = 0; i < nkw; ++i) {
        if (!insert_candidate(args[nargs + i], candidates, candidate_types,
                              false)) {
            return -1;
        }
    }
    if (candidates.empty()) {
        return 0;
    }

    PyObject* call_args = make_call_args(receiver, is_method, args, nargs);
    if (call_args == nullptr) return -1;
    PyObject* call_kwargs = make_call_kwargs(args, nargs, kwnames);
    if (call_kwargs == nullptr) {
        Py_DECREF(call_args);
        return -1;
    }

    PyObject* func = make_public_api(op_name, is_method);
    PyObject* types = PyTuple_New(static_cast<Py_ssize_t>(candidate_types.size()));
    if (func == nullptr || types == nullptr) {
        Py_XDECREF(types);
        Py_XDECREF(func);
        Py_DECREF(call_kwargs);
        Py_DECREF(call_args);
        return -1;
    }
    for (size_t i = 0; i < candidate_types.size(); ++i) {
        PyObject* type = reinterpret_cast<PyObject*>(candidate_types[i]);
        Py_INCREF(type);
        PyTuple_SET_ITEM(types, static_cast<Py_ssize_t>(i), type);
    }

    for (PyObject* candidate : candidates) {
        PyTypeObject* type = Py_TYPE(candidate);
        if (active_function_hook(op_name, type)) {
            PyErr_Format(PyExc_RuntimeError,
                         "recursive Tensor function dispatch for %s on %s",
                         op_name, type->tp_name);
            Py_DECREF(types);
            Py_DECREF(func);
            Py_DECREF(call_kwargs);
            Py_DECREF(call_args);
            return -1;
        }
        PyObject* hook = get_hook(candidate, true);
        if (hook == nullptr) continue;
        try {
            push_active_function_hook(op_name, type);
        } catch (...) {
            Py_DECREF(hook);
            Py_DECREF(types);
            Py_DECREF(func);
            Py_DECREF(call_kwargs);
            Py_DECREF(call_args);
            throw;
        }
        DispatchLayerGuard layer_guard(kDispatchFunctionLayer);
        PyObject* dispatched = call_dispatch_hook(
            hook, func, types, call_args, call_kwargs);
        pop_active_function_hook(op_name, type);
        Py_DECREF(hook);
        if (dispatched == nullptr) {
            Py_DECREF(types);
            Py_DECREF(func);
            Py_DECREF(call_kwargs);
            Py_DECREF(call_args);
            return -1;
        }
        if (dispatched != Py_NotImplemented) {
            *result = dispatched;
            Py_DECREF(types);
            Py_DECREF(func);
            Py_DECREF(call_kwargs);
            Py_DECREF(call_args);
            return 1;
        }
        Py_DECREF(dispatched);
    }

    PyErr_Format(PyExc_TypeError,
                 "all function hooks returned NotImplemented for tensorplay.%s",
                 op_name);
    Py_DECREF(types);
    Py_DECREF(func);
    Py_DECREF(call_kwargs);
    Py_DECREF(call_args);
    return -1;
}

int tpx_py_try_tensor_subclass_dispatch(
    const char* op_name, PyObject* receiver, bool is_method,
    PyObject* const* args, Py_ssize_t nargs, PyObject* kwnames,
    PyObject** result) {
    *result = nullptr;

    if (g_python_dispatch_tls.subclass_skip_next) {
        g_python_dispatch_tls.subclass_skip_next = false;
        return 0;
    }
    if (g_python_dispatch_tls.function_state == TPX_ALL_DISABLED ||
        g_python_dispatch_tls.function_state == TPX_SUBCLASSES_DISABLED) {
        return 0;
    }

    const Py_ssize_t nkw = kwnames == nullptr ? 0 : PyTuple_GET_SIZE(kwnames);

    std::vector<PyObject*> candidates;
    std::vector<PyTypeObject*> candidate_types;
    if (is_method && !collect_tensor_subclass_candidate(
                         receiver, candidates, candidate_types)) {
        return -1;
    }
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        if (!collect_tensor_subclass_candidate(
                args[i], candidates, candidate_types)) {
            return -1;
        }
    }
    for (Py_ssize_t i = 0; i < nkw; ++i) {
        if (!collect_tensor_subclass_candidate(
                args[nargs + i], candidates, candidate_types)) {
            return -1;
        }
    }

    if (candidates.empty()) {
        return 0;
    }

    PyObject* call_args = make_call_args(receiver, is_method, args, nargs);
    if (call_args == nullptr) return -1;
    PyObject* call_kwargs = make_call_kwargs(args, nargs, kwnames);
    if (call_kwargs == nullptr) {
        Py_DECREF(call_args);
        return -1;
    }

    PyObject* func = make_public_api(op_name, is_method);
    PyObject* types = PyTuple_New(static_cast<Py_ssize_t>(candidate_types.size()));
    if (func == nullptr || types == nullptr) {
        Py_XDECREF(types);
        Py_XDECREF(func);
        Py_DECREF(call_kwargs);
        Py_DECREF(call_args);
        return -1;
    }
    for (size_t i = 0; i < candidate_types.size(); ++i) {
        PyObject* type = reinterpret_cast<PyObject*>(candidate_types[i]);
        Py_INCREF(type);
        PyTuple_SET_ITEM(types, static_cast<Py_ssize_t>(i), type);
    }

    for (PyObject* candidate : candidates) {
        PyTypeObject* type = Py_TYPE(candidate);
        if (active_hook(op_name, type)) {
            PyErr_Format(PyExc_RuntimeError,
                         "recursive Tensor subclass dispatch for %s on %s",
                         op_name, type->tp_name);
            Py_DECREF(types);
            Py_DECREF(func);
            Py_DECREF(call_kwargs);
            Py_DECREF(call_args);
            return -1;
        }
        PyObject* hook = get_hook(candidate, false);
        if (hook == nullptr) {
            continue;
        }
        push_active_hook(op_name, type);
        DispatchLayerGuard layer_guard(kDispatchSubclassLayer);
        PyObject* dispatched = call_dispatch_hook(
            hook, func, types, call_args, call_kwargs);
        pop_active_hook(op_name, type);
        Py_DECREF(hook);
        if (dispatched == nullptr) {
            Py_DECREF(types);
            Py_DECREF(func);
            Py_DECREF(call_kwargs);
            Py_DECREF(call_args);
            return -1;
        }
        if (dispatched != Py_NotImplemented) {
            *result = dispatched;
            Py_DECREF(types);
            Py_DECREF(func);
            Py_DECREF(call_kwargs);
            Py_DECREF(call_args);
            return 1;
        }
        Py_DECREF(dispatched);
    }

    Py_DECREF(types);
    Py_DECREF(func);
    Py_DECREF(call_kwargs);
    Py_DECREF(call_args);
    return 0;
}

int tpx_py_get_function_state() {
    return g_python_dispatch_tls.function_state;
}

bool tpx_py_set_function_state(int state) {
    if (state < TPX_FUNCTION_ENABLED || state > TPX_ALL_DISABLED) {
        PyErr_SetString(PyExc_ValueError, "invalid Tensor function dispatch state");
        return false;
    }
    g_python_dispatch_tls.function_state = state;
    return true;
}

bool tpx_py_exchange_skip_next(bool value) {
    const bool old = g_python_dispatch_tls.function_skip_next;
    g_python_dispatch_tls.function_skip_next = value;
    return old;
}

bool tpx_py_peek_skip_next() {
    return g_python_dispatch_tls.function_skip_next;
}

bool tpx_py_exchange_subclass_skip_next(bool value) {
    const bool old = g_python_dispatch_tls.subclass_skip_next;
    g_python_dispatch_tls.subclass_skip_next = value;
    return old;
}

bool tpx_py_peek_subclass_skip_next() {
    return g_python_dispatch_tls.subclass_skip_next;
}

int tpx_py_get_dispatch_layer() {
    return g_python_dispatch_tls.dispatch_layer;
}

void tpx_py_push_function_mode(PyObject* mode) {
    if (mode == Py_None) return;
    Py_INCREF(mode);
    try {
        g_python_dispatch_tls.function_modes.push_back(mode);
    } catch (...) {
        Py_DECREF(mode);
        throw;
    }
}

PyObject* tpx_py_pop_function_mode() {
    if (g_python_dispatch_tls.function_modes.empty()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot pop an empty Tensor function mode stack");
        return nullptr;
    }
    PyObject* mode = g_python_dispatch_tls.function_modes.back();
    g_python_dispatch_tls.function_modes.pop_back();
    return mode;
}

PyObject* tpx_py_get_function_mode(Py_ssize_t index) {
    if (index < 0 || index >= static_cast<Py_ssize_t>(
                            g_python_dispatch_tls.function_modes.size())) {
        PyErr_SetString(PyExc_IndexError, "Tensor function mode index out of range");
        return nullptr;
    }
    PyObject* mode = g_python_dispatch_tls.function_modes[
        static_cast<size_t>(index)];
    Py_INCREF(mode);
    return mode;
}

Py_ssize_t tpx_py_function_mode_len() {
    return static_cast<Py_ssize_t>(g_python_dispatch_tls.function_modes.size());
}

int tpx_py_try_function_mode_dispatch(
    const char* op_name, PyObject* receiver, bool is_method,
    PyObject* const* args, Py_ssize_t nargs, PyObject* kwnames,
    PyObject** result) {
    *result = nullptr;
    if (g_python_dispatch_tls.function_skip_next) {
        return 0;
    }
    if (g_python_dispatch_tls.function_state == TPX_ALL_DISABLED ||
        g_python_dispatch_tls.function_modes.empty()) {
        return 0;
    }

    PyObject* call_args = make_call_args(receiver, is_method, args, nargs);
    if (call_args == nullptr) return -1;
    PyObject* call_kwargs = make_call_kwargs(args, nargs, kwnames);
    if (call_kwargs == nullptr) {
        Py_DECREF(call_args);
        return -1;
    }

    std::vector<PyObject*> candidates;
    std::vector<PyTypeObject*> candidate_types;
    if (is_method && !insert_candidate(receiver, candidates, candidate_types,
                                       false)) {
        Py_DECREF(call_kwargs);
        Py_DECREF(call_args);
        return -1;
    }
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        if (!insert_candidate(args[i], candidates, candidate_types, false)) {
            Py_DECREF(call_kwargs);
            Py_DECREF(call_args);
            return -1;
        }
    }
    const Py_ssize_t nkw = kwnames == nullptr ? 0 : PyTuple_GET_SIZE(kwnames);
    for (Py_ssize_t i = 0; i < nkw; ++i) {
        if (!insert_candidate(args[nargs + i], candidates, candidate_types,
                              false)) {
            Py_DECREF(call_kwargs);
            Py_DECREF(call_args);
            return -1;
        }
    }

    PyObject* types = PyTuple_New(static_cast<Py_ssize_t>(candidate_types.size()));
    PyObject* func = make_public_api(op_name, is_method);
    if (types == nullptr || func == nullptr) {
        Py_XDECREF(types);
        Py_XDECREF(func);
        Py_DECREF(call_kwargs);
        Py_DECREF(call_args);
        return -1;
    }
    for (size_t i = 0; i < candidate_types.size(); ++i) {
        PyObject* type = reinterpret_cast<PyObject*>(candidate_types[i]);
        Py_INCREF(type);
        PyTuple_SET_ITEM(types, static_cast<Py_ssize_t>(i), type);
    }

    PyObject* mode = g_python_dispatch_tls.function_modes.back();
    g_python_dispatch_tls.function_modes.pop_back();
    PyObject* hook = get_hook(mode, true);
    if (hook == nullptr) {
        try {
            g_python_dispatch_tls.function_modes.push_back(mode);
        } catch (...) {
            Py_DECREF(mode);
            Py_DECREF(types);
            Py_DECREF(func);
            Py_DECREF(call_kwargs);
            Py_DECREF(call_args);
            throw;
        }
        Py_DECREF(types);
        Py_DECREF(func);
        Py_DECREF(call_kwargs);
        Py_DECREF(call_args);
        return 0;
    }
    DispatchLayerGuard layer_guard(kDispatchModeLayer);
    PyObject* dispatched = call_dispatch_hook(
        hook, func, types, call_args, call_kwargs);
    Py_DECREF(hook);
    try {
        g_python_dispatch_tls.function_modes.push_back(mode);
    } catch (...) {
        Py_XDECREF(dispatched);
        Py_DECREF(mode);
        Py_DECREF(types);
        Py_DECREF(func);
        Py_DECREF(call_kwargs);
        Py_DECREF(call_args);
        throw;
    }
    if (dispatched == nullptr) {
        Py_DECREF(types);
        Py_DECREF(func);
        Py_DECREF(call_kwargs);
        Py_DECREF(call_args);
        return -1;
    }
    if (dispatched != Py_NotImplemented) {
        *result = dispatched;
        Py_DECREF(types);
        Py_DECREF(func);
        Py_DECREF(call_kwargs);
        Py_DECREF(call_args);
        return 1;
    }
    Py_DECREF(dispatched);
    Py_DECREF(types);
    Py_DECREF(func);
    Py_DECREF(call_kwargs);
    Py_DECREF(call_args);
    return 0;
}

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

bool tpx_py_kwnames_has(PyObject* kwnames, const char* name) {
    if (kwnames == nullptr) return false;
    const Py_ssize_t size = PyTuple_GET_SIZE(kwnames);
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* key = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_Check(key) &&
            PyUnicode_CompareWithASCIIString(key, name) == 0) {
            return true;
        }
    }
    return false;
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
    // Integral-valued floats are accepted for integer slots; non-integral
    // floats still raise.
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
        return t;
    } catch (const py::cast_error&) {
        type_error(obj, "op", 0, "a Tensor");
    }
}

Tensor& tensor_mref_slow(PyObject* obj) {
    try {
        Tensor& t = py::cast<Tensor&>(py::reinterpret_borrow<py::object>(obj));
        return t;
    } catch (const py::cast_error&) {
        type_error(obj, "op", 0, "a Tensor");
    }
}

}  // namespace

Tensor tpx_py_tensor(PyObject* obj) { return as_tensor(obj, "op", 0); }

const Tensor& tpx_py_tensor_cref(PyObject* obj) {
    if (g_tensor_type != nullptr && PyObject_TypeCheck(obj, g_tensor_type)) {
        // Registered wrappers use a simple value-holder layout, so direct
        // access avoids a registry lookup for the common case.
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
    // Only real bools are accepted by this conversion.  Truthiness of
    // arbitrary objects would silently change the call contract.
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
Generator tpx_py_generator(PyObject* obj) {
    try {
        return py::cast<Generator>(py::reinterpret_borrow<py::object>(obj));
    } catch (const py::cast_error&) {
        type_error(obj, "op", 0, "a Generator");
    }
}
std::optional<Generator> tpx_py_opt_generator(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return tpx_py_generator(obj);
}
Storage tpx_py_storage(PyObject* obj) {
    try {
        return py::cast<Storage>(py::reinterpret_borrow<py::object>(obj));
    } catch (const py::cast_error&) {
        type_error(obj, "op", 0, "a Storage");
    }
}
Device tpx_py_device(PyObject* obj) {
    try {
        return py::cast<Device>(py::reinterpret_borrow<py::object>(obj));
    } catch (const py::cast_error&) {
        type_error(obj, "op", 0, "a Device");
    }
}
std::optional<Device> tpx_py_opt_device(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return tpx_py_device(obj);
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
std::vector<bool> tpx_py_boollist(PyObject* obj) {
    std::vector<bool> r;
    if (PyBool_Check(obj)) { r.push_back(obj == Py_True); return r; }
    PyObject* seq = PySequence_Fast(obj, "expected a sequence of bools");
    if (!seq) { PyErr_Clear(); throw std::invalid_argument("expected a sequence of bools"); }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    r.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        const int truth = PyObject_IsTrue(item);
        if (truth < 0) {
            Py_DECREF(seq);
            PyErr_Clear();
            throw std::invalid_argument("expected a sequence of bools");
        }
        r.push_back(truth != 0);
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
std::optional<std::vector<double>> tpx_py_opt_doublelist(PyObject* obj) {
    if (obj == Py_None) return std::nullopt;
    return tpx_py_doublelist(obj);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

namespace {

[[noreturn]] void arg_type_error(const char* op_name, const char* name,
                                 int index, const char* want, PyObject* obj) {
    std::string msg = std::string(op_name) + "(): argument '" + (name ? name : "?")
                      + "'";
    // Report a position only for arguments that can be passed positionally;
    // keyword-only arguments get the bare form.
    if (index >= 0) {
        msg += " (position " + std::to_string(index + 1) + ")";
    }
    msg += std::string(" must be ") + want + ", not "
           + (obj ? Py_TYPE(obj)->tp_name : "NoneType");
    throw std::invalid_argument(msg);
}

bool obj_is_tensor(PyObject* obj) {
    return is_tensor_object(obj);
}

bool obj_is_storage(PyObject* obj) {
    try {
        return py::isinstance<Storage>(py::handle(obj));
    } catch (...) {
        return false;
    }
}

bool seq_item_is_number(PyObject* o) {
    // Python numbers plus the registered tensorplay.Scalar wrapper, which
    // generated wrappers (e.g. addmm's beta/alpha) pass through directly.
    // Complex numbers are part of the Number category.
    if (PyIndex_Check(o) || PyFloat_Check(o) || PyComplex_Check(o)) return true;
    try {
        return py::isinstance<tensorplay::Scalar>(py::handle(o));
    } catch (...) {
        return false;
    }
}

// For INT_LIST / FLOAT_LIST, a bare scalar folds to a singleton list;
// otherwise tuple/list and tensorplay._C.Size containers are accepted.
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
        case TPK_BOOLLIST:   return "bool[]";
        case TPK_TENSORLIST_OPTIONAL: return "Tensor?[]";
        case TPK_GENERATOR:  return "Generator";
        case TPK_STORAGE:    return "Storage";
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
            // Integral-valued floats are accepted for integer arguments.
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
        case TPK_TENSORLIST_OPTIONAL:
            if (!PyTuple_Check(obj) && !PyList_Check(obj)) return false;
            {
                Py_ssize_t m = PyTuple_Check(obj) ? PyTuple_GET_SIZE(obj)
                                                  : PyList_GET_SIZE(obj);
                for (Py_ssize_t j = 0; j < m; ++j) {
                    PyObject* el = PyTuple_Check(obj)
                                       ? PyTuple_GET_ITEM(obj, j)
                                       : PyList_GET_ITEM(obj, j);
                    if (el != Py_None && !obj_is_tensor(el)) return false;
                }
                return true;
            }
        case TPK_SCALARLIST: return check_list(obj, false);
        case TPK_BOOLLIST:
            if (PyBool_Check(obj)) return true;
            return PyTuple_Check(obj) || PyList_Check(obj);
        case TPK_GENERATOR:
            try {
                return py::isinstance<Generator>(py::handle(obj));
            } catch (...) {
                return false;
            }
        case TPK_STORAGE:    return obj_is_storage(obj);
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


// Only tuple/list containers qualify; each element must be a Tensor wrapper.
// Element errors stay std::invalid_argument so multi-overload dispatch can
// fall through to the next candidate signature.
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
std::vector<std::optional<Tensor>> tpx_py_opt_tensorlist(PyObject* obj) {
    std::vector<std::optional<Tensor>> r;
    if (!PyTuple_Check(obj) && !PyList_Check(obj)) {
        throw std::invalid_argument("expected a sequence of tensors or None");
    }
    Py_ssize_t n = PyTuple_Check(obj) ? PyTuple_GET_SIZE(obj)
                                      : PyList_GET_SIZE(obj);
    r.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PyTuple_Check(obj) ? PyTuple_GET_ITEM(obj, i)
                                            : PyList_GET_ITEM(obj, i);
        if (item == Py_None) {
            r.emplace_back(std::nullopt);
            continue;
        }
        try {
            r.emplace_back(tpx_py_tensor_cref(item));
        } catch (const std::invalid_argument&) {
            throw std::invalid_argument(
                std::string("expected Tensor or None as element ") +
                std::to_string(i) + ", but got " + Py_TYPE(item)->tp_name);
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

// Identity cache for returned wrappers, keyed by TensorImpl pointer:
// re-wrapping the same impl yields the *same* Python object instead of a fresh
// copy each call.  Invalidation needs no p10 changes -- each cached object
// carries an attribute capsule whose destructor runs when the wrapper dies and
// erases the entry.  The map stores borrowed pointers only (the owning
// reference is the object's own); all access happens under the GIL.
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
PyObject* tpx_py_wrap_optional_scalar(const std::optional<Scalar>& s) {
    if (!s.has_value()) {
        Py_RETURN_NONE;
    }
    return tpx_py_wrap_scalar(*s);
}
PyObject* tpx_py_wrap_symint(const SymInt& value) {
    return py::cast(value).release().ptr();
}
PyObject* tpx_py_wrap_symbool(const SymBool& value) {
    return py::cast(value).release().ptr();
}
PyObject* tpx_py_wrap_symfloat(const SymFloat& value) {
    return py::cast(value).release().ptr();
}
PyObject* tpx_py_wrap_optional_symint(const std::optional<SymInt>& value) {
    if (!value.has_value()) {
        Py_RETURN_NONE;
    }
    return tpx_py_wrap_symint(*value);
}
PyObject* tpx_py_wrap_optional_symbool(const std::optional<SymBool>& value) {
    if (!value.has_value()) {
        Py_RETURN_NONE;
    }
    return tpx_py_wrap_symbool(*value);
}
PyObject* tpx_py_wrap_optional_symfloat(const std::optional<SymFloat>& value) {
    if (!value.has_value()) {
        Py_RETURN_NONE;
    }
    return tpx_py_wrap_symfloat(*value);
}
PyObject* tpx_py_wrap_symintlist(const std::vector<SymInt>& values) {
    return py::cast(values).release().ptr();
}
PyObject* tpx_py_wrap_symboollist(const std::vector<SymBool>& values) {
    return py::cast(values).release().ptr();
}
PyObject* tpx_py_wrap_symfloatlist(const std::vector<SymFloat>& values) {
    return py::cast(values).release().ptr();
}
PyObject* tpx_py_wrap_optional_symintlist(
    const std::optional<std::vector<SymInt>>& values) {
    if (!values.has_value()) {
        Py_RETURN_NONE;
    }
    return tpx_py_wrap_symintlist(*values);
}
PyObject* tpx_py_wrap_optional_symboollist(
    const std::optional<std::vector<SymBool>>& values) {
    if (!values.has_value()) {
        Py_RETURN_NONE;
    }
    return tpx_py_wrap_symboollist(*values);
}
PyObject* tpx_py_wrap_optional_symfloatlist(
    const std::optional<std::vector<SymFloat>>& values) {
    if (!values.has_value()) {
        Py_RETURN_NONE;
    }
    return tpx_py_wrap_symfloatlist(*values);
}
PyObject* tpx_py_wrap_generator(const Generator& g) {
    return py::cast(g).release().ptr();
}
PyObject* tpx_py_wrap_storage(const Storage& storage) {
    return py::cast(storage).release().ptr();
}
PyObject* tpx_py_wrap_optional_tensor(const std::optional<Tensor>& t) {
    if (!t.has_value()) {
        Py_RETURN_NONE;
    }
    return tpx_py_wrap(*t);
}
PyObject* tpx_py_wrap_optional_generator(const std::optional<Generator>& g) {
    if (!g.has_value()) {
        Py_RETURN_NONE;
    }
    return tpx_py_wrap_generator(*g);
}
PyObject* tpx_py_wrap_optional_int64(const std::optional<int64_t>& v) {
    if (!v.has_value()) {
        Py_RETURN_NONE;
    }
    return PyLong_FromLongLong(*v);
}
PyObject* tpx_py_wrap_optional_double(const std::optional<double>& v) {
    if (!v.has_value()) {
        Py_RETURN_NONE;
    }
    return PyFloat_FromDouble(*v);
}
PyObject* tpx_py_wrap_optional_bool(const std::optional<bool>& v) {
    if (!v.has_value()) {
        Py_RETURN_NONE;
    }
    return PyBool_FromLong(*v);
}
PyObject* tpx_py_wrap_optional_string(const std::optional<std::string>& v) {
    if (!v.has_value()) {
        Py_RETURN_NONE;
    }
    return PyUnicode_FromString(v->c_str());
}
PyObject* tpx_py_wrap_dtype(const DType& dt) {
    return py::cast(dt).release().ptr();
}
PyObject* tpx_py_wrap_device(const Device& d) {
    return py::cast(d).release().ptr();
}
PyObject* tpx_py_wrap_optional_dtype(const std::optional<DType>& dt) {
    return py::cast(dt).release().ptr();
}
PyObject* tpx_py_wrap_optional_device(const std::optional<Device>& d) {
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
PyObject* tpx_py_wrap_optional_tensor_list(
    const std::vector<std::optional<Tensor>>& v) {
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(v.size()));
    if (list == nullptr) return nullptr;
    for (size_t i = 0; i < v.size(); ++i) {
        PyObject* item = v[i].has_value() ? tpx_py_wrap(*v[i]) : Py_None;
        if (item == nullptr) {
            Py_DECREF(list);
            return nullptr;
        }
        if (!v[i].has_value()) Py_INCREF(Py_None);
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), item);
    }
    return list;
}
PyObject* tpx_py_wrap_intlist(const std::vector<int64_t>& v) {
    return py::cast(v).release().ptr();
}
PyObject* tpx_py_wrap_doublelist(const std::vector<double>& v) {
    return py::cast(v).release().ptr();
}
PyObject* tpx_py_wrap_optional_intlist(
    const std::optional<std::vector<int64_t>>& v) {
    if (!v.has_value()) {
        Py_RETURN_NONE;
    }
    return tpx_py_wrap_intlist(*v);
}
PyObject* tpx_py_wrap_optional_doublelist(
    const std::optional<std::vector<double>>& v) {
    if (!v.has_value()) {
        Py_RETURN_NONE;
    }
    return tpx_py_wrap_doublelist(*v);
}
PyObject* tpx_py_wrap_boollist(const std::vector<bool>& v) {
    return py::cast(v).release().ptr();
}
PyObject* tpx_py_wrap_scalarlist(const std::vector<Scalar>& v) {
    return py::cast(v).release().ptr();
}

void tpx_py_keep_alive(PyObject*) {
    // No-op by construction: p10 views share their base's Storage
    // (shared_ptr) and VariableVersion (shared counter), so the returned
    // alias keeps both alive without an explicit keep-alive edge.  View
    // metadata can be added here when saved-view replay is enabled.
}

long long tpx_tensor_version(PyObject* obj) {
    // Version-counter read straight off the C++ value holder: the steady
    // state guard needs one integer per argument per call, so it must not
    // traverse Python attribute machinery.
    try {
        const Tensor& t = tpx_py_tensor_cref(obj);
        auto impl = t.unsafeGetTensorImpl();
        if (!impl) return -1;
        return static_cast<long long>(impl->version());
    } catch (...) {
        PyErr_Clear();
        return -1;
    }
}

int tpx_tensor_requires_grad(PyObject* obj) {
    try {
        const Tensor& t = tpx_py_tensor_cref(obj);
        if (!t.unsafeGetTensorImpl()) return -1;
        return t.requires_grad() ? 1 : 0;
    } catch (...) {
        PyErr_Clear();
        return -1;
    }
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

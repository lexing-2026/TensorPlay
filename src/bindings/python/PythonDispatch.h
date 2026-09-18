// Python dispatch key runtime.
//
// Every schema operator registers a generated kernel under
// DispatchKey::Python (tools/codegen/gen_python_dispatch.py).  The dispatcher
// selects it while a dispatch mode is on the thread's mode stack; the kernel
// converts its arguments to Python objects and hands the call to the
// innermost mode through ModeCall, then converts the result back.
#pragma once

#include <Python.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "CPythonBridge.h"
#include "LocalDispatchKeySet.h"
#include "PythonDispatchModeTLS.h"

namespace tensorplay {
namespace python_dispatch {

// Static description of one operator overload, emitted by codegen.
struct OpEntry {
    const char* name;            // "add.Tensor"
    const char* schema;          // full schema text
    const char* const* arg_names;
    int num_args;
    int num_positional;          // arguments before the keyword-only marker
    const char* tags;            // comma-separated schema tags
    // Lazily resolved OpOverload object; process lifetime.
    mutable PyObject* overload;
};

// One mode invocation.  Construction takes the interpreter lock, pops the
// innermost mode and excludes every dispatch layer above the Python key, so
// operators the handler re-enters reach the next mode (or the backend)
// without recording autograd history.  Destruction undoes all of it, also
// when the handler raised.
class ModeCall {
public:
    explicit ModeCall(const OpEntry& entry);
    ~ModeCall();

    ModeCall(const ModeCall&) = delete;
    ModeCall& operator=(const ModeCall&) = delete;

    // Stores argument ``index``; steals ``value``.  A null value means the
    // conversion raised, which is reported as a PythonError.
    void set_arg(int index, PyObject* value);

    // Calls the mode's __tensorplay_dispatch__ and returns a new reference.
    // Throws PythonError when the handler (or the conversion) raised.
    PyObject* invoke();

private:
    struct Gil {
        PyGILState_STATE state = PyGILState_Ensure();
        ~Gil() { PyGILState_Release(state); }
    };
    // Restores the popped mode, including when construction fails later.
    struct ModeRestore {
        impl::DispatchModeHandle mode = impl::DispatchModeTLS::pop();
        ~ModeRestore() { impl::DispatchModeTLS::push(std::move(mode)); }
    };

    const OpEntry& entry_;
    Gil gil_;                    // first in, last out
    ModeRestore mode_;
    impl::ExcludeDispatchKeyGuard exclude_;
    PyObject* args_ = nullptr;   // one slot per schema argument
};

// Converts a handler result, owning ``result``: the conversion runs under
// the lock ModeCall holds and releases the reference afterwards.
template <typename Convert>
auto convert_result(PyObject* result, Convert&& convert) {
    struct Release {
        PyObject* obj;
        ~Release() { Py_DECREF(obj); }
    } release{result};
    try {
        return convert(result);
    } catch (const python_c::PythonError&) {
        throw;
    } catch (const std::exception&) {
        if (PyErr_Occurred()) throw python_c::PythonError();
        throw;
    }
}

// Raises PythonError for a conversion that returned null.
inline PyObject* checked(PyObject* value) {
    if (value == nullptr) throw python_c::PythonError();
    return value;
}

// Conversions the bridge helpers do not cover.  Each returns a new
// reference (null with a Python error set) or throws.
PyObject* wrap_tensor_or_none(const Tensor& tensor);
PyObject* wrap_optional_tensor_list(const std::optional<std::vector<Tensor>>& tensors);
PyObject* wrap_memory_format(int64_t value);
PyObject* wrap_optional_memory_format(const std::optional<int64_t>& value);
PyObject* wrap_layout(int64_t value);
PyObject* wrap_optional_layout(const std::optional<int64_t>& value);
PyObject* wrap_string(const std::string& value);
PyObject* wrap_optional_int64_list(const std::vector<std::optional<int64_t>>& values);
PyObject* wrap_string_list(const std::vector<std::string>& values);
PyObject* wrap_optional_scalar_list(const std::optional<std::vector<Scalar>>& values);
PyObject* wrap_optional_bool_list(const std::optional<std::vector<bool>>& values);

SymInt parse_symint(PyObject* obj);
SymBool parse_symbool(PyObject* obj);
SymFloat parse_symfloat(PyObject* obj);
std::vector<SymInt> parse_symint_list(PyObject* obj);
std::vector<int64_t> parse_int64_list(PyObject* obj);
std::optional<Tensor> parse_optional_tensor(PyObject* obj);
std::vector<std::optional<Tensor>> parse_optional_tensor_list(PyObject* obj);

// Borrowed item ``index`` of a returned sequence of exactly ``size`` items.
PyObject* result_item(PyObject* result, Py_ssize_t size, Py_ssize_t index,
                      const char* op_name);

// Called once at import: installs the Python-object release hook used by
// p10 and registers every generated Python-key kernel.
void initialize();

// Generated (PythonDispatchGenerated.cpp).
void register_python_kernels();
const OpEntry* const* python_dispatch_entries(int64_t* count);

} // namespace python_dispatch
} // namespace tensorplay

// CPythonBridge.h -- conversion surface consumed by TensorCPythonGenerated.h
// (tools/codegen/gen_python_c.py).
//
// Implemented in src/bindings/python/CPythonBridge.cpp on top of pybind11
// casters; keeps the generated METH_FASTCALL layer free of pybind dispatch.
#pragma once

#include <Python.h>

#include <string>
#include <vector>

#include <Tensor.h>
#include <Scalar.h>
#include <DType.h>
#include <Device.h>

namespace tensorplay {
namespace python_c {

// ---- arg parsing -----------------------------------------------------------
struct ParsedArgs {
    // Owns merged positional+keyword slots in kwlist order.
    std::vector<PyObject*> owned;
    PyObject* pos(int i) const { return owned[static_cast<size_t>(i)]; }
};

ParsedArgs tpx_py_parse(PyObject* const* args, Py_ssize_t nargs,
                        PyObject* kwnames, const char* const* kwlist,
                        Py_ssize_t nkws, const char* op_name);

// Zero-allocation variant: merges positional/keyword args into caller-owned
// slots (out[i] == nullptr means "not supplied").  Hot path for generated
// METH_FASTCALL bindings; no heap traffic per call.
void tpx_py_parse_into(PyObject* const* args, Py_ssize_t nargs,
                       PyObject* kwnames, const char* const* kwlist,
                       Py_ssize_t nkws, const char* op_name,
                       PyObject** out);

// ---- eager argument type validation ----------------------------------------
// Generated bindings pass a parallel table of slot kinds so type mismatches
// raise with upstream python_arg_parser wording while full context (op name,
// argument name, positional index) is still available.  Kinds are the base
// category optionally ORed with TPK_OPTIONAL (None tolerated).
enum tpx_py_type_kind : unsigned char {
    TPK_TENSOR = 1,
    TPK_NUMBER,
    TPK_INT,
    TPK_FLOAT,
    TPK_BOOL,
    TPK_STR,
    TPK_DTYPE,
    TPK_DEVICE,
    TPK_INTLIST,
    TPK_FLOATLIST,
    TPK_TENSORLIST,
    TPK_SCALARLIST,
};
constexpr unsigned char TPK_OPTIONAL = 0x80;

// `slots[first .. first+n)` hold merged arguments whose names are `names`
// (the kwlist array) and kinds `kinds`; `max_pos` is how many leading slots
// may be passed positionally -- torch only annotates "(position N)" for
// those.  Null slots are skipped: required-missing and default injection are
// handled by the generated prologue around this call.
void tpx_py_check_types(PyObject* const* slots, Py_ssize_t n,
                        const char* op_name, const char* const* names,
                        const unsigned char* kinds, int max_pos);

// ---- unpacking -------------------------------------------------------------
// Tensor getters come in three flavours: by value (legacy), and by reference
// into the Python wrapper's storage.  The reference forms skip one
// refcount pair per tensor argument; the mutable form is required for
// in-place ops so writes land on the caller's tensor, not a copy.  The
// borrowed references stay valid while the caller holds the argument
// objects (always true inside a METH_FASTCALL entry point).
Tensor tpx_py_tensor(PyObject* obj);
const Tensor& tpx_py_tensor_cref(PyObject* obj);
Tensor& tpx_py_tensor_mref(PyObject* obj);
Scalar tpx_py_scalar(PyObject* obj);
std::optional<Tensor> tpx_py_opt_tensor(PyObject* obj);
int64_t tpx_py_int64(PyObject* obj);
double tpx_py_double(PyObject* obj);
bool tpx_py_bool(PyObject* obj);
std::optional<int64_t> tpx_py_opt_int64(PyObject* obj);
std::optional<double> tpx_py_opt_double(PyObject* obj);
std::optional<bool> tpx_py_opt_bool(PyObject* obj);
std::optional<Scalar> tpx_py_opt_scalar(PyObject* obj);
std::optional<Device> tpx_py_opt_device(PyObject* obj);
std::vector<int64_t> tpx_py_intlist(PyObject* obj);
std::vector<double> tpx_py_doublelist(PyObject* obj);
// Upstream parity (python_arg_parser.cpp TENSOR_LIST): only tuple/list are
// accepted -- arbitrary sequences fall through to overload dispatch.
std::vector<Tensor> tpx_py_tensorlist(PyObject* obj);
std::vector<Scalar> tpx_py_scalarlist(PyObject* obj);
std::optional<std::vector<int64_t>> tpx_py_opt_intlist(PyObject* obj);
std::string tpx_py_string(PyObject* obj);
std::optional<std::string> tpx_py_opt_string(PyObject* obj);
DType tpx_py_dtype(PyObject* obj);
std::optional<DType> tpx_py_opt_dtype(PyObject* obj);

// ---- GIL -------------------------------------------------------------------
// Upstream releases the GIL around every dispatched kernel
// (gen_python_functions emits `gil_scoped_release` unconditionally); this is
// the pybind-free equivalent for the FASTCALL layer.  Never hold it across
// a Python C-API call: every use site must restore before touching PyObject.
struct tpx_py_GilRelease {
    PyThreadState* state;
    tpx_py_GilRelease() : state(PyEval_SaveThread()) {}
    ~tpx_py_GilRelease() { PyEval_RestoreThread(state); }
    tpx_py_GilRelease(const tpx_py_GilRelease&) = delete;
    tpx_py_GilRelease& operator=(const tpx_py_GilRelease&) = delete;
};

// ---- packing ---------------------------------------------------------------
PyObject* tpx_py_wrap(const Tensor& t);
PyObject* tpx_py_wrap_scalar(const Scalar& s);
PyObject* tpx_py_wrap_dtype(const DType& dt);
PyObject* tpx_py_wrap_device(const Device& d);
PyObject* tpx_py_wrap_tuple(const std::tuple<Tensor, Tensor>& t);
PyObject* tpx_py_wrap_tuple3(const std::tuple<Tensor, Tensor, Tensor>& t);
PyObject* tpx_py_wrap_tuple4(
    const std::tuple<Tensor, Tensor, Tensor, Tensor>& t);
PyObject* tpx_py_wrap_list(const std::vector<Tensor>& v);

// keep `self` alive while the returned alias references its storage
void tpx_py_keep_alive(PyObject* self);

// exception translation: sets a Python error from a C++ exception
void tpx_py_set_error(const std::exception& e);

}  // namespace python_c
}  // namespace tensorplay

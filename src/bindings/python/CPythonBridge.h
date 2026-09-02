// CPythonBridge.h -- conversion surface consumed by TensorCPythonGenerated.h
// (tools/codegen/gen_python_c.py).
//
// Implemented in src/bindings/python/CPythonBridge.cpp on top of pybind11
// casters; keeps the generated METH_FASTCALL layer free of pybind dispatch.
#pragma once

#include <Python.h>

#include <string>
#include <optional>
#include <vector>

#include <Tensor.h>
#include <Scalar.h>
#include <DType.h>
#include <Device.h>
#include <Generator.h>
#include <Storage.h>
#include <SymBool.h>
#include <SymFloat.h>
#include <SymInt.h>

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

// Tests whether a FASTCALL keyword tuple contains a literal name.
// The tuple is supplied by the interpreter and is borrowed by the caller.
bool tpx_py_kwnames_has(PyObject* kwnames, const char* name);

// Give Python Tensor subclasses the first chance to handle an operator.
// Returns 1 when result is owned by the caller, 0 when native parsing should
// continue, and -1 when Python has already set an exception.
int tpx_py_try_tensor_subclass_dispatch(
    const char* op_name, PyObject* receiver, bool is_method,
    PyObject* const* args, Py_ssize_t nargs, PyObject* kwnames,
    PyObject** result);

int tpx_py_try_tensor_function_dispatch(
    const char* op_name, PyObject* receiver, bool is_method,
    PyObject* const* args, Py_ssize_t nargs, PyObject* kwnames,
    PyObject** result);

enum tpx_py_function_state : unsigned char {
    TPX_FUNCTION_ENABLED = 0,
    TPX_SUBCLASSES_DISABLED = 1,
    TPX_ALL_DISABLED = 2,
};

int tpx_py_get_function_state();
bool tpx_py_set_function_state(int state);
bool tpx_py_exchange_skip_next(bool value);
bool tpx_py_peek_skip_next();
bool tpx_py_exchange_subclass_skip_next(bool value);
bool tpx_py_peek_subclass_skip_next();
int tpx_py_get_dispatch_layer();

void tpx_py_push_function_mode(PyObject* mode);
PyObject* tpx_py_pop_function_mode();
PyObject* tpx_py_get_function_mode(Py_ssize_t index);
Py_ssize_t tpx_py_function_mode_len();

// Runs the top function mode for a generated operation.  The mode is
// temporarily removed while its hook runs so nested operations continue at
// the next level.  Return values have the same ownership convention as the
// subclass dispatch helper above.
int tpx_py_try_function_mode_dispatch(
    const char* op_name, PyObject* receiver, bool is_method,
    PyObject* const* args, Py_ssize_t nargs, PyObject* kwnames,
    PyObject** result);

// ---- eager argument type validation ----------------------------------------
// Generated bindings pass a parallel table of slot kinds so type mismatches
// include the operation name, argument name, and positional index.  Kinds are
// the base category optionally ORed with TPK_OPTIONAL (None tolerated).
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
    TPK_BOOLLIST,
    TPK_TENSORLIST_OPTIONAL,
    TPK_GENERATOR,
    TPK_STORAGE,
};
constexpr unsigned char TPK_OPTIONAL = 0x80;

// `slots[first .. first+n)` hold merged arguments whose names are `names`
// (the kwlist array) and kinds `kinds`; `max_pos` is how many leading slots
// may be passed positionally.  Null slots are skipped: required-missing and
// default injection are handled by the generated prologue around this call.
void tpx_py_check_types(PyObject* const* slots, Py_ssize_t n,
                        const char* op_name, const char* const* names,
                        const unsigned char* kinds, int max_pos);

// Non-throwing per-object kind predicate used by the generated multi-overload
// fast probe to pick a candidate overload without raising on a mismatch.
// Returns false for a null object; treats Py_None as matching only when the
// kind is flagged TPK_OPTIONAL.
bool tpx_py_obj_matches_kind(PyObject* obj, unsigned char kind);

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
// Non-throwing guard helpers for compiled-kernel entry points: read the
// version counter / autograd flag straight from the C++ value holder.
// Return -1 (version) / -1 (flag) when the object is not a plain tensor
// wrapper; PyErr is cleared so callers only test the integer.
long long tpx_tensor_version(PyObject* obj);
int tpx_tensor_requires_grad(PyObject* obj);
// Combined steady-state guard probe: classifies the tensor and reads its
// version in one call without letting the missing-counter case throw.
// Returns 0 for a versioned tensor (*version_out set), 1 for an inference
// tensor (no version counter, immutable -- fingerprint by identity alone),
// -1 when the object is not a plain tensor wrapper.  PyErr is cleared so
// callers only test the integer.
int tpx_tensor_guard_probe(PyObject* obj, long long* version_out);
Scalar tpx_py_scalar(PyObject* obj);
std::optional<Tensor> tpx_py_opt_tensor(PyObject* obj);
int64_t tpx_py_int64(PyObject* obj);
double tpx_py_double(PyObject* obj);
bool tpx_py_bool(PyObject* obj);
std::optional<int64_t> tpx_py_opt_int64(PyObject* obj);
std::optional<double> tpx_py_opt_double(PyObject* obj);
std::optional<bool> tpx_py_opt_bool(PyObject* obj);
std::optional<Scalar> tpx_py_opt_scalar(PyObject* obj);
Generator tpx_py_generator(PyObject* obj);
std::optional<Generator> tpx_py_opt_generator(PyObject* obj);
Storage tpx_py_storage(PyObject* obj);
Device tpx_py_device(PyObject* obj);
std::optional<Device> tpx_py_opt_device(PyObject* obj);
std::vector<int64_t> tpx_py_intlist(PyObject* obj);
std::vector<double> tpx_py_doublelist(PyObject* obj);
// Fixed-width bool lists (`bool[3] output_mask` and friends).
std::vector<bool> tpx_py_boollist(PyObject* obj);
// Tensor lists accept tuple/list containers; other sequences fall through to
// overload dispatch.
std::vector<Tensor> tpx_py_tensorlist(PyObject* obj);
std::vector<std::optional<Tensor>> tpx_py_opt_tensorlist(PyObject* obj);
std::vector<Scalar> tpx_py_scalarlist(PyObject* obj);
std::optional<std::vector<int64_t>> tpx_py_opt_intlist(PyObject* obj);
std::optional<std::vector<double>> tpx_py_opt_doublelist(PyObject* obj);
std::string tpx_py_string(PyObject* obj);
std::optional<std::string> tpx_py_opt_string(PyObject* obj);
DType tpx_py_dtype(PyObject* obj);
std::optional<DType> tpx_py_opt_dtype(PyObject* obj);

// ---- GIL -------------------------------------------------------------------
// The wrapper releases the GIL around every dispatched kernel.  This is the
// pybind-free equivalent for the FASTCALL layer.  Never hold it across a
// Python C-API call: every use site must restore before touching PyObject.
struct tpx_py_GilRelease {
    PyThreadState* state;
    tpx_py_GilRelease() : state(PyEval_SaveThread()) {}
    ~tpx_py_GilRelease() { PyEval_RestoreThread(state); }
    tpx_py_GilRelease(const tpx_py_GilRelease&) = delete;
    tpx_py_GilRelease& operator=(const tpx_py_GilRelease&) = delete;
};

// ---- packing ---------------------------------------------------------------
PyObject* tpx_py_wrap(const Tensor& t);
PyObject* tpx_py_wrap_optional_tensor(const std::optional<Tensor>& t);
PyObject* tpx_py_wrap_scalar(const Scalar& s);
PyObject* tpx_py_wrap_optional_scalar(const std::optional<Scalar>& s);
PyObject* tpx_py_wrap_symint(const SymInt& value);
PyObject* tpx_py_wrap_symbool(const SymBool& value);
PyObject* tpx_py_wrap_symfloat(const SymFloat& value);
PyObject* tpx_py_wrap_optional_symint(const std::optional<SymInt>& value);
PyObject* tpx_py_wrap_optional_symbool(const std::optional<SymBool>& value);
PyObject* tpx_py_wrap_optional_symfloat(const std::optional<SymFloat>& value);
PyObject* tpx_py_wrap_symintlist(const std::vector<SymInt>& values);
PyObject* tpx_py_wrap_symboollist(const std::vector<SymBool>& values);
PyObject* tpx_py_wrap_symfloatlist(const std::vector<SymFloat>& values);
PyObject* tpx_py_wrap_optional_symintlist(
    const std::optional<std::vector<SymInt>>& values);
PyObject* tpx_py_wrap_optional_symboollist(
    const std::optional<std::vector<SymBool>>& values);
PyObject* tpx_py_wrap_optional_symfloatlist(
    const std::optional<std::vector<SymFloat>>& values);
PyObject* tpx_py_wrap_generator(const Generator& g);
PyObject* tpx_py_wrap_storage(const Storage& storage);
PyObject* tpx_py_wrap_dtype(const DType& dt);
PyObject* tpx_py_wrap_device(const Device& d);
PyObject* tpx_py_wrap_optional_generator(const std::optional<Generator>& g);
PyObject* tpx_py_wrap_optional_int64(const std::optional<int64_t>& v);
PyObject* tpx_py_wrap_optional_double(const std::optional<double>& v);
PyObject* tpx_py_wrap_optional_bool(const std::optional<bool>& v);
PyObject* tpx_py_wrap_optional_string(const std::optional<std::string>& v);
PyObject* tpx_py_wrap_optional_dtype(const std::optional<DType>& dt);
PyObject* tpx_py_wrap_optional_device(const std::optional<Device>& d);
PyObject* tpx_py_wrap_tuple(const std::tuple<Tensor, Tensor>& t);
PyObject* tpx_py_wrap_tuple3(const std::tuple<Tensor, Tensor, Tensor>& t);
PyObject* tpx_py_wrap_tuple4(
    const std::tuple<Tensor, Tensor, Tensor, Tensor>& t);
PyObject* tpx_py_wrap_list(const std::vector<Tensor>& v);
PyObject* tpx_py_wrap_optional_tensor_list(
    const std::vector<std::optional<Tensor>>& v);
PyObject* tpx_py_wrap_intlist(const std::vector<int64_t>& v);
PyObject* tpx_py_wrap_doublelist(const std::vector<double>& v);
PyObject* tpx_py_wrap_optional_intlist(
    const std::optional<std::vector<int64_t>>& v);
PyObject* tpx_py_wrap_optional_doublelist(
    const std::optional<std::vector<double>>& v);
PyObject* tpx_py_wrap_boollist(const std::vector<bool>& v);
PyObject* tpx_py_wrap_scalarlist(const std::vector<Scalar>& v);

// keep `self` alive while the returned alias references its storage
void tpx_py_keep_alive(PyObject* self);

// exception translation: sets a Python error from a C++ exception
void tpx_py_set_error(const std::exception& e);

}  // namespace python_c
}  // namespace tensorplay

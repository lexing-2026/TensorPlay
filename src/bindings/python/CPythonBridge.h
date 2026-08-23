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
std::string tpx_py_string(PyObject* obj);
DType tpx_py_dtype(PyObject* obj);
std::optional<DType> tpx_py_opt_dtype(PyObject* obj);

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

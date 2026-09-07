#pragma once

#include <Python.h>

#include "DType.h"

namespace tensorplay {
namespace python {

// NumPy interop entry points for the bindings layer.  Implemented in
// TensorNumpy.cpp, the one translation unit that performs the NumPy C-API
// import; all other units call these helpers instead of touching the
// PyArray_* macros directly.

bool is_numpy_available();

int tp_to_numpy_dtype(DType scalar_type);
DType numpy_dtype_to_tp(int dtype);

bool is_numpy_int(PyObject* obj);
bool is_numpy_bool(PyObject* obj);
bool is_numpy_scalar(PyObject* obj);

// Convert a NumPy scalar (anything answering PyArray_CheckScalar) into a
// freshly created 0-dim NumPy array, or return nullptr with a Python error
// set when the scalar type cannot be represented.
PyObject* numpy_scalar_to_array(PyObject* obj);

}  // namespace python
}  // namespace tensorplay

#include "TensorNumpy.h"

#include "Exception.h"

#include <string>

#define WITH_NUMPY_IMPORT_ARRAY
#include "numpy_stub.h"

namespace tensorplay {
namespace python {

bool is_numpy_available() {
  static bool available = []() {
    if (_import_array() >= 0) {
      return true;
    }
    // Try to get exception message, print warning and return false
    std::string message = "Failed to initialize NumPy";
    PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    if (auto str = value ? PyObject_Str(value) : nullptr) {
      if (auto enc_str = PyUnicode_AsEncodedString(str, "utf-8", "strict")) {
        if (auto byte_str = PyBytes_AS_STRING(enc_str)) {
          message += ": " + std::string(byte_str);
        }
        Py_XDECREF(enc_str);
      }
      Py_XDECREF(str);
    }
    PyErr_Clear();
    PyErr_WarnEx(PyExc_UserWarning, message.c_str(), 0);
    return false;
  }();
  return available;
}

int tp_to_numpy_dtype(DType scalar_type) {
  switch (scalar_type) {
    case DType::Float64:
      return NPY_DOUBLE;
    case DType::Float32:
      return NPY_FLOAT;
    case DType::Float16:
      return NPY_HALF;
    case DType::ComplexDouble:
      return NPY_COMPLEX128;
    case DType::ComplexFloat:
      return NPY_COMPLEX64;
    case DType::Int16:
      return NPY_INT16;
    case DType::Int8:
      return NPY_INT8;
    case DType::UInt8:
      return NPY_UINT8;
    case DType::UInt16:
      return NPY_UINT16;
    case DType::UInt32:
      return NPY_UINT32;
    case DType::UInt64:
      return NPY_UINT64;
    case DType::Bool:
      return NPY_BOOL;
    default:
      TP_THROW(TypeError, "Got unsupported scalar type");
  }
}

DType numpy_dtype_to_tp(int dtype) {
  switch (dtype) {
    case NPY_DOUBLE:
      return DType::Float64;
    case NPY_FLOAT:
      return DType::Float32;
    case NPY_HALF:
      return DType::Float16;
    case NPY_COMPLEX64:
      return DType::ComplexFloat;
    case NPY_COMPLEX128:
      return DType::ComplexDouble;
    case NPY_INT16:
      return DType::Int16;
    case NPY_INT8:
      return DType::Int8;
    case NPY_UINT8:
      return DType::UInt8;
    case NPY_UINT16:
      return DType::UInt16;
    case NPY_UINT32:
      return DType::UInt32;
    case NPY_UINT64:
      return DType::UInt64;
    case NPY_BOOL:
      return DType::Bool;
    default:
      // Workaround: some toolchains reject two switch cases with the same
      // value, so the aliased integer widths are resolved here instead.
      if (dtype == NPY_INT || dtype == NPY_INT32) {
        // NPY_INT32 may alias NPY_INT, NPY_LONG, or NPY_INT64 depending on
        // the platform's integer widths; all of them are 32-bit signed.
        return DType::Int32;
      } else if (dtype == NPY_LONGLONG || dtype == NPY_INT64) {
        return DType::Int64;
      } else {
        break;
      }
  }
  PyObject* pytype = PyArray_TypeObjectFromType(dtype);
  if (!pytype) {
    PyErr_Clear();
    TP_THROW(TypeError, "can't convert np.ndarray of an unsupported type");
  }
  const char* name = ((PyTypeObject*)pytype)->tp_name;
  std::string message =
      "can't convert np.ndarray of type " + std::string(name) +
      ". The only supported types are: float64, float32, float16, "
      "complex64, complex128, int64, int32, int16, int8, uint64, uint32, "
      "uint16, uint8, and bool.";
  Py_DECREF(pytype);
  TP_THROW(TypeError, message);
}

bool is_numpy_int(PyObject* obj) {
  return is_numpy_available() && PyArray_IsScalar((obj), Integer);
}

bool is_numpy_bool(PyObject* obj) {
  return is_numpy_available() && PyArray_IsScalar((obj), Bool);
}

bool is_numpy_scalar(PyObject* obj) {
  return is_numpy_available() &&
      (is_numpy_int(obj) || PyArray_IsScalar(obj, Bool) ||
       PyArray_IsScalar(obj, Floating) ||
       PyArray_IsScalar(obj, ComplexFloating));
}

PyObject* numpy_scalar_to_array(PyObject* obj) {
  return PyArray_FromScalar(obj, nullptr);
}

}  // namespace python
}  // namespace tensorplay

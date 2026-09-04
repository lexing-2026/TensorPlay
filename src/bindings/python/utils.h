#pragma once
#include <Python.h>
#include "Autograd.h"
#include "Context.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <complex>
#include <type_traits>

namespace tensorplay {
namespace python {

namespace ops = tensorplay::tpx::ops;

using Tensor = tensorplay::Tensor;

// Helper function to parse shape from args
inline std::vector<int64_t> parse_shape_args(py::args args) {
    std::vector<int64_t> shape;
    if (args.size() == 1 && (py::isinstance<py::list>(args[0]) || py::isinstance<py::tuple>(args[0]) || py::isinstance<Size>(args[0]))) {
        py::object obj = args[0];
        for (auto item : obj) {
            shape.push_back(py::cast<int64_t>(item));
        }
    } else {
        for (auto item : args) {
            shape.push_back(py::cast<int64_t>(item));
        }
    }
    return shape;
}

// Recursively parse Python list shape, verify regularity 
// (e.g., [[1,2],[3]] is irregular, throw error)
inline bool IsListOrTuple(PyObject* obj) {
    return PyList_Check(obj) || PyTuple_Check(obj);
}

inline int64_t GetSize(PyObject* obj) {
    if (PyList_Check(obj)) return PyList_Size(obj);
    if (PyTuple_Check(obj)) return PyTuple_Size(obj);
    return 0;
}

inline PyObject* GetItem(PyObject* obj, int64_t i) {
    if (PyList_Check(obj)) return PyList_GetItem(obj, i); // Borrowed
    if (PyTuple_Check(obj)) return PyTuple_GetItem(obj, i); // Borrowed
    return nullptr;
}

inline void parse_shape(PyObject* list, std::vector<int64_t>& shape, int depth = 0) {
    if (depth > 128) {
        TP_THROW(RuntimeError, "Recursion depth exceeded in list_to_tensor");
    }
    if (!IsListOrTuple(list)) {
        TP_THROW(TypeError, "Input must be a list or tuple");
    }
    int64_t len = GetSize(list);
    shape.push_back(len);
    if (len == 0) return;

    // Ensure all sublists have the same length and recursively check shape
    PyObject* first = GetItem(list, 0);
    if (IsListOrTuple(first)) {
        std::vector<int64_t> sub_shape;
        parse_shape(first, sub_shape, depth + 1);
        // Verify all sublists have the same shape
        for (int64_t i = 1; i < len; ++i) {
            PyObject* sublist = GetItem(list, i);
            if (!IsListOrTuple(sublist)) {
                TP_THROW(ValueError, "Irregular list (mixed types)");
            }
            std::vector<int64_t> cur_sub_shape;
            parse_shape(sublist, cur_sub_shape, depth + 1);
            if (cur_sub_shape != sub_shape) {
                TP_THROW(ValueError, "Irregular list (sublists have different lengths)");
            }
        }
        // Merge sub-shape (e.g., [2] + [3] -> [2,3])
        shape.insert(shape.end(), sub_shape.begin(), sub_shape.end());
    }
}

// Infer data type of list elements 
// (uniform to highest precision, e.g., int and float mixed -> float64)
inline DType infer_dtype(PyObject* list, int depth = 0) {
    if (depth > 128) {
        TP_THROW(RuntimeError, "Recursion depth exceeded in infer_dtype");
    }
    if (!IsListOrTuple(list)) {
        TP_THROW(TypeError, "Can not transform ", std::string(Py_TYPE(list)->tp_name), " to tensor");
    }
    int64_t len = GetSize(list);
    if (len == 0) return DType::Float32;  // Empty list default to float32

    bool has_float = false;
    bool has_complex = false;
    bool has_int = false;
    bool has_bool = false;
    bool has_complex_double = false;
    
    // We only check the first element to guess type for optimization? 
    // Actually, we check all elements to be safe.
    // For example, for mixed types like [1.0, 2], we need to scan.
    // Let's scan all elements at this level. Recursive calls will scan sub-levels.
    
    for (int64_t i = 0; i < len; ++i) {
        PyObject* item = GetItem(list, i);
        if (IsListOrTuple(item)) {
            // Recursively infer dtype of sublists
            DType sub_dtype = infer_dtype(item, depth + 1);
            if (sub_dtype == DType::ComplexDouble) {
                has_complex = true;
                has_complex_double = true;
            } else if (sub_dtype == DType::ComplexFloat || sub_dtype == DType::ComplexHalf || sub_dtype == DType::BComplex32) {
                has_complex = true;
            } else if (sub_dtype == DType::Float32 || sub_dtype == DType::Float64) has_float = true;
            else if (sub_dtype == DType::Int64 || sub_dtype == DType::Int32) has_int = true;
            else if (sub_dtype == DType::Bool) has_bool = true;
        } else {
            if (PyComplex_Check(item)) {
                has_complex = true;
            } else if (PyFloat_Check(item)) {
                has_float = true;
            } else if (PyBool_Check(item)) {
                has_bool = true;
            } else if (PyLong_Check(item)) {
                has_int = true;
            } else {
                TP_THROW(TypeError, "Unsupported element type (only int/float/bool supported)");
            }
        }
    }
    
    if (has_complex) {
        // floating point dtype (float64 -> complex128, else complex64).
        if (has_complex_double || globalContext().defaultDType() == DType::Float64) {
            return DType::ComplexDouble;
        }
        return DType::ComplexFloat;
    }
    if (has_float) return globalContext().defaultDType();
    if (has_int) return DType::Int64;
    if (has_bool) return DType::Bool;

    return globalContext().defaultDType(); // Default if nothing found (e.g. empty sublists)
}

// Optimized flat copy for the last dimension
template <typename T>
inline T python_scalar_cast(PyObject* item) {
    if (PyComplex_Check(item)) {
        Py_complex value = PyComplex_AsCComplex(item);
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
        if constexpr (is_complex_type_v<T>) {
            using value_type = typename is_complex_type<T>::value_type;
            return T(static_cast<value_type>(value.real),
                     static_cast<value_type>(value.imag));
        } else {
            return static_cast<T>(value.real);
        }
    }
    if (PyFloat_Check(item)) return static_cast<T>(PyFloat_AsDouble(item));
    if (PyBool_Check(item)) return static_cast<T>(item == Py_True);
    if (PyLong_Check(item)) {
        if constexpr (std::is_unsigned_v<T>) {
            auto value = PyLong_AsUnsignedLongLong(item);
            if (PyErr_Occurred()) throw py::error_already_set();
            return static_cast<T>(value);
        } else {
            auto value = PyLong_AsLongLong(item);
            if (PyErr_Occurred()) throw py::error_already_set();
            return static_cast<T>(value);
        }
    }
    TP_THROW(TypeError, "Unsupported element type in flat copy");
}

template <typename T>
void copy_data_flat(PyObject* list, T* data, size_t& index) {
    int64_t len = GetSize(list);
    for (int64_t i = 0; i < len; ++i) {
         PyObject* item = GetItem(list, i);
         data[index++] = python_scalar_cast<T>(item);
    }
}

// Recursively copy list data to Tensor memory (row-major order)
template <typename T>
void copy_data(PyObject* list, T* data, size_t& index, const std::vector<int64_t>& shape, int dim) {
    // Optimization for 1D case (last dimension)
    if (dim == shape.size() - 1) {
        copy_data_flat(list, data, index);
        return;
    }

    int64_t len = GetSize(list);
    for (int64_t i = 0; i < len; ++i) {
        PyObject* item = GetItem(list, i);
        // We trust parse_shape so we know item is a list
        copy_data(item, data, index, shape, dim + 1);
    }
}

// Helper dispatch macro for local use
#define TP_DISPATCH_CASE(enum_type, type, ...) \
  case enum_type: { \
    using scalar_t = type; \
    __VA_ARGS__(); \
    break; \
  }

#define TP_DISPATCH_ALL_TYPES(dtype, NAME, ...) \
  switch (dtype) { \
    TP_DISPATCH_CASE(DType::UInt8, uint8_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Int8, int8_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Int16, int16_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Int32, int32_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Int64, int64_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::UInt16, uint16_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::UInt32, uint32_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::UInt64, uint64_t, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Float32, float, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Float64, double, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Float16, tensorplay::Half, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::BFloat16, tensorplay::BFloat16, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::ComplexHalf, std::complex<tensorplay::Half>, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::ComplexFloat, std::complex<float>, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::ComplexDouble, std::complex<double>, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::BComplex32, std::complex<tensorplay::BFloat16>, __VA_ARGS__) \
    TP_DISPATCH_CASE(DType::Bool, bool, __VA_ARGS__) \
    default: \
      TP_THROW(NotImplementedError, std::string(NAME) + " not implemented for this dtype"); \
  }

inline Tensor list_to_tensor(PyObject* list, std::optional<DType> requested_dtype = std::nullopt, std::optional<Device> device = std::nullopt) {
    std::vector<int64_t> shape;
    parse_shape(list, shape);
    
    DType dtype;
    if (requested_dtype.has_value()) {
        dtype = *requested_dtype;
    } else {
        dtype = infer_dtype(list);
    }
    
    // Determine target device
    Device target_device = device.value_or(globalContext().defaultDevice());
    
    // Staging buffer: pinned host memory when the list will be copied to the
    // GPU, so the H2D transfer runs asynchronously and is not pageable-
    // copy bound.
    Tensor t = Tensor(shape, dtype, Device(DeviceType::CPU));
    if (target_device.type() != DeviceType::CPU) {
        t = ops::empty(shape, dtype, t.device(), /*pin_memory=*/true);
    }
    
    // Dispatch copy_data based on dtype
    size_t index = 0;
    
    TP_DISPATCH_ALL_TYPES(dtype, "list_to_tensor", [&] {
        using T = scalar_t;
        T* data_ptr = t.data_ptr<T>();
        copy_data(list, data_ptr, index, shape, 0);
    });
    
    // Move to target device if needed
    if (target_device.type() != DeviceType::CPU) {
        return t.to(target_device);
    }
    
    return t;
}

inline void convert_tensor_data(const Tensor& src, Tensor& dst) {
    dst.copy_(src);
}

} // namespace python
} // namespace tensorplay

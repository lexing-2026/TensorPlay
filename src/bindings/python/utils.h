#pragma once
#include <Python.h>
#include "TPXTensor.h"
#include "Exception.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

namespace tensorplay {
namespace python {

using Tensor = tensorplay::tpx::Tensor;

// Helper function to parse shape from args
inline std::vector<int64_t> parse_shape_args(nanobind::args args) {
    std::vector<int64_t> shape;
    if (args.size() == 1 && (nanobind::isinstance<nanobind::list>(args[0]) || nanobind::isinstance<nanobind::tuple>(args[0]) || nanobind::isinstance<Size>(args[0]))) {
        nanobind::object obj = args[0];
        for (auto item : obj) {
            shape.push_back(nanobind::cast<int64_t>(item));
        }
    } else {
        for (auto item : args) {
            shape.push_back(nanobind::cast<int64_t>(item));
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
    bool has_int = false;
    bool has_bool = false;
    
    // We only check the first element to guess type for optimization? 
    // Actually, we check all elements to be safe.
    // For example, for mixed types like [1.0, 2], we need to scan.
    // Let's scan all elements at this level. Recursive calls will scan sub-levels.
    
    for (int64_t i = 0; i < len; ++i) {
        PyObject* item = GetItem(list, i);
        if (IsListOrTuple(item)) {
            // Recursively infer dtype of sublists
            DType sub_dtype = infer_dtype(item, depth + 1);
            if (sub_dtype == DType::Float32 || sub_dtype == DType::Float64) has_float = true;
            else if (sub_dtype == DType::Int64 || sub_dtype == DType::Int32) has_int = true;
            else if (sub_dtype == DType::Bool) has_bool = true;
        } else {
            if (PyFloat_Check(item)) {
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
    
    if (has_float) return DType::Float32;
    if (has_int) return DType::Int64;
    if (has_bool) return DType::Bool;
    
    return DType::Float32; // Default if nothing found (e.g. empty sublists)
}

// Optimized flat copy for the last dimension
template <typename T>
void copy_data_flat(PyObject* list, T* data, size_t& index) {
    int64_t len = GetSize(list);
    for (int64_t i = 0; i < len; ++i) {
         PyObject* item = GetItem(list, i);
         T val;
         // We trust parse_shape so we don't check for list again
         if (PyFloat_Check(item)) {
             val = static_cast<T>(PyFloat_AsDouble(item));
         } else if (PyBool_Check(item)) {
             val = static_cast<T>(item == Py_True);
         } else if (PyLong_Check(item)) {
             val = static_cast<T>(PyLong_AsLongLong(item));
         } else {
             // Fallback for safety
             TP_THROW(TypeError, "Unsupported element type in flat copy");
         }
         data[index++] = val;
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
    Device target_device = device.value_or(Device(DeviceType::CPU));
    
    // Optimization: If target is CPU, create directly.
    // If target is GPU, create on CPU first (staging) then copy.
    // TODO: Use pinned memory for staging if target is GPU
    
    Tensor t = Tensor(shape, dtype, Device(DeviceType::CPU));
    
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

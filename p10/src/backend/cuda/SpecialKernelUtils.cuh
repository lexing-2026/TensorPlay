#pragma once

#include "CUDALoops.cuh"
#include "Exception.h"
#include "Tensor.h"
#include "TypePromotion.h"
#include "Utils.h"

#include <cstdint>
#include <vector>

namespace tensorplay::cuda::special_detail {

inline std::vector<int64_t> shape_of(const Tensor& tensor) {
    return static_cast<std::vector<int64_t>>(tensor.shape());
}

template <typename F, typename T>
struct FloatMathUnary {
    F function;

    __device__ T operator()(T value) const {
        return static_cast<T>(function(static_cast<double>(value)));
    }
};

template <typename F, typename T>
struct FloatMathBinary {
    F function;

    __device__ T operator()(T lhs, T rhs) const {
        return static_cast<T>(function(static_cast<double>(lhs),
                                       static_cast<double>(rhs)));
    }
};

template <typename F, typename T>
struct TypedMathUnary {
    F function;

    __device__ T operator()(T value) const {
        return function(value);
    }
};

template <typename F, typename T>
struct TypedMathBinary {
    F function;

    __device__ T operator()(T lhs, T rhs) const {
        return function(lhs, rhs);
    }
};

template <typename F>
Tensor float_math_cuda(const Tensor& self, F function, const char*) {
    DType input_dtype = self.dtype();
    DType output_dtype = isFloatingType(input_dtype) ? input_dtype : DType::Float32;
    DType compute_dtype = (input_dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor input = (input_dtype == compute_dtype)
        ? self
        : self.to(compute_dtype);
    Tensor output = Tensor::empty(shape_of(input), compute_dtype, input.device());
    if (input.numel() > 0) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(output)
            .add_input(input)
            .build();
        if (compute_dtype == DType::Float64) {
            gpu_kernel(iter, FloatMathUnary<F, double>{function});
        } else {
            gpu_kernel(iter, FloatMathUnary<F, float>{function});
        }
    }
    return (output_dtype == compute_dtype) ? output : output.to(output_dtype);
}

template <typename F>
Tensor binary_float_cuda(const Tensor& lhs_in, const Tensor& rhs_in, F function,
                         const char* name) {
    if (lhs_in.device() != rhs_in.device()) {
        TP_THROW(DeviceMismatchError, name,
                 ": inputs must be on the same device");
    }
    DType dtype = promoteTypes(lhs_in.dtype(), rhs_in.dtype());
    if (!isFloatingType(dtype)) {
        dtype = DType::Float32;
    }
    DType compute_dtype = (dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    const std::vector<int64_t> output_shape =
        broadcast_shapes(shape_of(lhs_in), shape_of(rhs_in));
    Tensor lhs = lhs_in.to(compute_dtype);
    Tensor rhs = rhs_in.to(compute_dtype);
    Tensor output = Tensor::empty(output_shape, compute_dtype, lhs_in.device());
    if (output.numel() == 0) {
        return (dtype == compute_dtype) ? output : output.to(dtype);
    }
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(output)
        .add_input(lhs)
        .add_input(rhs)
        .build();
    if (compute_dtype == DType::Float64) {
        gpu_kernel(iter, FloatMathBinary<F, double>{function});
    } else {
        gpu_kernel(iter, FloatMathBinary<F, float>{function});
    }
    return (dtype == compute_dtype) ? output : output.to(dtype);
}

template <typename F>
Tensor typed_math_cuda(const Tensor& self, F function) {
    DType input_dtype = self.dtype();
    DType output_dtype = isFloatingType(input_dtype) ? input_dtype : DType::Float32;
    DType compute_dtype = (input_dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    Tensor input = (input_dtype == compute_dtype)
        ? self
        : self.to(compute_dtype);
    Tensor output = Tensor::empty(shape_of(input), compute_dtype, input.device());
    if (input.numel() > 0) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(output)
            .add_input(input)
            .build();
        if (compute_dtype == DType::Float64) {
            gpu_kernel(iter, TypedMathUnary<F, double>{function});
        } else {
            gpu_kernel(iter, TypedMathUnary<F, float>{function});
        }
    }
    return (output_dtype == compute_dtype) ? output : output.to(output_dtype);
}

template <typename F>
Tensor typed_binary_cuda(const Tensor& lhs_in, const Tensor& rhs_in, F function) {
    if (lhs_in.device() != rhs_in.device()) {
        TP_THROW(DeviceMismatchError,
                 "special binary inputs must be on the same device");
    }
    DType dtype = promoteTypes(lhs_in.dtype(), rhs_in.dtype());
    if (!isFloatingType(dtype)) {
        dtype = DType::Float32;
    }
    DType compute_dtype = (dtype == DType::Float64) ? DType::Float64 : DType::Float32;
    const std::vector<int64_t> output_shape =
        broadcast_shapes(shape_of(lhs_in), shape_of(rhs_in));
    Tensor lhs = lhs_in.to(compute_dtype);
    Tensor rhs = rhs_in.to(compute_dtype);
    Tensor output = Tensor::empty(output_shape, compute_dtype, lhs_in.device());
    if (output.numel() > 0) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(output)
            .add_input(lhs)
            .add_input(rhs)
            .build();
        if (compute_dtype == DType::Float64) {
            gpu_kernel(iter, TypedMathBinary<F, double>{function});
        } else {
            gpu_kernel(iter, TypedMathBinary<F, float>{function});
        }
    }
    return (dtype == compute_dtype) ? output : output.to(dtype);
}

}  // namespace tensorplay::cuda::special_detail

#pragma once

#include "Tensor.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include "Exception.h"

namespace tensorplay {

// Helper to broadcast shapes
inline std::vector<int64_t> broadcast_shapes(const std::vector<int64_t>& shape1, const std::vector<int64_t>& shape2) {
    int64_t ndim1 = shape1.size();
    int64_t ndim2 = shape2.size();
    int64_t ndim = std::max(ndim1, ndim2);
    std::vector<int64_t> result_shape(ndim);
    
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t dim1 = (i < ndim - ndim1) ? 1 : shape1[i - (ndim - ndim1)];
        int64_t dim2 = (i < ndim - ndim2) ? 1 : shape2[i - (ndim - ndim2)];
        
        if (dim1 == 1) result_shape[i] = dim2;
        else if (dim2 == 1) result_shape[i] = dim1;
        else if (dim1 == dim2) result_shape[i] = dim1;
        else TP_THROW(RuntimeError, "The size of tensor a must match the size of tensor b at non-singleton dimension");
    }
    return result_shape;
}

inline std::vector<int64_t> broadcast_shapes(
    const std::vector<int64_t>& shape1,
    const std::vector<int64_t>& shape2,
    const std::vector<int64_t>& shape3) {
    return broadcast_shapes(broadcast_shapes(shape1, shape2), shape3);
}

// Align an input's strides to an already-computed broadcast output shape.
// A zero stride represents a broadcast singleton dimension and keeps the
// kernel on the original storage without materialising expand().
inline std::vector<int64_t> broadcast_strides(
    const Tensor& tensor, const std::vector<int64_t>& output_shape) {
    const auto input_shape = static_cast<std::vector<int64_t>>(tensor.shape());
    const auto input_strides = tensor.strides();
    if (input_shape.size() > output_shape.size()) {
        TP_THROW(RuntimeError, "input rank is larger than broadcast output rank");
    }

    std::vector<int64_t> result(output_shape.size(), 0);
    const size_t leading = output_shape.size() - input_shape.size();
    for (size_t dim = 0; dim < input_shape.size(); ++dim) {
        const size_t output_dim = leading + dim;
        if (input_shape[dim] != 1 && input_shape[dim] != output_shape[output_dim]) {
            TP_THROW(RuntimeError, "input shape is not broadcastable to output shape");
        }
        result[output_dim] = input_shape[dim] == 1 ? 0 : input_strides[dim];
    }
    return result;
}

// Recursive application of binary op with different input/output types
template <typename OutT, typename InT, typename Op>
void apply_op_recursive(OutT* out_ptr, const std::vector<int64_t>& out_strides,
                       const Tensor& a, const std::vector<int64_t>& a_strides,
                       const Tensor& b, const std::vector<int64_t>& b_strides,
                       int dim, int64_t out_offset, int64_t a_offset, int64_t b_offset,
                       const std::vector<int64_t>& shape, Op op) {
    if (shape.empty()) {
        const InT* a_data = a.data_ptr<InT>();
        const InT* b_data = b.data_ptr<InT>();
        out_ptr[out_offset] = op(a_data[a_offset], b_data[b_offset]);
        return;
    }

    int64_t size = shape[dim];
    if (dim == shape.size() - 1) {
        // Base case: inner loop
        const InT* a_data = a.data_ptr<InT>();
        const InT* b_data = b.data_ptr<InT>();
        
        for (int64_t i = 0; i < size; ++i) {
            out_ptr[out_offset + i * out_strides[dim]] = op(
                a_data[a_offset + i * a_strides[dim]],
                b_data[b_offset + i * b_strides[dim]]
            );
        }
    } else {
        for (int64_t i = 0; i < size; ++i) {
            apply_op_recursive<OutT, InT, Op>(out_ptr, out_strides, a, a_strides, b, b_strides,
                                 dim + 1,
                                 out_offset + i * out_strides[dim],
                                 a_offset + i * a_strides[dim],
                                 b_offset + i * b_strides[dim],
                                 shape, op);
        }
    }
}

// Recursive application of binary op (same types)
template <typename T, typename Op>
void apply_op_recursive(T* out_ptr, const std::vector<int64_t>& out_strides,
                       const Tensor& a, const std::vector<int64_t>& a_strides,
                       const Tensor& b, const std::vector<int64_t>& b_strides,
                       int dim, int64_t out_offset, int64_t a_offset, int64_t b_offset,
                       const std::vector<int64_t>& shape, Op op) {
    apply_op_recursive<T, T, Op>(out_ptr, out_strides, a, a_strides, b, b_strides,
                                dim, out_offset, a_offset, b_offset, shape, op);
}

// Recursive ternary pointwise application used by native addcmul/addcdiv and
// where kernels.  Unlike a Python composition, this visits each output
// element once and reads broadcasted inputs through zero-aligned strides.
template <typename OutT, typename InT, typename Op>
void apply_ternary_op_recursive(
    OutT* out_ptr, const std::vector<int64_t>& out_strides,
    const Tensor& a, const std::vector<int64_t>& a_strides,
    const Tensor& b, const std::vector<int64_t>& b_strides,
    const Tensor& c, const std::vector<int64_t>& c_strides,
    int dim, int64_t out_offset, int64_t a_offset, int64_t b_offset,
    int64_t c_offset, const std::vector<int64_t>& shape, Op op) {
    if (shape.empty()) {
        const InT* a_data = a.data_ptr<InT>();
        const InT* b_data = b.data_ptr<InT>();
        const InT* c_data = c.data_ptr<InT>();
        out_ptr[out_offset] = op(a_data[a_offset], b_data[b_offset], c_data[c_offset]);
        return;
    }

    const int64_t size = shape[dim];
    if (dim == static_cast<int>(shape.size()) - 1) {
        const InT* a_data = a.data_ptr<InT>();
        const InT* b_data = b.data_ptr<InT>();
        const InT* c_data = c.data_ptr<InT>();
        for (int64_t i = 0; i < size; ++i) {
            out_ptr[out_offset + i * out_strides[dim]] = op(
                a_data[a_offset + i * a_strides[dim]],
                b_data[b_offset + i * b_strides[dim]],
                c_data[c_offset + i * c_strides[dim]]);
        }
        return;
    }

    for (int64_t i = 0; i < size; ++i) {
        apply_ternary_op_recursive<OutT, InT>(
            out_ptr, out_strides, a, a_strides, b, b_strides, c, c_strides,
            dim + 1,
            out_offset + i * out_strides[dim],
            a_offset + i * a_strides[dim],
            b_offset + i * b_strides[dim],
            c_offset + i * c_strides[dim], shape, op);
    }
}

// Ternary pointwise application with an independently typed condition/input.
// This is the native implementation primitive for where: the condition is a
// bool tensor while both selected values use the promoted output dtype.
template <typename OutT, typename CondT, typename InT, typename Op>
void apply_ternary_op_recursive_mixed(
    OutT* out_ptr, const std::vector<int64_t>& out_strides,
    const Tensor& condition, const std::vector<int64_t>& condition_strides,
    const Tensor& a, const std::vector<int64_t>& a_strides,
    const Tensor& b, const std::vector<int64_t>& b_strides,
    int dim, int64_t out_offset, int64_t condition_offset,
    int64_t a_offset, int64_t b_offset,
    const std::vector<int64_t>& shape, Op op) {
    if (shape.empty()) {
        const CondT* condition_data = condition.data_ptr<CondT>();
        const InT* a_data = a.data_ptr<InT>();
        const InT* b_data = b.data_ptr<InT>();
        out_ptr[out_offset] = op(
            condition_data[condition_offset], a_data[a_offset], b_data[b_offset]);
        return;
    }

    const int64_t size = shape[dim];
    if (dim == static_cast<int>(shape.size()) - 1) {
        const CondT* condition_data = condition.data_ptr<CondT>();
        const InT* a_data = a.data_ptr<InT>();
        const InT* b_data = b.data_ptr<InT>();
        for (int64_t i = 0; i < size; ++i) {
            out_ptr[out_offset + i * out_strides[dim]] = op(
                condition_data[condition_offset + i * condition_strides[dim]],
                a_data[a_offset + i * a_strides[dim]],
                b_data[b_offset + i * b_strides[dim]]);
        }
        return;
    }

    for (int64_t i = 0; i < size; ++i) {
        apply_ternary_op_recursive_mixed<OutT, CondT, InT>(
            out_ptr, out_strides, condition, condition_strides,
            a, a_strides, b, b_strides, dim + 1,
            out_offset + i * out_strides[dim],
            condition_offset + i * condition_strides[dim],
            a_offset + i * a_strides[dim],
            b_offset + i * b_strides[dim], shape, op);
    }
}

// Recursive application of unary op
template <typename OutT, typename InT, typename Op>
void apply_unary_op_recursive(OutT* out_ptr, const std::vector<int64_t>& out_strides,
                       const Tensor& a, const std::vector<int64_t>& a_strides,
                       int dim, int64_t out_offset, int64_t a_offset,
                       const std::vector<int64_t>& shape, Op op) {
    if (shape.empty()) {
        const InT* a_data = a.data_ptr<InT>();
        out_ptr[out_offset] = op(a_data[a_offset]);
        return;
    }

    int64_t size = shape[dim];
    if (dim == shape.size() - 1) {
        // Base case
        const InT* a_data = a.data_ptr<InT>();
        for (int64_t i = 0; i < size; ++i) {
            out_ptr[out_offset + i * out_strides[dim]] = op(a_data[a_offset + i * a_strides[dim]]);
        }
    } else {
        for (int64_t i = 0; i < size; ++i) {
            apply_unary_op_recursive<OutT, InT, Op>(out_ptr, out_strides, a, a_strides,
                                 dim + 1,
                                 out_offset + i * out_strides[dim],
                                 a_offset + i * a_strides[dim],
                                 shape, op);
        }
    }
}

template <typename T, typename Op>
void apply_unary_op_recursive(T* out_ptr, const std::vector<int64_t>& out_strides,
                       const Tensor& a, const std::vector<int64_t>& a_strides,
                       int dim, int64_t out_offset, int64_t a_offset,
                       const std::vector<int64_t>& shape, Op op) {
    apply_unary_op_recursive<T, T, Op>(out_ptr, out_strides, a, a_strides,
                                      dim, out_offset, a_offset, shape, op);
}

} // namespace tensorplay

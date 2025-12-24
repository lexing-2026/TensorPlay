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

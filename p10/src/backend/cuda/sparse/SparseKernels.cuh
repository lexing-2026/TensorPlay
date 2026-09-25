#pragma once

#include "Tensor.h"
#include "DType.h"
#include "Exception.h"

#include <cstdint>
#include <vector>

namespace tensorplay {
namespace cuda {

constexpr int kMaxSparseDims = 64;

struct SparseBlockLayoutInfo {
    int64_t shape[kMaxSparseDims];
};

struct SparseGatherInfo {
    int sparse_dim;
    int dense_dim;
    int64_t shape[kMaxSparseDims];
    int64_t strides[kMaxSparseDims];
};

constexpr int kCoalesceThreads = 128;

// Layout of a freshly allocated contiguous dense output, passed by value so
// kernels read it from parameter space.
struct DenseLayoutInfo {
    int64_t ndim;
    int64_t shape[kMaxSparseDims];
    int64_t strides[kMaxSparseDims];
};

// Cross-unit helpers.  The launch-shape arithmetic is tiny and shared by
// every unit in this family, so it stays inline in one place.

inline int coalesce_blocks(int64_t n) {
    return static_cast<int>((n + kCoalesceThreads - 1) / kCoalesceThreads);
}


inline SparseGatherInfo make_gather_info(const Tensor& dense, const Tensor& mask) {
    SparseGatherInfo info{};
    info.sparse_dim = static_cast<int>(mask.sparse_dim());
    info.dense_dim = static_cast<int>(mask.dense_dim());
    if (dense.dim() > kMaxSparseDims) {
        TP_THROW(RuntimeError, "sparse_mask(): tensor rank exceeds CUDA sparse limit");
    }
    for (int64_t d = 0; d < dense.dim(); ++d) {
        info.shape[d] = dense.size(d);
        info.strides[d] = dense.stride(d);
    }
    return info;
}


inline DenseLayoutInfo make_layout_info(const std::vector<int64_t>& sizes) {
    TP_CHECK(static_cast<int64_t>(sizes.size()) <= kMaxSparseDims,
             "to_dense(): tensor rank exceeds CUDA sparse limit");
    DenseLayoutInfo info{};
    info.ndim = static_cast<int64_t>(sizes.size());
    int64_t stride = 1;
    for (int64_t d = info.ndim - 1; d >= 0; --d) {
        info.shape[d] = sizes[static_cast<size_t>(d)];
        info.strides[d] = stride;
        stride *= sizes[static_cast<size_t>(d)];
    }
    return info;
}


inline int64_t product_of(const std::vector<int64_t>& dims) {
    int64_t result = 1;
    for (int64_t dim : dims) result *= dim;
    return result;
}

template <typename real_t>
struct CudaComplexPair {
    real_t real;
    real_t imag;
};

template <typename T>
struct CoalesceTypeTag { using type = T; };


template <typename F>
void dispatch_coalesce_dtype(DType dtype, F&& f) {
#define TP_COALESCE_DISPATCH(ctype, name) \
    case DType::name: f(CoalesceTypeTag<ctype>{}); return;
    switch (dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_COALESCE_DISPATCH)
        default:
            TP_THROW(NotImplementedError, "unsupported dtype in CUDA coalesce");
    }
#undef TP_COALESCE_DISPATCH
}

// Layout conversions that the conversion entry points hand to each other.
Tensor coo_to_csr_native(const Tensor& coalesced);
Tensor coo_to_csr_native_cuda(const Tensor& coalesced);
Tensor csr_to_coo_cuda(const Tensor& self);
Tensor to_sparse_coo_native(const Tensor& self);
Tensor to_sparse_coo_native_sparse_dim(const Tensor& self, int64_t sparse_dim);

} // namespace cuda
} // namespace tensorplay

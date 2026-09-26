#pragma once

// Reduction kernels - CUDA.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Context.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "SortingRadixSelect.cuh"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <tuple>
#include <utility>
#include <type_traits>
#include "Atomic.cuh"
#include "OutWrite.h"

namespace tensorplay {
namespace cuda {

namespace ops = tensorplay::tpx::ops;

extern std::tuple<Tensor, Tensor> sort_cuda(const Tensor& self, int64_t dim,
                                            bool descending);
extern std::tuple<Tensor, Tensor> var_mean_dim_kernel(
    const Tensor& self, const std::vector<int64_t>& dim,
    int64_t correction, bool keepdim);
extern Tensor mean_dim_kernel(const Tensor& self,
                              const std::vector<int64_t>& dim,
                              bool keepdim, DType dtype);
extern Tensor sum_dim_kernel(const Tensor& self,
                             const std::vector<int64_t>& dim,
                             bool keepdim, DType dtype);
extern Tensor nansum_dim_kernel(const Tensor& self,
                                const std::vector<int64_t>& dim,
                                bool keepdim, DType dtype);
extern Tensor amax_dim_kernel(const Tensor& self,
                              const std::vector<int64_t>& dim,
                              bool keepdim);
extern Tensor amin_dim_kernel(const Tensor& self,
                              const std::vector<int64_t>& dim,
                              bool keepdim);
extern Tensor var_dim_kernel(const Tensor& self,
                             const std::vector<int64_t>& dim,
                             int64_t correction, bool keepdim);

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

constexpr int kThreads = 256;


inline dim3 make_grid(int64_t work) {
    return dim3(static_cast<unsigned>((work + kThreads - 1) / kThreads));
}


inline int selection_threads(int64_t size) {
    return static_cast<int>(std::min<int64_t>(
        ((size + 31) / 32) * 32, 1024));
}


inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    // Dimension wrapping reports the original (unwrapped) value on error.
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 min, ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}


inline void outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                        int64_t& outer, int64_t& inner) {
    outer = 1; inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
}


inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}


// ---------------------------------------------------------------------------
// Reduction slice kernels (one thread per output slice)
// ---------------------------------------------------------------------------

template <typename T>
__device__ inline bool reduce_value_is_nan(T value) {
    if constexpr (std::is_same<T, float>::value ||
                  std::is_same<T, double>::value) {
        return ::isnan(value);
    } else if constexpr (std::is_same<T, Half>::value ||
                         std::is_same<T, BFloat16>::value) {
        return ::isnan(static_cast<float>(value));
    } else {
        return false;
    }
}


// ===========================================================================
// Reduction entry points
// ===========================================================================

// zero_numel_check_dims): reducing an empty tensor is only valid along an
// explicitly given non-empty dim; a full reduction has no identity.
static void zero_numel_check_dims(const Tensor& self, const std::vector<int64_t>& dims,
                                  const char* fn_name) {
    if (dims.empty()) {
        TP_THROW(RuntimeError, fn_name,
                 ": Expected reduction dim to be specified for input.numel() == 0. "
                 "Specify the reduction dim with the 'dim' argument.");
    }
    const int64_t nd = self.dim();
    for (int64_t d : dims) {
        if (d < 0) d += nd;
        TP_CHECK_INDEX(self.size(d) != 0, fn_name,
                       ": Expected reduction dim ", d, " to have non-zero size.");
    }
}

// Reduced entry points other units call into; defined next door in the
// sumprod / max / min units.
Tensor nansum_cuda2(const Tensor& self, const std::vector<int64_t>& dim_in, bool keepdim);
Tensor amax_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim);
Tensor amin_cuda2(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim);

} // namespace cuda
} // namespace tensorplay

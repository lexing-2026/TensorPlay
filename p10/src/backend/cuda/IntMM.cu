// Device kernel for the integer matrix product.
//
// _int_mm multiplies two 8-bit matrices and accumulates in 32 bits, the shape
// a quantized linear layer needs before its requantization step.  The tile
// walk stages a square block of each operand in shared memory so every 8-bit
// value loaded from global memory is reused across the tile, and the
// accumulator stays in a register for the whole K sweep.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDARuntime.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

#define TP_INT_MM_CHECK(expr)                                                  \
    do {                                                                       \
        cudaError_t status = (expr);                                           \
        if (status != cudaSuccess) {                                           \
            TP_THROW(RuntimeError, std::string("CUDA Error: ") +               \
                                       cudaGetErrorString(status));            \
        }                                                                      \
    } while (0)

constexpr int kIntMMTile = 16;

template <typename lhs_t>
__global__ void int_mm_kernel(int64_t M, int64_t K, int64_t N,
                              const lhs_t* __restrict__ a,
                              const int8_t* __restrict__ b,
                              int32_t* __restrict__ c) {
    __shared__ int32_t a_tile[kIntMMTile][kIntMMTile];
    __shared__ int32_t b_tile[kIntMMTile][kIntMMTile];

    const int64_t row = static_cast<int64_t>(blockIdx.y) * kIntMMTile + threadIdx.y;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * kIntMMTile + threadIdx.x;
    int32_t acc = 0;

    const int64_t tiles = (K + kIntMMTile - 1) / kIntMMTile;
    for (int64_t t = 0; t < tiles; ++t) {
        const int64_t a_col = t * kIntMMTile + threadIdx.x;
        const int64_t b_row = t * kIntMMTile + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K)
                ? static_cast<int32_t>(a[row * K + a_col])
                : 0;
        b_tile[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N)
                ? static_cast<int32_t>(b[b_row * N + col])
                : 0;
        __syncthreads();

        for (int i = 0; i < kIntMMTile; ++i) {
            acc += a_tile[threadIdx.y][i] * b_tile[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = acc;
    }
}

void check_int_mm_operands(const Tensor& self, const Tensor& mat2) {
    TP_CHECK(self.dim() == 2 && mat2.dim() == 2,
             "_int_mm: expected two matrices but got ", self.dim(), "-D and ",
             mat2.dim(), "-D tensors");
    TP_CHECK(self.size(1) == mat2.size(0),
             "_int_mm: cannot multiply a ", self.size(0), " by ", self.size(1),
             " matrix with a ", mat2.size(0), " by ", mat2.size(1), " matrix");
    TP_CHECK(self.dtype() == DType::Int8 || self.dtype() == DType::UInt8,
             "_int_mm: expected the left operand to be int8 or uint8");
    TP_CHECK(mat2.dtype() == DType::Int8,
             "_int_mm: expected the right operand to be int8");
}

}  // namespace

Tensor _int_mm_cuda(const Tensor& self, const Tensor& mat2) {
    check_int_mm_operands(self, mat2);
    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const int64_t N = mat2.size(1);
    Tensor out = Tensor::zeros({M, N}, DType::Int32, self.device());
    if (out.numel() == 0 || K == 0) return out;

    const Tensor lhs = self.contiguous();
    const Tensor rhs = mat2.contiguous();
    const dim3 block(kIntMMTile, kIntMMTile);
    const dim3 grid(static_cast<unsigned>((N + kIntMMTile - 1) / kIntMMTile),
                    static_cast<unsigned>((M + kIntMMTile - 1) / kIntMMTile));
    auto stream = getCurrentCUDAStream().stream();
    if (self.dtype() == DType::Int8) {
        int_mm_kernel<int8_t><<<grid, block, 0, stream>>>(
            M, K, N, lhs.data_ptr<int8_t>(), rhs.data_ptr<int8_t>(),
            out.data_ptr<int32_t>());
    } else {
        int_mm_kernel<uint8_t><<<grid, block, 0, stream>>>(
            M, K, N, lhs.data_ptr<uint8_t>(), rhs.data_ptr<int8_t>(),
            out.data_ptr<int32_t>());
    }
    TP_INT_MM_CHECK(cudaGetLastError());
    return out;
}

Tensor& _int_mm_out_cuda(const Tensor& self, const Tensor& mat2, Tensor& out) {
    const Tensor value = _int_mm_cuda(self, mat2);
    TP_CHECK(out.dtype() == DType::Int32,
             "_int_mm: expected an int32 destination");
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

#undef TP_INT_MM_CHECK

TENSORPLAY_LIBRARY_IMPL(CUDA, IntMMKernels) {
    m.impl("_int_mm", _int_mm_cuda);
    m.impl("_int_mm.out", _int_mm_out_cuda);
}

}  // namespace cuda
}  // namespace tensorplay

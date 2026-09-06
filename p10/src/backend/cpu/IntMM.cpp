// Host kernel for the integer matrix product.
//
// _int_mm multiplies two 8-bit matrices and accumulates in 32 bits, which is
// the shape a quantized linear layer needs before its requantization step.
// The walk is i-k-j: the inner loop streams one row of the right operand and
// one row of the destination contiguously, so every accumulation touches
// consecutive cache lines, and the batch of rows is split across threads.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

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

template <typename lhs_t>
void int_mm_loop(const Tensor& lhs, const Tensor& rhs, Tensor& out,
                 int64_t M, int64_t K, int64_t N) {
    const lhs_t* a = lhs.data_ptr<lhs_t>();
    const int8_t* b = rhs.data_ptr<int8_t>();
    int32_t* c = out.data_ptr<int32_t>();
    parallel::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            int32_t* row = c + i * N;
            std::memset(row, 0, sizeof(int32_t) * static_cast<size_t>(N));
            for (int64_t k = 0; k < K; ++k) {
                const int32_t scale = static_cast<int32_t>(a[i * K + k]);
                if (scale == 0) continue;
                const int8_t* rhs_row = b + k * N;
                for (int64_t j = 0; j < N; ++j) {
                    row[j] += scale * static_cast<int32_t>(rhs_row[j]);
                }
            }
        }
    });
}

}  // namespace

Tensor _int_mm_cpu(const Tensor& self, const Tensor& mat2) {
    check_int_mm_operands(self, mat2);
    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const int64_t N = mat2.size(1);
    Tensor out = Tensor::empty({M, N}, DType::Int32, self.device());
    if (out.numel() == 0) return out;
    if (K == 0) return Tensor::zeros({M, N}, DType::Int32, self.device());

    const Tensor lhs = self.contiguous();
    const Tensor rhs = mat2.contiguous();
    if (self.dtype() == DType::Int8) {
        int_mm_loop<int8_t>(lhs, rhs, out, M, K, N);
    } else {
        int_mm_loop<uint8_t>(lhs, rhs, out, M, K, N);
    }
    return out;
}

Tensor& _int_mm_out_cpu(const Tensor& self, const Tensor& mat2, Tensor& out) {
    const Tensor value = _int_mm_cpu(self, mat2);
    TP_CHECK(out.dtype() == DType::Int32,
             "_int_mm: expected an int32 destination");
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CPU, IntMMKernels) {
    m.impl("_int_mm", _int_mm_cpu);
    m.impl("_int_mm.out", _int_mm_out_cpu);
}

}  // namespace cpu
}  // namespace tensorplay

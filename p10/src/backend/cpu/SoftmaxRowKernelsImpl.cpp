// Tier-compiled last-dimension softmax row walk (f32/f64).
//
// Each CPU capability tier compiles its own copy; vec::maximum and the
// exponential member resolve to that tier's width.  The DEFAULT copy is the
// emulated width, still correct on a host without AVX2.
//
// The row is read twice in log mode (max, then the shifted values again after
// the log of the sum), which is the same traffic the scalar row makes.

#include "cpu/SoftmaxRowKernels.h"

#include "DType.h"
#include "Exception.h"
#include "Parallel.h"
#include "cpu/vec/vec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace tensorplay {
namespace cpu {

namespace {

template <typename T>
void softmax_rows(const T* in, T* out, int64_t rows, int64_t size,
                  bool log_mode) {
    using V = vec::Vectorized<T>;
    constexpr int64_t W = V::size();

    parallel::parallel_for(0, rows, 1, [&](int64_t rb, int64_t re) {
        // One scratch per worker: the lane fold of a row is written here and
        // read back as scalars, so it must not be shared across the pool.
        alignas(64) T buf[V::size()];
        for (int64_t r = rb; r < re; ++r) {
            const T* row = in + r * size;
            T* orow = out + r * size;
            int64_t j = 0;
            T mx = row[0];
            if (size >= W) {
                V m = V::loadu(row);
                for (j = W; j + W <= size; j += W) {
                    m = vec::maximum(m, V::loadu(row + j));
                }
                m.store(buf);
                mx = buf[0];
                for (int64_t l = 1; l < W; ++l) mx = std::max(mx, buf[l]);
            }
            for (; j < size; ++j) mx = std::max(mx, row[j]);

            const V vmx(mx);
            V sv(T(0));
            j = 0;
            if (size >= W) {
                for (; j + W <= size; j += W) {
                    const V e = (V::loadu(row + j) - vmx).exp();
                    if (!log_mode) e.store(orow + j);
                    sv = sv + e;
                }
                sv.store(buf);
            }
            T sum = size >= W ? buf[0] : T(0);
            if (size >= W) {
                for (int64_t l = 1; l < W; ++l) sum += buf[l];
            }
            for (; j < size; ++j) {
                const T e = std::exp(row[j] - mx);
                if (!log_mode) orow[j] = e;
                sum += e;
            }

            if (log_mode) {
                const T lse = std::log(sum);
                for (int64_t k = 0; k < size; ++k) orow[k] = (row[k] - mx) - lse;
            } else {
                const T inv = T(1) / sum;
                for (int64_t k = 0; k < size; ++k) orow[k] *= inv;
            }
        }
    });
}

void softmax_row_impl(const void* in, void* out, int64_t rows,
                      int64_t size, bool log_mode, int dtype) {
    if (rows <= 0 || size <= 0) {
        return;
    }
    switch (static_cast<DType>(dtype)) {
        case DType::Float32:
            softmax_rows(static_cast<const float*>(in),
                         static_cast<float*>(out), rows, size, log_mode);
            return;
        case DType::Float64:
            softmax_rows(static_cast<const double*>(in),
                         static_cast<double*>(out), rows, size, log_mode);
            return;
        default:
            TP_THROW(TypeError, "softmax_row: unsupported dtype");
    }
}

} // namespace

#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(softmax_row_stub, &softmax_row_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(softmax_row_stub, &softmax_row_impl);
#endif

} // namespace cpu
} // namespace tensorplay

// Tier-compiled layer_norm row walk (f32).
//
// Each row is reduced with two vector accumulators (sum and sum of squares over
// two independent lanes of the row), the same shape stats_f32_512 has, and the
// affine pass then walks `(x - mean) * rstd * w + b` a vector at a time.  The
// scalar remainders repeat both formulas element by element, so the last
// partial vector rounds like the loop it replaces.

#include "cpu/LayerNormRowKernels.h"

#include "DType.h"
#include "cpu/vec/vec.h"

#include <cmath>
#include <cstdint>

namespace tensorplay {
namespace cpu {

namespace {

template <typename T, typename StatsT>
void lnrow_chunk(const T* src, T* dst, int64_t begin, int64_t end,
                 int64_t inner, double eps_d, const T* w, const T* b,
                 StatsT* mean_out, StatsT* rstd_out) {
    using V = vec::Vectorized<T>;
    constexpr int64_t W = V::size();
    const T eps = static_cast<T>(eps_d);

    for (int64_t r = begin; r < end; ++r) {
        const T* row = src + r * inner;
        T* orow = dst + r * inner;

        V s0(T(0)), s1(T(0)), q0(T(0)), q1(T(0));
        int64_t i = 0;
        for (; i + 2 * W <= inner; i += 2 * W) {
            const V v0 = V::loadu(row + i);
            const V v1 = V::loadu(row + i + W);
            s0 = s0 + v0;
            s1 = s1 + v1;
            q0 = q0 + v0 * v0;
            q1 = q1 + v1 * v1;
        }
        const V s = s0 + s1;
        const V q = q0 + q1;
        T sum = static_cast<T>(s.reduce_add());
        T sq = static_cast<T>(q.reduce_add());
        for (; i < inner; ++i) {
            const T v = row[i];
            sum += v;
            sq += v * v;
        }

        const T mean = sum / static_cast<T>(inner);
        const T var = sq / static_cast<T>(inner) - mean * mean;
        const T rstd = T(1) / std::sqrt(var + eps);
        if (mean_out) mean_out[r] = static_cast<StatsT>(mean);
        if (rstd_out) rstd_out[r] = static_cast<StatsT>(rstd);

        const V vm(mean), vr(rstd);
        i = 0;
        if (w != nullptr && b != nullptr) {
            for (; i + W <= inner; i += W) {
                const V v = (V::loadu(row + i) - vm) * vr;
                (v * V::loadu(w + i) + V::loadu(b + i)).store(orow + i);
            }
            for (; i < inner; ++i) {
                orow[i] = (row[i] - mean) * rstd * w[i] + b[i];
            }
        } else if (w != nullptr) {
            for (; i + W <= inner; i += W) {
                ((V::loadu(row + i) - vm) * vr * V::loadu(w + i)).store(orow + i);
            }
            for (; i < inner; ++i) orow[i] = (row[i] - mean) * rstd * w[i];
        } else if (b != nullptr) {
            for (; i + W <= inner; i += W) {
                ((V::loadu(row + i) - vm) * vr + V::loadu(b + i)).store(orow + i);
            }
            for (; i < inner; ++i) orow[i] = (row[i] - mean) * rstd + b[i];
        } else {
            for (; i + W <= inner; i += W) {
                ((V::loadu(row + i) - vm) * vr).store(orow + i);
            }
            for (; i < inner; ++i) orow[i] = (row[i] - mean) * rstd;
        }
    }
}

void lnrow_impl(const void* src, void* dst, int64_t begin, int64_t end,
                int64_t inner, double eps, const void* w, const void* b,
                void* mean_out, void* rstd_out, int stats_dtype) {
    const int sd = stats_dtype;
    if (sd == static_cast<int>(DType::Float64)) {
        lnrow_chunk(static_cast<const float*>(src), static_cast<float*>(dst),
                    begin, end, inner, eps, static_cast<const float*>(w),
                    static_cast<const float*>(b),
                    static_cast<double*>(mean_out),
                    static_cast<double*>(rstd_out));
    } else {
        lnrow_chunk(static_cast<const float*>(src), static_cast<float*>(dst),
                    begin, end, inner, eps, static_cast<const float*>(w),
                    static_cast<const float*>(b),
                    static_cast<float*>(mean_out),
                    static_cast<float*>(rstd_out));
    }
}

} // namespace

#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(lnrow_stub, &lnrow_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(lnrow_stub, &lnrow_impl);
#endif

} // namespace cpu
} // namespace tensorplay

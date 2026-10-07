#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "ReductionKernels.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include "cpu/vec/vec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)
#include <immintrin.h>
#endif

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace ops = tensorplay::tpx::ops;
using namespace vec;

namespace {

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 min, ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

// Welford partial state: chunks reduce independently and merge pairwise,
// so a flat reduction parallelizes without a second pass over the data.
struct WelfordPartial {
    double mean;
    double m2;
    int64_t n;
};

inline void welford_step(WelfordPartial& s, double x) {
    const double delta = x - s.mean;
    ++s.n;
    s.mean += delta / static_cast<double>(s.n);
    s.m2 += delta * (x - s.mean);
}

inline void welford_merge(WelfordPartial& a, const WelfordPartial& b) {
    if (b.n == 0) return;
    if (a.n == 0) {
        a = b;
        return;
    }
    const double n_ab = static_cast<double>(a.n + b.n);
    const double delta = b.mean - a.mean;
    a.mean += delta * (static_cast<double>(b.n) / n_ab);
    a.m2 += b.m2 + delta * delta *
        (static_cast<double>(a.n) * static_cast<double>(b.n) / n_ab);
    a.n += b.n;
}

// One Welford chain per lane over interleaved elements: the per-element
// `delta / count` division is a long dependency chain inside a single lane,
// so independent lanes are what lets the pipeline stay busy.  Lanes merge
// pairwise at the end.
inline constexpr int64_t kWelfordLanes = 8;

template <typename Load>
inline WelfordPartial welford_reduce_strided(int64_t count, int64_t stride,
                                             Load load) {
    WelfordPartial lanes[kWelfordLanes];
    for (auto& lane : lanes) lane = WelfordPartial{0.0, 0.0, 0};
    int64_t c = 0;
    for (; c + kWelfordLanes <= count; c += kWelfordLanes)
        for (int64_t k = 0; k < kWelfordLanes; ++k)
            welford_step(lanes[k], load(c + k, stride));
    for (; c < count; ++c) welford_step(lanes[0], load(c, stride));
    WelfordPartial total = lanes[0];
    for (int64_t k = 1; k < kWelfordLanes; ++k) welford_merge(total, lanes[k]);
    return total;
}

// -------------------------------------------------------------------------
// Vector Welford over contiguous buffers.

// Widens one vector of contiguous floats to double lanes: a single convert
// instruction on the ISA tiers, a scalar round-trip on the generic tier.
inline Vectorized<double> widen_to_double(const float* p) {
#if defined(CPU_CAPABILITY_AVX512)
    return Vectorized<double>(_mm512_cvtps_pd(_mm256_loadu_ps(p)));
#elif defined(CPU_CAPABILITY_AVX2)
    return Vectorized<double>(_mm256_cvtps_pd(_mm_loadu_ps(p)));
#else
    __at_align__ double buf[Vectorized<double>::size()];
    for (int64_t i = 0; i < Vectorized<double>::size(); ++i)
        buf[i] = static_cast<double>(p[i]);
    return Vectorized<double>::loadu(buf);
#endif
}

inline Vectorized<double> widen_to_double(const double* p) {
    return Vectorized<double>::loadu(p);
}

// Vector moments: every lane of mean/m2 is an independent Welford chain over
// the elements that landed in it, and n counts that per-lane element stream.
struct WelfordVecPartial {
    int64_t n;
    Vectorized<double> mean;
    Vectorized<double> m2;
};

// Pairwise merge of two vector-lane moment sets.  Both counts are per-lane
// element counts, so the ratio and the cross term stay in one unit system;
// the two multiplies fold into the FMAs.
inline void welford_merge_vec(WelfordVecPartial& a, const WelfordVecPartial& b) {
    if (b.n == 0) return;
    if (a.n == 0) {
        a = b;
        return;
    }
    using Vec = Vectorized<double>;
    const int64_t n = a.n + b.n;
    const Vec c(static_cast<double>(b.n) / static_cast<double>(n));
    const Vec delta = b.mean - a.mean;
    a.mean = fmadd(c, delta, a.mean);
    a.m2 = fmadd(delta * Vec(static_cast<double>(a.n)), c * delta,
                 a.m2 + b.m2);
    a.n = n;
}

// Vector groups accumulated into one stack level before the pairwise
// cascade merges upward.
inline constexpr int64_t kWelfordVecChunk = 16;

// Runs one stack-chunk of vector Welford steps and merges the result into
// the given stack level.  Each step consumes two double vectors (one per
// chain, so the double accumulators stay wide), and the per-step update
// pulls its count ratio from a reciprocal table instead of dividing.
template <typename T>
inline void welford_update_vec(const T* X, int64_t groups,
                               const std::array<Vectorized<double>, kWelfordVecChunk>& c_vecs,
                               WelfordVecPartial& chain_a,
                               WelfordVecPartial& chain_b) {
    using Vec = Vectorized<double>;
    constexpr int64_t kVS = Vec::size();
    Vec m1_a(0.0), m2_a(0.0), m1_b(0.0), m2_b(0.0);
    for (int64_t j = 0; j < groups; ++j) {
        const T* p = X + j * (2 * kVS);
        const Vec x_a = widen_to_double(p);
        const Vec x_b = widen_to_double(p + kVS);
        const Vec c = c_vecs[j];
        const Vec d_a = x_a - m1_a;
        m1_a = fmadd(c, d_a, m1_a);
        m2_a = fmadd(d_a, x_a - m1_a, m2_a);
        const Vec d_b = x_b - m1_b;
        m1_b = fmadd(c, d_b, m1_b);
        m2_b = fmadd(d_b, x_b - m1_b, m2_b);
    }
    welford_merge_vec(chain_a, WelfordVecPartial{groups, m1_a, m2_a});
    welford_merge_vec(chain_b, WelfordVecPartial{groups, m1_b, m2_b});
}

// Whole-buffer Welford over contiguous data: vector chains with the
// reciprocal table do the bulk work, stack chunks merge pairwise so the
// rounding drift grows with log2 of the length instead of the length, and a
// scalar step covers the tail.  Returns the merged scalar state.
template <typename T>
inline WelfordPartial welford_reduce_contiguous(const T* data, int64_t n) {
    using Vec = Vectorized<double>;
    constexpr int64_t kVS = Vec::size();
    constexpr int64_t kGroup = 2 * kVS;
    constexpr int64_t kMaxDepth = 64;

    static const std::array<Vec, kWelfordVecChunk> c_vecs = [] {
        std::array<Vec, kWelfordVecChunk> table;
        for (int64_t j = 0; j < kWelfordVecChunk; ++j)
            table[j] = Vec(1.0 / static_cast<double>(j + 1));
        return table;
    }();

    const int64_t full_groups = n / kGroup;
    const int64_t nstk = full_groups > 0
        ? (full_groups + kWelfordVecChunk - 1) / kWelfordVecChunk
        : 0;
    int64_t depth = 0;
    while ((int64_t(1) << depth) < nstk) ++depth;

    const WelfordVecPartial zero{0, Vec(0.0), Vec(0.0)};
    std::array<WelfordVecPartial, kMaxDepth> stk_a, stk_b;
    stk_a.fill(zero);
    stk_b.fill(zero);

    for (int64_t i = 0; i < nstk; ++i) {
        const T* p = data + i * kWelfordVecChunk * kGroup;
        const int64_t groups =
            std::min(kWelfordVecChunk, full_groups - i * kWelfordVecChunk);
        welford_update_vec(p, groups, c_vecs, stk_a[0], stk_b[0]);
        int64_t mask = i + 1;
        for (int64_t j = 1; j < depth && (mask & 1) == 0; ++j) {
            welford_merge_vec(stk_a[j], stk_a[j - 1]);
            welford_merge_vec(stk_b[j], stk_b[j - 1]);
            stk_a[j - 1] = zero;
            stk_b[j - 1] = zero;
            mask >>= 1;
        }
    }
    for (int64_t j = 1; j < depth; ++j) {
        welford_merge_vec(stk_a[0], stk_a[j]);
        welford_merge_vec(stk_b[0], stk_b[j]);
    }

    WelfordPartial tail{0.0, 0.0, 0};
    for (int64_t i = full_groups * kGroup; i < n; ++i)
        welford_step(tail, static_cast<double>(data[i]));

    WelfordPartial total{0.0, 0.0, 0};
    if (full_groups > 0) {
        __at_align__ double m1_arr[2 * kVS];
        __at_align__ double m2_arr[2 * kVS];
        stk_a[0].mean.store(m1_arr);
        stk_b[0].mean.store(m1_arr + kVS);
        stk_a[0].m2.store(m2_arr);
        stk_b[0].m2.store(m2_arr + kVS);
        for (int64_t i = 0; i < 2 * kVS; ++i) {
            WelfordPartial lane{m1_arr[i], m2_arr[i], full_groups};
            welford_merge(total, lane);
        }
    }
    welford_merge(total, tail);
    return total;
}

}  // namespace

namespace {

// Compiled once per CPU capability (see the tier object libraries); each
// copy registers itself into the shared stub.  Internal linkage keeps the
// per-capability copies from colliding at link time.
std::pair<Tensor, Tensor> var_mean_kernel_impl(const Tensor& self,
                                               const std::vector<int64_t>& dims_in,
                                               int64_t correction, bool keepdim) {
    if (isComplexType(self.dtype())) {
        const Tensor real = ops::real(self);
        const Tensor imag = ops::imag(self);
        const auto real_stats =
            var_mean_kernel_impl(real, dims_in, correction, keepdim);
        const auto imag_stats =
            var_mean_kernel_impl(imag, dims_in, correction, keepdim);
        return {ops::add(real_stats.first, imag_stats.first),
                ops::complex(real_stats.second, imag_stats.second)};
    }

    const int64_t nd = self.dim();
    std::vector<int64_t> dims = dims_in;
    if (dims.empty()) {
        for (int64_t i = 0; i < nd; ++i) dims.push_back(i);
    }
    std::vector<bool> reduced(static_cast<size_t>(nd), false);
    for (auto& d : dims) {
        d = wrap_dim(d, nd);
        reduced[static_cast<size_t>(d)] = true;
    }
    bool all_reduced = true;
    for (const bool value : reduced) all_reduced = all_reduced && value;
    if (all_reduced) {
        for (int64_t i = 0; i < nd; ++i) reduced[static_cast<size_t>(i)] = true;
    }

    std::vector<int64_t> out_sizes;
    for (int64_t i = 0; i < nd; ++i) {
        if (reduced[static_cast<size_t>(i)]) {
            if (keepdim) out_sizes.push_back(1);
        } else {
            out_sizes.push_back(self.size(i));
        }
    }
    const DType dt = isFloatingType(self.dtype()) ? self.dtype() : DType::Float32;
    Tensor sc = self.to(dt).contiguous();
    Tensor mean = Tensor::empty(out_sizes, dt, self.device());
    Tensor var = Tensor::empty(out_sizes, dt, self.device());
    const int64_t out_numel = mean.numel();

    std::vector<int64_t> strides(static_cast<size_t>(nd), 0);
    int64_t stride = 1;
    for (int64_t i = nd - 1; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = stride;
        stride *= self.size(i);
    }
    std::vector<int64_t> red_dims;
    std::vector<int64_t> red_strides;
    for (int64_t i = 0; i < nd; ++i) {
        if (reduced[static_cast<size_t>(i)]) {
            red_dims.push_back(i);
            red_strides.push_back(strides[static_cast<size_t>(i)]);
        }
    }
    int64_t n_red = 1;
    for (const int64_t dim : red_dims) n_red *= self.size(dim);
    // Map each input dim to its output slot: with keepdim the size-1
    // reduced slots keep their position, without it the surviving dims
    // pack densely.
    std::vector<int64_t> out_slot(static_cast<size_t>(nd), -1);
    {
        int64_t slot = 0;
        for (int64_t i = 0; i < nd; ++i) {
            if (reduced[static_cast<size_t>(i)]) {
                if (keepdim) out_slot[static_cast<size_t>(i)] = slot++;
            } else {
                out_slot[static_cast<size_t>(i)] = slot++;
            }
        }
    }
    // Fast paths for contiguous float input.  A single reduced dim becomes
    // one fixed-stride Welford chain per output element; a full reduction
    // splits into independently reduced chunks merged pairwise.  Both skip
    // the per-element coordinate arithmetic of the generic walk below.
    if (n_red > 0 && (dt == DType::Float32 || dt == DType::Float64) &&
        (red_dims.size() == 1 ||
         static_cast<int64_t>(red_dims.size()) == nd)) {
        const auto run = [&](auto* sp) {
            using T = std::remove_cv_t<std::remove_pointer_t<decltype(sp)>>;
            T* mp = mean.data_ptr<T>();
            T* vp = var.data_ptr<T>();
            const T* data = sp;
            if (static_cast<int64_t>(red_dims.size()) == nd) {
                // Everything reduced: the output is one element, so the
                // parallelism has to come from splitting the data itself
                // into independently reduced chunks merged pairwise.
                const int64_t chunk = 32768;
                const int64_t nchunks = (n_red + chunk - 1) / chunk;
                std::vector<WelfordPartial> partials(nchunks);
                parallel_for(0, nchunks, 1, [&](int64_t b, int64_t e) {
                    for (int64_t ci = b; ci < e; ++ci) {
                        const int64_t beg = ci * chunk;
                        const int64_t fin = std::min(beg + chunk, n_red);
                        partials[ci] = welford_reduce_contiguous(
                            data + beg, fin - beg);
                    }
                });
                WelfordPartial total{0.0, 0.0, 0};
                for (const auto& ps : partials) welford_merge(total, ps);
                mp[0] = static_cast<T>(total.mean);
                vp[0] = static_cast<T>(total.m2 /
                    (static_cast<double>(total.n) - correction));
            } else {
                // Exactly one reduced dim.
                const int64_t d = red_dims[0];
                const int64_t d_size = self.size(d);
                int64_t outer = 1;
                int64_t inner = 1;
                for (int64_t i = 0; i < d; ++i) outer *= self.size(i);
                for (int64_t i = d + 1; i < nd; ++i) inner *= self.size(i);
                if (inner >= 64) {
                    // Wide trailing extent: stream the data row-major in two
                    // passes with column accumulators that stay L1-resident,
                    // instead of one chain per output whose loads would jump
                    // whole cache lines.  With several groups each task owns
                    // whole groups and needs no cross-task merge; with one
                    // group the parallelism splits the rows and merges.
                    if (outer > 1) {
                        const int64_t grain = std::max<int64_t>(
                            1, GRAIN_SIZE / std::max<int64_t>(d_size * inner, 1));
                        parallel_for(0, outer, grain, [&](int64_t b, int64_t e) {
                            std::vector<double> acc(static_cast<size_t>(inner));
                            std::vector<double> acc2(static_cast<size_t>(inner));
                            std::vector<double> meanv(static_cast<size_t>(inner));
                            for (int64_t g = b; g < e; ++g) {
                                std::fill(acc.begin(), acc.end(), 0.0);
                                const T* row = data + g * d_size * inner;
                                for (int64_t c = 0; c < d_size; ++c, row += inner)
                                    for (int64_t j = 0; j < inner; ++j)
                                        acc[static_cast<size_t>(j)] +=
                                            static_cast<double>(row[j]);
                                const int64_t base = g * inner;
                                for (int64_t j = 0; j < inner; ++j) {
                                    meanv[static_cast<size_t>(j)] =
                                        acc[static_cast<size_t>(j)] /
                                        static_cast<double>(d_size);
                                    mp[base + j] =
                                        static_cast<T>(meanv[static_cast<size_t>(j)]);
                                    acc2[static_cast<size_t>(j)] = 0.0;
                                }
                                row = data + g * d_size * inner;
                                for (int64_t c = 0; c < d_size; ++c, row += inner)
                                    for (int64_t j = 0; j < inner; ++j) {
                                        const double delta =
                                            static_cast<double>(row[j]) -
                                            meanv[static_cast<size_t>(j)];
                                        acc2[static_cast<size_t>(j)] += delta * delta;
                                    }
                                for (int64_t j = 0; j < inner; ++j)
                                    vp[base + j] = static_cast<T>(acc2[static_cast<size_t>(j)] /
                                        (static_cast<double>(d_size) - correction));
                            }
                        });
                    } else {
                        const int64_t rows_per_task =
                            std::max<int64_t>(1, GRAIN_SIZE / inner);
                        const int64_t ntasks =
                            (d_size + rows_per_task - 1) / rows_per_task;
                        std::vector<double> sums(
                            static_cast<size_t>(ntasks * inner), 0.0);
                        parallel_for(0, ntasks, 1, [&](int64_t b, int64_t e) {
                            for (int64_t ti = b; ti < e; ++ti) {
                                double* acc = &sums[static_cast<size_t>(ti * inner)];
                                const int64_t c0 = ti * rows_per_task;
                                const int64_t c1 = std::min(c0 + rows_per_task, d_size);
                                const T* row = data + c0 * inner;
                                for (int64_t c = c0; c < c1; ++c, row += inner)
                                    for (int64_t j = 0; j < inner; ++j)
                                        acc[j] += static_cast<double>(row[j]);
                            }
                        });
                        std::vector<double> mean_buf(static_cast<size_t>(inner), 0.0);
                        for (int64_t j = 0; j < inner; ++j) {
                            double total = 0.0;
                            for (int64_t ti = 0; ti < ntasks; ++ti)
                                total += sums[static_cast<size_t>(ti * inner + j)];
                            mean_buf[static_cast<size_t>(j)] =
                                total / static_cast<double>(d_size);
                        }
                        std::vector<double> squares(
                            static_cast<size_t>(ntasks * inner), 0.0);
                        parallel_for(0, ntasks, 1, [&](int64_t b, int64_t e) {
                            for (int64_t ti = b; ti < e; ++ti) {
                                double* acc =
                                    &squares[static_cast<size_t>(ti * inner)];
                                const int64_t c0 = ti * rows_per_task;
                                const int64_t c1 =
                                    std::min(c0 + rows_per_task, d_size);
                                const T* row = data + c0 * inner;
                                for (int64_t c = c0; c < c1; ++c, row += inner)
                                    for (int64_t j = 0; j < inner; ++j) {
                                        const double delta =
                                            static_cast<double>(row[j]) -
                                            mean_buf[static_cast<size_t>(j)];
                                        acc[j] += delta * delta;
                                    }
                            }
                        });
                        for (int64_t j = 0; j < inner; ++j) {
                            double m2 = 0.0;
                            for (int64_t ti = 0; ti < ntasks; ++ti)
                                m2 += squares[static_cast<size_t>(ti * inner + j)];
                            mp[j] = static_cast<T>(mean_buf[static_cast<size_t>(j)]);
                            vp[j] = static_cast<T>(m2 /
                                (static_cast<double>(d_size) - correction));
                        }
                    }
                } else {
                    // One fixed-stride chain per output element; the outputs
                    // split across threads.
                    const int64_t grain =
                        std::max<int64_t>(1, GRAIN_SIZE / std::max<int64_t>(d_size, 1));
                    parallel_for(0, out_numel, grain, [&](int64_t b, int64_t e) {
                        for (int64_t oi = b; oi < e; ++oi) {
                            const T* src = data + (oi / inner) * d_size * inner +
                                           (oi % inner);
                            const WelfordPartial s = welford_reduce_strided(
                                d_size, inner,
                                [src](int64_t idx, int64_t st) -> double {
                                    return static_cast<double>(src[idx * st]);
                                });
                            mp[oi] = static_cast<T>(s.mean);
                            vp[oi] = static_cast<T>(s.m2 /
                                (static_cast<double>(s.n) - correction));
                        }
                    });
                }
            }
        };
        if (dt == DType::Float32) run(sc.data_ptr<float>());
        else run(sc.data_ptr<double>());
        return {var, mean};
    }

    auto compute = [&](auto* sp, auto* mp, auto* vp) {
        using output_t = std::remove_cv_t<std::remove_pointer_t<decltype(mp)>>;
        parallel_for(0, out_numel, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            std::vector<int64_t> coords(red_dims.size(), 0);
            std::vector<int64_t> out_coords(out_sizes.size(), 0);
            for (int64_t oi = begin; oi < end; ++oi) {
                int64_t rest = oi;
                for (int64_t i = static_cast<int64_t>(out_sizes.size()) - 1;
                     i >= 0; --i) {
                    out_coords[static_cast<size_t>(i)] =
                        rest % out_sizes[static_cast<size_t>(i)];
                    rest /= out_sizes[static_cast<size_t>(i)];
                }
                int64_t base = 0;
                for (int64_t i = 0; i < nd; ++i) {
                    const int64_t slot = out_slot[static_cast<size_t>(i)];
                    if (slot < 0) continue;
                    base += out_coords[static_cast<size_t>(slot)] *
                            strides[static_cast<size_t>(i)];
                }
                if (n_red == 0) {
                    const double nan = std::numeric_limits<double>::quiet_NaN();
                    mp[oi] = static_cast<output_t>(nan);
                    vp[oi] = static_cast<output_t>(nan);
                    continue;
                }
                double mean_value = 0.0;
                double m2 = 0.0;
                int64_t count = 0;
                std::fill(coords.begin(), coords.end(), 0);
                for (int64_t c = 0; c < n_red; ++c) {
                    int64_t offset = base;
                    for (size_t r = 0; r < red_dims.size(); ++r)
                        offset += coords[r] * red_strides[r];
                    const double value = static_cast<double>(sp[offset]);
                    const int64_t new_count = ++count;
                    const double delta = value - mean_value;
                    mean_value += delta / static_cast<double>(new_count);
                    const double new_delta = value - mean_value;
                    m2 += delta * new_delta;
                    for (int64_t r = static_cast<int64_t>(red_dims.size()) - 1;
                         r >= 0; --r) {
                        if (++coords[static_cast<size_t>(r)] <
                            self.size(red_dims[static_cast<size_t>(r)]))
                            break;
                        coords[static_cast<size_t>(r)] = 0;
                    }
                }
                mp[oi] = static_cast<output_t>(mean_value);
                vp[oi] = static_cast<output_t>(
                    m2 / (static_cast<double>(n_red) - correction));
            }
        });
    };

    switch (dt) {
        case DType::Float16:
            compute(sc.data_ptr<Half>(), mean.data_ptr<Half>(), var.data_ptr<Half>());
            break;
        case DType::Float32:
            compute(sc.data_ptr<float>(), mean.data_ptr<float>(), var.data_ptr<float>());
            break;
        case DType::Float64:
            compute(sc.data_ptr<double>(), mean.data_ptr<double>(), var.data_ptr<double>());
            break;
        case DType::BFloat16:
            compute(sc.data_ptr<BFloat16>(), mean.data_ptr<BFloat16>(),
                    var.data_ptr<BFloat16>());
            break;
        default:
            TP_THROW(TypeError, "std_mean: unsupported dtype ", toString(dt));
    }
    return {var, mean};
}

}  // namespace

REGISTER_DISPATCH(var_mean_stub, &var_mean_kernel_impl);

} // namespace cpu
} // namespace tensorplay

// Batch normalization and instance normalization CPU kernels.
//
// Layout of this file groups the ops by
//   - batch_norm forward: stats collection + affine apply, specialized for
//     contiguous (N, C, ...) and channels-last (N, ..., C) memory formats and
//     for Float32/Float64 (native math) and Float16/BFloat16 (fp32 opmath).
//   - batch_norm backward: per-channel fused reductions (sum dy, sum dy * x)
//     followed by the vectorized grad_input formula, same format/dtype matrix.
//   - instance_norm forward is a composite: reshape (N, C, ...) to
//     (1, N*C, ...) so each (sample, channel) plane becomes one batch-norm
//     channel, run batch_norm, then fold the updated running stats back by
//     averaging the per-plane statistics over the batch.  instance_norm
//     backward reuses group_norm backward (input-stats mode) or batch_norm
//     backward (eval mode).
//
// All hot loops are parallelized with the intra-op pool; the fp32 contiguous
// path additionally uses the AVX-512 row helpers from NormRowHelpers.h.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "MemoryFormat.h"
#include "NormRowHelpers.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace tensorplay {
namespace cpu {

namespace {

using parallel::parallel_for;
using parallel::get_num_threads;
using parallel::get_thread_num;
using parallel::GRAIN_SIZE;

// ---------------------------------------------------------------------------
// AVX-512 kernels need the target attribute on the function that inlines the
// intrinsics; the runtime gate stays at the call site (norm_row::avx512_ok).
// ---------------------------------------------------------------------------
#if defined(__x86_64__)
__attribute__((target("avx512f")))
inline void bn_stats_cl_f32x8(double* row, const float* p, int64_t C) {
    int64_t c = 0;
    for (; c + 8 <= C; c += 8) {
        __m512d v = _mm512_cvtps_pd(_mm256_loadu_ps(p + c));
        _mm512_storeu_pd(row + c, _mm512_add_pd(v, _mm512_loadu_pd(row + c)));
    }
    for (; c < C; ++c) row[c] += static_cast<double>(p[c]);
}

__attribute__((target("avx512f")))
inline void bn_affine_f32x16(float* op, const float* ip, const float* alpha,
                             const float* beta, int64_t C) {
    int64_t c = 0;
    for (; c + 16 <= C; c += 16) {
        __m512 va = _mm512_loadu_ps(alpha + c);
        __m512 vb = _mm512_loadu_ps(beta + c);
        __m512 v = _mm512_loadu_ps(ip + c);
        _mm512_storeu_ps(op + c, _mm512_fmadd_ps(v, va, vb));
    }
    for (; c < C; ++c) op[c] = static_cast<float>(ip[c]) * alpha[c] + beta[c];
}

__attribute__((target("avx512f,avx512bw")))
inline void bn_bwd_f32x16(float* op, const float* xp, const float* yp,
                          const double* mean, const double* dotp,
                          const double* invstd, const double* sum,
                          const float* wp, int64_t C, double M, bool training) {
    int64_t c = 0;
    if (training) {
        for (; c + 16 <= C; c += 16) {
            __m512 vx = _mm512_loadu_ps(xp + c);
            __m512 vdy = _mm512_loadu_ps(yp + c);
            __m512 vm = _mm512_set1_ps(static_cast<float>(mean[c]));
            __m512 vk = _mm512_set1_ps(static_cast<float>(dotp[c] * invstd[c] * invstd[c] / M));
            __m512 vg = _mm512_set1_ps(static_cast<float>(sum[c] / M));
            __m512 vr = _mm512_set1_ps(static_cast<float>(invstd[c]));
            __m512 vw = _mm512_set1_ps(wp ? wp[c] : 1.0f);
            __m512 t = _mm512_mul_ps(_mm512_sub_ps(vx, vm), vk);
            __m512 u = _mm512_sub_ps(_mm512_sub_ps(vdy, vg), t);
            _mm512_storeu_ps(op + c, _mm512_mul_ps(_mm512_mul_ps(u, vr), vw));
        }
    } else {
        for (; c + 16 <= C; c += 16) {
            __m512 vdy = _mm512_loadu_ps(yp + c);
            __m512 vs = _mm512_set1_ps(static_cast<float>(invstd[c]) * (wp ? wp[c] : 1.0f));
            _mm512_storeu_ps(op + c, _mm512_mul_ps(vdy, vs));
        }
    }
    for (; c < C; ++c) {
        const float w = wp ? wp[c] : 1.0f;
        if (training) {
            const float k = static_cast<float>(dotp[c] * invstd[c] * invstd[c] / M);
            const float gm = static_cast<float>(sum[c] / M);
            op[c] = (yp[c] - gm - (xp[c] - static_cast<float>(mean[c])) * k) *
                    static_cast<float>(invstd[c]) * w;
        } else {
            op[c] = yp[c] * static_cast<float>(invstd[c]) * w;
        }
    }
}

#endif  // __x86_64__


// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

bool is_channels_last(const Tensor& t) {
    return t.dim() == 4 ? t.is_contiguous(MemoryFormat::ChannelsLast)
                        : t.dim() == 5 ? t.is_contiguous(MemoryFormat::ChannelsLast3d)
                                       : false;
}

// Allocate an empty tensor with the same shape/device and memory format as
// `like` (dtype override supported for grad buffers).
Tensor empty_like_format(const Tensor& like, std::optional<DType> dtype = std::nullopt) {
    std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(like.shape());
    const DType dt = dtype.value_or(like.dtype());
    Tensor out = Tensor::empty(sizes, dt, like.device());
    if (like.dim() == 4 && like.is_contiguous(MemoryFormat::ChannelsLast)) {
        out = out.as_strided(sizes, get_channels_last_strides(sizes), 0);
    } else if (like.dim() == 5 && like.is_contiguous(MemoryFormat::ChannelsLast3d)) {
        out = out.as_strided(sizes, get_channels_last_strides(sizes), 0);
    }
    return out;
}

bool is_reduced(DType t) { return t == DType::Float16 || t == DType::BFloat16; }

// Dtype of the saved/running statistics for a given input: reduced types keep
// stats in fp32; float/double keep their own type.  Affine params follow the
// same rule, so a half input may legally carry fp32 weight/bias/running stats.
DType stats_dtype_for(DType input_dt) {
    return is_reduced(input_dt) ? DType::Float32 : input_dt;
}

void check_param_dtype(DType input_dt, const std::optional<Tensor>& p,
                       const char* name) {
    if (!p.has_value() || !p->defined()) return;
    // Reduced-precision inputs accept their own dtype (homogeneous) or the
    // wider stats dtype (mixed); float/double inputs require an exact match.
    DType want = stats_dtype_for(input_dt);
    if (p->dtype() != want && !(is_reduced(input_dt) && p->dtype() == input_dt)) {
        TP_THROW(RuntimeError,
                 std::string("batch_norm: ") + name + " must be " +
                 toString(want) + " (or the input dtype) for this input, got " +
                 toString(p->dtype()));
    }
}

Tensor repeat_channels(const Tensor& t, int64_t b, DType dt) {
    // Tile a (C,) parameter vector b times into a (b*C,) tensor of dtype dt.
    Tensor out = Tensor::empty({b * t.size(0)}, dt, t.device());
    const int64_t C = t.size(0);
    auto read_as_double = [&](int64_t i) -> double {
        switch (t.dtype()) {
            case DType::Float64: return static_cast<double>(t.data_ptr<double>()[i]);
            case DType::Float32: return static_cast<double>(t.data_ptr<float>()[i]);
            case DType::Float16: return static_cast<double>(t.data_ptr<tensorplay::Half>()[i]);
            case DType::BFloat16: return static_cast<double>(t.data_ptr<tensorplay::BFloat16>()[i]);
            default:
                TP_THROW(RuntimeError, "repeat_channels: unexpected source dtype");
        }
    };
    auto write_value = [&](int64_t i, double v) {
        switch (dt) {
            case DType::Float64: out.data_ptr<double>()[i] = v; break;
            case DType::Float32: out.data_ptr<float>()[i] = static_cast<float>(v); break;
            case DType::Float16: out.data_ptr<tensorplay::Half>()[i] = tensorplay::Half(static_cast<float>(v)); break;
            case DType::BFloat16: out.data_ptr<tensorplay::BFloat16>()[i] = tensorplay::BFloat16(static_cast<float>(v)); break;
            default:
                TP_THROW(RuntimeError, "repeat_channels: unexpected dtype");
        }
    };
    for (int64_t n = 0; n < b; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            write_value(n * C + c, read_as_double(c));
        }
    }
    return out;
}

// Read one element of a stats vector as double (handles fp32/fp64 storage).
double stats_at(const Tensor& t, int64_t i) {
    switch (t.dtype()) {
        case DType::Float64: return static_cast<double>(t.data_ptr<double>()[i]);
        case DType::Float32: return static_cast<double>(t.data_ptr<float>()[i]);
        case DType::Float16: return static_cast<double>(t.data_ptr<tensorplay::Half>()[i]);
        case DType::BFloat16: return static_cast<double>(t.data_ptr<tensorplay::BFloat16>()[i]);
        default:
            TP_THROW(RuntimeError, "batch_norm: unsupported stats dtype");
    }
}

void stats_set(Tensor& t, int64_t i, double v) {
    switch (t.dtype()) {
        case DType::Float64: t.data_ptr<double>()[i] = v; break;
        case DType::Float32: t.data_ptr<float>()[i] = static_cast<float>(v); break;
        case DType::Float16: t.data_ptr<tensorplay::Half>()[i] = tensorplay::Half(static_cast<float>(v)); break;
        case DType::BFloat16: t.data_ptr<tensorplay::BFloat16>()[i] = tensorplay::BFloat16(static_cast<float>(v)); break;
        default:
            TP_THROW(RuntimeError, "batch_norm: unsupported stats dtype");
    }
}

// ---------------------------------------------------------------------------
// Contiguous forward: stats + apply over (N, C, S) planes.
// ---------------------------------------------------------------------------

// Accumulate per-channel sum/sum-of-squares with double accumulators.
// `count` is N * S (elements per channel).
void bn_stats_contiguous(const Tensor& input, std::vector<double>& mean,
                         std::vector<double>& var) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t S = input.numel() / (N * C);
    const double count = static_cast<double>(N * S);

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in = input.data_ptr<float>();
            parallel_for(0, C, 1, [&](int64_t cb, int64_t ce) {
                for (int64_t c = cb; c < ce; ++c) {
                    double s = 0.0, q = 0.0;
#if defined(__x86_64__)
                    if (norm_row::avx512_ok()) {
                        for (int64_t n = 0; n < N; ++n) {
                            norm_row::acc_stats_f64_512(in + n * C * S + c * S, S, s, q);
                        }
                    } else
#endif
                    {
                        for (int64_t n = 0; n < N; ++n) {
                            const float* p = in + n * C * S + c * S;
                            for (int64_t i = 0; i < S; ++i) {
                                double v = static_cast<double>(p[i]);
                                s += v;
                                q += v * v;
                            }
                        }
                    }
                    const double m = s / count;
                    mean[c] = m;
                    var[c] = q / count - m * m;
                }
            });
            break;
        }
        case DType::Float64: {
            const double* in = input.data_ptr<double>();
            parallel_for(0, C, 1, [&](int64_t cb, int64_t ce) {
                for (int64_t c = cb; c < ce; ++c) {
                    double s = 0.0, q = 0.0;
                    for (int64_t n = 0; n < N; ++n) {
                        const double* p = in + n * C * S + c * S;
                        for (int64_t i = 0; i < S; ++i) { s += p[i]; q += p[i] * p[i]; }
                    }
                    const double m = s / count;
                    mean[c] = m;
                    var[c] = q / count - m * m;
                }
            });
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            const bool bf = input.dtype() == DType::BFloat16;
            const void* base = input.data_ptr();
            parallel_for(0, C, 1, [&](int64_t cb, int64_t ce) {
                for (int64_t c = cb; c < ce; ++c) {
                    double s = 0.0, q = 0.0;
                    for (int64_t n = 0; n < N; ++n) {
                        for (int64_t i = 0; i < S; ++i) {
                            const int64_t idx = n * C * S + c * S + i;
                            double v = bf ? static_cast<double>(static_cast<const tensorplay::BFloat16*>(base)[idx])
                                          : static_cast<double>(static_cast<const tensorplay::Half*>(base)[idx]);
                            s += v;
                            q += v * v;
                        }
                    }
                    const double m = s / count;
                    mean[c] = m;
                    var[c] = q / count - m * m;
                }
            });
            break;
        }
        default:
            TP_THROW(NotImplementedError, "batch_norm: unsupported dtype");
    }
}

// Affine apply over contiguous planes: out = x * alpha[c] + beta[c].
template <typename scalar_t, typename opmath_t>
void bn_apply_contiguous_typed(const Tensor& input, Tensor& out,
                               const std::vector<float>& alpha,
                               const std::vector<float>& beta) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t S = input.numel() / (N * C);
    const scalar_t* in = input.data_ptr<scalar_t>();
    scalar_t* outp = out.data_ptr<scalar_t>();
    constexpr bool native = std::is_same_v<scalar_t, opmath_t>;

    parallel_for(0, N * C, 1, [&](int64_t rb, int64_t re) {
        for (int64_t rc = rb; rc < re; ++rc) {
            const int64_t c = rc % C;
            const float a = alpha[c];
            const float b = beta[c];
            const scalar_t* ip = in + rc * S;
            scalar_t* op = outp + rc * S;
            if constexpr (native) {
                if constexpr (std::is_same_v<scalar_t, float>) {
#if defined(__x86_64__)
                    if (norm_row::avx512_ok()) {
                        norm_row::plane_affine_f32_512(ip, op, S, a, b);
                        continue;
                    }
#endif
                }
                const opmath_t av = static_cast<opmath_t>(a);
                const opmath_t bv = static_cast<opmath_t>(b);
                for (int64_t i = 0; i < S; ++i) op[i] = static_cast<scalar_t>(static_cast<opmath_t>(ip[i]) * av + bv);
            } else {
                for (int64_t i = 0; i < S; ++i) {
                    const opmath_t v = static_cast<opmath_t>(ip[i]) * a + b;
                    op[i] = static_cast<scalar_t>(v);
                }
            }
        }
    });
}

void bn_apply_contiguous(const Tensor& input, Tensor& out,
                         const std::vector<float>& alpha, const std::vector<float>& beta) {
    switch (input.dtype()) {
        case DType::Float32: bn_apply_contiguous_typed<float, float>(input, out, alpha, beta); break;
        case DType::Float64: bn_apply_contiguous_typed<double, double>(input, out, alpha, beta); break;
        case DType::Float16: bn_apply_contiguous_typed<tensorplay::Half, float>(input, out, alpha, beta); break;
        case DType::BFloat16: bn_apply_contiguous_typed<tensorplay::BFloat16, float>(input, out, alpha, beta); break;
        default: TP_THROW(NotImplementedError, "batch_norm: unsupported dtype");
    }
}

// ---------------------------------------------------------------------------
// Channels-last forward: (N, C, ...) stored as (N, ..., C); every (n, image)
// position is a contiguous C-vector.  Statistics reduce vertically over the
// image positions into per-thread buffers, then reduce over threads.
// ---------------------------------------------------------------------------

void bn_stats_channels_last(const Tensor& input, std::vector<double>& mean,
                            std::vector<double>& var) {
    const int64_t C = input.size(1);
    const int64_t N = input.numel() / C;
    const int th = get_num_threads();

    // Pass 1: per-thread sums over contiguous C-vectors.
    std::vector<double> buf(static_cast<size_t>(th) * C, 0.0);
    switch (input.dtype()) {
        case DType::Float32: {
            const float* in = input.data_ptr<float>();
            parallel_for(0, N, 1, [&](int64_t b, int64_t e) {
                double* row = buf.data() + static_cast<size_t>(get_thread_num()) * C;
                for (int64_t i = b; i < e; ++i) {
                    const float* p = in + i * C;
#if defined(__x86_64__)
                    if (norm_row::avx512_ok() && C >= 8) {
                        bn_stats_cl_f32x8(row, p, C);
                    } else
#endif
                    {
                        for (int64_t c = 0; c < C; ++c) row[c] += static_cast<double>(p[c]);
                    }
                }
            });
            break;
        }
        case DType::Float64: {
            const double* in = input.data_ptr<double>();
            parallel_for(0, N, 1, [&](int64_t b, int64_t e) {
                double* row = buf.data() + static_cast<size_t>(get_thread_num()) * C;
                for (int64_t i = b; i < e; ++i) {
                    const double* p = in + i * C;
                    for (int64_t c = 0; c < C; ++c) row[c] += p[c];
                }
            });
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            const bool bf = input.dtype() == DType::BFloat16;
            const void* base = input.data_ptr();
            parallel_for(0, N, 1, [&](int64_t b, int64_t e) {
                double* row = buf.data() + static_cast<size_t>(get_thread_num()) * C;
                for (int64_t i = b; i < e; ++i) {
                    for (int64_t c = 0; c < C; ++c) {
                        double v = bf ? static_cast<double>(static_cast<const tensorplay::BFloat16*>(base)[i * C + c])
                                      : static_cast<double>(static_cast<const tensorplay::Half*>(base)[i * C + c]);
                        row[c] += v;
                    }
                }
            });
            break;
        }
        default:
            TP_THROW(NotImplementedError, "batch_norm: unsupported dtype");
    }

    std::vector<double> sum(C, 0.0);
    for (int64_t c = 0; c < C; ++c) {
        double s = 0.0;
        for (int64_t t = 0; t < th; ++t) s += buf[t * C + c];
        sum[c] = s;
        mean[c] = s / static_cast<double>(N);
    }

    // Pass 2: variance around the mean, same two-phase structure.
    std::fill(buf.begin(), buf.end(), 0.0);
    switch (input.dtype()) {
        case DType::Float32: {
            const float* in = input.data_ptr<float>();
            parallel_for(0, N, 1, [&](int64_t b, int64_t e) {
                double* row = buf.data() + static_cast<size_t>(get_thread_num()) * C;
                for (int64_t i = b; i < e; ++i) {
                    const float* p = in + i * C;
                    for (int64_t c = 0; c < C; ++c) {
                        const double d = static_cast<double>(p[c]) - mean[c];
                        row[c] += d * d;
                    }
                }
            });
            break;
        }
        case DType::Float64: {
            const double* in = input.data_ptr<double>();
            parallel_for(0, N, 1, [&](int64_t b, int64_t e) {
                double* row = buf.data() + static_cast<size_t>(get_thread_num()) * C;
                for (int64_t i = b; i < e; ++i) {
                    const double* p = in + i * C;
                    for (int64_t c = 0; c < C; ++c) {
                        const double d = p[c] - mean[c];
                        row[c] += d * d;
                    }
                }
            });
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            const bool bf = input.dtype() == DType::BFloat16;
            const void* base = input.data_ptr();
            parallel_for(0, N, 1, [&](int64_t b, int64_t e) {
                double* row = buf.data() + static_cast<size_t>(get_thread_num()) * C;
                for (int64_t i = b; i < e; ++i) {
                    for (int64_t c = 0; c < C; ++c) {
                        double v = bf ? static_cast<double>(static_cast<const tensorplay::BFloat16*>(base)[i * C + c])
                                      : static_cast<double>(static_cast<const tensorplay::Half*>(base)[i * C + c]);
                        const double d = v - mean[c];
                        row[c] += d * d;
                    }
                }
            });
            break;
        }
        default:
            TP_THROW(NotImplementedError, "batch_norm: unsupported dtype");
    }
    for (int64_t c = 0; c < C; ++c) {
        double q = 0.0;
        for (int64_t t = 0; t < th; ++t) q += buf[t * C + c];
        var[c] = q / static_cast<double>(N);
    }
    (void)sum;
}

template <typename scalar_t, typename opmath_t>
void bn_apply_channels_last_typed(const Tensor& input, Tensor& out,
                                  const std::vector<float>& alpha,
                                  const std::vector<float>& beta) {
    const int64_t C = input.size(1);
    const int64_t N = input.numel() / C;
    const scalar_t* in = input.data_ptr<scalar_t>();
    scalar_t* outp = out.data_ptr<scalar_t>();
    constexpr bool native = std::is_same_v<scalar_t, opmath_t>;

    parallel_for(0, N, 1, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const scalar_t* ip = in + i * C;
            scalar_t* op = outp + i * C;
            if constexpr (native) {
                if constexpr (std::is_same_v<scalar_t, float>) {
#if defined(__x86_64__)
                    if (norm_row::avx512_ok() && C >= 16) {
                        bn_affine_f32x16(op, ip, alpha.data(), beta.data(), C);
                        continue;
                    }
#endif
                }
                for (int64_t c = 0; c < C; ++c) op[c] = static_cast<opmath_t>(ip[c]) * alpha[c] + beta[c];
            } else {
                for (int64_t c = 0; c < C; ++c) {
                    op[c] = static_cast<scalar_t>(static_cast<opmath_t>(ip[c]) * alpha[c] + beta[c]);
                }
            }
        }
    });
}

void bn_apply_channels_last(const Tensor& input, Tensor& out,
                            const std::vector<float>& alpha, const std::vector<float>& beta) {
    switch (input.dtype()) {
        case DType::Float32: bn_apply_channels_last_typed<float, float>(input, out, alpha, beta); break;
        case DType::Float64: bn_apply_channels_last_typed<double, double>(input, out, alpha, beta); break;
        case DType::Float16: bn_apply_channels_last_typed<tensorplay::Half, float>(input, out, alpha, beta); break;
        case DType::BFloat16: bn_apply_channels_last_typed<tensorplay::BFloat16, float>(input, out, alpha, beta); break;
        default: TP_THROW(NotImplementedError, "batch_norm: unsupported dtype");
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// batch_norm forward
// ---------------------------------------------------------------------------

static Tensor batch_norm_cpu_impl(
                      const Tensor& input, const std::optional<Tensor>& weight_opt,
                      const std::optional<Tensor>& bias_opt,
                      const std::optional<Tensor>& running_mean_opt,
                      const std::optional<Tensor>& running_var_opt,
                      bool training, double momentum, double eps,
                      Tensor* save_mean, Tensor* save_invstd) {
    if (input.dim() < 2 || input.dim() > 5)
        TP_THROW(RuntimeError, "batch_norm: Input must be between 2D and 5D");
    if (!isFloatingType(input.dtype()))
        TP_THROW(NotImplementedError,
                 std::string("batch_norm: only floating point dtypes are supported (got ") +
                 toString(input.dtype()) + ")");

    check_param_dtype(input.dtype(), weight_opt, "weight");
    check_param_dtype(input.dtype(), bias_opt, "bias");
    check_param_dtype(input.dtype(), running_mean_opt, "running_mean");
    check_param_dtype(input.dtype(), running_var_opt, "running_var");

    const int64_t C = input.size(1);
    const bool has_running_mean = running_mean_opt.has_value() &&
        running_mean_opt->defined();
    const bool has_running_var = running_var_opt.has_value() &&
        running_var_opt->defined();
    if (has_running_mean != has_running_var) {
        TP_THROW(RuntimeError,
                 "batch_norm: running_mean and running_var must either both be defined or both be absent");
    }
    if (!training &&
        !(has_running_mean && has_running_var)) {
        TP_THROW(RuntimeError,
                 "batch_norm: running_mean and running_var must be defined in evaluation mode");
    }
    if (save_mean && save_invstd && save_mean->numel() != C && training) {
        TP_THROW(RuntimeError, "batch_norm: invalid saved statistics shape");
    }
    if (input.numel() == 0) {
        if (save_mean && save_invstd && training) {
            for (int64_t c = 0; c < C; ++c) {
                stats_set(*save_mean, c, 0.0);
                stats_set(*save_invstd, c, 1.0 / std::sqrt(eps));
            }
        }
        return empty_like_format(input);
    }

    std::vector<double> mean(C, 0.0), var(C, 0.0);
    if (training) {
        if (is_channels_last(input)) bn_stats_channels_last(input, mean, var);
        else bn_stats_contiguous(input, mean, var);

        if (has_running_mean && has_running_var) {
            const double count = static_cast<double>(input.numel() / C);
            const double unbiased = count > 1.0 ? count / (count - 1.0) : 1.0;
            for (int64_t c = 0; c < C; ++c) {
                stats_set(const_cast<Tensor&>(*running_mean_opt), c,
                          (1.0 - momentum) * stats_at(*running_mean_opt, c) + momentum * mean[c]);
                stats_set(const_cast<Tensor&>(*running_var_opt), c,
                          (1.0 - momentum) * stats_at(*running_var_opt, c) + momentum * var[c] * unbiased);
            }
        }
    } else {
        for (int64_t c = 0; c < C; ++c) {
            mean[c] = stats_at(*running_mean_opt, c);
            var[c] = stats_at(*running_var_opt, c);
        }
    }

    if (save_mean && save_invstd && training) {
        for (int64_t c = 0; c < C; ++c) {
            stats_set(*save_mean, c, mean[c]);
            stats_set(*save_invstd, c, 1.0 / std::sqrt(var[c] + eps));
        }
    }

    // Fold the normalization into per-channel affine coefficients:
    // y = (x - mean) / sqrt(var + eps) * w + b = x * alpha + beta.
    std::vector<float> alpha(C), beta(C);
    for (int64_t c = 0; c < C; ++c) {
        const double invstd = 1.0 / std::sqrt(var[c] + eps);
        const double w = weight_opt.has_value() && weight_opt->defined() ? stats_at(*weight_opt, c) : 1.0;
        const double b = bias_opt.has_value() && bias_opt->defined() ? stats_at(*bias_opt, c) : 0.0;
        alpha[c] = static_cast<float>(invstd * w);
        beta[c] = static_cast<float>(b - mean[c] * invstd * w);
    }

    Tensor out = empty_like_format(input);
    if (is_channels_last(input)) bn_apply_channels_last(input, out, alpha, beta);
    else bn_apply_contiguous(input, out, alpha, beta);
    return out;
}

Tensor batch_norm_cpu(const Tensor& input, std::optional<Tensor> weight_opt,
                      std::optional<Tensor> bias_opt,
                      std::optional<Tensor> running_mean_opt,
                      std::optional<Tensor> running_var_opt,
                      bool training, double momentum, double eps) {
    return batch_norm_cpu_impl(input, weight_opt, bias_opt, running_mean_opt,
                               running_var_opt, training, momentum, eps,
                               nullptr, nullptr);
}

// ---------------------------------------------------------------------------
// batch_norm backward
// ---------------------------------------------------------------------------

namespace {

// Per-channel fused reduction: sum(dy) and sum(dy * (x - mean)) with double
// accumulators, then grad_input, grad_weight, grad_bias.  Contiguous layout:
// parallel over channels; each channel owns N planes of S elements.
void bn_backward_contiguous(const Tensor& grad_output, const Tensor& input,
                            const std::optional<Tensor>& weight_opt,
                            const std::optional<Tensor>& running_mean_opt,
                            const std::optional<Tensor>& running_var_opt,
                            const std::vector<double>& mean, const std::vector<double>& invstd,
                            bool training, Tensor& grad_input, Tensor& grad_weight,
                            Tensor& grad_bias) {
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t S = input.numel() / (N * C);
    const double M = static_cast<double>(N * S);

    const bool has_w = weight_opt.has_value() && weight_opt->defined();

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in = input.data_ptr<float>();
            const float* dy = grad_output.data_ptr<float>();
            float* gx = grad_input.defined() ? grad_input.data_ptr<float>() : nullptr;
            float* gw = grad_weight.defined() ? grad_weight.data_ptr<float>() : nullptr;
            float* gb = grad_bias.defined() ? grad_bias.data_ptr<float>() : nullptr;
            const float* wp = has_w ? weight_opt->data_ptr<float>() : nullptr;

            parallel_for(0, C, 1, [&](int64_t cb, int64_t ce) {
                for (int64_t c = cb; c < ce; ++c) {
                    const float mu = static_cast<float>(mean[c]);
                    const float r = static_cast<float>(invstd[c]);
                    double s = 0.0, dotp = 0.0;
#if defined(__x86_64__)
                    if (norm_row::avx512_ok()) {
                        for (int64_t n = 0; n < N; ++n) {
                            const int64_t off = n * C * S + c * S;
                            norm_row::acc_dot2_f64_512(dy + off, in + off, S, s, dotp);
                        }
                        // acc_dot2 accumulates sum(dy * x); recenter it.
                        dotp -= static_cast<double>(mu) * s;
                    } else
#endif
                    {
                        for (int64_t n = 0; n < N; ++n) {
                            const int64_t off = n * C * S + c * S;
                            for (int64_t i = 0; i < S; ++i) {
                                const double y = static_cast<double>(dy[off + i]);
                                s += y;
                                dotp += y * static_cast<double>(in[off + i] - mu);
                            }
                        }
                    }
                    if (gb) gb[c] = static_cast<float>(s);
                    if (gw) gw[c] = static_cast<float>(dotp * static_cast<double>(r));
                    if (gx) {
                        const float w = wp ? wp[c] : 1.0f;
                        if (training) {
                            const float k = static_cast<float>(dotp * static_cast<double>(r) * r / M);
                            const float gm = static_cast<float>(s / M);
                            for (int64_t n = 0; n < N; ++n) {
                                const int64_t off = n * C * S + c * S;
#if defined(__x86_64__)
                                if (norm_row::avx512_ok()) {
                                    norm_row::plane_bn_dx_f32_512(in + off, dy + off, gx + off, S,
                                                                  mu, k, gm, r, w);
                                    continue;
                                }
#endif
                                for (int64_t i = 0; i < S; ++i) {
                                    gx[off + i] = (dy[off + i] - gm - (in[off + i] - mu) * k) * r * w;
                                }
                            }
                        } else {
                            const float scale = r * w;
                            for (int64_t n = 0; n < N; ++n) {
                                const int64_t off = n * C * S + c * S;
#if defined(__x86_64__)
                                if (norm_row::avx512_ok()) {
                                    norm_row::plane_scale_f32_512(dy + off, gx + off, S, scale);
                                    continue;
                                }
#endif
                                for (int64_t i = 0; i < S; ++i) gx[off + i] = dy[off + i] * scale;
                            }
                        }
                    }
                }
            });
            break;
        }
        case DType::Float64: {
            const double* in = input.data_ptr<double>();
            const double* dy = grad_output.data_ptr<double>();
            double* gx = grad_input.defined() ? grad_input.data_ptr<double>() : nullptr;
            double* gw = grad_weight.defined() ? grad_weight.data_ptr<double>() : nullptr;
            double* gb = grad_bias.defined() ? grad_bias.data_ptr<double>() : nullptr;
            const double* wp = has_w ? weight_opt->data_ptr<double>() : nullptr;

            parallel_for(0, C, 1, [&](int64_t cb, int64_t ce) {
                for (int64_t c = cb; c < ce; ++c) {
                    const double mu = mean[c];
                    const double r = invstd[c];
                    double s = 0.0, dotp = 0.0;
                    for (int64_t n = 0; n < N; ++n) {
                        const int64_t off = n * C * S + c * S;
                        for (int64_t i = 0; i < S; ++i) {
                            const double y = dy[off + i];
                            s += y;
                            dotp += y * (in[off + i] - mu);
                        }
                    }
                    if (gb) gb[c] = s;
                    if (gw) gw[c] = dotp * r;
                    if (gx) {
                        const double w = wp ? wp[c] : 1.0;
                        if (training) {
                            const double k = dotp * r * r / M;
                            const double gm = s / M;
                            for (int64_t n = 0; n < N; ++n) {
                                const int64_t off = n * C * S + c * S;
                                for (int64_t i = 0; i < S; ++i) {
                                    gx[off + i] = (dy[off + i] - gm - (in[off + i] - mu) * k) * r * w;
                                }
                            }
                        } else {
                            const double scale = r * w;
                            for (int64_t n = 0; n < N; ++n) {
                                const int64_t off = n * C * S + c * S;
                                for (int64_t i = 0; i < S; ++i) gx[off + i] = dy[off + i] * scale;
                            }
                        }
                    }
                }
            });
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            const bool bf = input.dtype() == DType::BFloat16;
            const void* inb = input.data_ptr();
            const void* dyb = grad_output.data_ptr();
            auto readf = [&](const void* p, int64_t i) -> float {
                return bf ? static_cast<float>(static_cast<const tensorplay::BFloat16*>(p)[i])
                          : static_cast<float>(static_cast<const tensorplay::Half*>(p)[i]);
            };
            void* gxb = grad_input.defined() ? grad_input.data_ptr() : nullptr;
            const bool gxbf = grad_input.defined() && grad_input.dtype() == DType::BFloat16;
            auto writeg = [&](int64_t i, float v) {
                if (gxbf) static_cast<tensorplay::BFloat16*>(gxb)[i] = tensorplay::BFloat16(v);
                else static_cast<tensorplay::Half*>(gxb)[i] = tensorplay::Half(v);
            };
            // Parameter grads / affine params may be half, bf16 or float:
            // access them dtype-aware, never reinterpret.
            auto gwset = [&](int64_t c, double v) {
                if (grad_weight.defined()) stats_set(grad_weight, c, v);
            };
            auto gbset = [&](int64_t c, double v) {
                if (grad_bias.defined()) stats_set(grad_bias, c, v);
            };
            auto wread = [&](int64_t c) -> float {
                return has_w ? static_cast<float>(stats_at(*weight_opt, c)) : 1.0f;
            };

            parallel_for(0, C, 1, [&](int64_t cb, int64_t ce) {
                for (int64_t c = cb; c < ce; ++c) {
                    const float mu = static_cast<float>(mean[c]);
                    const float r = static_cast<float>(invstd[c]);
                    double s = 0.0, dotp = 0.0;
                    for (int64_t n = 0; n < N; ++n) {
                        const int64_t off = n * C * S + c * S;
                        for (int64_t i = 0; i < S; ++i) {
                            const double y = static_cast<double>(readf(dyb, off + i));
                            s += y;
                            dotp += y * (static_cast<double>(readf(inb, off + i)) - mu);
                        }
                    }
                    gbset(c, s);
                    gwset(c, dotp * static_cast<double>(r));
                    if (gxb) {
                        const float w = wread(c);
                        if (training) {
                            const float k = static_cast<float>(dotp * static_cast<double>(r) * r / M);
                            const float gm = static_cast<float>(s / M);
                            for (int64_t n = 0; n < N; ++n) {
                                const int64_t off = n * C * S + c * S;
                                for (int64_t i = 0; i < S; ++i) {
                                    writeg(off + i, (readf(dyb, off + i) - gm -
                                                     (readf(inb, off + i) - mu) * k) * r * w);
                                }
                            }
                        } else {
                            const float scale = r * w;
                            for (int64_t n = 0; n < N; ++n) {
                                const int64_t off = n * C * S + c * S;
                                for (int64_t i = 0; i < S; ++i) {
                                    writeg(off + i, readf(dyb, off + i) * scale);
                                }
                            }
                        }
                    }
                }
            });
            break;
        }
        default:
            TP_THROW(NotImplementedError, "batch_norm_backward: unsupported dtype");
    }
}

// Channels-last backward: vertical reductions into per-thread buffers, then
// grad_input over (N*image) contiguous C-vectors.
void bn_backward_channels_last(const Tensor& grad_output, const Tensor& input,
                               const std::optional<Tensor>& weight_opt,
                               const std::vector<double>& mean,
                               const std::vector<double>& invstd,
                               bool training, Tensor& grad_input, Tensor& grad_weight,
                               Tensor& grad_bias) {
    const int64_t C = input.size(1);
    const int64_t N = input.numel() / C;
    const double M = static_cast<double>(N);
    const int th = get_num_threads();
    const bool has_w = weight_opt.has_value() && weight_opt->defined();

    // Per-thread sum / dotp buffers.
    std::vector<double> sbuf(static_cast<size_t>(th) * C, 0.0);
    std::vector<double> dbuf(static_cast<size_t>(th) * C, 0.0);

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in = input.data_ptr<float>();
            const float* dy = grad_output.data_ptr<float>();
            parallel_for(0, N, 1, [&](int64_t b, int64_t e) {
                double* srow = sbuf.data() + static_cast<size_t>(get_thread_num()) * C;
                double* drow = dbuf.data() + static_cast<size_t>(get_thread_num()) * C;
                for (int64_t i = b; i < e; ++i) {
                    const float* xp = in + i * C;
                    const float* yp = dy + i * C;
                    for (int64_t c = 0; c < C; ++c) {
                        const double y = static_cast<double>(yp[c]);
                        srow[c] += y;
                        drow[c] += y * (static_cast<double>(xp[c]) - mean[c]);
                    }
                }
            });
            break;
        }
        case DType::Float64: {
            const double* in = input.data_ptr<double>();
            const double* dy = grad_output.data_ptr<double>();
            parallel_for(0, N, 1, [&](int64_t b, int64_t e) {
                double* srow = sbuf.data() + static_cast<size_t>(get_thread_num()) * C;
                double* drow = dbuf.data() + static_cast<size_t>(get_thread_num()) * C;
                for (int64_t i = b; i < e; ++i) {
                    for (int64_t c = 0; c < C; ++c) {
                        srow[c] += dy[i * C + c];
                        drow[c] += dy[i * C + c] * (in[i * C + c] - mean[c]);
                    }
                }
            });
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            const bool bf = input.dtype() == DType::BFloat16;
            const void* inb = input.data_ptr();
            const void* dyb = grad_output.data_ptr();
            parallel_for(0, N, 1, [&](int64_t b, int64_t e) {
                double* srow = sbuf.data() + static_cast<size_t>(get_thread_num()) * C;
                double* drow = dbuf.data() + static_cast<size_t>(get_thread_num()) * C;
                for (int64_t i = b; i < e; ++i) {
                    for (int64_t c = 0; c < C; ++c) {
                        const double y = bf ? static_cast<double>(static_cast<const tensorplay::BFloat16*>(dyb)[i * C + c])
                                            : static_cast<double>(static_cast<const tensorplay::Half*>(dyb)[i * C + c]);
                        const double x = bf ? static_cast<double>(static_cast<const tensorplay::BFloat16*>(inb)[i * C + c])
                                            : static_cast<double>(static_cast<const tensorplay::Half*>(inb)[i * C + c]);
                        srow[c] += y;
                        drow[c] += y * (x - mean[c]);
                    }
                }
            });
            break;
        }
        default:
            TP_THROW(NotImplementedError, "batch_norm_backward: unsupported dtype");
    }

    // Reduce over threads.
    std::vector<double> sum(C, 0.0), dotp(C, 0.0);
    for (int64_t c = 0; c < C; ++c) {
        double s = 0.0, d = 0.0;
        for (int64_t t = 0; t < th; ++t) {
            s += sbuf[t * C + c];
            d += dbuf[t * C + c];
        }
        sum[c] = s;
        dotp[c] = d;
        if (grad_bias.defined()) stats_set(grad_bias, c, s);
        if (grad_weight.defined()) stats_set(grad_weight, c, d * invstd[c]);
    }

    // grad_input over rows.
    if (!grad_input.defined()) return;
    switch (input.dtype()) {
        case DType::Float32: {
            const float* in = input.data_ptr<float>();
            const float* dy = grad_output.data_ptr<float>();
            float* gx = grad_input.data_ptr<float>();
            const float* wp = has_w ? weight_opt->data_ptr<float>() : nullptr;
            parallel_for(0, N, 1, [&](int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) {
                    const float* xp = in + i * C;
                    const float* yp = dy + i * C;
                    float* op = gx + i * C;
#if defined(__x86_64__)
                    if (norm_row::avx512_ok() && C >= 16) {
                        bn_bwd_f32x16(op, xp, yp, mean.data(), dotp.data(),
                                      invstd.data(), sum.data(),
                                      wp, C, M, training);
                        continue;
                    }
#endif
                    for (int64_t c = 0; c < C; ++c) {
                        const float w = wp ? wp[c] : 1.0f;
                        if (training) {
                            const float k = static_cast<float>(dotp[c] * invstd[c] * invstd[c] / M);
                            const float gm = static_cast<float>(sum[c] / M);
                            op[c] = (yp[c] - gm - (xp[c] - static_cast<float>(mean[c])) * k) *
                                    static_cast<float>(invstd[c]) * w;
                        } else {
                            op[c] = yp[c] * static_cast<float>(invstd[c]) * w;
                        }
                    }
                }
            });
            break;
        }
        default:
            // Reduced and double types: shared scalar row loop in fp32 math.
        {
            const bool bf = input.dtype() == DType::BFloat16;
            const bool f64 = input.dtype() == DType::Float64;
            const void* inb = input.data_ptr();
            const void* dyb = grad_output.data_ptr();
            void* gxb = grad_input.data_ptr();
            const auto wrow = [&](int64_t c) -> float {
                return has_w ? static_cast<float>(stats_at(*weight_opt, c)) : 1.0f;
            };
            parallel_for(0, N, 1, [&](int64_t b, int64_t e) {
                for (int64_t i = b; i < e; ++i) {
                    for (int64_t c = 0; c < C; ++c) {
                        const double w = static_cast<double>(wrow(c));
                        double y, x;
                        if (f64) {
                            y = static_cast<const double*>(dyb)[i * C + c];
                            x = static_cast<const double*>(inb)[i * C + c];
                        } else if (bf) {
                            y = static_cast<double>(static_cast<const tensorplay::BFloat16*>(dyb)[i * C + c]);
                            x = static_cast<double>(static_cast<const tensorplay::BFloat16*>(inb)[i * C + c]);
                        } else {
                            y = static_cast<double>(static_cast<const tensorplay::Half*>(dyb)[i * C + c]);
                            x = static_cast<double>(static_cast<const tensorplay::Half*>(inb)[i * C + c]);
                        }
                        double v = training
                            ? (y - sum[c] / M - (x - mean[c]) * (dotp[c] * invstd[c] * invstd[c] / M)) * invstd[c] * w
                            : y * invstd[c] * w;
                        if (f64) static_cast<double*>(gxb)[i * C + c] = v;
                        else if (bf) static_cast<tensorplay::BFloat16*>(gxb)[i * C + c] = tensorplay::BFloat16(static_cast<float>(v));
                        else static_cast<tensorplay::Half*>(gxb)[i * C + c] = tensorplay::Half(static_cast<float>(v));
                    }
                }
            });
        }
    }
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cpu(
    const Tensor& grad_output, const Tensor& input,
    std::optional<Tensor> weight_opt,
    std::optional<Tensor> running_mean_opt,
    std::optional<Tensor> running_var_opt,
    bool training, double eps) {
    const int64_t C = input.size(1);

    check_param_dtype(input.dtype(), weight_opt, "weight");
    check_param_dtype(input.dtype(), running_mean_opt, "running_mean");
    check_param_dtype(input.dtype(), running_var_opt, "running_var");

    const bool has_w = weight_opt.has_value() && weight_opt->defined();

    // grad_input follows the grad_output dtype (matches autograd expectations);
    // parameter grads follow the affine param dtype.
    const DType gx_dt = grad_output.dtype();
    Tensor grad_input, grad_weight, grad_bias;
    grad_input = empty_like_format(input, gx_dt);
    if (has_w) grad_weight = Tensor::empty({C}, weight_opt->dtype(), weight_opt->device());
    const DType grad_param_dtype = has_w
        ? weight_opt->dtype() : stats_dtype_for(input.dtype());
    grad_bias = Tensor::empty({C}, grad_param_dtype, input.device());

    // Resolve mean/invstd (recompute in training mode, running stats in eval).
    std::vector<double> mean(C), invstd(C);
    if (training) {
        std::vector<double> var(C);
        if (is_channels_last(input)) bn_stats_channels_last(input, mean, var);
        else bn_stats_contiguous(input, mean, var);
        for (int64_t c = 0; c < C; ++c) invstd[c] = 1.0 / std::sqrt(var[c] + eps);
    } else {
        for (int64_t c = 0; c < C; ++c) {
            mean[c] = (running_mean_opt.has_value() && running_mean_opt->defined())
                          ? stats_at(*running_mean_opt, c) : 0.0;
            const double v = (running_var_opt.has_value() && running_var_opt->defined())
                                 ? stats_at(*running_var_opt, c) : 0.0;
            invstd[c] = 1.0 / std::sqrt(v + eps);
        }
    }

    if (is_channels_last(input))
        bn_backward_channels_last(grad_output, input, weight_opt, mean, invstd,
                                  training, grad_input, grad_weight, grad_bias);
    else
        bn_backward_contiguous(grad_output, input, weight_opt, running_mean_opt,
                               running_var_opt, mean, invstd, training,
                               grad_input, grad_weight, grad_bias);

    if (!has_w) grad_weight = Tensor();
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_cpu(
    const Tensor& input, std::optional<Tensor> weight_opt,
    std::optional<Tensor> bias_opt,
    std::optional<Tensor> running_mean_opt,
    std::optional<Tensor> running_var_opt, bool training,
    double momentum, double eps) {
    std::optional<Tensor> running_mean = running_mean_opt;
    std::optional<Tensor> running_var = running_var_opt;
    Tensor out = batch_norm_cpu(input, weight_opt, bias_opt, running_mean,
                                running_var, training, momentum, eps);

    const int64_t channels = input.size(1);
    const DType stats_dtype = stats_dtype_for(input.dtype());
    if (input.numel() == 0) {
        Tensor mean = Tensor::empty({channels}, stats_dtype, input.device());
        Tensor invstd = Tensor::empty({channels}, stats_dtype, input.device());
        return std::make_tuple(out, mean, invstd);
    }
    if (!training) {
        Tensor mean = Tensor::empty({0}, stats_dtype, input.device());
        Tensor invstd = Tensor::empty({0}, stats_dtype, input.device());
        return std::make_tuple(out, mean, invstd);
    }

    std::vector<double> mean_values(channels, 0.0);
    std::vector<double> variance_values(channels, 0.0);
    if (is_channels_last(input)) {
        bn_stats_channels_last(input, mean_values, variance_values);
    } else {
        bn_stats_contiguous(input, mean_values, variance_values);
    }
    Tensor mean = Tensor::empty({channels}, stats_dtype, input.device());
    Tensor invstd = Tensor::empty({channels}, stats_dtype, input.device());
    for (int64_t channel = 0; channel < channels; ++channel) {
        stats_set(mean, channel, mean_values[channel]);
        stats_set(invstd, channel,
                  1.0 / std::sqrt(variance_values[channel] + eps));
    }
    return std::make_tuple(out, mean, invstd);
}

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_backward_cpu(
    const Tensor& grad_out, const Tensor& input,
    std::optional<Tensor> weight_opt,
    std::optional<Tensor> running_mean_opt,
    std::optional<Tensor> running_var_opt,
    std::optional<Tensor> save_mean_opt,
    std::optional<Tensor> save_invstd_opt, bool train, double eps,
    const std::vector<bool>& output_mask) {
    (void)save_mean_opt;
    (void)save_invstd_opt;
    auto grads = batch_norm_backward_cpu(
        grad_out, input, weight_opt, running_mean_opt, running_var_opt,
        train, eps);
    Tensor grad_input = output_mask.size() > 0 && output_mask[0]
        ? std::get<0>(grads) : Tensor();
    Tensor grad_weight = output_mask.size() > 1 && output_mask[1]
        ? std::get<1>(grads) : Tensor();
    Tensor grad_bias = output_mask.size() > 2 && output_mask[2]
        ? std::get<2>(grads) : Tensor();
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

// ---------------------------------------------------------------------------
// instance_norm: composite over batch_norm (reshape to (1, N*C, S)).
// ---------------------------------------------------------------------------

Tensor instance_norm_cpu(const Tensor& input, std::optional<Tensor> weight_opt,
                         std::optional<Tensor> bias_opt,
                         std::optional<Tensor> running_mean_opt,
                         std::optional<Tensor> running_var_opt,
                         bool use_input_stats, double momentum, double eps) {
    if (input.dim() < 3)
        TP_THROW(RuntimeError, "instance_norm: input must have at least 3 dimensions");
    if (!isFloatingType(input.dtype()))
        TP_THROW(NotImplementedError,
                 std::string("instance_norm: only floating point dtypes are supported (got ") +
                 toString(input.dtype()) + ")");
    std::optional<Tensor> running_mean = running_mean_opt;
    std::optional<Tensor> running_var = running_var_opt;
    if (!use_input_stats &&
        !(running_mean.has_value() && running_mean->defined() &&
          running_var.has_value() && running_var->defined()))
        TP_THROW(RuntimeError,
                 "instance_norm: running_mean and running_var must be defined when use_input_stats is false");

    const int64_t b = input.size(0);
    const int64_t c = input.size(1);
    const DType sdt = stats_dtype_for(input.dtype());

    // Affine params and running stats are broadcast per (sample, channel).
    std::optional<Tensor> weight_r, bias_r, rm_r, rv_r;
    if (weight_opt.has_value() && weight_opt->defined())
        weight_r = repeat_channels(*weight_opt, b, sdt);
    if (bias_opt.has_value() && bias_opt->defined())
        bias_r = repeat_channels(*bias_opt, b, sdt);
    if (running_mean.has_value() && running_mean->defined())
        rm_r = repeat_channels(*running_mean, b, sdt);
    if (running_var.has_value() && running_var->defined())
        rv_r = repeat_channels(*running_var, b, sdt);

    Tensor input_c = input.contiguous();
    Tensor reshaped = input_c.view({1, b * c, input_c.numel() / (b * c)});
    Tensor out = batch_norm_cpu(reshaped, weight_r, bias_r, rm_r, rv_r,
                                use_input_stats, momentum, eps);

    if (use_input_stats) {
        // Fold the updated per-plane stats back: running stats are the average
        // over the batch of the per-sample statistics.
        if (running_mean.has_value() && running_mean->defined()) {
            for (int64_t ch = 0; ch < c; ++ch) {
                double acc = 0.0;
                for (int64_t n = 0; n < b; ++n) acc += stats_at(*rm_r, n * c + ch);
                stats_set(*running_mean, ch, acc / static_cast<double>(b));
            }
        }
        if (running_var.has_value() && running_var->defined()) {
            for (int64_t ch = 0; ch < c; ++ch) {
                double acc = 0.0;
                for (int64_t n = 0; n < b; ++n) acc += stats_at(*rv_r, n * c + ch);
                stats_set(*running_var, ch, acc / static_cast<double>(b));
            }
        }
    }

    return out.view(static_cast<std::vector<int64_t>>(input.shape()));
}

// ---------------------------------------------------------------------------
// instance_norm backward
// ---------------------------------------------------------------------------

// Provided by NormalizationKernels.cpp.
std::tuple<Tensor, Tensor, Tensor> group_norm_backward_cpu(
    const Tensor& grad_output, const Tensor& input, int64_t num_groups,
    const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    double eps);

std::tuple<Tensor, Tensor, Tensor> instance_norm_backward_cpu(
    const Tensor& grad_output, const Tensor& input,
    std::optional<Tensor> weight_opt, std::optional<Tensor> bias_opt,
    std::optional<Tensor> running_mean_opt,
    std::optional<Tensor> running_var_opt,
    bool use_input_stats, double eps) {
    if (use_input_stats) {
        const int64_t C = input.size(1);
        return group_norm_backward_cpu(grad_output, input, C, weight_opt, bias_opt, eps);
    }
    return batch_norm_backward_cpu(grad_output, input, weight_opt,
                                   running_mean_opt, running_var_opt, false, eps);
}

}  // namespace cpu

TENSORPLAY_LIBRARY_IMPL(CPU, BatchNormKernels) {
    using namespace tensorplay::cpu;
    m.impl("batch_norm", batch_norm_cpu);
    m.impl("batch_norm_backward", batch_norm_backward_cpu);
    m.impl("native_batch_norm", native_batch_norm_cpu);
    m.impl("native_batch_norm_backward", native_batch_norm_backward_cpu);
    m.impl("instance_norm", instance_norm_cpu);
    m.impl("instance_norm_backward", instance_norm_backward_cpu);
}

}  // namespace tensorplay

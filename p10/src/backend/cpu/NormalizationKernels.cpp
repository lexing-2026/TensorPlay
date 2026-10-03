#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "NormRowHelpers.h"
#include "MemoryFormat.h"

namespace tensorplay {
namespace cpu {

namespace {

DType normalization_stats_dtype(DType input_dtype) {
    return input_dtype == DType::Float16 || input_dtype == DType::BFloat16
        ? DType::Float32
        : input_dtype;
}

double normalization_stats_read(const Tensor& tensor, int64_t index) {
    switch (tensor.dtype()) {
        case DType::Float64:
            return tensor.data_ptr<double>()[index];
        case DType::Float32:
            return static_cast<double>(tensor.data_ptr<float>()[index]);
        case DType::Float16:
            return static_cast<double>(tensor.data_ptr<tensorplay::Half>()[index]);
        case DType::BFloat16:
            return static_cast<double>(tensor.data_ptr<tensorplay::BFloat16>()[index]);
        default:
            TP_THROW(RuntimeError, "normalization: unsupported statistics dtype");
    }
}

void normalization_stats_write(Tensor& tensor, int64_t index, double value) {
    switch (tensor.dtype()) {
        case DType::Float64:
            tensor.data_ptr<double>()[index] = value;
            return;
        case DType::Float32:
            tensor.data_ptr<float>()[index] = static_cast<float>(value);
            return;
        case DType::Float16:
            tensor.data_ptr<tensorplay::Half>()[index] =
                tensorplay::Half(static_cast<float>(value));
            return;
        case DType::BFloat16:
            tensor.data_ptr<tensorplay::BFloat16>()[index] =
                tensorplay::BFloat16(static_cast<float>(value));
            return;
        default:
            TP_THROW(RuntimeError, "normalization: unsupported statistics dtype");
    }
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cpu(
    const Tensor& grad_output, const Tensor& input,
    const std::optional<Tensor>& weight_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool training, double eps);

}  // namespace cpu
}  // namespace tensorplay
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#if defined(__x86_64__)
#include <immintrin.h>
#include <limits>
#endif

namespace tensorplay {
namespace cpu {

// ---------------------------------------------------------------------------
// Group norm over channels-last storage.
//
// A channels-last activation stores the C channels of one position next to
// each other, so a group's values are interleaved with the other groups'.
// Rather than repacking the activation into planes, the kernels below walk a
// sample position by position: per-channel accumulators (C wide, contiguous)
// collect the moments, the groups are folded out of them, and a second walk
// applies per-channel coefficient arrays.  Both walks vectorize across the
// channel axis, and the result is written channels-last.  Samples are shared
// over the intra-op pool; one sample is small enough that its second walk
// reads what the first left in cache.
// ---------------------------------------------------------------------------
namespace {

#if defined(__GNUC__) || defined(__clang__)
#define TP_GN_BODY static inline __attribute__((always_inline))
#else
#define TP_GN_BODY static inline
#endif

// Per-channel sum and sum of squares over the S positions of one sample.
TP_GN_BODY void gn_nhwc_moments_body(const float* x, int64_t S, int64_t C,
                                     float* sum, float* sq) {
    for (int64_t c = 0; c < C; ++c) { sum[c] = 0.0f; sq[c] = 0.0f; }
    for (int64_t s = 0; s < S; ++s) {
        const float* xs = x + s * C;
        for (int64_t c = 0; c < C; ++c) {
            const float v = xs[c];
            sum[c] += v;
            sq[c] += v * v;
        }
    }
}

// y = (x - mean_c) * scale_c + shift_c
TP_GN_BODY void gn_nhwc_apply_body(const float* x, float* y, int64_t S,
                                   int64_t C, const float* mean_c,
                                   const float* scale_c, const float* shift_c) {
    for (int64_t s = 0; s < S; ++s) {
        const float* xs = x + s * C;
        float* ys = y + s * C;
        for (int64_t c = 0; c < C; ++c) {
            ys[c] = (xs[c] - mean_c[c]) * scale_c[c] + shift_c[c];
        }
    }
}

// Per-channel sum(dy) and sum(dy * x), accumulated in double.
TP_GN_BODY void gn_nhwc_bwd_sums_body(const float* dy, const float* x,
                                      int64_t S, int64_t C, double* sd,
                                      double* dot) {
    for (int64_t c = 0; c < C; ++c) { sd[c] = 0.0; dot[c] = 0.0; }
    for (int64_t s = 0; s < S; ++s) {
        const float* ds = dy + s * C;
        const float* xs = x + s * C;
        for (int64_t c = 0; c < C; ++c) {
            const double y = static_cast<double>(ds[c]);
            sd[c] += y;
            dot[c] += y * static_cast<double>(xs[c]);
        }
    }
}

// gi = term1_c * (M * (dy * w_c) - sdy_c - ((x - mean_c) * inv_c) * sxh_c)
TP_GN_BODY void gn_nhwc_bwd_apply_body(const float* dy, const float* x,
                                       float* gi, int64_t S, int64_t C,
                                       const float* w_c, const float* mean_c,
                                       const float* inv_c, const float* term1_c,
                                       const float* sdy_c, const float* sxh_c,
                                       float M) {
    for (int64_t s = 0; s < S; ++s) {
        const float* ds = dy + s * C;
        const float* xs = x + s * C;
        float* gs = gi + s * C;
        for (int64_t c = 0; c < C; ++c) {
            const float dyw = ds[c] * w_c[c];
            const float x_hat = (xs[c] - mean_c[c]) * inv_c[c];
            gs[c] = term1_c[c] * (M * dyw - sdy_c[c] - x_hat * sxh_c[c]);
        }
    }
}

// Each walk exists once for the base instruction set and once for AVX2+FMA;
// the wider one is reached only after the running CPU reported both.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define TP_GN_WIDE __attribute__((target("avx2,fma")))
TP_GN_WIDE void gn_nhwc_moments_wide(const float* x, int64_t S, int64_t C,
                                     float* sum, float* sq) {
    gn_nhwc_moments_body(x, S, C, sum, sq);
}
TP_GN_WIDE void gn_nhwc_apply_wide(const float* x, float* y, int64_t S,
                                   int64_t C, const float* mean_c,
                                   const float* scale_c, const float* shift_c) {
    gn_nhwc_apply_body(x, y, S, C, mean_c, scale_c, shift_c);
}
TP_GN_WIDE void gn_nhwc_bwd_sums_wide(const float* dy, const float* x, int64_t S,
                                      int64_t C, double* sd, double* dot) {
    gn_nhwc_bwd_sums_body(dy, x, S, C, sd, dot);
}
TP_GN_WIDE void gn_nhwc_bwd_apply_wide(const float* dy, const float* x, float* gi,
                                       int64_t S, int64_t C, const float* w_c,
                                       const float* mean_c, const float* inv_c,
                                       const float* term1_c, const float* sdy_c,
                                       const float* sxh_c, float M) {
    gn_nhwc_bwd_apply_body(dy, x, gi, S, C, w_c, mean_c, inv_c, term1_c, sdy_c,
                           sxh_c, M);
}
bool gn_nhwc_wide_ok() {
    static const bool ok =
        __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return ok;
}
#else
bool gn_nhwc_wide_ok() { return false; }
#endif

void gn_nhwc_moments(const float* x, int64_t S, int64_t C, float* sum, float* sq) {
#ifdef TP_GN_WIDE
    if (gn_nhwc_wide_ok()) return gn_nhwc_moments_wide(x, S, C, sum, sq);
#endif
    gn_nhwc_moments_body(x, S, C, sum, sq);
}
void gn_nhwc_apply(const float* x, float* y, int64_t S, int64_t C,
                   const float* mean_c, const float* scale_c,
                   const float* shift_c) {
#ifdef TP_GN_WIDE
    if (gn_nhwc_wide_ok()) return gn_nhwc_apply_wide(x, y, S, C, mean_c, scale_c, shift_c);
#endif
    gn_nhwc_apply_body(x, y, S, C, mean_c, scale_c, shift_c);
}
void gn_nhwc_bwd_sums(const float* dy, const float* x, int64_t S, int64_t C,
                      double* sd, double* dot) {
#ifdef TP_GN_WIDE
    if (gn_nhwc_wide_ok()) return gn_nhwc_bwd_sums_wide(dy, x, S, C, sd, dot);
#endif
    gn_nhwc_bwd_sums_body(dy, x, S, C, sd, dot);
}
void gn_nhwc_bwd_apply(const float* dy, const float* x, float* gi, int64_t S,
                       int64_t C, const float* w_c, const float* mean_c,
                       const float* inv_c, const float* term1_c,
                       const float* sdy_c, const float* sxh_c, float M) {
#ifdef TP_GN_WIDE
    if (gn_nhwc_wide_ok()) {
        return gn_nhwc_bwd_apply_wide(dy, x, gi, S, C, w_c, mean_c, inv_c,
                                      term1_c, sdy_c, sxh_c, M);
    }
#endif
    gn_nhwc_bwd_apply_body(dy, x, gi, S, C, w_c, mean_c, inv_c, term1_c, sdy_c,
                           sxh_c, M);
}

// The channels-last walk shares samples over the pool, so it is taken when
// there are enough samples to occupy it; a batch too small for that keeps the
// plane kernels, which share (sample, group) rows.
bool gn_use_nhwc(const Tensor& input, int64_t N) {
    return input.dim() == 4 && input.dtype() == DType::Float32 &&
           input.is_contiguous(MemoryFormat::ChannelsLast) &&
           !input.is_contiguous() &&
           2 * N >= tensorplay::parallel::get_num_threads();
}

Tensor gn_empty_channels_last(const Tensor& like) {
    const auto sizes = static_cast<std::vector<int64_t>>(like.shape());
    Tensor out = Tensor::empty(sizes, like.dtype(), like.device());
    return out.as_strided(sizes, get_channels_last_strides(sizes), 0);
}

Tensor gn_group_norm_nhwc(const Tensor& input, int64_t G, const float* w_ptr,
                          const float* b_ptr, float feps, Tensor* mean_out,
                          Tensor* rstd_out) {
    const int64_t N = input.size(0), C = input.size(1);
    const int64_t S = input.size(2) * input.size(3);
    const int64_t D = C / G;
    const float M = static_cast<float>(D * S);
    Tensor out = gn_empty_channels_last(input);
    const float* in_ptr = input.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    tensorplay::parallel::parallel_for(0, N, 1, [&](int64_t nb, int64_t ne) {
        std::vector<float> scratch(static_cast<size_t>(5 * C));
        float* sum = scratch.data();
        float* sq = sum + C;
        float* mean_c = sq + C;
        float* scale_c = mean_c + C;
        float* shift_c = scale_c + C;
        for (int64_t n = nb; n < ne; ++n) {
            const float* x = in_ptr + n * S * C;
            gn_nhwc_moments(x, S, C, sum, sq);
            for (int64_t g = 0; g < G; ++g) {
                float gs = 0.0f, gq = 0.0f;
                for (int64_t c = g * D; c < (g + 1) * D; ++c) {
                    gs += sum[c];
                    gq += sq[c];
                }
                const float mean = gs / M;
                const float var = gq / M - mean * mean;
                const float inv_std = 1.0f / std::sqrt(var + feps);
                if (mean_out) normalization_stats_write(*mean_out, n * G + g, mean);
                if (rstd_out) normalization_stats_write(*rstd_out, n * G + g, inv_std);
                for (int64_t c = g * D; c < (g + 1) * D; ++c) {
                    mean_c[c] = mean;
                    scale_c[c] = inv_std * (w_ptr ? w_ptr[c] : 1.0f);
                    shift_c[c] = b_ptr ? b_ptr[c] : 0.0f;
                }
            }
            gn_nhwc_apply(x, out_ptr + n * S * C, S, C, mean_c, scale_c, shift_c);
        }
    });
    return out;
}

// Fills grad_in (channels-last, may be null) and adds each thread's
// dgamma/dbeta partials into its (C,) slice of gw_buf/gb_buf.
void gn_group_norm_backward_nhwc(const float* dy, const float* x, float* grad_in,
                                 int64_t N, int64_t C, int64_t S, int64_t G,
                                 const float* w_ptr, float feps,
                                 const Tensor* mean_opt, const Tensor* rstd_opt,
                                 float* gw_buf, float* gb_buf) {
    const int64_t D = C / G;
    const float M = static_cast<float>(D * S);
    tensorplay::parallel::parallel_for(0, N, 1, [&](int64_t nb, int64_t ne) {
        const size_t tid = static_cast<size_t>(tensorplay::parallel::get_thread_num());
        float* gw_row = gw_buf ? gw_buf + tid * C : nullptr;
        float* gb_row = gb_buf ? gb_buf + tid * C : nullptr;
        std::vector<double> dscratch(static_cast<size_t>(2 * C));
        double* sd = dscratch.data();
        double* dot = sd + C;
        std::vector<float> scratch(static_cast<size_t>(8 * C));
        float* sum = scratch.data();
        float* sq = sum + C;
        float* w_c = sq + C;
        float* mean_c = w_c + C;
        float* inv_c = mean_c + C;
        float* term1_c = inv_c + C;
        float* sdy_c = term1_c + C;
        float* sxh_c = sdy_c + C;
        for (int64_t n = nb; n < ne; ++n) {
            const float* xs = x + n * S * C;
            const float* ds = dy + n * S * C;
            gn_nhwc_bwd_sums(ds, xs, S, C, sd, dot);
            if (!mean_opt) gn_nhwc_moments(xs, S, C, sum, sq);
            for (int64_t g = 0; g < G; ++g) {
                float mean, inv_std;
                if (mean_opt) {
                    mean = static_cast<float>(normalization_stats_read(*mean_opt, n * G + g));
                    inv_std = static_cast<float>(normalization_stats_read(*rstd_opt, n * G + g));
                } else {
                    float gs = 0.0f, gq = 0.0f;
                    for (int64_t c = g * D; c < (g + 1) * D; ++c) {
                        gs += sum[c];
                        gq += sq[c];
                    }
                    mean = gs / M;
                    const float var = gq / M - mean * mean;
                    inv_std = 1.0f / std::sqrt(var + feps);
                }
                float s_dy = 0.0f, s_dy_xhat = 0.0f;
                for (int64_t c = g * D; c < (g + 1) * D; ++c) {
                    const float s_dy_c = static_cast<float>(sd[c]);
                    const float s_dy_xhat_c =
                        static_cast<float>((dot[c] - mean * sd[c]) * inv_std);
                    if (gb_row) gb_row[c] += s_dy_c;
                    if (gw_row) gw_row[c] += s_dy_xhat_c;
                    const float w = w_ptr ? w_ptr[c] : 1.0f;
                    s_dy += w * s_dy_c;
                    s_dy_xhat += w * s_dy_xhat_c;
                }
                const float term1 = inv_std / M;
                for (int64_t c = g * D; c < (g + 1) * D; ++c) {
                    w_c[c] = w_ptr ? w_ptr[c] : 1.0f;
                    mean_c[c] = mean;
                    inv_c[c] = inv_std;
                    term1_c[c] = term1;
                    sdy_c[c] = s_dy;
                    sxh_c[c] = s_dy_xhat;
                }
            }
            if (grad_in) {
                gn_nhwc_bwd_apply(ds, xs, grad_in + n * S * C, S, C, w_c, mean_c,
                                  inv_c, term1_c, sdy_c, sxh_c, M);
            }
        }
    });
}

#undef TP_GN_BODY

}  // namespace

// Helper to check input validity
static void check_dims(const Tensor& input, int64_t expected_dim, const char* name) {
    if (input.dim() != expected_dim) {
        TP_THROW(RuntimeError, std::string(name) + ": Expected " + std::to_string(expected_dim) + "D input");
    }
}

// Backward for GroupNorm.  The native entry point supplies the saved moments;
// the public helper passes null moments and computes them in this kernel.
static std::tuple<Tensor, Tensor, Tensor> group_norm_backward_cpu_impl(
        const Tensor& grad_output, const Tensor& input, int64_t num_groups,
        const std::optional<Tensor>& weight_opt,
        const std::optional<Tensor>& bias_opt, double eps,
        const Tensor* mean_opt, const Tensor* rstd_opt,
        const std::vector<bool>* output_mask) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);

    if (input.dim() < 2)
        TP_THROW(RuntimeError, "group_norm_backward requires at least 2 dims");
    if (num_groups <= 0)
        TP_THROW(RuntimeError, "group_norm_backward: num_groups must be positive");
    if (C <= 0 || C % num_groups != 0)
        TP_THROW(RuntimeError, "group_norm_backward: C must be positive and divisible by num_groups");

    int64_t G = num_groups;
    int64_t D = C / G;

    int64_t spatial_size = 1;
    for (int64_t dim = 2; dim < input.dim(); ++dim)
        spatial_size *= input.size(dim);
    if (spatial_size <= 0)
        TP_THROW(RuntimeError, "group_norm_backward: spatial dimensions must be positive");
    int64_t group_size = D * spatial_size; // Normalization size

    if (input.dtype() != DType::Float32) TP_THROW(NotImplementedError, "group_norm_backward only supports Float32");
    if (grad_output.dtype() != input.dtype())
        TP_THROW(RuntimeError, "group_norm_backward: grad_output dtype must match input dtype");

    const bool has_weight = weight_opt.has_value() && weight_opt->defined();
    const bool has_bias = bias_opt.has_value() && bias_opt->defined();
    const bool want_input = output_mask == nullptr || output_mask->empty() ||
        (output_mask->size() > 0 && (*output_mask)[0]);
    const bool want_weight = has_weight &&
        (output_mask == nullptr || output_mask->size() <= 1 || (*output_mask)[1]);
    const bool want_bias = output_mask == nullptr
        ? has_bias
        : (output_mask->size() > 2 && (*output_mask)[2]);

    // A channels-last activation is differentiated where it lies; the
    // gradient is brought to the same order if it is not already in it.
    const bool nhwc = gn_use_nhwc(input, N);
    const Tensor go_c = nhwc
        ? (grad_output.is_contiguous(MemoryFormat::ChannelsLast)
               ? grad_output
               : detail::contiguous_impl(
                     grad_output, static_cast<int64_t>(MemoryFormat::ChannelsLast)))
        : grad_output.contiguous();
    const Tensor in_c = nhwc ? input : input.contiguous();

    Tensor grad_input = want_input
        ? (nhwc ? gn_empty_channels_last(input) : Tensor::empty_like(in_c))
        : Tensor();
    Tensor grad_weight;
    Tensor grad_bias;

    if (want_weight) grad_weight = Tensor::empty_like(*weight_opt);
    if (want_bias) {
        if (has_bias) grad_bias = Tensor::empty_like(*bias_opt);
        else grad_bias = Tensor::empty({C}, input.dtype(), input.device());
    }

    const int64_t rows = N * G;
    if (N == 0) {
        if (want_input) grad_input = Tensor::zeros_like(input);
        if (want_weight) grad_weight = Tensor::zeros_like(*weight_opt);
        if (want_bias) {
            if (has_bias) grad_bias = Tensor::zeros_like(*bias_opt);
            else grad_bias = Tensor::zeros({C}, input.dtype(), input.device());
        }
        return std::make_tuple(grad_input, grad_weight, grad_bias);
    }

    if (mean_opt || rstd_opt) {
        if (!mean_opt || !rstd_opt || !mean_opt->defined() ||
            !rstd_opt->defined() || mean_opt->numel() != rows ||
            rstd_opt->numel() != rows) {
            TP_THROW(RuntimeError,
                     "group_norm_backward: saved statistics have an invalid shape");
        }
    }

    float* grad_in_ptr = grad_input.defined() ? grad_input.data_ptr<float>() : nullptr;
    const float* grad_out_ptr = go_c.data_ptr<float>();
    const float* in_ptr = in_c.data_ptr<float>();

    float* gw_ptr = (grad_weight.defined()) ? grad_weight.data_ptr<float>() : nullptr;
    float* gb_ptr = (grad_bias.defined()) ? grad_bias.data_ptr<float>() : nullptr;
    const float* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<float>() : nullptr;

    // Per-thread dgamma/dbeta partials: rows are independent, so each thread
    // accumulates into its own (C,) slices and a final pass reduces them.
    const int th = tensorplay::parallel::get_num_threads();
    std::vector<float> gw_buf;
    std::vector<float> gb_buf;
    if (gw_ptr) gw_buf.assign(static_cast<size_t>(th) * C, 0.0f);
    if (gb_ptr) gb_buf.assign(static_cast<size_t>(th) * C, 0.0f);

    const float feps = static_cast<float>(eps);
    const int64_t row_grain = std::max<int64_t>(
        1, tensorplay::parallel::GRAIN_SIZE / std::max<int64_t>(group_size, 1));

    if (nhwc) {
        gn_group_norm_backward_nhwc(
            grad_out_ptr, in_ptr, grad_in_ptr, N, C, spatial_size, G, w_ptr, feps,
            mean_opt, rstd_opt, gw_ptr ? gw_buf.data() : nullptr,
            gb_ptr ? gb_buf.data() : nullptr);
    } else
    tensorplay::parallel::parallel_for(0, rows, row_grain, [&](int64_t rb, int64_t re) {
        float* gw_row = gw_ptr ? gw_buf.data() + static_cast<size_t>(tensorplay::parallel::get_thread_num()) * C : nullptr;
        float* gb_row = gb_ptr ? gb_buf.data() + static_cast<size_t>(tensorplay::parallel::get_thread_num()) * C : nullptr;
        for (int64_t row = rb; row < re; ++row) {
            const int64_t n = row / G;
            const int64_t g = row % G;
            const int64_t c_start = g * D;
            const int64_t group_offset = n * C * spatial_size + g * D * spatial_size;

            // 1. Use the saved moments for native backward; public backward
            // computes the same moments when no saved tensors are available.
            float mean;
            float inv_std;
            if (mean_opt) {
                mean = static_cast<float>(normalization_stats_read(*mean_opt, row));
                inv_std = static_cast<float>(normalization_stats_read(*rstd_opt, row));
            } else {
#if defined(__x86_64__)
                if (norm_row::avx512_ok() && group_size >= 16) {
                    norm_row::stats_f32_512(in_ptr + group_offset, group_size, feps,
                                            &mean, &inv_std);
                } else
#endif
                {
                    float sum = 0.0f, sq_sum = 0.0f;
                    for (int64_t i = 0; i < group_size; ++i) {
                        const float val = in_ptr[group_offset + i];
                        sum += val;
                        sq_sum += val * val;
                    }
                    mean = sum / group_size;
                    const float var = (sq_sum / group_size) - mean * mean;
                    inv_std = 1.0f / std::sqrt(var + feps);
                }
            }

            // 2. Per-channel reductions: sum(dy), sum(dy * x) fold into the
            //    dgamma/dbeta partials and the row-local terms.
            float s_dy = 0.0f;
            float s_dy_xhat = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                const int64_t c = c_start + d;
                const int64_t c_offset = group_offset + d * spatial_size;
                double sd = 0.0, dotp = 0.0;
#if defined(__x86_64__)
                if (norm_row::avx512_ok()) {
                    norm_row::acc_dot2_f64_512(grad_out_ptr + c_offset, in_ptr + c_offset,
                                               spatial_size, sd, dotp);
                } else
#endif
                {
                    for (int64_t s = 0; s < spatial_size; ++s) {
                        const double y = static_cast<double>(grad_out_ptr[c_offset + s]);
                        sd += y;
                        dotp += y * static_cast<double>(in_ptr[c_offset + s]);
                    }
                }
                const float s_dy_c = static_cast<float>(sd);
                const float s_dy_xhat_c = static_cast<float>((dotp - mean * sd) * inv_std);
                if (gb_row) gb_row[c] += s_dy_c;
                if (gw_row) gw_row[c] += s_dy_xhat_c;
                const float w = (w_ptr) ? w_ptr[c] : 1.0f;
                s_dy += w * s_dy_c;
                s_dy_xhat += w * s_dy_xhat_c;
            }

            // 3. grad_input for the whole group.
            if (grad_in_ptr) {
                const float term1 = inv_std / group_size;
                const float M = static_cast<float>(group_size);
                for (int64_t d = 0; d < D; ++d) {
                    const int64_t c = c_start + d;
                    const int64_t c_offset = group_offset + d * spatial_size;
                    const float w = (w_ptr) ? w_ptr[c] : 1.0f;
#if defined(__x86_64__)
                    if (norm_row::avx512_ok()) {
                        if (w_ptr) {
                            norm_row::gn_bwd_plane_f32_512<true>(
                                in_ptr + c_offset, grad_out_ptr + c_offset,
                                grad_in_ptr + c_offset, spatial_size, mean,
                                inv_std, w, term1, M, s_dy, s_dy_xhat);
                        } else {
                            norm_row::gn_bwd_plane_f32_512<false>(
                                in_ptr + c_offset, grad_out_ptr + c_offset,
                                grad_in_ptr + c_offset, spatial_size, mean,
                                inv_std, 1.0f, term1, M, s_dy, s_dy_xhat);
                        }
                    } else
#endif
                    {
                        for (int64_t s = 0; s < spatial_size; ++s) {
                            const float dy = grad_out_ptr[c_offset + s] * w;
                            const float x_hat = (in_ptr[c_offset + s] - mean) * inv_std;
                            grad_in_ptr[c_offset + s] =
                                term1 * (M * dy - s_dy - x_hat * s_dy_xhat);
                        }
                    }
                }
            }
        }
    });

    // Reduce per-thread partials into the output gradients.
    if (gw_ptr) {
        tensorplay::parallel::parallel_for(0, C, std::max<int64_t>(1, C / (th * 4)), [&](int64_t cb, int64_t ce) {
            for (int64_t c = cb; c < ce; ++c) {
                float acc = 0.0f;
                for (int64_t t = 0; t < th; ++t) acc += gw_buf[t * C + c];
                gw_ptr[c] = acc;
            }
        });
    }
    if (gb_ptr) {
        tensorplay::parallel::parallel_for(0, C, std::max<int64_t>(1, C / (th * 4)), [&](int64_t cb, int64_t ce) {
            for (int64_t c = cb; c < ce; ++c) {
                float acc = 0.0f;
                for (int64_t t = 0; t < th; ++t) acc += gb_buf[t * C + c];
                gb_ptr[c] = acc;
            }
        });
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> group_norm_backward_cpu(
        const Tensor& grad_output, const Tensor& input, int64_t num_groups,
        const std::optional<Tensor>& weight_opt,
        const std::optional<Tensor>& bias_opt,
        double eps) {
    return group_norm_backward_cpu_impl(
        grad_output, input, num_groups, weight_opt, bias_opt, eps, nullptr,
        nullptr, nullptr);
}

// Layer Normalization
// ============================================================================


// The kernels below walk an affine term as one dense run of normalized
// elements; a term laid out otherwise (an expanded or strided view) is made
// into one first.
static std::optional<Tensor> dense_affine(const std::optional<Tensor>& term) {
    if (!term.has_value() || !term->defined() || term->is_contiguous()) return term;
    return term->contiguous();
}

static Tensor layer_norm_cpu_impl(
        const Tensor& input, const std::vector<int64_t>& normalized_shape,
        const std::optional<Tensor>& weight_opt,
        const std::optional<Tensor>& bias_opt, double eps,
        Tensor* mean_out, Tensor* rstd_out) {
    
    // normalized_shape defines the last D dimensions to normalize over.
    // e.g. input (N, C, H, W), normalized_shape (C, H, W) -> normalize over C,H,W (per N)
    // e.g. input (N, L, D), normalized_shape (D) -> normalize over D (per N, L)
    
    int64_t norm_ndim = normalized_shape.size();
    int64_t input_ndim = input.dim();
    
    if (norm_ndim > input_ndim) TP_THROW(RuntimeError, "layer_norm: normalized_shape dim larger than input dim");
    
    // Check shapes match last dims
    int64_t outer_dims = input_ndim - norm_ndim;
    int64_t inner_size = 1;
    for (int64_t i = 0; i < norm_ndim; ++i) {
        if (input.size(outer_dims + i) != normalized_shape[i]) {
            TP_THROW(RuntimeError, "layer_norm: Input shape mismatch with normalized_shape");
        }
        inner_size *= normalized_shape[i];
    }
    // The affine terms are read one per normalized element: a shape of any
    // other extent would be read past its end.
    for (const auto* affine : {&weight_opt, &bias_opt}) {
        if (affine->has_value() && (*affine)->defined() &&
            static_cast<std::vector<int64_t>>((*affine)->shape()) != normalized_shape) {
            TP_THROW(RuntimeError, affine == &weight_opt
                ? "layer_norm: weight shape mismatch with normalized_shape"
                : "layer_norm: bias shape mismatch with normalized_shape");
        }
    }
    
    int64_t outer_size = inner_size == 0 ? 0 : input.numel() / inner_size;
    
    Tensor input_c = input.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(input.shape()), input.dtype(), input.device());
    
    // Rows are independent; partition by whole rows so each pass streams
    // once through the row's data.
    const int64_t row_grain = std::max<int64_t>(
        1, tensorplay::parallel::GRAIN_SIZE / std::max<int64_t>(inner_size, 1));
    
    if (input.dtype() == DType::Float32) {
        float* out_ptr = out.data_ptr<float>();
        const float* in_ptr = input_c.data_ptr<float>();
        const float* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<float>() : nullptr;
        const float* b_ptr = (bias_opt.has_value() && bias_opt->defined()) ? bias_opt->data_ptr<float>() : nullptr;
        
        tensorplay::parallel::parallel_for(0, outer_size, row_grain, [&](int64_t rb, int64_t re) {
            for (int64_t i = rb; i < re; ++i) {
                int64_t offset = i * inner_size;
                const float* row = in_ptr + offset;
                float* orow = out_ptr + offset;
#if defined(__x86_64__)
                if (norm_row::avx512_ok() && inner_size >= 16) {
                    float mean, rstd;
                    norm_row::stats_f32_512(row, inner_size, static_cast<float>(eps), &mean, &rstd);
                    if (mean_out) normalization_stats_write(*mean_out, i, mean);
                    if (rstd_out) normalization_stats_write(*rstd_out, i, rstd);
                    if (w_ptr && b_ptr) norm_row::apply_f32_512<true, true>(row, orow, inner_size, mean, rstd, w_ptr, b_ptr);
                    else if (w_ptr) norm_row::apply_f32_512<true, false>(row, orow, inner_size, mean, rstd, w_ptr, b_ptr);
                    else if (b_ptr) norm_row::apply_f32_512<false, true>(row, orow, inner_size, mean, rstd, w_ptr, b_ptr);
                    else norm_row::apply_f32_512<false, false>(row, orow, inner_size, mean, rstd, w_ptr, b_ptr);
                    continue;
                }
#endif
                float sum = 0.0f;
                float sq_sum = 0.0f;
                for (int64_t j = 0; j < inner_size; ++j) {
                    float val = row[j];
                    sum += val;
                    sq_sum += val * val;
                }
                float mean = sum / inner_size;
                float var = (sq_sum / inner_size) - (mean * mean);
                float inv_std = 1.0f / std::sqrt(var + (float)eps);
                if (mean_out) normalization_stats_write(*mean_out, i, mean);
                if (rstd_out) normalization_stats_write(*rstd_out, i, inv_std);
                for (int64_t j = 0; j < inner_size; ++j) {
                    float normalized = (row[j] - mean) * inv_std;
                    if (w_ptr) normalized *= w_ptr[j];
                    if (b_ptr) normalized += b_ptr[j];
                    orow[j] = normalized;
                }
            }
        });
    } else if (input.dtype() == DType::Float64) {
        double* out_ptr = out.data_ptr<double>();
        const double* in_ptr = input_c.data_ptr<double>();
        const double* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<double>() : nullptr;
        const double* b_ptr = (bias_opt.has_value() && bias_opt->defined()) ? bias_opt->data_ptr<double>() : nullptr;

        tensorplay::parallel::parallel_for(0, outer_size, row_grain, [&](int64_t rb, int64_t re) {
            for (int64_t i = rb; i < re; ++i) {
                int64_t offset = i * inner_size;
                const double* row = in_ptr + offset;
                double* orow = out_ptr + offset;
#if defined(__x86_64__)
                if (norm_row::avx512_ok() && inner_size >= 8) {
                    double mean, rstd;
                    norm_row::stats_f64_512(row, inner_size, eps, &mean, &rstd);
                    if (mean_out) normalization_stats_write(*mean_out, i, mean);
                    if (rstd_out) normalization_stats_write(*rstd_out, i, rstd);
                    if (w_ptr && b_ptr) norm_row::apply_f64_512<true, true>(row, orow, inner_size, mean, rstd, w_ptr, b_ptr);
                    else if (w_ptr) norm_row::apply_f64_512<true, false>(row, orow, inner_size, mean, rstd, w_ptr, b_ptr);
                    else if (b_ptr) norm_row::apply_f64_512<false, true>(row, orow, inner_size, mean, rstd, w_ptr, b_ptr);
                    else norm_row::apply_f64_512<false, false>(row, orow, inner_size, mean, rstd, w_ptr, b_ptr);
                    continue;
                }
#endif
                double sum = 0.0;
                double sq_sum = 0.0;
                for (int64_t j = 0; j < inner_size; ++j) {
                    double val = row[j];
                    sum += val;
                    sq_sum += val * val;
                }
                double mean = sum / inner_size;
                double var = (sq_sum / inner_size) - (mean * mean);
                double inv_std = 1.0 / std::sqrt(var + eps);
                if (mean_out) normalization_stats_write(*mean_out, i, mean);
                if (rstd_out) normalization_stats_write(*rstd_out, i, inv_std);
                for (int64_t j = 0; j < inner_size; ++j) {
                    double normalized = (row[j] - mean) * inv_std;
                    if (w_ptr) normalized *= w_ptr[j];
                    if (b_ptr) normalized += b_ptr[j];
                    orow[j] = normalized;
                }
            }
        });
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        if (input.dtype() == DType::Float16) {
            tensorplay::Half* out_ptr = out.data_ptr<tensorplay::Half>();
            const tensorplay::Half* in_ptr = input.data_ptr<tensorplay::Half>();
            const tensorplay::Half* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<tensorplay::Half>() : nullptr;
            const tensorplay::Half* b_ptr = (bias_opt.has_value() && bias_opt->defined()) ? bias_opt->data_ptr<tensorplay::Half>() : nullptr;

            tensorplay::parallel::parallel_for(0, outer_size, row_grain, [&](int64_t rb, int64_t re) {
            for (int64_t i = rb; i < re; ++i) {
                float sum = 0.0f;
                float sq_sum = 0.0f;
                int64_t offset = i * inner_size;

                for (int64_t j = 0; j < inner_size; ++j) {
                    float val = static_cast<float>(in_ptr[offset + j]);
                    sum += val;
                    sq_sum += val * val;
                }

                float mean = sum / inner_size;
                float var = (sq_sum / inner_size) - (mean * mean);
                float inv_std = 1.0f / std::sqrt(var + static_cast<float>(eps));
                if (mean_out) normalization_stats_write(*mean_out, i, mean);
                if (rstd_out) normalization_stats_write(*rstd_out, i, inv_std);

                for (int64_t j = 0; j < inner_size; ++j) {
                    float val = static_cast<float>(in_ptr[offset + j]);
                    float normalized = (val - mean) * inv_std;

                    if (w_ptr) normalized *= static_cast<float>(w_ptr[j]);
                    if (b_ptr) normalized += static_cast<float>(b_ptr[j]);

                    out_ptr[offset + j] = static_cast<tensorplay::Half>(normalized);
                }
            }
            });
        } else {
            tensorplay::BFloat16* out_ptr = out.data_ptr<tensorplay::BFloat16>();
            const tensorplay::BFloat16* in_ptr = input.data_ptr<tensorplay::BFloat16>();
            const tensorplay::BFloat16* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<tensorplay::BFloat16>() : nullptr;
            const tensorplay::BFloat16* b_ptr = (bias_opt.has_value() && bias_opt->defined()) ? bias_opt->data_ptr<tensorplay::BFloat16>() : nullptr;

            tensorplay::parallel::parallel_for(0, outer_size, row_grain, [&](int64_t rb, int64_t re) {
            for (int64_t i = rb; i < re; ++i) {
                float sum = 0.0f;
                float sq_sum = 0.0f;
                int64_t offset = i * inner_size;

                for (int64_t j = 0; j < inner_size; ++j) {
                    float val = static_cast<float>(in_ptr[offset + j]);
                    sum += val;
                    sq_sum += val * val;
                }

                float mean = sum / inner_size;
                float var = (sq_sum / inner_size) - (mean * mean);
                float inv_std = 1.0f / std::sqrt(var + static_cast<float>(eps));
                if (mean_out) normalization_stats_write(*mean_out, i, mean);
                if (rstd_out) normalization_stats_write(*rstd_out, i, inv_std);

                for (int64_t j = 0; j < inner_size; ++j) {
                    float val = static_cast<float>(in_ptr[offset + j]);
                    float normalized = (val - mean) * inv_std;

                    if (w_ptr) normalized *= static_cast<float>(w_ptr[j]);
                    if (b_ptr) normalized += static_cast<float>(b_ptr[j]);

                    out_ptr[offset + j] = static_cast<tensorplay::BFloat16>(normalized);
                }
            }
            });
        }
    } else {
        TP_THROW(NotImplementedError,
                 "layer_norm only supports Float32/Float64/Float16/BFloat16");
    }
    
    return out;
}

// The public spelling composes over the stats-producing native op so the
// autograd node saves the per-row moments for the backward pass.  The call
// goes through the generated ops wrapper, which records that node.
Tensor layer_norm_cpu(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                      const std::optional<Tensor>& weight_opt,
                      const std::optional<Tensor>& bias_opt, double eps) {
    return std::get<0>(tensorplay::tpx::ops::native_layer_norm(
        input, normalized_shape, weight_opt, bias_opt, eps));
}

// ============================================================================
// Group Normalization
// ============================================================================

static Tensor group_norm_cpu_impl(
        const Tensor& input, int64_t num_groups,
        const std::optional<Tensor>& weight_opt,
        const std::optional<Tensor>& bias_opt, double eps,
        Tensor* mean_out, Tensor* rstd_out) {
    
    // input: (N, C, *)
    if (input.dim() < 2) TP_THROW(RuntimeError, "group_norm requires at least 2 dims");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    
    if (num_groups <= 0)
        TP_THROW(RuntimeError, "group_norm: num_groups must be positive");
    if (C <= 0)
        TP_THROW(RuntimeError, "group_norm: num_channels must be positive");
    if (C % num_groups != 0) TP_THROW(RuntimeError, "group_norm: num_channels must be divisible by num_groups");
    
    int64_t channels_per_group = C / num_groups;
    int64_t spatial_size = 1;
    for (int64_t dim = 2; dim < input.dim(); ++dim) {
        spatial_size *= input.size(dim);
    }
    if (spatial_size <= 0)
        TP_THROW(RuntimeError, "group_norm: spatial dimensions must be positive");
    
    // Effectively we reshape (N, G, C/G, *) and normalize over (C/G, *)
    // inner_size = (C/G) * spatial_size
    int64_t inner_size = channels_per_group * spatial_size;
    
    if (N > 0 && gn_use_nhwc(input, N)) {
        // A channels-last activation is normalized where it lies and the
        // result keeps its order.
        const float* w_nhwc = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<float>() : nullptr;
        const float* b_nhwc = (bias_opt.has_value() && bias_opt->defined()) ? bias_opt->data_ptr<float>() : nullptr;
        return gn_group_norm_nhwc(input, num_groups, w_nhwc, b_nhwc,
                                  static_cast<float>(eps), mean_out, rstd_out);
    }

    Tensor input_c = input.contiguous();
    Tensor out = Tensor::empty_like(input_c);
    
    if (input.dtype() == DType::Float32) {
        float* out_ptr = out.data_ptr<float>();
        const float* in_ptr = input_c.data_ptr<float>();
        const float* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<float>() : nullptr;
        const float* b_ptr = (bias_opt.has_value() && bias_opt->defined()) ? bias_opt->data_ptr<float>() : nullptr;

        const int64_t group_rows = N * num_groups;
        const int64_t row_grain = std::max<int64_t>(
            1, tensorplay::parallel::GRAIN_SIZE / std::max<int64_t>(inner_size, 1));
        tensorplay::parallel::parallel_for(0, group_rows, row_grain,
            [&](int64_t rb, int64_t re) {
            for (int64_t row = rb; row < re; ++row) {
                const int64_t n = row / num_groups;
                const int64_t g = row % num_groups;
                const int64_t c_start = g * channels_per_group;
                const int64_t offset = n * C * spatial_size + c_start * spatial_size;
                const float* group = in_ptr + offset;
                float* group_out = out_ptr + offset;
                float mean;
                float inv_std;
#if defined(__x86_64__)
                if (norm_row::avx512_ok() && inner_size >= 16) {
                    norm_row::stats_f32_512(
                        group, inner_size, static_cast<float>(eps), &mean, &inv_std);
                    if (mean_out) normalization_stats_write(
                        *mean_out, row, mean);
                    if (rstd_out) normalization_stats_write(
                        *rstd_out, row, inv_std);
                    if (w_ptr && b_ptr) {
                        norm_row::apply_group_f32_512<true, true>(
                            group, group_out, channels_per_group, spatial_size,
                            mean, inv_std, w_ptr + c_start, b_ptr + c_start);
                    } else if (w_ptr) {
                        norm_row::apply_group_f32_512<true, false>(
                            group, group_out, channels_per_group, spatial_size,
                            mean, inv_std, w_ptr + c_start, nullptr);
                    } else if (b_ptr) {
                        norm_row::apply_group_f32_512<false, true>(
                            group, group_out, channels_per_group, spatial_size,
                            mean, inv_std, nullptr, b_ptr + c_start);
                    } else {
                        norm_row::apply_group_f32_512<false, false>(
                            group, group_out, channels_per_group, spatial_size,
                            mean, inv_std, nullptr, nullptr);
                    }
                    continue;
                }
#endif
                float sum = 0.0f;
                float sq_sum = 0.0f;
                for (int64_t i = 0; i < inner_size; ++i) {
                    const float val = group[i];
                    sum += val;
                    sq_sum += val * val;
                }
                mean = sum / inner_size;
                const float var = (sq_sum / inner_size) - mean * mean;
                inv_std = 1.0f / std::sqrt(var + static_cast<float>(eps));
                if (mean_out) normalization_stats_write(*mean_out, row, mean);
                if (rstd_out) normalization_stats_write(*rstd_out, row, inv_std);
                for (int64_t c = 0; c < channels_per_group; ++c) {
                    const float w = w_ptr ? w_ptr[c_start + c] : 1.0f;
                    const float b = b_ptr ? b_ptr[c_start + c] : 0.0f;
                    const float* ip = group + c * spatial_size;
                    float* op = group_out + c * spatial_size;
                    for (int64_t s = 0; s < spatial_size; ++s)
                        op[s] = (ip[s] - mean) * inv_std * w + b;
                }
            }
        });
    } else {
        TP_THROW(NotImplementedError, "group_norm only supports Float32");
    }
    
    return out;
}

Tensor group_norm_cpu(const Tensor& input, int64_t num_groups,
                      const std::optional<Tensor>& weight_opt,
                      const std::optional<Tensor>& bias_opt, double eps) {
    return group_norm_cpu_impl(input, num_groups, weight_opt, bias_opt, eps,
                               nullptr, nullptr);
}

// ============================================================================
// Backward for LayerNorm
// Rows are independent: statistics are recomputed per row, per-row reduction
// terms are computed in the same pass that accumulates the dgamma/dbeta
// partials, and grad_input is applied in a second streaming pass.  Parallel
// over rows with per-thread dgamma/dbeta buffers reduced at the end.
template <typename T>
static std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cpu_typed(
        const Tensor& grad_output, const Tensor& input,
        const std::vector<int64_t>& normalized_shape,
        const std::optional<Tensor>& weight_opt,
        const std::optional<Tensor>& bias_opt,
        double eps, const Tensor* mean_opt, const Tensor* rstd_opt,
        const std::vector<bool>* output_mask) {

    int64_t norm_ndim = normalized_shape.size();
    int64_t input_ndim = input.dim();
    int64_t inner_size = 1;
    for (auto s : normalized_shape) inner_size *= s;
    int64_t outer_size = inner_size == 0 ? 0 : input.numel() / inner_size;

    if (mean_opt || rstd_opt) {
        if (!mean_opt || !rstd_opt || !mean_opt->defined() ||
            !rstd_opt->defined() || mean_opt->numel() != outer_size ||
            rstd_opt->numel() != outer_size) {
            TP_THROW(RuntimeError,
                     "layer_norm_backward: saved statistics have an invalid shape");
        }
    }

    const bool has_weight = weight_opt.has_value() && weight_opt->defined();
    const bool has_bias = bias_opt.has_value() && bias_opt->defined();
    const bool want_input = output_mask == nullptr || output_mask->empty() ||
        (output_mask->size() > 0 && (*output_mask)[0]);
    const bool want_weight = has_weight &&
        (output_mask == nullptr || output_mask->size() <= 1 || (*output_mask)[1]);
    const bool want_bias = output_mask == nullptr
        ? has_bias
        : (output_mask->size() > 2 && (*output_mask)[2]);

    Tensor grad_input = want_input ? Tensor::empty_like(input) : Tensor();
    Tensor grad_weight;
    Tensor grad_bias;

    if (want_weight) grad_weight = Tensor::empty_like(*weight_opt);
    if (want_bias) {
        if (has_bias) grad_bias = Tensor::empty_like(*bias_opt);
        else grad_bias = Tensor::empty({inner_size}, input.dtype(), input.device());
    }

    T* grad_in_ptr = grad_input.defined() ? grad_input.data_ptr<T>() : nullptr;
    const T* grad_out_ptr = grad_output.data_ptr<T>();
    const T* in_ptr = input.data_ptr<T>();

    T* gw_ptr = (grad_weight.defined()) ? grad_weight.data_ptr<T>() : nullptr;
    T* gb_ptr = (grad_bias.defined()) ? grad_bias.data_ptr<T>() : nullptr;
    const T* w_ptr = (weight_opt.has_value() && weight_opt->defined()) ? weight_opt->data_ptr<T>() : nullptr;

    // Per-thread dgamma/dbeta partials (T storage for float/double).
    const int th = tensorplay::parallel::get_num_threads();
    std::vector<T> gw_buf, gb_buf;
    if (gw_ptr) gw_buf.assign(static_cast<size_t>(th) * inner_size, T(0));
    if (gb_ptr) gb_buf.assign(static_cast<size_t>(th) * inner_size, T(0));

    const T Teps = static_cast<T>(eps);
    const bool is_f32 = std::is_same_v<T, float>;
    const int64_t row_grain = std::max<int64_t>(
        1, tensorplay::parallel::GRAIN_SIZE / std::max<int64_t>(inner_size, 1));

    tensorplay::parallel::parallel_for(0, outer_size, row_grain, [&](int64_t rb, int64_t re) {
        T* gw_row = gw_ptr ? gw_buf.data() + static_cast<size_t>(tensorplay::parallel::get_thread_num()) * inner_size : nullptr;
        T* gb_row = gb_ptr ? gb_buf.data() + static_cast<size_t>(tensorplay::parallel::get_thread_num()) * inner_size : nullptr;

        for (int64_t i = rb; i < re; ++i) {
            int64_t offset = i * inner_size;

            // 1. Native backward consumes the forward moments.  The public
            // helper computes them here because its spelling has no moments.
            T mean, inv_std;
            if (mean_opt) {
                mean = static_cast<T>(normalization_stats_read(*mean_opt, i));
                inv_std = static_cast<T>(normalization_stats_read(*rstd_opt, i));
            } else {
#if defined(__x86_64__)
            if (is_f32 && norm_row::avx512_ok() && inner_size >= 16) {
                // The branch runs only for T == float; reinterpret the row
                // pointers so the generic instantiation still compiles.
                norm_row::stats_f32_512(reinterpret_cast<const float*>(in_ptr + offset), inner_size, static_cast<float>(eps),
                                        reinterpret_cast<float*>(&mean),
                                        reinterpret_cast<float*>(&inv_std));
            } else
#endif
            {
                T sum = T(0);
                T sq_sum = T(0);
                for (int64_t j = 0; j < inner_size; ++j) {
                    T val = in_ptr[offset + j];
                    sum += val;
                    sq_sum += val * val;
                }
                mean = sum / inner_size;
                T var = (sq_sum / inner_size) - (mean * mean);
                inv_std = T(1) / std::sqrt(var + Teps);
            }
            }
            (void)Teps;

            // 2. Row reductions: s_dy = sum(dy*w), s_dy_xhat = sum(dy*w*x_hat)
            //    while accumulating the dgamma/dbeta partials for this row.
            T s_dy = T(0);
            T s_dy_x_hat = T(0);
#if defined(__x86_64__)
            if (is_f32 && norm_row::avx512_ok() && inner_size >= 16) {
                if (w_ptr) {
                    norm_row::ln_bwd_stats_f32_512<true>(
                        reinterpret_cast<const float*>(grad_out_ptr + offset),
                        reinterpret_cast<const float*>(in_ptr + offset),
                        reinterpret_cast<const float*>(w_ptr), inner_size,
                        reinterpret_cast<const float&>(mean), reinterpret_cast<const float&>(inv_std),
                        reinterpret_cast<float*>(&s_dy), reinterpret_cast<float*>(&s_dy_x_hat));
                } else {
                    norm_row::ln_bwd_stats_f32_512<false>(
                        reinterpret_cast<const float*>(grad_out_ptr + offset),
                        reinterpret_cast<const float*>(in_ptr + offset), reinterpret_cast<const float*>(w_ptr), inner_size,
                        reinterpret_cast<const float&>(mean), reinterpret_cast<const float&>(inv_std),
                        reinterpret_cast<float*>(&s_dy), reinterpret_cast<float*>(&s_dy_x_hat));
                }
                if (gw_row || gb_row) {
                    for (int64_t j = 0; j < inner_size; ++j) {
                        const T dy = grad_out_ptr[offset + j];
                        const T x_hat = (in_ptr[offset + j] - mean) * inv_std;
                        if (gw_row) gw_row[j] += dy * x_hat;
                        if (gb_row) gb_row[j] += dy;
                    }
                }
            } else
#endif
            {
                for (int64_t j = 0; j < inner_size; ++j) {
                    T dy = grad_out_ptr[offset + j];
                    T x = in_ptr[offset + j];
                    T x_hat = (x - mean) * inv_std;

                    if (gw_row) gw_row[j] += dy * x_hat;
                    if (gb_row) gb_row[j] += dy;

                    T gamma = (w_ptr) ? w_ptr[j] : T(1);
                    T dy_eff = dy * gamma;

                    s_dy += dy_eff;
                    s_dy_x_hat += dy_eff * x_hat;
                }
            }

            // 3. grad_input for this row.
            if (grad_in_ptr) {
            const T term1 = inv_std / inner_size;
            const T M = static_cast<T>(inner_size);
#if defined(__x86_64__)
            if (is_f32 && norm_row::avx512_ok() && inner_size >= 16) {
                if (w_ptr) {
                    norm_row::ln_bwd_apply_f32_512<true>(
                        reinterpret_cast<const float*>(grad_out_ptr + offset),
                        reinterpret_cast<const float*>(in_ptr + offset),
                        reinterpret_cast<const float*>(w_ptr),
                        reinterpret_cast<float*>(grad_in_ptr + offset),
                        inner_size, reinterpret_cast<const float&>(mean),
                        reinterpret_cast<const float&>(inv_std),
                        reinterpret_cast<const float&>(term1),
                        reinterpret_cast<const float&>(M),
                        reinterpret_cast<const float&>(s_dy),
                        reinterpret_cast<const float&>(s_dy_x_hat));
                } else {
                    norm_row::ln_bwd_apply_f32_512<false>(
                        reinterpret_cast<const float*>(grad_out_ptr + offset),
                        reinterpret_cast<const float*>(in_ptr + offset), reinterpret_cast<const float*>(w_ptr),
                        reinterpret_cast<float*>(grad_in_ptr + offset),
                        inner_size, reinterpret_cast<const float&>(mean),
                        reinterpret_cast<const float&>(inv_std),
                        reinterpret_cast<const float&>(term1),
                        reinterpret_cast<const float&>(M),
                        reinterpret_cast<const float&>(s_dy),
                        reinterpret_cast<const float&>(s_dy_x_hat));
                }
                continue;
            }
#endif
            for (int64_t j = 0; j < inner_size; ++j) {
                T dy = grad_out_ptr[offset + j];
                T x = in_ptr[offset + j];
                T x_hat = (x - mean) * inv_std;
                T gamma = (w_ptr) ? w_ptr[j] : T(1);
                T dy_eff = dy * gamma;

                grad_in_ptr[offset + j] = term1 * (M * dy_eff - s_dy - x_hat * s_dy_x_hat);
            }
            }
        }
    });

    // Reduce per-thread partials.
    const int64_t grain = std::max<int64_t>(1, inner_size / (th * 4));
    if (gw_ptr) {
        tensorplay::parallel::parallel_for(0, inner_size, grain, [&](int64_t b, int64_t e) {
            for (int64_t j = b; j < e; ++j) {
                T acc = T(0);
                for (int64_t t = 0; t < th; ++t) acc += gw_buf[t * inner_size + j];
                gw_ptr[j] = acc;
            }
        });
    }
    if (gb_ptr) {
        tensorplay::parallel::parallel_for(0, inner_size, grain, [&](int64_t b, int64_t e) {
            for (int64_t j = b; j < e; ++j) {
                T acc = T(0);
                for (int64_t t = 0; t < th; ++t) acc += gb_buf[t * inner_size + j];
                gb_ptr[j] = acc;
            }
        });
    }

    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

static std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cpu_reduced(
        const Tensor& grad_output, const Tensor& input,
        const std::vector<int64_t>& normalized_shape,
        const std::optional<Tensor>& weight_opt,
        const std::optional<Tensor>& bias_opt,
        double eps, const Tensor* mean_opt, const Tensor* rstd_opt,
        const std::vector<bool>* output_mask) {
    // Reduced precision: promote to float32, reuse the typed kernel,
    const DType act_dt = input.dtype();
    Tensor in_f = input.to(DType::Float32);
    Tensor gy_f = grad_output.to(DType::Float32);
    const bool has_w = weight_opt.has_value() && weight_opt->defined();
    const bool has_b = bias_opt.has_value() && bias_opt->defined();
    std::optional<Tensor> w_f = has_w
        ? std::optional<Tensor>(weight_opt->to(DType::Float32)) : std::nullopt;
    std::optional<Tensor> b_f = has_b
        ? std::optional<Tensor>(bias_opt->to(DType::Float32)) : std::nullopt;
    auto g = layer_norm_backward_cpu_typed<float>(
        gy_f, in_f, normalized_shape, w_f, b_f, eps, mean_opt, rstd_opt,
        output_mask);
    return std::make_tuple(
        std::get<0>(g).to(act_dt),
        std::get<1>(g).defined()
            ? std::get<1>(g).to(has_w ? weight_opt->dtype() : act_dt)
            : std::get<1>(g),
        std::get<2>(g).defined()
            ? std::get<2>(g).to(has_b ? bias_opt->dtype() : act_dt)
            : std::get<2>(g));
}

std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_cpu(
                              const Tensor& grad_output, const Tensor& input,
                              const std::vector<int64_t>& normalized_shape,
                              const std::optional<Tensor>& weight_in,
                              const std::optional<Tensor>& bias_in, double eps) {
    const std::optional<Tensor> weight_opt = dense_affine(weight_in);
    const std::optional<Tensor> bias_opt = dense_affine(bias_in);
    switch (input.dtype()) {
        case DType::Float32:
            return layer_norm_backward_cpu_typed<float>(
                grad_output, input, normalized_shape, weight_opt, bias_opt, eps,
                nullptr, nullptr, nullptr);
        case DType::Float64:
            return layer_norm_backward_cpu_typed<double>(
                grad_output, input, normalized_shape, weight_opt, bias_opt, eps,
                nullptr, nullptr, nullptr);
        case DType::Float16:
        case DType::BFloat16:
            return layer_norm_backward_cpu_reduced(
                grad_output, input, normalized_shape, weight_opt, bias_opt, eps,
                nullptr, nullptr, nullptr);
        default:
            TP_THROW(NotImplementedError,
                     "layer_norm_backward only supports Float32/Float64/Float16/BFloat16");
    }
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_cpu(
        const Tensor& input, const std::vector<int64_t>& normalized_shape,
        const std::optional<Tensor>& weight_in,
        const std::optional<Tensor>& bias_in, double eps) {
    const std::optional<Tensor> weight_opt = dense_affine(weight_in);
    const std::optional<Tensor> bias_opt = dense_affine(bias_in);
    int64_t inner_size = 1;
    if (normalized_shape.empty())
        TP_THROW(RuntimeError, "native_layer_norm: normalized_shape must not be empty");
    for (int64_t size : normalized_shape) inner_size *= size;
    const int64_t outer_size = inner_size == 0 ? 0 : input.numel() / inner_size;
    const DType stats_dtype = normalization_stats_dtype(input.dtype());
    Tensor mean = Tensor::empty({outer_size}, stats_dtype, input.device());
    Tensor rstd = Tensor::empty({outer_size}, stats_dtype, input.device());
    Tensor out = layer_norm_cpu_impl(input, normalized_shape, weight_opt, bias_opt,
                                     eps, &mean, &rstd);
    return std::make_tuple(out, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward_cpu(
        const Tensor& grad_output, const Tensor& input,
        const std::vector<int64_t>& normalized_shape, const Tensor& mean,
    const Tensor& rstd, const std::optional<Tensor>& weight_in,
        const std::optional<Tensor>& bias_in,
        const std::vector<bool>& output_mask) {
    const std::optional<Tensor> weight_opt = dense_affine(weight_in);
    const std::optional<Tensor> bias_opt = dense_affine(bias_in);
    switch (input.dtype()) {
        case DType::Float32:
            return layer_norm_backward_cpu_typed<float>(
                grad_output, input, normalized_shape, weight_opt, bias_opt, 0.0,
                &mean, &rstd, &output_mask);
        case DType::Float64:
            return layer_norm_backward_cpu_typed<double>(
                grad_output, input, normalized_shape, weight_opt, bias_opt, 0.0,
                &mean, &rstd, &output_mask);
        case DType::Float16:
        case DType::BFloat16:
            return layer_norm_backward_cpu_reduced(
                grad_output, input, normalized_shape, weight_opt, bias_opt, 0.0,
                &mean, &rstd, &output_mask);
        default:
            TP_THROW(NotImplementedError,
                     "native_layer_norm_backward only supports Float32/Float64/Float16/BFloat16");
    }
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm_cpu(
        const Tensor& input, const std::optional<Tensor>& weight_opt,
        const std::optional<Tensor>& bias_opt, int64_t N, int64_t C,
        int64_t HxW, int64_t group, double eps) {
    TP_CHECK(N == input.size(0) && C == input.size(1),
             "native_group_norm: supplied dimensions do not match input");
    TP_CHECK(HxW > 0 && group > 0 && C > 0 && C % group == 0,
             "native_group_norm: invalid group dimensions");
    const DType stats_dtype = normalization_stats_dtype(input.dtype());
    Tensor mean = Tensor::empty({N, group}, stats_dtype, input.device());
    Tensor rstd = Tensor::empty({N, group}, stats_dtype, input.device());
    Tensor out = group_norm_cpu_impl(input, group, weight_opt, bias_opt, eps,
                                     &mean, &rstd);
    return std::make_tuple(out, mean, rstd);
}

std::tuple<Tensor, Tensor, Tensor> native_group_norm_backward_cpu(
        const Tensor& grad_out, const Tensor& input, const Tensor& mean,
        const Tensor& rstd, const std::optional<Tensor>& weight_opt,
        int64_t N, int64_t C, int64_t HxW, int64_t group,
        const std::vector<bool>& output_mask) {
    TP_CHECK(N == input.size(0) && C == input.size(1),
             "native_group_norm_backward: supplied dimensions do not match input");
    TP_CHECK(HxW > 0 && group > 0 && C > 0 && C % group == 0,
             "native_group_norm_backward: invalid group dimensions");
    return group_norm_backward_cpu_impl(
        grad_out, input, group, weight_opt, std::nullopt, 0.0, &mean, &rstd,
        &output_mask);
}

// rms_norm over the trailing normalized_shape dims: y = x * rsqrt(mean(x^2)+eps) * w.
// Native single kernel replaces a 6-op python composite that cost ~24 extra
// dispatches per Llama layer per token in the e2e profile.
Tensor rms_norm_cpu(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                    const std::optional<Tensor>& weight_opt, std::optional<double> eps_opt) {
    // An unset epsilon is the machine epsilon of the computation type.
    const double eps = eps_opt.has_value() ? *eps_opt
        : (input.dtype() == DType::Float64 ? std::numeric_limits<double>::epsilon()
                                           : static_cast<double>(std::numeric_limits<float>::epsilon()));
    const int64_t norm_ndim = (int64_t)normalized_shape.size();
    const int64_t input_ndim = input.dim();
    if (norm_ndim > input_ndim)
        TP_THROW(RuntimeError, "rms_norm: normalized_shape dim larger than input dim");
    int64_t inner_size = 1;
    for (int64_t i = 0; i < norm_ndim; ++i) {
        if (input.size(input_ndim - norm_ndim + i) != normalized_shape[i])
            TP_THROW(RuntimeError, "rms_norm: Input shape mismatch with normalized_shape");
        inner_size *= normalized_shape[i];
    }
    const int64_t outer_size = input.numel() / inner_size;
    const bool has_w = weight_opt.has_value() && weight_opt->defined();

    Tensor out = Tensor::empty_like(input);
    Tensor wc = has_w ? weight_opt->contiguous() : Tensor();

    const DType dt = input.dtype();
    if (dt == DType::Float32 || dt == DType::Float64) {
        const bool is_f64 = dt == DType::Float64;
        const auto* in = is_f64 ? static_cast<const void*>(input.data_ptr<double>())
                                : static_cast<const void*>(input.data_ptr<float>());
        auto* op = is_f64 ? static_cast<void*>(out.data_ptr<double>())
                          : static_cast<void*>(out.data_ptr<float>());
        const auto* wp = has_w ? (is_f64 ? static_cast<const void*>(wc.data_ptr<double>())
                                         : static_cast<const void*>(wc.data_ptr<float>()))
                               : nullptr;
        auto body = [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                const int64_t off = i * inner_size;
                long double acc = 0.0L;
                if (is_f64) {
                    const double* r = static_cast<const double*>(in) + off;
                    for (int64_t j = 0; j < inner_size; ++j) acc += static_cast<long double>(r[j]) * r[j];
                    const double inv = 1.0 / std::sqrt(double(acc) / inner_size + eps);
                    double* o = static_cast<double*>(op) + off;
                    const double* w = static_cast<const double*>(wp);
                    for (int64_t j = 0; j < inner_size; ++j) o[j] = r[j] * inv * (w ? w[j] : 1.0);
                } else {
                    const float* r = static_cast<const float*>(in) + off;
                    float acc32 = 0.0f;
                    for (int64_t j = 0; j < inner_size; ++j) acc32 += r[j] * r[j];
                    const float inv = 1.0f / std::sqrt(acc32 / inner_size + (float)eps);
                    float* o = static_cast<float*>(op) + off;
                    const float* w = static_cast<const float*>(wp);
                    for (int64_t j = 0; j < inner_size; ++j) o[j] = r[j] * inv * (w ? w[j] : 1.0f);
                }
            }
        };
        if (outer_size > 1) {
            const int64_t th = tensorplay::parallel::get_num_threads();
            const int64_t grain = std::max<int64_t>(1, outer_size / (th * 4));
            tensorplay::parallel::parallel_for(0, outer_size, grain, body);
        } else body(0, outer_size);
        return out;
    }
    if (dt == DType::Float16 || dt == DType::BFloat16) {
        // fp32 accumulate + scale, store back at input precision.
        const bool is_bf16 = dt == DType::BFloat16;
        const auto readv = [&](const void* p, int64_t i) -> float {
            return is_bf16 ? float(static_cast<const tensorplay::BFloat16*>(p)[i])
                           : float(static_cast<const tensorplay::Half*>(p)[i]);
        };
        const auto writev = [&](void* p, int64_t i, float v) {
            if (is_bf16) static_cast<tensorplay::BFloat16*>(p)[i] = tensorplay::BFloat16(v);
            else static_cast<tensorplay::Half*>(p)[i] = tensorplay::Half(v);
        };
        const void* ip = input.data_ptr();
        void* optr = out.data_ptr();
        const void* wp = has_w ? wc.data_ptr() : nullptr;
        const bool wb16 = has_w && wc.dtype() == DType::BFloat16;
        const bool whalf = has_w && wc.dtype() == DType::Float16;
        for (int64_t i = 0; i < outer_size; ++i) {
            const int64_t off = i * inner_size;
            float acc = 0.0f;
            for (int64_t j = 0; j < inner_size; ++j) { float v = readv(ip, off + j); acc += v * v; }
            const float inv = 1.0f / std::sqrt(acc / inner_size + (float)eps);
            for (int64_t j = 0; j < inner_size; ++j) {
                float v = readv(ip, off + j) * inv;
                if (has_w) {
                    float w = wb16 ? float(static_cast<const tensorplay::BFloat16*>(wp)[j])
                             : whalf ? float(static_cast<const tensorplay::Half*>(wp)[j])
                                     : static_cast<const float*>(wp)[j];
                    v *= w;
                }
                writev(optr, off + j, v);
            }
        }
        return out;
    }
    TP_THROW(NotImplementedError, "rms_norm_cpu: unsupported dtype");
}

// Registration
TENSORPLAY_LIBRARY_IMPL(CPU, NormalizationKernels) {
    m.impl("layer_norm", layer_norm_cpu);
    m.impl("group_norm", group_norm_cpu);
    m.impl("rms_norm", rms_norm_cpu);

    m.impl("layer_norm_backward", layer_norm_backward_cpu);
    m.impl("group_norm_backward", group_norm_backward_cpu);
    m.impl("native_layer_norm", native_layer_norm_cpu);
    m.impl("native_layer_norm_backward", native_layer_norm_backward_cpu);
    m.impl("native_group_norm", native_group_norm_cpu);
    m.impl("native_group_norm_backward", native_group_norm_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay

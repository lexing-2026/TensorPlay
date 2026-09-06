// Loss operators use device-side elementwise loops and reduction kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "CUDALoops.cuh"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <utility>

namespace tensorplay {
namespace cuda {

namespace ops = tensorplay::tpx::ops;

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace {

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

Tensor mean_from_elems(const Tensor& elems, int64_t n, DType dt, const Device& dev) {
    const DType out_dtype = dt == DType::Float64 ? DType::Float64 : DType::Float32;
    if (n == 0) return Tensor::full({}, Scalar(0), out_dtype, dev);
    Tensor result = elems.mean();
    return result.dtype() == out_dtype ? result : result.to(out_dtype);
}

std::pair<Tensor, Tensor> pair_f64_dev(const Tensor& a, const Tensor& b) {
    std::vector<int64_t> shape = broadcast_shapes(shape_of(a), shape_of(b));
    return {a.expand(shape).contiguous().to(DType::Float64),
            b.expand(shape).contiguous().to(DType::Float64)};
}

Tensor expand_f64_dev(const Tensor& a, const std::vector<int64_t>& shape) {
    return a.expand(shape).contiguous().to(DType::Float64);
}

void tp_validate_reduction(int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2) {
        TP_THROW(ValueError, "Invalid reduction mode");
    }
}

Tensor tp_loss_reduce(const Tensor& loss, int64_t reduction) {
    tp_validate_reduction(reduction);
    if (reduction == 0) return loss;
    if (reduction == 1) return loss.mean();
    return loss.sum();
}

Tensor tp_scale_grad(const Tensor& grad, int64_t reduction, int64_t numel) {
    tp_validate_reduction(reduction);
    if (reduction == 1) return grad / static_cast<double>(numel);
    return grad;
}

// ---------------------------------------------------------------------------
// elementwise kernels (one thread per pair element)
// ---------------------------------------------------------------------------

__host__ __device__ inline double dsp(double y) {
    return ::fmax(y, 0.0) + ::log1p(::exp(-::fabs(y)));
}

} // anonymous namespace

// ===========================================================================
// Public entry points
// ===========================================================================

Tensor l1_loss_cuda(const Tensor& input, const Tensor& target) {
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, input.device());
    int64_t n = elems.numel();
    if (n) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .build();
        gpu_kernel(iter, [] __host__ __device__(double a, double b) -> double {
            return ::fabs(a - b);
        });
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, input.dtype(), input.device());
}

Tensor kl_div_cuda(const Tensor& input, const Tensor& target) {
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, input.device());
    int64_t n = elems.numel();
    if (n) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .build();
        gpu_kernel(iter, [] __host__ __device__(double x, double t) -> double {
            return t > 0 ? t * (::log(t) - x) : 0.0;
        });
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, input.dtype(), input.device());
}

Tensor binary_cross_entropy_with_logits_cuda(const Tensor& self, const Tensor& target,
                                             const std::optional<Tensor>& weight_opt,
                                             const std::optional<Tensor>& pos_weight_opt) {
    Tensor weight = weight_opt.value_or(Tensor());
    Tensor pos_weight = pos_weight_opt.value_or(Tensor());
    auto pr = pair_f64_dev(self, target);
    bool has_w = weight.defined() && weight.numel() > 0;
    bool has_pw = pos_weight.defined() && pos_weight.numel() > 0;
    Tensor w = has_w ? expand_f64_dev(weight, shape_of(pr.first))
                     : pr.first;  // dummy pointer when absent
    Tensor pw = has_pw ? expand_f64_dev(pos_weight, shape_of(pr.first))
                       : pr.first;
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, self.device());
    int64_t n = elems.numel();
    if (n) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .add_const_input(w)
            .add_const_input(pw)
            .build();
        gpu_kernel(iter, [has_w, has_pw] __host__ __device__(
            double x, double t, double wv, double pwv) -> double {
            const double wi = has_w ? wv : 1.0;
            const double pi = has_pw ? pwv : 1.0;
            return wi * (pi * t * dsp(-x) + (1.0 - t) * dsp(x));
        });
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, self.dtype(), self.device());
}

Tensor hinge_embedding_loss_cuda(const Tensor& input, const Tensor& target, Scalar margin) {
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, input.device());
    int64_t n = elems.numel();
    if (n) {
        const double margin_value = margin.toDouble();
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .build();
        gpu_kernel(iter, [margin_value] __host__ __device__(double x, double t) -> double {
            return (t == 1.0) ? x : ::fmax(0.0, margin_value - x);
        });
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, input.dtype(), input.device());
}

Tensor margin_ranking_loss_cuda(const Tensor& input1, const Tensor& input2,
                                const Tensor& target, Scalar margin) {
    auto pr = pair_f64_dev(input1, input2);
    Tensor tg = expand_f64_dev(target, shape_of(pr.first));
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, input1.device());
    int64_t n = elems.numel();
    if (n) {
        const double margin_value = margin.toDouble();
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .add_const_input(tg)
            .build();
        gpu_kernel(iter, [margin_value] __host__ __device__(
            double a, double b, double target_value) -> double {
            return ::fmax(0.0, margin_value - target_value * (a - b));
        });
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, input1.dtype(), input1.device());
}

Tensor soft_margin_loss_cuda(const Tensor& input, const Tensor& target) {
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, input.device());
    int64_t n = elems.numel();
    if (n) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .build();
        gpu_kernel(iter, [] __host__ __device__(double x, double t) -> double {
            return dsp(-t * x);
        });
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, input.dtype(), input.device());
}

Tensor poisson_nll_loss_cuda(const Tensor& input, const Tensor& target, bool log_input,
                             bool full, double eps) {
    auto pr = pair_f64_dev(input, target);
    Tensor elems = Tensor::empty(shape_of(pr.first), DType::Float64, input.device());
    int64_t n = elems.numel();
    if (n) {
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(elems)
            .add_const_input(pr.first)
            .add_const_input(pr.second)
            .build();
        gpu_kernel(iter, [log_input, full, eps] __host__ __device__(double x, double z) -> double {
            double l2 = log_input ? (::exp(x) - z * x)
                                  : (x - z * ::log(::exp(x) + eps));
            if (full && z > 0) l2 += z * ::log(z) - ::lgamma(z + 1.0);
            return l2;
        });
        CUDA_CHECK(cudaGetLastError());
    }
    return mean_from_elems(elems, n, input.dtype(), input.device());
}

Tensor cosine_embedding_loss_cuda(const Tensor& x1, const Tensor& x2, const Tensor& target,
                                  Scalar margin) {
    const std::vector<int64_t> reduce_dims{1};
    Tensor prod_sum = (x1 * x2).sum(reduce_dims);
    Tensor mag_square1 = (x1 * x1).sum(reduce_dims) + Scalar(1e-12);
    Tensor mag_square2 = (x2 * x2).sum(reduce_dims) + Scalar(1e-12);
    Tensor cosine = prod_sum / (mag_square1 * mag_square2).sqrt();
    Tensor zeros = Tensor::zeros_like(cosine);
    Tensor negative = Tensor::where(
        (cosine - margin).lt(Scalar(0)), Scalar(0), cosine - margin);
    Tensor loss = Tensor::where(
        target.eq(Scalar(1)), Scalar(1) - cosine,
        Tensor::where(target.eq(Scalar(-1)), negative, zeros));
    return mean_from_elems(loss, loss.numel(), x1.dtype(), x1.device());
}

Tensor triplet_margin_loss_cuda(const Tensor& anchor, const Tensor& positive,
                                const Tensor& negative, Scalar margin, double p) {
    Tensor dist_pos = ops::pairwise_distance(anchor, positive, p, 1e-6, false);
    Tensor dist_neg = ops::pairwise_distance(anchor, negative, p, 1e-6, false);
    Tensor raw = dist_pos - dist_neg + margin;
    Tensor loss = Tensor::where(raw.lt(Scalar(0)), Scalar(0), raw);
    return mean_from_elems(loss, loss.numel(), anchor.dtype(), anchor.device());
}

Tensor multilabel_soft_margin_loss_cuda(const Tensor& input, const Tensor& target) {
    Tensor positive = target * (-ops::log_sigmoid(input));
    Tensor negative = (Scalar(1) - target) * (-ops::log_sigmoid(-input));
    Tensor row_loss = ops::mean(positive + negative, {1}, false);
    return mean_from_elems(row_loss, row_loss.numel(), input.dtype(), input.device());
}

Tensor tp_l1_loss_cuda(const Tensor& input, const Tensor& target,
                       int64_t reduction) {
    return tp_loss_reduce((input - target).abs(), reduction);
}

Tensor tp_l1_loss_backward_cuda(const Tensor& grad_output, const Tensor& input,
                                const Tensor& target, int64_t reduction) {
    Tensor grad = (input - target).sign() * grad_output;
    return tp_scale_grad(grad, reduction, input.numel());
}

Tensor tp_kl_div_cuda(const Tensor& input, const Tensor& target,
                      int64_t reduction, bool log_target) {
    Tensor loss;
    if (log_target) {
        loss = target.exp() * (target - input);
    } else {
        Tensor nonzero = target.ne(Scalar(0)).to(input.dtype());
        Tensor xlogy = Tensor::where(target.eq(Scalar(0)), target * 0,
                                     target * target.log());
        loss = (xlogy - target * input) * nonzero;
    }
    return tp_loss_reduce(loss, reduction);
}

Tensor tp_kl_div_backward_cuda(const Tensor& grad_output, const Tensor& input,
                               const Tensor& target, int64_t reduction,
                               bool log_target) {
    Tensor grad;
    if (log_target) {
        grad = -target.exp() * grad_output;
    } else {
        Tensor nonzero = target.ne(Scalar(0)).to(input.dtype());
        grad = (-target * nonzero) * grad_output;
    }
    return tp_scale_grad(grad, reduction, input.numel());
}

Tensor tp_margin_ranking_loss_cuda(const Tensor& input1, const Tensor& input2,
                                   const Tensor& target, double margin,
                                   int64_t reduction) {
    Tensor raw = -(input1 - input2) * target + margin;
    Tensor loss = Tensor::where(raw.lt(Scalar(0)), Scalar(0), raw);
    return tp_loss_reduce(loss, reduction);
}

std::tuple<Tensor, Tensor> tp_margin_ranking_loss_backward_cuda(
        const Tensor& grad_output, const Tensor& input1, const Tensor& input2,
        const Tensor& target, double margin, int64_t reduction) {
    tp_validate_reduction(reduction);
    Tensor raw = -(input1 - input2) * target + margin;
    Tensor active = raw.gt(Scalar(0)).to(input1.dtype());
    Tensor grad = -active * target * grad_output;
    grad = tp_scale_grad(grad, reduction, input1.numel());
    return std::make_tuple(grad, -grad);
}

Tensor tp_hinge_embedding_loss_cuda(const Tensor& input, const Tensor& target,
                                    double margin, int64_t reduction) {
    Tensor zeros = Tensor::zeros_like(input);
    Tensor margin_diff = margin - input;
    Tensor margin_part = Tensor::where(
        target.ne(Scalar(1)),
        Tensor::where(margin_diff.lt(Scalar(0)), Scalar(0), margin_diff),
        zeros);
    Tensor self_part = Tensor::where(target.ne(Scalar(-1)), input, zeros);
    return tp_loss_reduce(margin_part + self_part, reduction);
}

Tensor tp_hinge_embedding_loss_backward_cuda(const Tensor& grad_output,
                                             const Tensor& input,
                                             const Tensor& target, double margin,
                                             int64_t reduction) {
    Tensor ones = Tensor::ones_like(input);
    Tensor active = (margin - input).gt(Scalar(0)).to(input.dtype());
    Tensor grad = Tensor::where(
        target.eq(Scalar(1)), ones,
        Tensor::where(target.eq(Scalar(-1)), -active, ones - active));
    grad = grad * grad_output;
    return tp_scale_grad(grad, reduction, input.numel());
}

Tensor tp_soft_margin_loss_cuda(const Tensor& input, const Tensor& target,
                                int64_t reduction) {
    Tensor loss = ((input * target) * Scalar(-1)).exp().add(Scalar(1)).log();
    return tp_loss_reduce(loss, reduction);
}

Tensor tp_soft_margin_loss_backward_cuda(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& target,
                                         int64_t reduction) {
    Tensor z = ((input * target) * Scalar(-1)).exp();
    Tensor grad = -target * z.sigmoid() * grad_output;
    return tp_scale_grad(grad, reduction, input.numel());
}

Tensor tp_cosine_embedding_loss_cuda(const Tensor& input1, const Tensor& input2,
                                     const Tensor& target, double margin,
                                     int64_t reduction) {
    const std::vector<int64_t> reduce_dims{1};
    Tensor n1 = (input1 * input1).sum(reduce_dims) + 1e-12;
    Tensor n2 = (input2 * input2).sum(reduce_dims) + 1e-12;
    Tensor denom = (n1 * n2).sqrt();
    Tensor cosine = (input1 * input2).sum(reduce_dims) / denom;
    Tensor zeros = Tensor::zeros_like(cosine);
    Tensor negative = Tensor::where((cosine - margin).lt(Scalar(0)), Scalar(0),
                                    cosine - margin);
    Tensor loss = Tensor::where(target.eq(Scalar(1)), Scalar(1) - cosine,
                                Tensor::where(target.eq(Scalar(-1)), negative,
                                              zeros));
    return tp_loss_reduce(loss, reduction);
}

std::tuple<Tensor, Tensor> tp_cosine_embedding_loss_backward_cuda(
        const Tensor& grad_output, const Tensor& input1, const Tensor& input2,
        const Tensor& target, double margin, int64_t reduction) {
    tp_validate_reduction(reduction);
    const std::vector<int64_t> reduce_dims{1};
    Tensor n1 = (input1 * input1).sum(reduce_dims) + 1e-12;
    Tensor n2 = (input2 * input2).sum(reduce_dims) + 1e-12;
    Tensor denom = (n1 * n2).sqrt();
    Tensor cosine = (input1 * input2).sum(reduce_dims) / denom;

    Tensor ones = Tensor::ones({input1.size(0)}, input1.dtype(), input1.device());
    Tensor dl_dcos = Tensor::where(
        target.eq(Scalar(1)), -ones,
        Tensor::where(target.eq(Scalar(-1)),
                      (cosine - margin).gt(Scalar(0)).to(input1.dtype()),
                      (Scalar(1) - cosine - margin)
                          .gt(Scalar(0))
                          .to(input1.dtype()) * Scalar(-1)));
    if (reduction == 1) {
        dl_dcos = dl_dcos / static_cast<double>(input1.size(0));
    }

    Tensor cosine_col = cosine.unsqueeze(1);
    Tensor denom_col = denom.unsqueeze(1);
    Tensor grad1 = (input2 / denom_col) -
                   cosine_col * (input1 / n1.unsqueeze(1));
    Tensor grad2 = (input1 / denom_col) -
                   cosine_col * (input2 / n2.unsqueeze(1));
    Tensor multiplier = (dl_dcos * grad_output).unsqueeze(1);
    return std::make_tuple(grad1 * multiplier, grad2 * multiplier);
}

Tensor tp_poisson_nll_loss_cuda(const Tensor& input, const Tensor& target,
                                bool log_input, bool full, double eps,
                                int64_t reduction) {
    Tensor loss = log_input ? input.exp() - target * input
                            : input - target * (input + eps).log();
    if (full) {
        Tensor active = target.gt(Scalar(1)).to(input.dtype());
        Tensor safe_target = Tensor::where(active, target,
                                           Tensor::ones_like(target));
        Tensor stirling = target * safe_target.log() - target +
                          (safe_target * (2.0 * M_PI)).log() * 0.5;
        loss = loss + Tensor::where(active, stirling,
                                    Tensor::zeros_like(stirling));
    }
    return tp_loss_reduce(loss, reduction);
}

Tensor tp_poisson_nll_loss_backward_cuda(const Tensor& grad_output,
                                         const Tensor& input,
                                         const Tensor& target, bool log_input,
                                         bool full, double eps,
                                         int64_t reduction) {
    static_cast<void>(full);
    Tensor grad = log_input ? input.exp() - target
                            : Scalar(1) - target / (input + eps);
    grad = grad * grad_output;
    return tp_scale_grad(grad, reduction, input.numel());
}

TENSORPLAY_LIBRARY_IMPL(CUDA, LossKernels) {
    m.impl("l1_loss", l1_loss_cuda);
    m.impl("kl_div", kl_div_cuda);
    m.impl("binary_cross_entropy_with_logits", binary_cross_entropy_with_logits_cuda);
    m.impl("cosine_embedding_loss", cosine_embedding_loss_cuda);
    m.impl("hinge_embedding_loss", hinge_embedding_loss_cuda);
    m.impl("margin_ranking_loss", margin_ranking_loss_cuda);
    m.impl("soft_margin_loss", soft_margin_loss_cuda);
    m.impl("triplet_margin_loss", triplet_margin_loss_cuda);
    m.impl("poisson_nll_loss", poisson_nll_loss_cuda);
    m.impl("multilabel_soft_margin_loss", multilabel_soft_margin_loss_cuda);
    m.impl("tp_l1_loss", tp_l1_loss_cuda);
    m.impl("tp_l1_loss_backward", tp_l1_loss_backward_cuda);
    m.impl("tp_kl_div", tp_kl_div_cuda);
    m.impl("tp_kl_div_backward", tp_kl_div_backward_cuda);
    m.impl("tp_margin_ranking_loss", tp_margin_ranking_loss_cuda);
    m.impl("tp_margin_ranking_loss_backward", tp_margin_ranking_loss_backward_cuda);
    m.impl("tp_hinge_embedding_loss", tp_hinge_embedding_loss_cuda);
    m.impl("tp_hinge_embedding_loss_backward", tp_hinge_embedding_loss_backward_cuda);
    m.impl("tp_soft_margin_loss", tp_soft_margin_loss_cuda);
    m.impl("tp_soft_margin_loss_backward", tp_soft_margin_loss_backward_cuda);
    m.impl("tp_cosine_embedding_loss", tp_cosine_embedding_loss_cuda);
    m.impl("tp_cosine_embedding_loss_backward", tp_cosine_embedding_loss_backward_cuda);
    m.impl("tp_poisson_nll_loss", tp_poisson_nll_loss_cuda);
    m.impl("tp_poisson_nll_loss_backward", tp_poisson_nll_loss_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay

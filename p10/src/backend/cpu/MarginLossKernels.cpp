// multi_margin_loss / multilabel_margin_loss CPU kernels.
//
// LossMultiLabelMargin.cpp (shape checks from LossMulti.h):
//
// * multi_margin_loss: 0-D/1-D/2-D input with class-index target; per-row
//   hinge sum_{d != y} max(0, margin - x_y + x_d)^p * w[y] / C with
//   reduction 0=none (per-row output), 1=mean, 2=sum.  1-D (and 0-D) input
//   always produces a scalar; with reduction=none and a >0-dim target the
//   output is (nframe).
// * multilabel_margin_loss_forward: rows of target list class indices
//   terminated by -1; emits the loss plus the is_target mask (input dtype,
//   target shape) consumed by the backward.  multilabel_margin_loss is the
//
// 1/(nframe*dim) for mean else 1/dim and then multiplies by grad_output
// (scalar or per-row); multilabel accumulates +-g for every active
// (target, non-target) pair with positive margin.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "LinearAlgebraNames.h"

#include <vector>
#include <string>
#include <cmath>
#include <tuple>
#include <optional>

// Dispatcher-level entry point for the composite (defined in
// TPXOpsGenerated.cpp; declared locally -- same pattern as
// PoolingKernels.cpp).
namespace tensorplay {
namespace tpx {
namespace ops {
TENSORPLAY_API std::tuple<Tensor, Tensor> multilabel_margin_loss_forward(const Tensor& self, const Tensor& target, int64_t reduction);
} // namespace ops
} // namespace tpx
} // namespace tensorplay

namespace tensorplay {
namespace cpu {

namespace {

constexpr int64_t kReductionNone = 0;
constexpr int64_t kReductionMean = 1;
constexpr int64_t kReductionSum = 2;

void check_reduction(int64_t reduction, const char* name) {
    if (reduction != kReductionNone && reduction != kReductionMean && reduction != kReductionSum)
        TP_THROW(ValueError, std::string(name) + ": invalid reduction, expected 0 (none), 1 (mean) or 2 (sum) but got " + std::to_string(reduction));
}

Tensor as_long_contiguous(const Tensor& target) {
    Tensor t = target.contiguous();
    if (t.dtype() != DType::Int64) t = t.to(DType::Int64);
    return t;
}

void multi_margin_shape_check(int64_t& nframe, int64_t& dim, const Tensor& input,
                              const Tensor& target, const std::optional<Tensor>& weight) {
    const int64_t ndims = input.dim();
    if (!((ndims == 2 && input.size(1) != 0) || (ndims == 1 && input.size(0) != 0) || ndims == 0))
        TP_THROW(RuntimeError, std::string("multi_margin_loss: Expected non-empty vector or matrix with optional 0-dim batch size, but got: ") + input.shape().toString());
    if (ndims <= 1) {
        nframe = 1;
        dim = ndims == 0 ? 1 : input.size(0);
    } else {
        nframe = input.size(0);
        dim = input.size(1);
    }
    if (!(target.dim() <= 1 && target.numel() == nframe))
        TP_THROW(RuntimeError, std::string("multi_margin_loss: target tensor should be 1-D with size equal to the number of input samples (batch size). Expected target size [") +
                     std::to_string(nframe) + "], but got " + target.shape().toString() +
                     ". Input has shape " + input.shape().toString() + ".");
    if (weight.has_value() && weight->defined()) {
        if (!(weight->dim() <= 1 && weight->numel() == dim))
            TP_THROW(RuntimeError, std::string("multi_margin_loss: inconsistent weight size, expected ") +
                         std::to_string(dim) + " but got " + weight->shape().toString());
    }
}

void multilabel_shape_check(int64_t& nframe, int64_t& dim, const Tensor& input,
                            const Tensor& target) {
    const int64_t ndims = input.dim();
    if (!((ndims == 2 && input.size(1) != 0) || (ndims == 1 && input.size(0) != 0) || ndims == 0))
        TP_THROW(RuntimeError, std::string("multilabel_margin_loss: Expected non-empty vector or matrix with optional 0-dim batch size, but got: ") + input.shape().toString());
    if (ndims <= 1) {
        nframe = 1;
        dim = ndims == 0 ? 1 : input.size(0);
        if (!(target.dim() <= 1 && target.numel() == dim))
            TP_THROW(RuntimeError, std::string("multilabel_margin_loss: inconsistent target size: ") +
                         target.shape().toString() + " for input of size: " + input.shape().toString());
    } else {
        nframe = input.size(0);
        dim = input.size(1);
        if (!(target.dim() == 2 && target.size(0) == nframe && target.size(1) == dim))
            TP_THROW(RuntimeError, std::string("multilabel_margin_loss: inconsistent target size: ") +
                         target.shape().toString() + " for input of size: " + input.shape().toString());
    }
}

int64_t target_index_checked(const int64_t* target_data, int64_t index, int64_t dim) {
    const int64_t idx = target_data[index];
    if (idx < 0 || idx >= dim)
        TP_THROW(RuntimeError, "multi_margin_loss: target out of range");
    return idx;
}

void check_target_range(const Tensor& target_c, int64_t dim, const char* name) {
    const int64_t n = target_c.numel();
    const int64_t* data = target_c.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
        if (data[i] < -1 || data[i] >= dim)
            TP_THROW(RuntimeError, std::string(name) + ": target is out of range");
    }
}

template <typename scalar_t>
Tensor scalar_tensor(double v, DType dt, const Device& dev) {
    return Tensor::full({}, Scalar(static_cast<scalar_t>(v)), dt, dev);
}

template <typename scalar_t>
Tensor multi_margin_loss_impl(const Tensor& input, const Tensor& target, int64_t p,
                              double margin, const std::optional<Tensor>& weight,
                              int64_t reduction) {
    int64_t nframe = 0, dim = 0;
    multi_margin_shape_check(nframe, dim, input, target, weight);

    const bool per_row = reduction == kReductionNone && target.dim() > 0;
    Tensor output = per_row ? Tensor::empty({nframe}, input.dtype(), input.device())
                            : Tensor::empty({}, input.dtype(), input.device());
    if (input.numel() == 0) {
        if (per_row) return output;
        return scalar_tensor<scalar_t>(0.0, input.dtype(), input.device());
    }

    const Tensor input_c = input.contiguous();
    const Tensor target_c = as_long_contiguous(target);
    Tensor weight_c;
    if (weight.has_value() && weight->defined()) weight_c = weight->contiguous();

    const scalar_t* input_data = input_c.data_ptr<scalar_t>();
    const int64_t* target_data = target_c.data_ptr<int64_t>();
    const scalar_t* weight_data = weight_c.defined() ? weight_c.data_ptr<scalar_t>() : nullptr;
    const scalar_t mg = static_cast<scalar_t>(margin);

    double total = 0;
    for (int64_t t = 0; t < nframe; ++t) {
        const int64_t idx = target_index_checked(target_data, t, dim);
        const scalar_t input_target = input_data[t * dim + idx];
        double sum = 0;
        for (int64_t d = 0; d < dim; ++d) {
            if (d == idx) continue;
            const double z = static_cast<double>(mg - input_target + input_data[t * dim + d]);
            if (z > 0) {
                double h = (p == 1) ? z : z * z;
                if (weight_data != nullptr) h *= static_cast<double>(weight_data[idx]);
                sum += h;
            }
        }
        sum /= static_cast<double>(dim);
        if (per_row) output.data_ptr<scalar_t>()[t] = static_cast<scalar_t>(sum);
        total += sum;
    }
    if (per_row) return output;
    if (reduction == kReductionMean) total /= static_cast<double>(nframe);
    return scalar_tensor<scalar_t>(total, input.dtype(), input.device());
}

template <typename scalar_t>
Tensor multi_margin_loss_backward_impl(const Tensor& grad_output, const Tensor& input,
                                       const Tensor& target, int64_t p, double margin,
                                       const std::optional<Tensor>& weight, int64_t reduction) {
    int64_t nframe = 0, dim = 0;
    multi_margin_shape_check(nframe, dim, input, target, weight);

    Tensor grad_input = Tensor::empty(input.shape(), input.dtype(), input.device());
    if (input.numel() == 0) return grad_input;

    const Tensor input_c = input.contiguous();
    const Tensor target_c = as_long_contiguous(target);
    const Tensor grad_output_c = grad_output.contiguous();
    Tensor weight_c;
    if (weight.has_value() && weight->defined()) weight_c = weight->contiguous();

    const scalar_t* input_data = input_c.data_ptr<scalar_t>();
    const int64_t* target_data = target_c.data_ptr<int64_t>();
    const scalar_t* weight_data = weight_c.defined() ? weight_c.data_ptr<scalar_t>() : nullptr;
    scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();
    const scalar_t mg = static_cast<scalar_t>(margin);

    const double g = reduction == kReductionMean ? 1.0 / (static_cast<double>(nframe) * dim)
                                                 : 1.0 / static_cast<double>(dim);

    for (int64_t t = 0; t < nframe; ++t) {
        const int64_t idx = target_index_checked(target_data, t, dim);
        const scalar_t input_target = input_data[t * dim + idx];
        double grad_input_target = 0;
        for (int64_t d = 0; d < dim; ++d) {
            if (d == idx) continue;
            const double z = static_cast<double>(mg - input_target + input_data[t * dim + d]);
            if (z > 0) {
                double h = (p == 1) ? g : 2 * g * z;
                if (weight_data != nullptr) h *= static_cast<double>(weight_data[idx]);
                grad_input_target -= h;
                grad_input_data[t * dim + d] = static_cast<scalar_t>(h);
            } else {
                grad_input_data[t * dim + d] = static_cast<scalar_t>(0);
            }
        }
        grad_input_data[t * dim + idx] = static_cast<scalar_t>(grad_input_target);
    }

    if (reduction != kReductionNone || grad_output.dim() == 0) {
        const double d0 = static_cast<double>(grad_output_c.data_ptr<scalar_t>()[0]);
        for (int64_t i = 0; i < nframe * dim; ++i)
            grad_input_data[i] = static_cast<scalar_t>(static_cast<double>(grad_input_data[i]) * d0);
    } else {
        const scalar_t* grad_output_data = grad_output_c.data_ptr<scalar_t>();
        for (int64_t t = 0; t < nframe; ++t)
            for (int64_t d = 0; d < dim; ++d)
                grad_input_data[t * dim + d] = static_cast<scalar_t>(
                    static_cast<double>(grad_input_data[t * dim + d]) *
                    static_cast<double>(grad_output_data[t]));
    }
    return grad_input;
}

template <typename scalar_t>
std::tuple<Tensor, Tensor> multilabel_margin_loss_forward_impl(const Tensor& input,
                                                               const Tensor& target,
                                                               int64_t reduction) {
    int64_t nframe = 0, dim = 0;
    multilabel_shape_check(nframe, dim, input, target);

    const bool scalar_out = reduction != kReductionNone || target.dim() <= 1;
    Tensor output = scalar_out ? Tensor::empty({}, input.dtype(), input.device())
                               : Tensor::empty({nframe}, input.dtype(), input.device());
    Tensor is_target = Tensor::zeros(target.shape(), input.dtype(), input.device());
    if (input.numel() == 0) {
        if (scalar_out) return std::make_tuple(scalar_tensor<scalar_t>(0.0, input.dtype(), input.device()), is_target);
        return std::make_tuple(output, is_target);
    }

    const Tensor input_c = input.contiguous();
    const Tensor target_c = as_long_contiguous(target);
    check_target_range(target_c, dim, "multilabel_margin_loss_forward");

    const scalar_t* input_data = input_c.data_ptr<scalar_t>();
    const int64_t* target_data = target_c.data_ptr<int64_t>();
    scalar_t* is_target_data = is_target.data_ptr<scalar_t>();
    scalar_t* output_data = output.data_ptr<scalar_t>();

    double total = 0;
    for (int64_t t = 0; t < nframe; ++t) {
        const scalar_t* row_in = input_data + t * dim;
        const int64_t* row_tg = target_data + t * dim;
        scalar_t* row_is = is_target_data + t * dim;
        for (int64_t dt = 0; dt < dim; ++dt) {
            const int64_t idx = row_tg[dt];
            if (idx < 0) break;
            row_is[idx] = static_cast<scalar_t>(1);
        }
        double sum = 0;
        for (int64_t dt = 0; dt < dim; ++dt) {
            const int64_t idx = row_tg[dt];
            if (idx < 0) break;
            const scalar_t input_target = row_in[idx];
            for (int64_t d = 0; d < dim; ++d) {
                if (static_cast<double>(row_is[d]) == 0.0) {
                    const double z = 1.0 - static_cast<double>(input_target) + static_cast<double>(row_in[d]);
                    if (z > 0) sum += z;
                }
            }
        }
        sum /= static_cast<double>(dim);
        if (!scalar_out) output_data[t] = static_cast<scalar_t>(sum);
        total += sum;
    }
    if (scalar_out) {
        if (reduction == kReductionMean) total /= static_cast<double>(nframe);
        return std::make_tuple(scalar_tensor<scalar_t>(total, input.dtype(), input.device()), is_target);
    }
    return std::make_tuple(output, is_target);
}

template <typename scalar_t>
Tensor multilabel_margin_loss_backward_impl(const Tensor& grad_output, const Tensor& input,
                                            const Tensor& target, int64_t reduction,
                                            const Tensor& is_target) {
    int64_t nframe = 0, dim = 0;
    multilabel_shape_check(nframe, dim, input, target);
    if (is_target.shape() != target.shape())
        TP_THROW(RuntimeError, "multilabel_margin_loss_backward: inconsistent is_target size");

    Tensor grad_input = Tensor::zeros(input.shape(), input.dtype(), input.device());
    if (input.numel() == 0) return grad_input;

    const Tensor input_c = input.contiguous();
    const Tensor target_c = as_long_contiguous(target);
    const Tensor is_target_c = is_target.contiguous();
    const Tensor grad_output_c = grad_output.contiguous();
    check_target_range(target_c, dim, "multilabel_margin_loss_backward");

    const scalar_t* is_target_data = is_target_c.data_ptr<scalar_t>();
    for (int64_t i = 0; i < is_target_c.numel(); ++i) {
        const double v = static_cast<double>(is_target_data[i]);
        if (v < 0.0 || v > 1.0)
            TP_THROW(RuntimeError, "multilabel_margin_loss_backward: is_target is out of range");
    }

    const scalar_t* input_data = input_c.data_ptr<scalar_t>();
    const int64_t* target_data = target_c.data_ptr<int64_t>();
    scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();

    const double g = reduction == kReductionMean ? 1.0 / (static_cast<double>(nframe) * dim)
                                                 : 1.0 / static_cast<double>(dim);

    for (int64_t t = 0; t < nframe; ++t) {
        const scalar_t* row_in = input_data + t * dim;
        const int64_t* row_tg = target_data + t * dim;
        const scalar_t* row_is = is_target_data + t * dim;
        scalar_t* row_gi = grad_input_data + t * dim;
        for (int64_t dt = 0; dt < dim; ++dt) {
            const int64_t idx = row_tg[dt];
            if (idx < 0) break;
            const scalar_t input_target = row_in[idx];
            for (int64_t d = 0; d < dim; ++d) {
                if (static_cast<double>(row_is[d]) == 0.0) {
                    const double z = 1.0 - static_cast<double>(input_target) + static_cast<double>(row_in[d]);
                    if (z > 0) {
                        row_gi[idx] = static_cast<scalar_t>(static_cast<double>(row_gi[idx]) - g);
                        row_gi[d] = static_cast<scalar_t>(static_cast<double>(row_gi[d]) + g);
                    }
                }
            }
        }
    }

    if (reduction != kReductionNone || grad_output.dim() == 0) {
        const double d0 = static_cast<double>(grad_output_c.data_ptr<scalar_t>()[0]);
        for (int64_t i = 0; i < nframe * dim; ++i)
            grad_input_data[i] = static_cast<scalar_t>(static_cast<double>(grad_input_data[i]) * d0);
    } else {
        const scalar_t* grad_output_data = grad_output_c.data_ptr<scalar_t>();
        for (int64_t t = 0; t < nframe; ++t)
            for (int64_t d = 0; d < dim; ++d)
                grad_input_data[t * dim + d] = static_cast<scalar_t>(
                    static_cast<double>(grad_input_data[t * dim + d]) *
                    static_cast<double>(grad_output_data[t]));
    }
    return grad_input;
}

#define DISPATCH_FLOAT(dt, fn, ...)                                    \
    do {                                                               \
        if ((dt) == DType::Float32) return fn<float>(__VA_ARGS__);     \
        if ((dt) == DType::Float64) return fn<double>(__VA_ARGS__);    \
        TP_THROW(NotImplementedError, std::string(#fn) + ": only supports Float32/Float64, got " + pretty_dtype_name(dt)); \
    } while (0)

} // namespace

Tensor multi_margin_loss_cpu(const Tensor& input, const Tensor& target, Scalar p,
                             Scalar margin, const std::optional<Tensor>& weight,
                             int64_t reduction) {
    check_reduction(reduction, "multi_margin_loss");
    const int64_t pint = p.to<int64_t>();
    if (pint != 1 && pint != 2)
        TP_THROW(RuntimeError, "multi_margin_loss: only p == 1 and p == 2 supported");
    DISPATCH_FLOAT(input.dtype(), multi_margin_loss_impl, input, target, pint,
                   margin.toDouble(), weight, reduction);
}

Tensor multi_margin_loss_cpu_backward(const Tensor& grad_output, const Tensor& input,
                                      const Tensor& target, Scalar p, Scalar margin,
                                      const std::optional<Tensor>& weight, int64_t reduction) {
    check_reduction(reduction, "multi_margin_loss_backward");
    const int64_t pint = p.to<int64_t>();
    if (pint != 1 && pint != 2)
        TP_THROW(RuntimeError, "multi_margin_loss_backward: only p == 1 and p == 2 supported");
    DISPATCH_FLOAT(input.dtype(), multi_margin_loss_backward_impl, grad_output, input,
                   target, pint, margin.toDouble(), weight, reduction);
}

std::tuple<Tensor, Tensor> multilabel_margin_loss_forward_cpu(const Tensor& input,
                                                              const Tensor& target,
                                                              int64_t reduction) {
    check_reduction(reduction, "multilabel_margin_loss_forward");
    if (input.dtype() == DType::Float32)
        return multilabel_margin_loss_forward_impl<float>(input, target, reduction);
    if (input.dtype() == DType::Float64)
        return multilabel_margin_loss_forward_impl<double>(input, target, reduction);
    TP_THROW(NotImplementedError, std::string("multilabel_margin_loss_forward: only supports Float32/Float64, got ") + pretty_dtype_name(input.dtype()));
}

Tensor multilabel_margin_loss_backward_cpu(const Tensor& grad_output, const Tensor& input,
                                           const Tensor& target, int64_t reduction,
                                           const Tensor& is_target) {
    check_reduction(reduction, "multilabel_margin_loss_backward");
    DISPATCH_FLOAT(input.dtype(), multilabel_margin_loss_backward_impl, grad_output,
                   input, target, reduction, is_target);
}

Tensor multilabel_margin_loss_composite(const Tensor& input, const Tensor& target,
                                        int64_t reduction) {
    return std::get<0>(tpx::ops::multilabel_margin_loss_forward(input, target, reduction));
}

TENSORPLAY_LIBRARY_IMPL(CPU, MarginLossKernels) {
    m.impl("multi_margin_loss", multi_margin_loss_cpu);
    m.impl("multi_margin_loss_backward", multi_margin_loss_cpu_backward);
    m.impl("multilabel_margin_loss_forward", multilabel_margin_loss_forward_cpu);
    m.impl("multilabel_margin_loss_backward", multilabel_margin_loss_backward_cpu);
}

TENSORPLAY_LIBRARY_IMPL(Composite, MarginLossComposite) {
    m.impl("multilabel_margin_loss", multilabel_margin_loss_composite);
}

} // namespace cpu
} // namespace tensorplay

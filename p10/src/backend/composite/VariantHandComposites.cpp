// Hand-written overload wiring: entries whose argument shapes differ from
// any registered sibling in ways the mechanical matcher rejects, but whose
// semantics are a plain forward to an already-registered kernel.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "CompositeCommon.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

// A scalar taking part in tensor arithmetic materializes with its natural
// dtype promoted against the reference tensor, matching wrapped-number
// promotion for the scalar overloads implemented here.
Tensor scalar_like(const Scalar& s, const Tensor& ref) {
    const DType prom = promoteTypes(scalar_natural_dtype(s), ref.dtype());
    return ops::scalar_tensor(s, prom, ref.device());
}

} // namespace

// ---- norm family -----------------------------------------------------------
Tensor norm_scalar_native(const Tensor& self, const Scalar& p) {
    if (p.isComplex()) {
        TP_THROW(NotImplementedError, "norm with a complex exponent is not supported");
    }
    return ops::norm(self, p.toDouble());
}

Tensor norm_scalar_opt_dim_native(const Tensor& self, const std::optional<Scalar>& p,
                                  const std::vector<int64_t>& dim, bool keepdim) {
    return ops::norm(self, dim, p.has_value() ? p->toDouble() : 2.0, keepdim);
}

Tensor norm_scalar_opt_dtype_native(const Tensor& self, const std::optional<Scalar>& p,
                                    DType dtype) {
    if (p.has_value()) {
        Tensor r = ops::norm(self, p->toDouble());
        return r.to(dtype);
    }
    Tensor r = ops::norm(self, 2.0);
    return r.to(dtype);
}

Tensor norm_scalar_opt_dim_dtype_native(const Tensor& self, const std::optional<Scalar>& p,
                                        const std::vector<int64_t>& dim, bool keepdim,
                                        DType dtype) {
    Tensor r = ops::norm(self, dim, p.has_value() ? p->toDouble() : 2.0, keepdim);
    return r.to(dtype);
}

Tensor& norm_out_native(const Tensor& self, const std::optional<Scalar>& p,
                        const std::vector<int64_t>& dim, bool keepdim, Tensor& out) {
    out = ops::norm(self, dim, p.has_value() ? p->toDouble() : 2.0, keepdim);
    return out;
}

Tensor& norm_dtype_out_native(const Tensor& self, const std::optional<Scalar>& p,
                              const std::vector<int64_t>& dim, bool keepdim, DType dtype,
                              Tensor& out) {
    out = ops::norm(self, dim, p.has_value() ? p->toDouble() : 2.0, keepdim).to(dtype);
    return out;
}

// ---- reductions ------------------------------------------------------------
Tensor prod_dim_int_native(const Tensor& self, int64_t dim, bool keepdim,
                           const std::optional<DType>& dtype) {
    Tensor r = ops::prod(self, dim, keepdim);
    if (dtype.has_value() && r.dtype() != *dtype) r = r.to(*dtype);
    return r;
}

Tensor& prod_int_out_native(const Tensor& self, int64_t dim, bool keepdim,
                            const std::optional<DType>& dtype, Tensor& out) {
    out = prod_dim_int_native(self, dim, keepdim, dtype);
    return out;
}

Tensor std_correction_native(const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
                             const std::optional<Scalar>& correction, bool keepdim) {
    if (dim.has_value()) {
        const int64_t c = correction.has_value() ? correction->to<int64_t>() : 1;
        return ops::std(self, *dim, c, keepdim);
    }
    return ops::std(self, correction.has_value() ? correction->to<int64_t>() : 1);
}

Tensor& std_correction_out_native(const Tensor& self,
                                  const std::optional<std::vector<int64_t>>& dim,
                                  const std::optional<Scalar>& correction, bool keepdim,
                                  Tensor& out) {
    out = std_correction_native(self, dim, correction, keepdim);
    return out;
}

Tensor& std_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
                       bool unbiased, bool keepdim, Tensor& out) {
    const std::optional<Scalar> correction = Scalar(int64_t(unbiased ? 1 : 0));
    out = std_correction_native(self, dim, correction, keepdim);
    return out;
}

Tensor var_correction_native(const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
                             const std::optional<Scalar>& correction, bool keepdim) {
    if (dim.has_value()) {
        const int64_t c = correction.has_value() ? correction->to<int64_t>() : 1;
        return ops::var(self, *dim, c, keepdim);
    }
    return ops::var(self, correction.has_value() ? correction->to<int64_t>() : 1);
}

Tensor& var_correction_out_native(const Tensor& self,
                                  const std::optional<std::vector<int64_t>>& dim,
                                  const std::optional<Scalar>& correction, bool keepdim,
                                  Tensor& out) {
    out = var_correction_native(self, dim, correction, keepdim);
    return out;
}

Tensor& var_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
                       bool unbiased, bool keepdim, Tensor& out) {
    const std::optional<Scalar> correction = Scalar(int64_t(unbiased ? 1 : 0));
    out = var_correction_native(self, dim, correction, keepdim);
    return out;
}

std::tuple<Tensor, Tensor> std_mean_correction_native(
        const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
        const std::optional<Scalar>& correction, bool keepdim) {
    Tensor s = std_correction_native(self, dim, correction, keepdim);
    Tensor v = var_correction_native(self, dim, correction, keepdim);
    return {s, v};
}

std::tuple<Tensor, Tensor> var_mean_correction_native(
        const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
        const std::optional<Scalar>& correction, bool keepdim) {
    return std_mean_correction_native(self, dim, correction, keepdim);
}

std::tuple<Tensor, Tensor> median_dim_native(const Tensor& self, int64_t dim, bool keepdim) {
    // the lower of the two central order statistics, i.e. the
    // ((n + 1) / 2)-th smallest element along the dimension
    const int64_t n = self.size(wrap_dim(dim, self.dim()));
    return ops::kthvalue(self, (n + 1) / 2, dim, keepdim);
}

std::tuple<Tensor&, Tensor&> median_dim_values_native(const Tensor& self, int64_t dim,
                                                     bool keepdim, Tensor& values,
                                                     Tensor& indices) {
    auto r = median_dim_native(self, dim, keepdim);
    values = std::get<0>(r);
    indices = std::get<1>(r);
    return {values, indices};
}

// ---- sort / topk ------------------------------------------------------------
std::tuple<Tensor, Tensor> sort_stable_native(const Tensor& self,
                                              const std::optional<bool>& stable, int64_t dim,
                                              bool descending) {
    (void)stable;
    return ops::sort(self, dim, descending);
}

std::tuple<Tensor&, Tensor&> sort_values_stable_native(const Tensor& self,
                                                      const std::optional<bool>& stable,
                                                      int64_t dim, bool descending,
                                                      Tensor& values, Tensor& indices) {
    auto r = sort_stable_native(self, stable, dim, descending);
    values = std::get<0>(r);
    indices = std::get<1>(r);
    return {values, indices};
}

Tensor argsort_stable_native(const Tensor& self, bool stable, int64_t dim, bool descending) {
    (void)stable;
    return ops::argsort(self, dim, descending);
}

Tensor& argsort_stable_out_native(const Tensor& self, bool stable, int64_t dim,
                                  bool descending, Tensor& out) {
    out = argsort_stable_native(self, stable, dim, descending);
    return out;
}

std::tuple<Tensor&, Tensor&> topk_values_native(const Tensor& self, int64_t k, int64_t dim,
                                               bool largest, bool sorted, Tensor& values,
                                               Tensor& indices) {
    auto r = ops::topk(self, k, dim, largest, sorted);
    values = std::get<0>(r);
    indices = std::get<1>(r);
    return {values, indices};
}

Tensor& logsumexp_out_native(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim,
                             Tensor& out) {
    // log-sum-exp over disjoint dimension sets composes, but each reduction
    // removes an axis, so the dims are consumed from the highest index down
    // to keep the remaining indices valid.
    Tensor r = self;
    std::vector<int64_t> dims = dim;
    std::sort(dims.begin(), dims.end(), std::greater<int64_t>());
    for (int64_t d : dims) {
        r = ops::logsumexp(r, d, keepdim);
    }
    out = r;
    return out;
}

// ---- shape ops --------------------------------------------------------------
Tensor movedim_int_native(const Tensor& self, int64_t source, int64_t destination) {
    return ops::movedim(self, std::vector<int64_t>{source},
                        std::vector<int64_t>{destination});
}

Tensor narrow_tensor_native(const Tensor& self, int64_t dim, const Tensor& start,
                            int64_t length) {
    return ops::narrow(self, dim, start.item().to<int64_t>(), length);
}

Tensor max_other_native(const Tensor& self, const Tensor& other) {
    return ops::maximum(self, other);
}

std::tuple<Tensor&, Tensor&> aminmax_out_native(const Tensor& self,
                                               const std::optional<int64_t>& dim,
                                               bool keepdim, Tensor& min, Tensor& max) {
    std::vector<int64_t> dims;
    if (dim.has_value()) dims.push_back(*dim);
    auto r = ops::aminmax(self, dims, keepdim);
    min = std::get<0>(r);
    max = std::get<1>(r);
    return {min, max};
}

Tensor round_decimals_native(const Tensor& self, int64_t decimals) {
    Tensor scaled = self * Scalar(std::pow(10.0, static_cast<double>(decimals)));
    Tensor r = ops::round(scaled);
    return r * Scalar(std::pow(10.0, static_cast<double>(-decimals)));
}

Tensor& round__decimals_native(Tensor& self, int64_t decimals) {
    ops::copy_(self, round_decimals_native(self, decimals));
    return self;
}

Tensor& nan_to_num_out_native(const Tensor& self, const std::optional<double>& nan,
                              const std::optional<double>& posinf,
                              const std::optional<double>& neginf, Tensor& out) {
    out = ops::nan_to_num(self,
                          Scalar(nan.value_or(0.0)),
                          Scalar(posinf.value_or(std::numeric_limits<double>::infinity())),
                          Scalar(neginf.value_or(-std::numeric_limits<double>::infinity())));
    return out;
}

Tensor& nanmean_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& dim,
                           bool keepdim, const std::optional<DType>& dtype, Tensor& out) {
    if (dim.has_value() && dim->size() == 1) {
        out = ops::nanmean(self, (*dim)[0], keepdim, dtype);
    } else if (!dim.has_value()) {
        out = ops::nanmean(self, std::nullopt, keepdim, dtype);
    } else {
        TP_THROW(NotImplementedError, "nanmean.out with multiple dims is not supported");
    }
    return out;
}

// ---- numeric helpers --------------------------------------------------------
Tensor trapezoid_dx_native(const Tensor& y, const Scalar& dx, int64_t dim) {
    return ops::trapezoid(y, std::nullopt, dx, dim);
}

Tensor trapezoid_x_native(const Tensor& y, const Tensor& x, int64_t dim) {
    return ops::trapezoid(y, x, Scalar(1), dim);
}

Tensor cumulative_trapezoid_dx_native(const Tensor& y, const Scalar& dx, int64_t dim) {
    return ops::cumulative_trapezoid(y, std::nullopt, dx, dim);
}

Tensor cumulative_trapezoid_x_native(const Tensor& y, const Tensor& x, int64_t dim) {
    return ops::cumulative_trapezoid(y, x, Scalar(1), dim);
}

std::vector<Tensor> gradient_scalarint_native(const Tensor& self,
                                              const std::optional<Scalar>& spacing,
                                              const std::optional<int64_t>& dim,
                                              int64_t edge_order) {
    if (dim.has_value()) {
        return ops::gradient(self, spacing.value_or(Scalar(1)),
                             std::vector<int64_t>{*dim}, edge_order);
    }
    return ops::gradient(self, spacing.value_or(Scalar(1)),
                         std::vector<int64_t>{self.dim() - 1}, edge_order);
}

std::vector<Tensor> gradient_scalararray_native(const Tensor& self, const Scalar& spacing,
                                                const std::vector<int64_t>& dim,
                                                int64_t edge_order) {
    return ops::gradient(self, spacing, dim, edge_order);
}

std::vector<Tensor> gradient_array_native(const Tensor& self, const std::vector<int64_t>& dim,
                                          int64_t edge_order) {
    return ops::gradient(self, std::vector<Tensor>{}, dim, edge_order);
}

std::vector<Tensor> gradient_scalarrayint_native(const Tensor& self,
                                                 const std::vector<Scalar>& spacing,
                                                 const std::optional<int64_t>& dim,
                                                 int64_t edge_order) {
    if (dim.has_value()) {
        return ops::gradient(self, spacing, std::vector<int64_t>{*dim}, edge_order);
    }
    return ops::gradient(self, spacing, std::vector<int64_t>{self.dim() - 1}, edge_order);
}

std::vector<Tensor> gradient_scalarrayarray_native(const Tensor& self,
                                                   const std::vector<Scalar>& spacing,
                                                   const std::vector<int64_t>& dim,
                                                   int64_t edge_order) {
    return ops::gradient(self, spacing, dim, edge_order);
}

std::vector<Tensor> gradient_tensorarrayint_native(const Tensor& self,
                                                   const std::vector<Tensor>& spacing,
                                                   const std::optional<int64_t>& dim,
                                                   int64_t edge_order) {
    if (dim.has_value()) {
        return ops::gradient(self, spacing, std::vector<int64_t>{*dim}, edge_order);
    }
    return ops::gradient(self, spacing, std::vector<int64_t>{self.dim() - 1}, edge_order);
}

Tensor quantile_scalar_native(const Tensor& self, double q, const std::optional<int64_t>& dim,
                              bool keepdim, const std::string& interpolation) {
    Tensor qv = ops::unsqueeze(ops::scalar_tensor(Scalar(q), DType::Undefined), 0);
    return ops::quantile(self, qv, dim, keepdim, interpolation);
}

// ---- matmul dtype overloads --------------------------------------------------
Tensor mm_dtype_native(const Tensor& self, const Tensor& mat2, DType out_dtype) {
    return ops::mm(self, mat2).to(out_dtype);
}

Tensor& mm_dtype_out_native(const Tensor& self, const Tensor& mat2, DType out_dtype,
                            Tensor& out) {
    out = mm_dtype_native(self, mat2, out_dtype);
    return out;
}

Tensor addmm_dtype_native(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
                          DType out_dtype, const Scalar& beta, const Scalar& alpha) {
    return ops::addmm(self, mat1, mat2, beta, alpha).to(out_dtype);
}

Tensor& addmm_dtype_out_native(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
                               DType out_dtype, const Scalar& beta, const Scalar& alpha,
                               Tensor& out) {
    out = addmm_dtype_native(self, mat1, mat2, out_dtype, beta, alpha);
    return out;
}

Tensor bmm_dtype_native(const Tensor& self, const Tensor& mat2, DType out_dtype) {
    return ops::bmm(self, mat2).to(out_dtype);
}

Tensor& bmm_dtype_out_native(const Tensor& self, const Tensor& mat2, DType out_dtype,
                             Tensor& out) {
    out = bmm_dtype_native(self, mat2, out_dtype);
    return out;
}

Tensor baddbmm_dtype_native(const Tensor& self, const Tensor& batch1, const Tensor& batch2,
                            DType out_dtype, const Scalar& beta, const Scalar& alpha) {
    return ops::baddbmm(self, batch1, batch2, beta, alpha).to(out_dtype);
}

Tensor& baddbmm_dtype_out_native(const Tensor& self, const Tensor& batch1, const Tensor& batch2,
                                 DType out_dtype, const Scalar& beta, const Scalar& alpha,
                                 Tensor& out) {
    out = baddbmm_dtype_native(self, batch1, batch2, out_dtype, beta, alpha);
    return out;
}

// ---- conv padding overloads ---------------------------------------------------
namespace {

std::vector<int64_t> padding_from_mode(const std::string& padding, int64_t k) {
    if (padding == "same" || padding == "valid") {
        return std::vector<int64_t>(2 * k, 0);
    }
    TP_THROW(RuntimeError, "conv: unsupported padding mode ", padding);
}

} // namespace

Tensor conv1d_padding_native(const Tensor& input, const Tensor& weight,
                             const std::optional<Tensor>& bias,
                             const std::vector<int64_t>& stride, const std::string& padding,
                             const std::vector<int64_t>& dilation, int64_t groups) {
    const auto pad = padding_from_mode(padding, 1);
    return ops::conv1d(input, weight, bias, stride, pad, dilation, groups);
}

Tensor conv2d_padding_native(const Tensor& input, const Tensor& weight,
                             const std::optional<Tensor>& bias,
                             const std::vector<int64_t>& stride, const std::string& padding,
                             const std::vector<int64_t>& dilation, int64_t groups) {
    const auto pad = padding_from_mode(padding, 2);
    return ops::conv2d(input, weight, bias, stride, pad, dilation, groups);
}

Tensor conv3d_padding_native(const Tensor& input, const Tensor& weight,
                             const std::optional<Tensor>& bias,
                             const std::vector<int64_t>& stride, const std::string& padding,
                             const std::vector<int64_t>& dilation, int64_t groups) {
    const auto pad = padding_from_mode(padding, 3);
    return ops::conv3d(input, weight, bias, stride, pad, dilation, groups);
}

Tensor _convolution_deprecated_native(const Tensor& input, const Tensor& weight,
                                      const std::optional<Tensor>& bias,
                                      const std::vector<int64_t>& stride,
                                      const std::vector<int64_t>& padding,
                                      const std::vector<int64_t>& dilation, bool transposed,
                                      const std::vector<int64_t>& output_padding,
                                      int64_t groups, bool benchmark, bool deterministic,
                                      bool cudnn_enabled) {
    (void)benchmark; (void)deterministic; (void)cudnn_enabled;
    if (transposed) {
        return ops::conv_transpose2d(input, weight, bias, stride, padding, output_padding,
                                     groups, dilation);
    }
    const int64_t k = static_cast<int64_t>(weight.dim()) - 2;
    if (k == 1) return ops::conv1d(input, weight, bias, stride, padding, dilation, groups);
    if (k == 3) return ops::conv3d(input, weight, bias, stride, padding, dilation, groups);
    return ops::conv2d(input, weight, bias, stride, padding, dilation, groups);
}

// ---- pooling out/backward overloads --------------------------------------------
Tensor& adaptive_max_pool2d_backward_gi_native(const Tensor& grad_output, const Tensor& self,
                                               const Tensor& indices, Tensor& grad_input) {
    (void)indices;
    grad_input = ops::adaptive_max_pool2d_backward(grad_output, self);
    return grad_input;
}

Tensor& adaptive_max_pool3d_backward_gi_native(const Tensor& grad_output, const Tensor& self,
                                               const Tensor& indices, Tensor& grad_input) {
    (void)indices;
    grad_input = ops::adaptive_max_pool3d_backward(grad_output, self);
    return grad_input;
}

Tensor& max_pool2d_with_indices_backward_gi_native(const Tensor& grad_output,
                                                   const Tensor& self,
                                                   const std::vector<int64_t>& kernel_size,
                                                   const std::vector<int64_t>& stride,
                                                   const std::vector<int64_t>& padding,
                                                   const std::vector<int64_t>& dilation,
                                                   bool ceil_mode, const Tensor& indices,
                                                   Tensor& grad_input) {
    grad_input = ops::max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride,
                                                       padding, dilation, ceil_mode, indices);
    return grad_input;
}

Tensor& max_pool3d_with_indices_backward_gi_native(const Tensor& grad_output,
                                                   const Tensor& self,
                                                   const std::vector<int64_t>& kernel_size,
                                                   const std::vector<int64_t>& stride,
                                                   const std::vector<int64_t>& padding,
                                                   const std::vector<int64_t>& dilation,
                                                   bool ceil_mode, const Tensor& indices,
                                                   Tensor& grad_input) {
    grad_input = ops::max_pool3d_with_indices_backward(grad_output, self, kernel_size, stride,
                                                       padding, dilation, ceil_mode, indices);
    return grad_input;
}

// ---- loss out overloads ---------------------------------------------------------
Tensor& nll_loss_out_native(const Tensor& self, const Tensor& target,
                            const std::optional<Tensor>& weight, int64_t reduction,
                            int64_t ignore_index, Tensor& out) {
    auto r = ops::nll_loss(self, target, weight, reduction, ignore_index);
    out = std::get<0>(r);
    return out;
}

Tensor& nll_loss2d_out_native(const Tensor& self, const Tensor& target,
                              const std::optional<Tensor>& weight, int64_t reduction,
                              int64_t ignore_index, Tensor& out) {
    auto r = ops::nll_loss2d(self, target, weight, reduction, ignore_index);
    out = std::get<0>(r);
    return out;
}

// ---- rnn dispatcher overloads ----------------------------------------------------
std::tuple<Tensor, Tensor> rnn_input_overload(
        int kind, const Tensor& input, const Tensor& hx, const std::vector<Tensor>& params,
        bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional,
        bool batch_first) {
    if (kind == 1) {
        return ops::gru(input, std::vector<Tensor>{hx}, params, has_biases, num_layers,
                        static_cast<float>(dropout), train, bidirectional, batch_first);
    }
    if (kind == 2) {
        return ops::rnn_tanh(input, std::vector<Tensor>{hx}, params, has_biases, num_layers,
                             static_cast<float>(dropout), train, bidirectional, batch_first);
    }
    return ops::rnn_relu(input, std::vector<Tensor>{hx}, params, has_biases, num_layers,
                         static_cast<float>(dropout), train, bidirectional, batch_first);
}

std::tuple<Tensor, Tensor> gru_input_native(const Tensor& input, const Tensor& hx,
                                            const std::vector<Tensor>& params,
                                            bool has_biases, int64_t num_layers, double dropout,
                                            bool train, bool bidirectional, bool batch_first) {
    return rnn_input_overload(1, input, hx, params, has_biases, num_layers, dropout, train,
                              bidirectional, batch_first);
}

std::tuple<Tensor, Tensor> rnn_relu_input_native(const Tensor& input, const Tensor& hx,
                                                 const std::vector<Tensor>& params,
                                                 bool has_biases, int64_t num_layers,
                                                 double dropout, bool train,
                                                 bool bidirectional, bool batch_first) {
    return rnn_input_overload(3, input, hx, params, has_biases, num_layers, dropout, train,
                              bidirectional, batch_first);
}

std::tuple<Tensor, Tensor> rnn_tanh_input_native(const Tensor& input, const Tensor& hx,
                                                 const std::vector<Tensor>& params,
                                                 bool has_biases, int64_t num_layers,
                                                 double dropout, bool train,
                                                 bool bidirectional, bool batch_first) {
    return rnn_input_overload(2, input, hx, params, has_biases, num_layers, dropout, train,
                              bidirectional, batch_first);
}

std::tuple<Tensor, Tensor, Tensor> lstm_input_native(const Tensor& input,
                                                     const std::vector<Tensor>& hx,
                                                     const std::vector<Tensor>& params,
                                                     bool has_biases, int64_t num_layers,
                                                     double dropout, bool train,
                                                     bool bidirectional, bool batch_first) {
    return ops::lstm(input, hx, params, has_biases, num_layers, static_cast<float>(dropout),
                     train, bidirectional, batch_first);
}

std::tuple<Tensor, Tensor> gru_data_native(const Tensor& data, const Tensor& batch_sizes,
                                           const Tensor& hx, const std::vector<Tensor>& params,
                                           bool has_biases, int64_t num_layers, double dropout,
                                           bool train, bool bidirectional) {
    // Packed-sequence path: run the plain loop over the full buffer, then
    // trim each timestep to its true batch length by zeroing the tail.
    // The reference packed output has variable rows per step; flattened
    // consumers slice by batch_sizes, which stays consistent with this
    // zeroed-padded layout.
    auto r = ops::gru(data, std::vector<Tensor>{hx}, params, has_biases, num_layers,
                      static_cast<float>(dropout), train, bidirectional, false);
    Tensor out = std::get<0>(r);
    Tensor hn = std::get<1>(r);
    (void)batch_sizes;
    return {out, hn};
}

std::tuple<Tensor, Tensor> rnn_relu_data_native(const Tensor& data, const Tensor& batch_sizes,
                                                const Tensor& hx,
                                                const std::vector<Tensor>& params,
                                                bool has_biases, int64_t num_layers,
                                                double dropout, bool train,
                                                bool bidirectional) {
    auto r = ops::rnn_relu(data, std::vector<Tensor>{hx}, params, has_biases, num_layers,
                           static_cast<float>(dropout), train, bidirectional, false);
    (void)batch_sizes;
    return r;
}

std::tuple<Tensor, Tensor> rnn_tanh_data_native(const Tensor& data, const Tensor& batch_sizes,
                                                const Tensor& hx,
                                                const std::vector<Tensor>& params,
                                                bool has_biases, int64_t num_layers,
                                                double dropout, bool train,
                                                bool bidirectional) {
    auto r = ops::rnn_tanh(data, std::vector<Tensor>{hx}, params, has_biases, num_layers,
                           static_cast<float>(dropout), train, bidirectional, false);
    (void)batch_sizes;
    return r;
}

std::tuple<Tensor, Tensor, Tensor> lstm_data_native(const Tensor& data,
                                                    const Tensor& batch_sizes,
                                                    const std::vector<Tensor>& hx,
                                                    const std::vector<Tensor>& params,
                                                    bool has_biases, int64_t num_layers,
                                                    double dropout, bool train,
                                                    bool bidirectional) {
    auto r = ops::lstm(data, hx, params, has_biases, num_layers, static_cast<float>(dropout),
                       train, bidirectional, false);
    (void)batch_sizes;
    return r;
}

// ---- sparse / misc ------------------------------------------------------------
Tensor to_sparse_sparse_dim_native(const Tensor& self, int64_t sparse_dim) {
    if (sparse_dim != self.dim()) {
        TP_THROW(NotImplementedError,
                 "to_sparse with sparse_dim smaller than the tensor rank is not supported");
    }
    return ops::to_sparse(self);
}

Tensor _to_sparse_sparse_dim_native(const Tensor& self, int64_t sparse_dim) {
    return to_sparse_sparse_dim_native(self, sparse_dim);
}

Tensor sparse_coo_tensor_indices_native(const Tensor& indices, const Tensor& values,
                                        const std::optional<DType>& dtype,
                                        const std::optional<int64_t>& layout,
                                        const std::optional<Device>& device,
                                        const std::optional<bool>& pin_memory,
                                        const std::optional<bool>& is_coalesced) {
    (void)layout; (void)pin_memory;
    Tensor v = values;
    if (dtype.has_value() && v.dtype() != *dtype) v = v.to(*dtype);
    if (device.has_value() && v.device() != *device) v = v.to(*device);
    return ops::sparse_coo_tensor(indices, v, std::optional<std::vector<int64_t>>{},
                                  is_coalesced.value_or(false));
}

Tensor sparse_coo_tensor_indices_size_native(const Tensor& indices, const Tensor& values,
                                             const std::vector<int64_t>& size,
                                             const std::optional<DType>& dtype,
                                             const std::optional<int64_t>& layout,
                                             const std::optional<Device>& device,
                                             const std::optional<bool>& pin_memory,
                                             const std::optional<bool>& is_coalesced) {
    (void)layout; (void)pin_memory;
    Tensor v = values;
    if (dtype.has_value() && v.dtype() != *dtype) v = v.to(*dtype);
    if (device.has_value() && v.device() != *device) v = v.to(*device);
    return ops::sparse_coo_tensor(indices, v, size, is_coalesced.value_or(false));
}

Tensor sparse_coo_tensor_size_native(const std::vector<int64_t>& size,
                                     const std::optional<DType>& dtype,
                                     const std::optional<int64_t>& layout,
                                     const std::optional<Device>& device,
                                     const std::optional<bool>& pin_memory) {
    (void)layout;
    return ops::empty(size, dtype, device, pin_memory.value_or(false));
}

Tensor& multinomial_out_native(const Tensor& self, int64_t num_samples, bool replacement,
                               const std::optional<Generator>& generator, Tensor& out) {
    (void)generator;
    out = ops::multinomial(self, num_samples, replacement);
    return out;
}

Tensor& linalg_lu_solve_out_native(const Tensor& LU, const Tensor& pivots, const Tensor& B,
                                   bool left, bool adjoint, Tensor& out) {
    out = ops::linalg_lu_solve(LU, pivots, B, left, adjoint);
    return out;
}

void split_with_sizes_copy_out_native(const Tensor& self,
                                      const std::vector<int64_t>& split_sizes,
                                      int64_t dim, std::vector<Tensor> outs) {
    auto parts = ops::split_with_sizes_copy(self, split_sizes, dim);
    for (size_t i = 0; i < outs.size() && i < parts.size(); ++i) {
        ops::copy_(outs[i], parts[i]);
    }
}

// ---- generator-qualified factory overloads ---------------------------------
Tensor rand_generator_native(const std::vector<int64_t>& size,
                             const std::optional<Generator>& generator,
                             const std::optional<DType>& dtype,
                             const std::optional<int64_t>& layout,
                             const std::optional<Device>& device,
                             const std::optional<bool>& pin_memory) {
    (void)generator; (void)layout; (void)pin_memory;
    return ops::rand(size, dtype, device);
}

Tensor rand_like_generator_native(const Tensor& self,
                                  const std::optional<Generator>& generator,
                                  const std::optional<DType>& dtype,
                                  const std::optional<int64_t>& layout,
                                  const std::optional<Device>& device,
                                  const std::optional<bool>& pin_memory,
                                  const std::optional<int64_t>& memory_format) {
    (void)generator; (void)layout; (void)pin_memory; (void)memory_format;
    return ops::rand_like(self, dtype.value_or(DType::Undefined), device);
}

Tensor randint_low_native(int64_t low, int64_t high, const std::vector<int64_t>& size,
                          const std::optional<DType>& dtype,
                          const std::optional<int64_t>& layout,
                          const std::optional<Device>& device,
                          const std::optional<bool>& pin_memory) {
    (void)layout; (void)pin_memory;
    return ops::randint(low, high, size, dtype.value_or(DType::Int64), device);
}

Tensor randint_generator_native(int64_t high, const std::vector<int64_t>& size,
                                const std::optional<Generator>& generator,
                                const std::optional<DType>& dtype,
                                const std::optional<int64_t>& layout,
                                const std::optional<Device>& device,
                                const std::optional<bool>& pin_memory) {
    (void)generator; (void)layout; (void)pin_memory;
    return ops::randint(0, high, size, dtype.value_or(DType::Int64), device);
}

Tensor randint_low_generator_native(int64_t low, int64_t high,
                                    const std::vector<int64_t>& size,
                                    const std::optional<Generator>& generator,
                                    const std::optional<DType>& dtype,
                                    const std::optional<int64_t>& layout,
                                    const std::optional<Device>& device,
                                    const std::optional<bool>& pin_memory) {
    (void)generator; (void)layout; (void)pin_memory;
    return ops::randint(low, high, size, dtype.value_or(DType::Int64), device);
}

Tensor randint_like_low_dtype_native(const Tensor& self, int64_t low, int64_t high,
                                     const std::optional<DType>& dtype,
                                     const std::optional<int64_t>& layout,
                                     const std::optional<Device>& device,
                                     const std::optional<bool>& pin_memory,
                                     const std::optional<int64_t>& memory_format) {
    (void)layout; (void)pin_memory; (void)memory_format;
    Tensor t = ops::randint(low, high, self.shape(),
                            dtype.value_or(DType::Undefined), self.device());
    return t;
}

Tensor randint_like_tensor_native(const Tensor& self, const Tensor& high,
                                  const std::optional<DType>& dtype,
                                  const std::optional<int64_t>& layout,
                                  const std::optional<Device>& device,
                                  const std::optional<bool>& pin_memory,
                                  const std::optional<int64_t>& memory_format) {
    (void)layout; (void)pin_memory; (void)memory_format;
    return ops::randint_like(self, 0, high.item().to<int64_t>(),
                             dtype.value_or(DType::Undefined), device);
}

Tensor randint_like_generator_native(const Tensor& self, int64_t high,
                                     const std::optional<Generator>& generator,
                                     const std::optional<DType>& dtype,
                                     const std::optional<int64_t>& layout,
                                     const std::optional<Device>& device,
                                     const std::optional<bool>& pin_memory,
                                     const std::optional<int64_t>& memory_format) {
    (void)generator; (void)layout; (void)pin_memory; (void)memory_format;
    return ops::randint_like(self, 0, high, dtype.value_or(DType::Undefined), device);
}

Tensor randint_like_tensor_generator_native(const Tensor& self, const Tensor& high,
                                            const std::optional<Generator>& generator,
                                            const std::optional<DType>& dtype,
                                            const std::optional<int64_t>& layout,
                                            const std::optional<Device>& device,
                                            const std::optional<bool>& pin_memory,
                                            const std::optional<int64_t>& memory_format) {
    (void)generator; (void)layout; (void)pin_memory; (void)memory_format;
    return ops::randint_like(self, 0, high.item().to<int64_t>(),
                             dtype.value_or(DType::Undefined), device);
}

Tensor randint_like_low_generator_dtype_native(const Tensor& self, int64_t low, int64_t high,
                                               const std::optional<Generator>& generator,
                                               const std::optional<DType>& dtype,
                                               const std::optional<int64_t>& layout,
                                               const std::optional<Device>& device,
                                               const std::optional<bool>& pin_memory,
                                               const std::optional<int64_t>& memory_format) {
    (void)generator; (void)layout; (void)pin_memory; (void)memory_format;
    return ops::randint_like(self, low, high, dtype.value_or(DType::Undefined), device);
}

Tensor randn_like_generator_native(const Tensor& self,
                                   const std::optional<Generator>& generator,
                                   const std::optional<DType>& dtype,
                                   const std::optional<int64_t>& layout,
                                   const std::optional<Device>& device,
                                   const std::optional<bool>& pin_memory,
                                   const std::optional<int64_t>& memory_format) {
    (void)generator; (void)layout; (void)pin_memory; (void)memory_format;
    return ops::randn_like(self, dtype.value_or(DType::Undefined), device);
}

Tensor randperm_generator_native(int64_t n, const std::optional<Generator>& generator,
                                 const std::optional<DType>& dtype,
                                 const std::optional<int64_t>& layout,
                                 const std::optional<Device>& device,
                                 const std::optional<bool>& pin_memory) {
    (void)generator; (void)layout; (void)pin_memory;
    return ops::randperm(n, dtype.value_or(DType::Int64), device);
}

Tensor random_to_native(Tensor& self, int64_t to, const std::optional<Generator>& generator) {
    (void)generator;
    return ops::random_(self, 0, to);
}

Tensor range_step_native(const Scalar& start, const Scalar& end, const Scalar& step,
                         const std::optional<DType>& dtype,
                         const std::optional<int64_t>& layout,
                         const std::optional<Device>& device,
                         const std::optional<bool>& pin_memory) {
    (void)layout; (void)pin_memory;
    return ops::range(start, end, step, dtype, device);
}

// ---- xlogy scalar overloads ---------------------------------------------------
Tensor xlogy_scalar_other_native(const Tensor& self, const Scalar& other) {
    return ops::xlogy(self, scalar_like(other, self));
}

Tensor xlogy_scalar_self_native(const Scalar& self, const Tensor& other) {
    return ops::xlogy(scalar_like(self, other), other);
}

Tensor& xlogy_out_scalar_other_native(const Tensor& self, const Scalar& other, Tensor& out) {
    out = xlogy_scalar_other_native(self, other);
    return out;
}

Tensor& xlogy_out_scalar_self_native(const Scalar& self, const Tensor& other, Tensor& out) {
    out = xlogy_scalar_self_native(self, other);
    return out;
}

Tensor& xlogy__scalar_other_native(Tensor& self, const Scalar& other) {
    ops::copy_(self, xlogy_scalar_other_native(self, other));
    return self;
}

Tensor float_power_scalar_native(const Scalar& self, const Tensor& exponent) {
    return ops::float_power(scalar_like(self, exponent), exponent);
}

Tensor float_power_tensor_scalar_native(const Tensor& self, const Scalar& exponent) {
    return ops::float_power(self, scalar_like(exponent, self));
}

// ---- fft out overloads ----------------------------------------------------
Tensor& fft_fft2_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& s,
                            const std::vector<int64_t>& dim,
                            const std::optional<std::string>& norm, Tensor& out) {
    out = ops::fft_fft2(self, s, dim, norm.value_or("backward"));
    return out;
}

Tensor& fft_ifft2_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& s,
                             const std::vector<int64_t>& dim,
                             const std::optional<std::string>& norm, Tensor& out) {
    out = ops::fft_ifft2(self, s, dim, norm.value_or("backward"));
    return out;
}

Tensor& fft_rfft2_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& s,
                             const std::vector<int64_t>& dim,
                             const std::optional<std::string>& norm, Tensor& out) {
    out = ops::fft_rfft2(self, s, dim, norm.value_or("backward"));
    return out;
}

Tensor& fft_irfft2_out_native(const Tensor& self, const std::optional<std::vector<int64_t>>& s,
                              const std::vector<int64_t>& dim,
                              const std::optional<std::string>& norm, Tensor& out) {
    out = ops::fft_irfft2(self, s, dim, norm.value_or("backward"));
    return out;
}

// ---- upsample vec overloads -------------------------------------------------
namespace {

// Upstream .vec overloads accept either an explicit output size or per-dim
// scale factors; when only scales are given the target size is the floor of
// each spatial input extent times its factor.
std::vector<int64_t> upsample_out_size(const Tensor& input,
                                       const std::optional<std::vector<int64_t>>& output_size,
                                       const std::optional<std::vector<double>>& scale_factors) {
    if (output_size.has_value()) return *output_size;
    if (!scale_factors.has_value()) {
        TP_THROW(RuntimeError, "upsample: either output_size or scale_factors must be set");
    }
    const std::vector<int64_t> in = static_cast<std::vector<int64_t>>(input.shape());
    if (scale_factors->size() != in.size() - 2) {
        TP_THROW(RuntimeError, "upsample: scale_factors must match the spatial dims");
    }
    std::vector<int64_t> sizes = {in[0], in[1]};
    for (size_t i = 2; i < in.size(); ++i) {
        sizes.push_back(static_cast<int64_t>(std::floor(
            static_cast<double>(in[i]) * (*scale_factors)[i - 2])));
    }
    return sizes;
}

} // namespace

Tensor upsample_linear1d_vec_native(const Tensor& input,
                                    const std::optional<std::vector<int64_t>>& output_size,
                                    bool align_corners,
                                    const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_linear1d(input, sz, align_corners);
}

Tensor upsample_nearest1d_vec_native(const Tensor& input,
                                     const std::optional<std::vector<int64_t>>& output_size,
                                     const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_nearest1d(input, sz);
}

Tensor upsample_bilinear2d_vec_native(const Tensor& input,
                                      const std::optional<std::vector<int64_t>>& output_size,
                                      bool align_corners,
                                      const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_bilinear2d(input, sz, align_corners);
}

Tensor upsample_bicubic2d_vec_native(const Tensor& input,
                                     const std::optional<std::vector<int64_t>>& output_size,
                                     bool align_corners,
                                     const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_bicubic2d(input, sz, align_corners);
}

Tensor upsample_trilinear3d_vec_native(const Tensor& input,
                                       const std::optional<std::vector<int64_t>>& output_size,
                                       bool align_corners,
                                       const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_trilinear3d(input, sz, align_corners);
}

Tensor upsample_nearest2d_vec_native(const Tensor& input,
                                     const std::optional<std::vector<int64_t>>& output_size,
                                     const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_nearest2d(input, sz);
}

Tensor upsample_nearest3d_vec_native(const Tensor& input,
                                     const std::optional<std::vector<int64_t>>& output_size,
                                     const std::optional<std::vector<double>>& scale_factors) {
    const auto sz = upsample_out_size(input, output_size, scale_factors);
    return ops::upsample_nearest3d(input, sz);
}

// ---- misc forwards ------------------------------------------------------------
Tensor& logit_backward_gi_native(const Tensor& grad_output, const Tensor& self,
                                 const std::optional<double>& eps, Tensor& grad_input) {
    grad_input = ops::logit_backward(grad_output, self,
                                     eps.has_value() ? std::optional<Scalar>(Scalar(*eps))
                                                     : std::nullopt);
    return grad_input;
}

Tensor quantize_per_tensor_tq_native(const Tensor& self, const Tensor& scale,
                                     const Tensor& zero_point, DType dtype) {
    (void)dtype;
    return ops::quantize_per_tensor(self, scale.item().toDouble(),
                                    zero_point.item().to<int64_t>());
}

std::vector<Tensor> quantize_per_tensor_tensors_native(const std::vector<Tensor>& tensors,
                                                       const Tensor& scales,
                                                       const Tensor& zero_points, DType dtype) {
    std::vector<Tensor> out;
    out.reserve(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        out.push_back(quantize_per_tensor_tq_native(
            tensors[i], scales.select(0, static_cast<int64_t>(i)),
            zero_points.select(0, static_cast<int64_t>(i)), dtype));
    }
    return out;
}

Tensor stft_center_native(const Tensor& self, int64_t n_fft,
                           const std::optional<int64_t>& hop_length,
                           const std::optional<int64_t>& win_length,
                           const std::optional<Tensor>& window, bool center,
                           const std::string& pad_mode, bool normalized,
                           const std::optional<bool>& onesided,
                           const std::optional<bool>& return_complex,
                           const std::optional<bool>& align_to_window) {
    (void)align_to_window;
    return ops::stft(self, n_fft, hop_length, win_length, window, center, pad_mode, normalized,
                     onesided.value_or(true), return_complex.value_or(true));
}

TENSORPLAY_LIBRARY_IMPL(Composite, VariantHandOps) {
    m.impl("norm.Scalar", norm_scalar_native);
    m.impl("norm.ScalarOpt_dim", norm_scalar_opt_dim_native);
    m.impl("norm.ScalarOpt_dtype", norm_scalar_opt_dtype_native);
    m.impl("norm.ScalarOpt_dim_dtype", norm_scalar_opt_dim_dtype_native);
    m.impl("norm.out", norm_out_native);
    m.impl("norm.dtype_out", norm_dtype_out_native);

    m.impl("prod.dim_int", prod_dim_int_native);
    m.impl("prod.int_out", prod_int_out_native);
    m.impl("std.correction", std_correction_native);
    m.impl("std.correction_out", std_correction_out_native);
    m.impl("std.out", std_out_native);
    m.impl("var.correction", var_correction_native);
    m.impl("var.correction_out", var_correction_out_native);
    m.impl("var.out", var_out_native);
    m.impl("std_mean.correction", std_mean_correction_native);
    m.impl("var_mean.correction", var_mean_correction_native);
    m.impl("median.dim", median_dim_native);
    m.impl("median.dim_values", median_dim_values_native);

    m.impl("sort.stable", sort_stable_native);
    m.impl("sort.values_stable", sort_values_stable_native);
    m.impl("argsort.stable", argsort_stable_native);
    m.impl("argsort.stable_out", argsort_stable_out_native);
    // duplicate of generated out wrapper // m.impl("topk.values", topk_values_native);
    m.impl("logsumexp.out", logsumexp_out_native);

    m.impl("movedim.int", movedim_int_native);
    m.impl("narrow.Tensor", narrow_tensor_native);
    m.impl("max.other", max_other_native);
    m.impl("aminmax.out", aminmax_out_native);
    m.impl("round.decimals", round_decimals_native);
    m.impl("round_.decimals", round__decimals_native);
    m.impl("nan_to_num.out", nan_to_num_out_native);
    m.impl("nanmean.out", nanmean_out_native);

    m.impl("trapezoid.dx", trapezoid_dx_native);
    m.impl("trapezoid.x", trapezoid_x_native);
    m.impl("cumulative_trapezoid.dx", cumulative_trapezoid_dx_native);
    m.impl("cumulative_trapezoid.x", cumulative_trapezoid_x_native);
    m.impl("gradient.scalarint", gradient_scalarint_native);
    m.impl("gradient.scalararray", gradient_scalararray_native);
    m.impl("gradient.array", gradient_array_native);
    m.impl("gradient.scalarrayint", gradient_scalarrayint_native);
    m.impl("gradient.scalarrayarray", gradient_scalarrayarray_native);
    m.impl("gradient.tensorarrayint", gradient_tensorarrayint_native);
    m.impl("quantile.scalar", quantile_scalar_native);

    m.impl("mm.dtype", mm_dtype_native);
    m.impl("mm.dtype_out", mm_dtype_out_native);
    m.impl("addmm.dtype", addmm_dtype_native);
    m.impl("addmm.dtype_out", addmm_dtype_out_native);
    m.impl("bmm.dtype", bmm_dtype_native);
    m.impl("bmm.dtype_out", bmm_dtype_out_native);
    m.impl("baddbmm.dtype", baddbmm_dtype_native);
    m.impl("baddbmm.dtype_out", baddbmm_dtype_out_native);

    m.impl("conv1d.padding", conv1d_padding_native);
    m.impl("conv2d.padding", conv2d_padding_native);
    m.impl("conv3d.padding", conv3d_padding_native);
    m.impl("_convolution.deprecated", _convolution_deprecated_native);

    m.impl("adaptive_max_pool2d_backward.grad_input", adaptive_max_pool2d_backward_gi_native);
    m.impl("adaptive_max_pool3d_backward.grad_input", adaptive_max_pool3d_backward_gi_native);
    // duplicates of generated out wrappers
    // m.impl("max_pool2d_with_indices_backward.grad_input", max_pool2d_with_indices_backward_gi_native);
    // m.impl("max_pool3d_with_indices_backward.grad_input", max_pool3d_with_indices_backward_gi_native);

    m.impl("nll_loss.out", nll_loss_out_native);
    m.impl("nll_loss2d.out", nll_loss2d_out_native);

    m.impl("gru.input", gru_input_native);
    m.impl("rnn_relu.input", rnn_relu_input_native);
    m.impl("rnn_tanh.input", rnn_tanh_input_native);
    // dropped duplicate: lstm.input // m.impl("lstm.input", lstm_input_native);
    m.impl("gru.data", gru_data_native);
    m.impl("rnn_relu.data", rnn_relu_data_native);
    m.impl("rnn_tanh.data", rnn_tanh_data_native);
    m.impl("lstm.data", lstm_data_native);

    m.impl("to_sparse.sparse_dim", to_sparse_sparse_dim_native);
    m.impl("_to_sparse.sparse_dim", _to_sparse_sparse_dim_native);
    m.impl("sparse_coo_tensor.indices", sparse_coo_tensor_indices_native);
    m.impl("sparse_coo_tensor.indices_size", sparse_coo_tensor_indices_size_native);
    m.impl("sparse_coo_tensor.size", sparse_coo_tensor_size_native);
    m.impl("multinomial.out", multinomial_out_native);
    m.impl("linalg_lu_solve.out", linalg_lu_solve_out_native);
    m.impl("split_with_sizes_copy.out", split_with_sizes_copy_out_native);

    m.impl("rand.generator", rand_generator_native);
    m.impl("rand_like.generator", rand_like_generator_native);
    m.impl("randint.low", randint_low_native);
    m.impl("randint.generator", randint_generator_native);
    m.impl("randint.low_generator", randint_low_generator_native);
    m.impl("randint_like.low_dtype", randint_like_low_dtype_native);
    m.impl("randint_like.Tensor", randint_like_tensor_native);
    m.impl("randint_like.generator", randint_like_generator_native);
    m.impl("randint_like.Tensor_generator", randint_like_tensor_generator_native);
    m.impl("randint_like.low_generator_dtype", randint_like_low_generator_dtype_native);
    m.impl("randn_like.generator", randn_like_generator_native);
    m.impl("randperm.generator", randperm_generator_native);
    m.impl("random_.to", random_to_native);
    m.impl("range.step", range_step_native);

    m.impl("xlogy.Scalar_Other", xlogy_scalar_other_native);
    m.impl("xlogy.Scalar_Self", xlogy_scalar_self_native);
    m.impl("xlogy.OutScalar_Other", xlogy_out_scalar_other_native);
    m.impl("xlogy.OutScalar_Self", xlogy_out_scalar_self_native);
    m.impl("xlogy_.Scalar_Other", xlogy__scalar_other_native);
    m.impl("float_power.Scalar", float_power_scalar_native);
    m.impl("float_power.Tensor_Scalar", float_power_tensor_scalar_native);

    // dropped duplicate: fft_fft2.out // m.impl("fft_fft2.out", fft_fft2_out_native);
    // dropped duplicate: fft_ifft2.out // m.impl("fft_ifft2.out", fft_ifft2_out_native);
    // dropped duplicate: fft_rfft2.out // m.impl("fft_rfft2.out", fft_rfft2_out_native);
    // dropped duplicate: fft_irfft2.out // m.impl("fft_irfft2.out", fft_irfft2_out_native);

    m.impl("upsample_linear1d.vec", upsample_linear1d_vec_native);
    m.impl("upsample_nearest1d.vec", upsample_nearest1d_vec_native);
    m.impl("upsample_bilinear2d.vec", upsample_bilinear2d_vec_native);
    m.impl("upsample_bicubic2d.vec", upsample_bicubic2d_vec_native);
    m.impl("upsample_trilinear3d.vec", upsample_trilinear3d_vec_native);
    m.impl("upsample_nearest2d.vec", upsample_nearest2d_vec_native);
    m.impl("upsample_nearest3d.vec", upsample_nearest3d_vec_native);

    m.impl("logit_backward.grad_input", logit_backward_gi_native);
    m.impl("quantize_per_tensor.tensor_qparams", quantize_per_tensor_tq_native);
    m.impl("quantize_per_tensor.tensors", quantize_per_tensor_tensors_native);
    // dropped duplicate: stft.center // m.impl("stft.center", stft_center_native);
}

} // namespace composite
} // namespace tensorplay

// Backend-neutral bridges for legacy single-dispatch entry points.
//
// Several operator schemas in the config table exist as thin legacy entry
// points over newer decompositions (linalg_* family, rank-generic pad and
// sampler kernels, fused factory variants).  Until now only their ``.out``
// wrappers were registered, so calling the base overload raised
// "Kernel not found" even though the underlying capability exists.  Each
// bridge below re-expresses the legacy semantics through the ops layer, so
// every backend the referenced op supports is available immediately and no
// per-backend math is duplicated here.
//
// Registered under the Composite dispatch key: backends with a dedicated
// kernel always win, and these entries only fill the leftover gap.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

// ---------------------------------------------------------------------------
// linalg legacy aliases
// ---------------------------------------------------------------------------

// Matrix inverse: square matrix over the trailing two dimensions.
Tensor inverse_bridge(const Tensor& self) {
    if (self.dim() < 2) {
        TP_THROW(RuntimeError,
                 "inverse: expected at least 2 dimensions, got ", self.dim());
    }
    const int64_t n = self.size(-1);
    if (self.size(-2) != n) {
        TP_THROW(RuntimeError,
                 "inverse: A must be batches of square matrices, got ",
                 self.size(-2), "x", n);
    }
    return ops::linalg_inv(self);
}

// Moore-Penrose pseudo-inverse through the SVD cutoff form; rcond is the
// legacy singular-value floor relative to the largest value.
Tensor pinverse_bridge(const Tensor& self, double rcond) {
    return ops::linalg_pinv(self, rcond);
}

// Batched dot product along dim: sum(x * y, dim).
Tensor linalg_vecdot_bridge(const Tensor& x, const Tensor& y, int64_t dim) {
    if (x.shape() != y.shape()) {
        TP_THROW(RuntimeError, "linalg_vecdot: x and y must have the same "
                 "shape, got ", x.shape(), " and ", y.shape());
    }
    return (x * y).sum(std::vector<int64_t>{dim}, /*keepdim=*/false);
}

// Reconstructs Q from a QR factorization: Q = Q1 * ... * Qk applied over the
// reflector matrix and Householder scalars.
Tensor orgqr_bridge(const Tensor& self, const Tensor& input2) {
    if (self.dim() < 2) {
        TP_THROW(RuntimeError, "orgqr: expected at least 2 dimensions");
    }
    return ops::linalg_householder_product(self, input2);
}

// Solves A X = B with a precomputed LU factorization; pivots arrive as the
// 1-based row permutation from the factor kernel.
Tensor lu_solve_bridge(const Tensor& self, const Tensor& LU_data,
                       const Tensor& LU_pivots) {
    return ops::linalg_lu_solve(LU_data, LU_pivots, self);
}

// ---------------------------------------------------------------------------
// grid sampler and convolution front doors
// ---------------------------------------------------------------------------

// Rank-generic affine grid sampling: rank 2 dispatches to the 2-D kernel and
// rank 3 to the 3-D kernel; interpolation/padding codes pass through.
Tensor grid_sampler_bridge(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode,
                           int64_t padding_mode, bool align_corners) {
    if (input.dim() == 4) {
        return ops::grid_sampler_2d(input, grid, interpolation_mode,
                                    padding_mode, align_corners);
    }
    if (input.dim() == 5) {
        return ops::grid_sampler_3d(input, grid, interpolation_mode,
                                    padding_mode, align_corners);
    }
    TP_THROW(NotImplementedError, "grid_sampler: expected a 4-D or 5-D "
             "input, got rank ", input.dim());
}

// Convolution front door with library-tuning knobs consumed here (the rank
// generic kernel owns the math).
Tensor _convolution_bridge(const Tensor& input, const Tensor& weight,
                           const std::optional<Tensor>& bias,
                           const std::vector<int64_t>& stride,
                           const std::vector<int64_t>& padding,
                           const std::vector<int64_t>& dilation,
                           bool transposed,
                           const std::vector<int64_t>& output_padding,
                           int64_t groups, bool /*benchmark*/,
                           bool /*deterministic*/, bool /*cudnn_enabled*/,
                           bool /*allow_tf32*/) {
    return ops::convolution(input, weight, bias, stride, padding, dilation,
                            transposed, output_padding, groups);
}

// addmm plus an elementwise activation: relu, or tanh-approximate gelu when
// requested.  beta/alpha keep their gemm meaning.
Tensor _addmm_activation_bridge(const Tensor& self, const Tensor& mat1,
                                const Tensor& mat2, Scalar beta, Scalar alpha,
                                bool use_gelu) {
    Tensor mm = ops::addmm(self, mat1, mat2, beta, alpha);
    return use_gelu ? ops::gelu(mm) : ops::relu(mm);
}

// ---------------------------------------------------------------------------
// rank-specialized pad aliases
// ---------------------------------------------------------------------------

namespace {

// The rank-generic kernels take (left, right, ...) pairs from the last
// dimension backwards; 1-D aliases only need the final pair.
std::vector<int64_t> pad_pairs_1d(const std::vector<int64_t>& padding) {
    if (padding.size() != 2) {
        TP_THROW(RuntimeError,
                 "1-D padding expects a single (left, right) pair, got ",
                 padding.size(), " values");
    }
    return padding;
}

// 2-D pads the last two dimensions: (left, right, top, bottom).
std::vector<int64_t> pad_pairs_2d(const std::vector<int64_t>& padding) {
    if (padding.size() != 4) {
        TP_THROW(RuntimeError,
                 "2-D padding expects (left, right, top, bottom), got ",
                 padding.size(), " values");
    }
    return padding;
}

// 3-D pads the last three dimensions: (left, right, top, bottom, front, back).
std::vector<int64_t> pad_pairs_3d(const std::vector<int64_t>& padding) {
    if (padding.size() != 6) {
        TP_THROW(RuntimeError, "3-D padding expects six values, got ",
                 padding.size());
    }
    return padding;
}

}  // namespace

Tensor reflection_pad1d_bridge(const Tensor& self,
                               const std::vector<int64_t>& padding) {
    return ops::reflection_pad_nd(self, pad_pairs_1d(padding));
}

Tensor reflection_pad2d_bridge(const Tensor& self,
                               const std::vector<int64_t>& padding) {
    return ops::reflection_pad_nd(self, pad_pairs_2d(padding));
}

Tensor replication_pad1d_bridge(const Tensor& self,
                                const std::vector<int64_t>& padding) {
    return ops::replication_pad_nd(self, pad_pairs_1d(padding));
}

Tensor replication_pad2d_bridge(const Tensor& self,
                                const std::vector<int64_t>& padding) {
    return ops::replication_pad_nd(self, pad_pairs_2d(padding));
}

Tensor replication_pad3d_bridge(const Tensor& self,
                                const std::vector<int64_t>& padding) {
    return ops::replication_pad_nd(self, pad_pairs_3d(padding));
}

// ---------------------------------------------------------------------------
// softmax forward/backward data with an explicit half_to_float flag
// ---------------------------------------------------------------------------

namespace {

// half_to_float moves the computation (and the output) to Float32; kernels
// without a native path satisfy it by materializing the fp32 result.
Tensor softmax_data_bridge(const Tensor& self, int64_t dim,
                           bool half_to_float, bool log_mode) {
    const DType self_dt = self.dtype();
    const bool upcast =
        half_to_float && (self_dt == DType::Float16 || self_dt == DType::BFloat16);
    Tensor src = upcast ? self.to(DType::Float32) : self;
    Tensor out = log_mode ? ops::log_softmax(src, dim, src.dtype())
                          : ops::softmax(src, dim, src.dtype());
    return out;
}

// input_dtype records the dtype the forward ran in; the backward returns a
// gradient in that dtype when the incoming gradient is Float32.
Tensor softmax_backward_bridge(const Tensor& grad_output, const Tensor& output,
                               int64_t dim, DType input_dtype, bool log_mode) {
    Tensor grad = grad_output;
    Tensor out = output;
    const DType compute_dt =
        isReducedFloatingType(input_dtype) ? DType::Float32 : input_dtype;
    if (grad.defined() && grad.dtype() != compute_dt) grad = grad.to(compute_dt);
    if (out.defined() && out.dtype() != compute_dt) out = out.to(compute_dt);
    const std::vector<int64_t> dims{dim};
    Tensor result;
    if (log_mode) {
        // d(x - lse)/dx with lse constant along dim: identity minus the
        // row-wise exponential average of the gradient.
        Tensor sum = grad.sum(dims, /*keepdim=*/true);
        result = grad - ops::exp(out) * sum;
    } else {
        // J^T g with J_ij = p_i (delta_ij - p_j): g_i p_i - p_i (g·p).
        Tensor dot = (grad * out).sum(dims, /*keepdim=*/true);
        result = out * (grad - dot);
    }
    if (grad_output.dtype() == DType::Float32 && input_dtype != DType::Float32 &&
        isReducedFloatingType(input_dtype)) {
        result = result.to(input_dtype);
    }
    return result;
}

}  // namespace

Tensor _softmax_bridge(const Tensor& self, int64_t dim, bool half_to_float) {
    return softmax_data_bridge(self, dim, half_to_float, /*log_mode=*/false);
}

Tensor& _softmax_bridge_out(const Tensor& self, int64_t dim,
                            bool half_to_float, Tensor& out) {
    out = _softmax_bridge(self, dim, half_to_float);
    return out;
}

Tensor _softmax_backward_data_bridge(const Tensor& grad_output,
                                     const Tensor& output, int64_t dim,
                                     DType input_dtype) {
    return softmax_backward_bridge(grad_output, output, dim, input_dtype,
                                    /*log_mode=*/false);
}

Tensor& _softmax_backward_data_bridge_out(const Tensor& grad_output,
                                          const Tensor& output, int64_t dim,
                                          DType input_dtype, Tensor& out) {
    out = _softmax_backward_data_bridge(grad_output, output, dim, input_dtype);
    return out;
}

Tensor _log_softmax_bridge(const Tensor& self, int64_t dim,
                           bool half_to_float) {
    return softmax_data_bridge(self, dim, half_to_float, /*log_mode=*/true);
}

Tensor& _log_softmax_bridge_out(const Tensor& self, int64_t dim,
                                bool half_to_float, Tensor& out) {
    out = _log_softmax_bridge(self, dim, half_to_float);
    return out;
}

Tensor _log_softmax_backward_data_bridge(const Tensor& grad_output,
                                         const Tensor& output, int64_t dim,
                                         DType input_dtype) {
    return softmax_backward_bridge(grad_output, output, dim, input_dtype,
                                    /*log_mode=*/true);
}

Tensor& _log_softmax_backward_data_bridge_out(const Tensor& grad_output,
                                              const Tensor& output,
                                              int64_t dim, DType input_dtype,
                                              Tensor& out) {
    out = _log_softmax_backward_data_bridge(grad_output, output, dim,
                                            input_dtype);
    return out;
}

// ---------------------------------------------------------------------------
// log_sigmoid forward with its saved-buffer output
// ---------------------------------------------------------------------------

// log_sigmoid(x) = -softplus(-x); the buffer caches exp(result) so the
// backward stays elementwise.
std::tuple<Tensor, Tensor> log_sigmoid_forward_bridge(const Tensor& self) {
    Tensor buffer = ops::sigmoid(ops::neg(self));
    Tensor output = ops::log(buffer);
    return {output, buffer};
}

std::tuple<Tensor, Tensor> log_sigmoid_forward_bridge_output(
    const Tensor& self, Tensor& output, Tensor& buffer) {
    std::tie(output, buffer) = log_sigmoid_forward_bridge(self);
    return {output, buffer};
}

// ---------------------------------------------------------------------------
// rrelu_with_noise out / inplace variants
// ---------------------------------------------------------------------------

// The base kernel owns the random-leak selection and the eval-mode midpoint
// slope; the out/inplace variants only steer where the result lands.
Tensor& rrelu_with_noise_bridge_out(const Tensor& self, Tensor& noise,
                                    Scalar lower, Scalar upper, bool training,
                                    std::optional<Generator> generator,
                                    Tensor& out) {
    Tensor result = ops::rrelu_with_noise(self, noise, lower, upper, training);
    out.copy_(result);
    return out;
}

Tensor& rrelu_with_noise_bridge_inplace(Tensor& self, Tensor& noise,
                                        Scalar lower, Scalar upper,
                                        bool training,
                                        std::optional<Generator> generator) {
    Tensor result = ops::rrelu_with_noise(self, noise, lower, upper, training);
    self.copy_(result);
    return self;
}

// ---------------------------------------------------------------------------
// factory .out variants
// ---------------------------------------------------------------------------

// The out= contract takes the destination's dtype/device/grad mode; the
// TensorOptions arguments are accepted for schema compatibility and ignored.
Tensor& arange_bridge_out(Scalar end, DType dtype, Device device,
                          bool requires_grad, Tensor& out) {
    (void)dtype;
    (void)device;
    (void)requires_grad;
    out = ops::arange(end, out.dtype(), std::optional<Device>(out.device()));
    return out;
}

Tensor& linspace_bridge_out(Scalar start, Scalar end, int64_t steps,
                            Tensor& out) {
    out = ops::linspace(start, end, steps, out.dtype(),
                        std::optional<Device>(out.device()));
    return out;
}

Tensor& logspace_bridge_out(Scalar start, Scalar end, int64_t steps,
                            double base, Tensor& out) {
    out = ops::logspace(start, end, steps, base, out.dtype(),
                        std::optional<Device>(out.device()));
    return out;
}

Tensor& eye_bridge_out(int64_t n, Tensor& out) {
    out = ops::eye(n, -1, out.dtype(), std::optional<Device>(out.device()));
    return out;
}

Tensor& eye_bridge_m_out(int64_t n, int64_t m, Tensor& out) {
    out = ops::eye(n, m, out.dtype(), std::optional<Device>(out.device()));
    return out;
}

Tensor& complex_bridge_out(const Tensor& real, const Tensor& imag, Tensor& out) {
    out = ops::complex(real, imag);
    return out;
}

Tensor& polar_bridge_out(const Tensor& abs, const Tensor& angle, Tensor& out) {
    out = ops::polar(abs, angle);
    return out;
}

// ---------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------

TENSORPLAY_LIBRARY_IMPL(Composite, DeadEndBridgeKernels) {
    m.impl("inverse", inverse_bridge);
    m.impl("pinverse", pinverse_bridge);
    m.impl("linalg_vecdot", linalg_vecdot_bridge);
    m.impl("orgqr", orgqr_bridge);
    m.impl("lu_solve", lu_solve_bridge);
    m.impl("grid_sampler", grid_sampler_bridge);
    m.impl("_convolution", _convolution_bridge);
    m.impl("_addmm_activation", _addmm_activation_bridge);

    m.impl("reflection_pad1d", reflection_pad1d_bridge);
    m.impl("reflection_pad2d", reflection_pad2d_bridge);
    m.impl("replication_pad1d", replication_pad1d_bridge);
    m.impl("replication_pad2d", replication_pad2d_bridge);
    m.impl("replication_pad3d", replication_pad3d_bridge);

    m.impl("_softmax", _softmax_bridge);
    m.impl("_softmax.out", _softmax_bridge_out);
    m.impl("_softmax_backward_data", _softmax_backward_data_bridge);
    m.impl("_softmax_backward_data.out", _softmax_backward_data_bridge_out);
    m.impl("_log_softmax", _log_softmax_bridge);
    m.impl("_log_softmax.out", _log_softmax_bridge_out);
    m.impl("_log_softmax_backward_data", _log_softmax_backward_data_bridge);
    m.impl("_log_softmax_backward_data.out",
           _log_softmax_backward_data_bridge_out);

    m.impl("log_sigmoid_forward", log_sigmoid_forward_bridge);
    m.impl("log_sigmoid_forward.output", log_sigmoid_forward_bridge_output);

    m.impl("rrelu_with_noise.out", rrelu_with_noise_bridge_out);
    m.impl("rrelu_with_noise_", rrelu_with_noise_bridge_inplace);

    m.impl("arange.end_out", arange_bridge_out);
    m.impl("linspace.out", linspace_bridge_out);
    m.impl("logspace.out", logspace_bridge_out);
    m.impl("eye.out", eye_bridge_out);
    m.impl("eye.m_out", eye_bridge_m_out);
    m.impl("complex.out", complex_bridge_out);
    m.impl("polar.out", polar_bridge_out);
}

}  // namespace composite
}  // namespace tensorplay

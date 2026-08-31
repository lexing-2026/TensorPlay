// CPU kernels for the extended activation family:
//   - _softmax / _log_softmax (thin wrappers over the row-wise softmax
//     kernels in PointwiseKernels.cpp) and their *_backward_data kernels,
//     which recompute grad_input = output * (grad - sum(grad*output)) for
//     softmax and grad_input = grad - exp(output) * sum(grad) for
//     log_softmax along the reduced dim.
//   - _prelu_kernel: PReLU evaluation with the (pre-shaped) weight
//     broadcast elementwise over the input; negative positions scale the
//     input by the weight, positive positions pass through.
//   - log_sigmoid_forward: computes output = min(x, 0) - log1p(exp(-|x|))
//     together with buffer = exp(-|x|); the branch split keeps exp() bounded
//     for large-magnitude inputs of either sign.  log_sigmoid_backward
//     consumes that buffer: with x the input and b the buffer, the gradient
//     is (max_deriv - sign * b / (1 + b)) * grad, where
//     max_deriv = [x < 0], sign = -[x >= 0] + [x < 0].
//   - rrelu_with_noise out/inplace: training draws a slope per non-positive
//     element from U(lower, upper) on the supplied generator (default
//     generator when none is passed), writes it into noise and scales the
//     element; positive elements record noise = 1 and pass through.  Eval
//     collapses to leaky_relu with the midpoint slope (lower + upper) / 2.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "Parallel.h"
#include "Generator.h"
#include "DistributionsHelper.h"
#include "TensorIteratorOps.h"

#include <cmath>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {

namespace cpu {

// Row-wise softmax kernels live in PointwiseKernels.cpp (same namespace).
Tensor softmax_kernel(const Tensor& self, int64_t dim, DType dtype);
Tensor log_softmax_kernel(const Tensor& self, int64_t dim, DType dtype);
// Leaky ReLU elementwise kernel lives in PointwiseKernels.cpp.
Tensor leaky_relu_kernel_impl(const Tensor& self, Scalar negative_slope);

using tensorplay::parallel::GRAIN_SIZE;
using tensorplay::parallel::parallel_for;

namespace {

// Split a shape into (outer, dim_size, inner) around the reduced dimension.
inline void dim_outer_inner(const std::vector<int64_t>& shape, int64_t dim,
                            int64_t& outer, int64_t& inner) {
    outer = 1;
    inner = 1;
    for (int64_t i = 0; i < dim; ++i) outer *= shape[i];
    for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) inner *= shape[i];
}

template <bool LogMode, typename scalar_t>
void softmax_backward_data_loop(const scalar_t* grad, const scalar_t* out,
                                scalar_t* grad_in, int64_t outer, int64_t inner,
                                int64_t dim_size) {
    const int64_t outer_stride = dim_size * inner;
    parallel_for(0, outer * inner, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t o = i / inner;
            const int64_t in = i % inner;
            const int64_t base = o * outer_stride + in;
            const scalar_t* gcol = grad + base;
            const scalar_t* ocol = out + base;
            scalar_t* rcol = grad_in + base;
            if constexpr (LogMode) {
                scalar_t sum = scalar_t(0);
                for (int64_t k = 0; k < dim_size; ++k) sum += gcol[k * inner];
                for (int64_t k = 0; k < dim_size; ++k) {
                    rcol[k * inner] =
                        gcol[k * inner] - std::exp(ocol[k * inner]) * sum;
                }
            } else {
                scalar_t sum = scalar_t(0);
                for (int64_t k = 0; k < dim_size; ++k) {
                    sum += gcol[k * inner] * ocol[k * inner];
                }
                for (int64_t k = 0; k < dim_size; ++k) {
                    rcol[k * inner] =
                        ocol[k * inner] * (gcol[k * inner] - sum);
                }
            }
        }
    });
}

// Shared body of the two backward-data kernels; log_mode selects the
// log_softmax formula.  Reduced-width dtypes accumulate in float.
Tensor softmax_backward_data_core(const Tensor& grad_output, const Tensor& output,
                                  int64_t dim, bool log_mode) {
    Tensor g = grad_output.dim() == 0 ? grad_output.view({1}) : grad_output;
    Tensor o = output.dim() == 0 ? output.view({1}) : output;
    const int64_t nd = g.dim();
    const int64_t d = dim < 0 ? dim + nd : dim;
    if (d < 0 || d >= nd) {
        TP_THROW(IndexError,
                 "dim must be non-negative and less than input dimensions");
    }
    const DType compute_dt =
        isReducedFloatingType(g.dtype()) ? DType::Float32 : g.dtype();
    if (compute_dt != DType::Float32 && compute_dt != DType::Float64) {
        TP_THROW(TypeError, "unsupported dtype for softmax backward");
    }
    if (o.dtype() != compute_dt) o = o.to(compute_dt);
    if (g.dtype() != compute_dt) g = g.to(compute_dt);

    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(g.shape()),
                                  compute_dt, g.device());
    if (g.numel() == 0) return result;

    const Tensor gc = g.contiguous();
    const Tensor oc = o.contiguous();
    const std::vector<int64_t> shape =
        static_cast<std::vector<int64_t>>(gc.shape());
    int64_t outer = 1, inner = 1;
    dim_outer_inner(shape, d, outer, inner);
    const int64_t dim_size = shape[d];

    if (compute_dt == DType::Float32) {
        if (log_mode) {
            softmax_backward_data_loop<true, float>(
                gc.data_ptr<float>(), oc.data_ptr<float>(),
                result.data_ptr<float>(), outer, inner, dim_size);
        } else {
            softmax_backward_data_loop<false, float>(
                gc.data_ptr<float>(), oc.data_ptr<float>(),
                result.data_ptr<float>(), outer, inner, dim_size);
        }
    } else {
        if (log_mode) {
            softmax_backward_data_loop<true, double>(
                gc.data_ptr<double>(), oc.data_ptr<double>(),
                result.data_ptr<double>(), outer, inner, dim_size);
        } else {
            softmax_backward_data_loop<false, double>(
                gc.data_ptr<double>(), oc.data_ptr<double>(),
                result.data_ptr<double>(), outer, inner, dim_size);
        }
    }
    return result;
}

// The half_to_float flag only exists for accelerators; on CPU the output
// dtype of the backward pass falls back to the grad dtype (or Half when the
// original input was Half and the grad came back Float32).
inline DType softmax_backward_out_dtype(const Tensor& grad_output,
                                        DType input_dtype) {
    DType out_dtype = grad_output.dtype();
    if (out_dtype != input_dtype && out_dtype == DType::Float32 &&
        input_dtype == DType::Float16) {
        out_dtype = DType::Float16;
    }
    return out_dtype;
}

template <typename scalar_t, typename opmath_t>
void log_sigmoid_forward_loop(const scalar_t* in, scalar_t* out, scalar_t* buf,
                              int64_t n) {
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const opmath_t x = static_cast<opmath_t>(in[i]);
            const opmath_t buffer = std::exp(-std::abs(x));
            const opmath_t clipped = x < opmath_t(0) ? x : opmath_t(0);
            buf[i] = static_cast<scalar_t>(buffer);
            out[i] = static_cast<scalar_t>(clipped - std::log1p(buffer));
        }
    });
}

template <typename scalar_t, typename opmath_t>
void log_sigmoid_backward_loop(const scalar_t* grad, const scalar_t* input,
                               const scalar_t* buffer, scalar_t* grad_in,
                               int64_t n) {
    parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const opmath_t x = static_cast<opmath_t>(input[i]);
            const opmath_t b = static_cast<opmath_t>(buffer[i]);
            const opmath_t c = static_cast<opmath_t>(grad[i]);
            const bool in_negative = x < opmath_t(0);
            const opmath_t max_deriv = in_negative ? opmath_t(1) : opmath_t(0);
            const opmath_t sign = in_negative ? opmath_t(1) : opmath_t(-1);
            grad_in[i] = static_cast<scalar_t>(
                (max_deriv - sign * (b / (opmath_t(1) + b))) * c);
        }
    });
}

// Training-mode randomized ReLU: one U(lower, upper) draw per non-positive
// element, consumed from the generator's stream in element order (serial,
// because each draw advances the shared RNG state).
template <typename scalar_t, typename opmath_t>
void rrelu_with_noise_train_draw(const Tensor& input, Tensor& noise,
                                 Tensor& result, double lower, double upper,
                                 Generator& gen) {
    const scalar_t* in_data = input.data_ptr<scalar_t>();
    scalar_t* out_data = result.data_ptr<scalar_t>();
    scalar_t* noise_data = noise.data_ptr<scalar_t>();
    uniform_real_distribution<double> uniform(lower, upper);
    const int64_t n = input.numel();
    for (int64_t i = 0; i < n; ++i) {
        if (static_cast<opmath_t>(in_data[i]) <= opmath_t(0)) {
            const opmath_t r = static_cast<opmath_t>(uniform(&gen));
            out_data[i] =
                static_cast<scalar_t>(static_cast<opmath_t>(in_data[i]) * r);
            noise_data[i] = static_cast<scalar_t>(r);
        } else {
            noise_data[i] = scalar_t(1);
            out_data[i] = in_data[i];
        }
    }
}

Tensor rrelu_with_noise_train_cpu(const Tensor& input, Tensor& noise,
                                  Scalar lower, Scalar upper,
                                  std::optional<Generator>& generator) {
    TP_CHECK(noise.is_contiguous(),
             "rrelu_with_noise: noise tensor must be contiguous");
    const Tensor input_c = input.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(input_c.shape()),
                                  input_c.dtype(), input_c.device());
    if (input_c.numel() > 0) {
        Generator& gen =
            generator.has_value() ? *generator : default_generator();
        const double lo = lower.toDouble();
        const double hi = upper.toDouble();
        switch (input_c.dtype()) {
            case DType::Float32:
                rrelu_with_noise_train_draw<float, float>(
                    input_c, noise, result, lo, hi, gen);
                break;
            case DType::Float64:
                rrelu_with_noise_train_draw<double, double>(
                    input_c, noise, result, lo, hi, gen);
                break;
            case DType::Float16:
                rrelu_with_noise_train_draw<Half, float>(
                    input_c, noise, result, lo, hi, gen);
                break;
            case DType::BFloat16:
                rrelu_with_noise_train_draw<BFloat16, float>(
                    input_c, noise, result, lo, hi, gen);
                break;
            default:
                TP_THROW(TypeError,
                         "rrelu_with_noise: only floating dtypes are supported");
        }
    }
    return result;
}

Tensor rrelu_with_noise_core(const Tensor& self, Tensor& noise, Scalar lower,
                             Scalar upper, bool training,
                             std::optional<Generator>& generator) {
    if (training) {
        return rrelu_with_noise_train_cpu(self, noise, lower, upper, generator);
    }
    // Eval mode has no randomness: the expected slope is the midpoint.
    const double negative = (lower.toDouble() + upper.toDouble()) / 2.0;
    return leaky_relu_kernel_impl(self, Scalar(negative));
}

}  // namespace

// ---------------------------------------------------------------------------
// _softmax / _log_softmax (+ out)
// ---------------------------------------------------------------------------

Tensor _softmax_cpu(const Tensor& self, int64_t dim, bool half_to_float) {
    if (half_to_float) {
        TP_THROW(RuntimeError,
                 "softmax with half to float conversion is not supported on CPU");
    }
    return softmax_kernel(self, dim, self.dtype());
}

Tensor& _softmax_out_cpu(const Tensor& self, int64_t dim, bool half_to_float,
                         Tensor& out) {
    out = _softmax_cpu(self, dim, half_to_float);
    return out;
}

Tensor _log_softmax_cpu(const Tensor& self, int64_t dim, bool half_to_float) {
    if (half_to_float) {
        TP_THROW(RuntimeError,
                 "log_softmax with half to float conversion is not supported on CPU");
    }
    return log_softmax_kernel(self, dim, self.dtype());
}

Tensor& _log_softmax_out_cpu(const Tensor& self, int64_t dim, bool half_to_float,
                             Tensor& out) {
    out = _log_softmax_cpu(self, dim, half_to_float);
    return out;
}

// ---------------------------------------------------------------------------
// _softmax_backward_data / _log_softmax_backward_data (+ out)
// ---------------------------------------------------------------------------

Tensor _softmax_backward_data_cpu(const Tensor& grad_output, const Tensor& output,
                                  int64_t dim, DType input_dtype) {
    Tensor result = softmax_backward_data_core(grad_output, output, dim, false);
    const DType out_dtype = softmax_backward_out_dtype(grad_output, input_dtype);
    if (result.dtype() != out_dtype) result = result.to(out_dtype);
    return result;
}

Tensor& _softmax_backward_data_out_cpu(const Tensor& grad_output,
                                       const Tensor& output, int64_t dim,
                                       DType input_dtype, Tensor& grad_input) {
    grad_input =
        _softmax_backward_data_cpu(grad_output, output, dim, input_dtype);
    return grad_input;
}

Tensor _log_softmax_backward_data_cpu(const Tensor& grad_output,
                                      const Tensor& output, int64_t dim,
                                      DType input_dtype) {
    Tensor result = softmax_backward_data_core(grad_output, output, dim, true);
    const DType out_dtype = softmax_backward_out_dtype(grad_output, input_dtype);
    if (result.dtype() != out_dtype) result = result.to(out_dtype);
    return result;
}

Tensor& _log_softmax_backward_data_out_cpu(const Tensor& grad_output,
                                           const Tensor& output, int64_t dim,
                                           DType input_dtype, Tensor& grad_input) {
    grad_input =
        _log_softmax_backward_data_cpu(grad_output, output, dim, input_dtype);
    return grad_input;
}

// ---------------------------------------------------------------------------
// _prelu_kernel
// ---------------------------------------------------------------------------

Tensor _prelu_kernel_cpu(const Tensor& self, const Tensor& weight_in) {
    // The caller pre-shapes the weight (rank equal to self, channel dim at
    // position 1); broadcasting handles the trailing singleton dims.
    Tensor weight =
        weight_in.dtype() == self.dtype() ? weight_in : weight_in.to(self.dtype());
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                                  self.dtype(), self.device());
    ti_apply_binary(result, self, weight, [](auto x, auto w) {
        using T = decltype(x);
        return x > static_cast<T>(0) ? x : static_cast<T>(w * x);
    });
    return result;
}

// ---------------------------------------------------------------------------
// log_sigmoid_forward (+ out) and log_sigmoid_backward (+ out)
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> log_sigmoid_forward_cpu(const Tensor& input) {
    const Tensor input_c = input.contiguous();
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(input_c.shape()),
                                  input_c.dtype(), input_c.device());
    Tensor buffer = Tensor::empty(static_cast<std::vector<int64_t>>(input_c.shape()),
                                  input_c.dtype(), input_c.device());
    const int64_t n = input_c.numel();
    if (n > 0) {
        switch (input_c.dtype()) {
            case DType::Float32:
                log_sigmoid_forward_loop<float, float>(
                    input_c.data_ptr<float>(), result.data_ptr<float>(),
                    buffer.data_ptr<float>(), n);
                break;
            case DType::Float64:
                log_sigmoid_forward_loop<double, double>(
                    input_c.data_ptr<double>(), result.data_ptr<double>(),
                    buffer.data_ptr<double>(), n);
                break;
            case DType::Float16:
                log_sigmoid_forward_loop<Half, float>(
                    input_c.data_ptr<Half>(), result.data_ptr<Half>(),
                    buffer.data_ptr<Half>(), n);
                break;
            case DType::BFloat16:
                log_sigmoid_forward_loop<BFloat16, float>(
                    input_c.data_ptr<BFloat16>(), result.data_ptr<BFloat16>(),
                    buffer.data_ptr<BFloat16>(), n);
                break;
            default:
                TP_THROW(TypeError, "log_sigmoid_forward: unsupported dtype");
        }
    }
    return {result, buffer};
}

std::tuple<Tensor, Tensor> log_sigmoid_forward_out_cpu(const Tensor& input,
                                                       Tensor& result,
                                                       Tensor& buffer) {
    std::tie(result, buffer) = log_sigmoid_forward_cpu(input);
    return {result, buffer};
}

Tensor& log_sigmoid_backward_out_cpu(const Tensor& grad_output,
                                     const Tensor& input, const Tensor& buffer,
                                     Tensor& grad_input) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()),
                                  grad_output.dtype(), grad_output.device());
    const int64_t n = grad_output.numel();
    if (n > 0) {
        const Tensor gc = grad_output.contiguous();
        const Tensor ic = input.to(grad_output.dtype()).contiguous();
        const Tensor bc = buffer.to(grad_output.dtype()).contiguous();
        switch (grad_output.dtype()) {
            case DType::Float32:
                log_sigmoid_backward_loop<float, float>(
                    gc.data_ptr<float>(), ic.data_ptr<float>(),
                    bc.data_ptr<float>(), result.data_ptr<float>(), n);
                break;
            case DType::Float64:
                log_sigmoid_backward_loop<double, double>(
                    gc.data_ptr<double>(), ic.data_ptr<double>(),
                    bc.data_ptr<double>(), result.data_ptr<double>(), n);
                break;
            case DType::Float16:
                log_sigmoid_backward_loop<Half, float>(
                    gc.data_ptr<Half>(), ic.data_ptr<Half>(),
                    bc.data_ptr<Half>(), result.data_ptr<Half>(), n);
                break;
            case DType::BFloat16:
                log_sigmoid_backward_loop<BFloat16, float>(
                    gc.data_ptr<BFloat16>(), ic.data_ptr<BFloat16>(),
                    bc.data_ptr<BFloat16>(), result.data_ptr<BFloat16>(), n);
                break;
            default:
                TP_THROW(TypeError, "log_sigmoid_backward: unsupported dtype");
        }
    }
    grad_input = result;
    return grad_input;
}

// ---------------------------------------------------------------------------
// rrelu_with_noise (+ out / inplace)
// ---------------------------------------------------------------------------

Tensor& rrelu_with_noise_out_cpu(const Tensor& self, Tensor& noise,
                                 Scalar lower, Scalar upper, bool training,
                                 std::optional<Generator> generator,
                                 Tensor& output) {
    TP_CHECK(self.shape() == noise.shape(),
             "noise tensor shape must match self tensor shape. Got self.shape = ",
             self.shape(), " noise.shape = ", noise.shape());
    output = rrelu_with_noise_core(self, noise, lower, upper, training, generator);
    return output;
}

Tensor& rrelu_with_noise__cpu(Tensor& self, Tensor& noise, Scalar lower,
                              Scalar upper, bool training,
                              std::optional<Generator> generator) {
    TP_CHECK(self.shape() == noise.shape(),
             "noise tensor shape must match self tensor shape. Got self.shape = ",
             self.shape(), " noise.shape = ", noise.shape());
    Tensor result =
        rrelu_with_noise_core(self, noise, lower, upper, training, generator);
    self.copy_(result);
    return self;
}

TENSORPLAY_LIBRARY_IMPL(CPU, ActivationMoreOps) {
    m.impl("_softmax", _softmax_cpu);
    m.impl("_softmax.out", _softmax_out_cpu);
    m.impl("_softmax_backward_data", _softmax_backward_data_cpu);
    m.impl("_softmax_backward_data.out", _softmax_backward_data_out_cpu);
    m.impl("_log_softmax", _log_softmax_cpu);
    m.impl("_log_softmax.out", _log_softmax_out_cpu);
    m.impl("_log_softmax_backward_data", _log_softmax_backward_data_cpu);
    m.impl("_log_softmax_backward_data.out", _log_softmax_backward_data_out_cpu);
    m.impl("_prelu_kernel", _prelu_kernel_cpu);
    m.impl("log_sigmoid_forward", log_sigmoid_forward_cpu);
    m.impl("log_sigmoid_forward.output", log_sigmoid_forward_out_cpu);
    m.impl("log_sigmoid_backward.grad_input", log_sigmoid_backward_out_cpu);
    m.impl("rrelu_with_noise.out", rrelu_with_noise_out_cpu);
    m.impl("rrelu_with_noise_", rrelu_with_noise__cpu);
}

}  // namespace cpu
}  // namespace tensorplay

#include "QuantTensorOperators.h"
// CPU kernels for the extended activation family:
//   - _softmax / _log_softmax (thin wrappers over the row-wise softmax
//     kernels in PointwiseKernels.cpp) and their *_backward_data kernels,
//     which recompute grad_input = output * (grad - sum(grad*output)) for
//     softmax and grad_input = grad - exp(output) * sum(grad) for
//     log_softmax along the reduced dim.
//   - _prelu_kernel: PReLU evaluation with the (pre-shaped) weight
//     broadcast elementwise over the input; negative positions scale the
//     input by the weight, positive positions pass through.
//     _prelu_kernel_backward walks the same broadcast once and emits both
//     gradients: grad_input = x > 0 ? g : w * g and, per input element,
//     grad_weight = x > 0 ? 0 : x * g.
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
#include "OpMathType.h"
#include "Utils.h"

#include <cmath>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>
#include "OutWrite.h"

namespace tensorplay {

namespace cpu {

// Row-wise softmax kernels live in PointwiseKernels.cpp (same namespace).
Tensor softmax_kernel(const Tensor& self, int64_t dim, DType dtype);
Tensor log_softmax_kernel(const Tensor& self, int64_t dim, DType dtype);
// Leaky ReLU elementwise kernel lives in PointwiseKernels.cpp.
Tensor leaky_relu_kernel_impl(const Tensor& self, const Scalar& negative_slope);
// Its destination-taking form, which writes the caller's tensor in one pass
// wherever that map can walk it.
Tensor& leaky_relu_out_cpu(const Tensor& self, const Scalar& negative_slope,
                           Tensor& out);

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
    // One task walks a whole column, so the grain counts the columns a chunk
    // covers rather than the elements it touches.
    const int64_t grain = std::max<int64_t>(
        1, GRAIN_SIZE / std::max<int64_t>(dim_size * inner, 1));
    parallel_for(0, outer * inner, grain, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t o = i / inner;
            const int64_t in = i % inner;
            const int64_t base = o * outer_stride + in;
            const scalar_t* gcol = grad + base;
            const scalar_t* ocol = out + base;
            scalar_t* rcol = grad_in + base;
            if (inner == 1) {
                // A unit stride lets the sum fold several elements at once,
                // while a runtime stride keeps it to one element per step.
                if constexpr (LogMode) {
                    scalar_t s0 = scalar_t(0), s1 = scalar_t(0);
                    scalar_t s2 = scalar_t(0), s3 = scalar_t(0);
                    int64_t k = 0;
                    for (; k + 3 < dim_size; k += 4) {
                        s0 += gcol[k + 0];
                        s1 += gcol[k + 1];
                        s2 += gcol[k + 2];
                        s3 += gcol[k + 3];
                    }
                    scalar_t sum = (s0 + s1) + (s2 + s3);
                    for (; k < dim_size; ++k) sum += gcol[k];
                    for (int64_t j = 0; j < dim_size; ++j) {
                        rcol[j] = gcol[j] - std::exp(ocol[j]) * sum;
                    }
                } else {
                    scalar_t s0 = scalar_t(0), s1 = scalar_t(0);
                    scalar_t s2 = scalar_t(0), s3 = scalar_t(0);
                    int64_t k = 0;
                    for (; k + 3 < dim_size; k += 4) {
                        s0 += gcol[k + 0] * ocol[k + 0];
                        s1 += gcol[k + 1] * ocol[k + 1];
                        s2 += gcol[k + 2] * ocol[k + 2];
                        s3 += gcol[k + 3] * ocol[k + 3];
                    }
                    scalar_t sum = (s0 + s1) + (s2 + s3);
                    for (; k < dim_size; ++k) sum += gcol[k] * ocol[k];
                    for (int64_t j = 0; j < dim_size; ++j) {
                        rcol[j] = ocol[j] * (gcol[j] - sum);
                    }
                }
                continue;
            }
            if constexpr (LogMode) {
                // The sum runs over four accumulators: one chain of additions
                // would keep the walk from folding several elements at once.
                scalar_t s0 = scalar_t(0), s1 = scalar_t(0);
                scalar_t s2 = scalar_t(0), s3 = scalar_t(0);
                int64_t k = 0;
                for (; k + 3 < dim_size; k += 4) {
                    s0 += gcol[(k + 0) * inner];
                    s1 += gcol[(k + 1) * inner];
                    s2 += gcol[(k + 2) * inner];
                    s3 += gcol[(k + 3) * inner];
                }
                scalar_t sum = (s0 + s1) + (s2 + s3);
                for (; k < dim_size; ++k) sum += gcol[k * inner];
                for (int64_t j = 0; j < dim_size; ++j) {
                    rcol[j * inner] =
                        gcol[j * inner] - std::exp(ocol[j * inner]) * sum;
                }
            } else {
                scalar_t s0 = scalar_t(0), s1 = scalar_t(0);
                scalar_t s2 = scalar_t(0), s3 = scalar_t(0);
                int64_t k = 0;
                for (; k + 3 < dim_size; k += 4) {
                    s0 += gcol[(k + 0) * inner] * ocol[(k + 0) * inner];
                    s1 += gcol[(k + 1) * inner] * ocol[(k + 1) * inner];
                    s2 += gcol[(k + 2) * inner] * ocol[(k + 2) * inner];
                    s3 += gcol[(k + 3) * inner] * ocol[(k + 3) * inner];
                }
                scalar_t sum = (s0 + s1) + (s2 + s3);
                for (; k < dim_size; ++k) {
                    sum += gcol[k * inner] * ocol[k * inner];
                }
                for (int64_t j = 0; j < dim_size; ++j) {
                    rcol[j * inner] =
                        ocol[j * inner] * (gcol[j * inner] - sum);
                }
            }
        }
    });
}

// The width the column walk itself runs at: a gradient narrower than single
// precision is computed at single precision and narrowed to the answer afterwards.
inline DType softmax_backward_walk_dtype(DType grad_dtype) {
    return isReducedFloatingType(grad_dtype) ? DType::Float32 : grad_dtype;
}

// Shared body of the two backward-data kernels; log_mode selects the
// log_softmax formula.  Reduced-width dtypes accumulate in float.  A caller that
// has already judged its own destination hands it over, so the columns are stored
// where the values have to end up instead of being copied there.
Tensor softmax_backward_data_core(const Tensor& grad_output, const Tensor& output,
                                  int64_t dim, bool log_mode,
                                  const Tensor* destination = nullptr) {
    Tensor g = grad_output.dim() == 0 ? grad_output.view({1}) : grad_output;
    Tensor o = output.dim() == 0 ? output.view({1}) : output;
    const int64_t nd = g.dim();
    const int64_t d = dim < 0 ? dim + nd : dim;
    if (d < 0 || d >= nd) {
        TP_THROW(IndexError,
                 "dim must be non-negative and less than input dimensions");
    }
    const DType compute_dt = softmax_backward_walk_dtype(g.dtype());
    if (compute_dt != DType::Float32 && compute_dt != DType::Float64) {
        TP_THROW(TypeError, "unsupported dtype for softmax backward");
    }
    if (o.dtype() != compute_dt) o = o.to(compute_dt);
    if (g.dtype() != compute_dt) g = g.to(compute_dt);

    Tensor result = destination != nullptr
        ? *destination
        : Tensor::empty(static_cast<std::vector<int64_t>>(g.shape()), compute_dt,
                        g.device());
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

// Whether the column walk can answer into the gradient the caller holds.  It
// reads the gradient and the saved output at one offset and stores the answer at
// that same offset, so the destination has to be packed with the shape those two
// are walked at and of the walk's own width -- a gradient narrower than single
// precision is computed wide and narrowed afterwards, which is a second pass over
// other storage.  Each operand only collides with the destination when it is
// reached inside the destination's own block: the walk sums a column before it
// stores into it, then reads and writes that column element by element.
inline bool softmax_backward_destination_takes(const Tensor& grad_output,
                                               const Tensor& output, DType want,
                                               const Tensor& grad_input) {
    const DType compute_dt = softmax_backward_walk_dtype(grad_output.dtype());
    if (want != compute_dt || grad_output.dim() == 0 ||
        (compute_dt != DType::Float32 && compute_dt != DType::Float64)) {
        return false;
    }
    const auto shape = static_cast<std::vector<int64_t>>(grad_output.shape());
    if (!destination_takes(grad_input, grad_output.device(), compute_dt, shape) ||
        !grad_input.is_contiguous()) {
        return false;
    }
    if (grad_output.is_contiguous() &&
        !read_while_write_is_safe(grad_output, grad_input)) {
        return false;
    }
    // An operand of another width is materialised at the walk's own before the
    // walk sees it, so it cannot be a byte of the destination.
    return !output.is_contiguous() || output.dtype() != compute_dt ||
           read_while_write_is_safe(output, grad_input);
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

// The factors are stored at the input's own width, so a noise of another one
// would be written through a buffer of the wrong element size.
Tensor rrelu_with_noise_train_cpu(const Tensor& input, Tensor& noise,
                                  Scalar lower, Scalar upper,
                                  std::optional<Generator>& generator,
                                  const Tensor* destination = nullptr) {
    TP_CHECK(noise.is_contiguous(),
             "rrelu_with_noise: noise tensor must be contiguous");
    TP_CHECK(noise.dtype() == input.dtype(),
             "rrelu_with_noise: noise tensor must have the same dtype as the "
             "input, got noise.dtype = ", toString(noise.dtype()),
             " input.dtype = ", toString(input.dtype()));
    const Tensor input_c = input.contiguous();
    Tensor result = destination != nullptr
        ? *destination
        : Tensor::empty(static_cast<std::vector<int64_t>>(input_c.shape()),
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

// Whether two tensors are filled from separate blocks of storage.
inline bool stored_apart(const Tensor& first, const Tensor& second) {
    const auto first_impl = first.unsafeGetTensorImpl();
    const auto second_impl = second.unsafeGetTensorImpl();
    if (!first_impl || !second_impl) return false;
    const auto& first_storage = first_impl->storage();
    const auto& second_storage = second_impl->storage();
    return first_storage.defined() && second_storage.defined() &&
           first_storage.data() != second_storage.data();
}

// Whether the draw can answer into the pair the caller holds.  It walks the
// input in element order and lays two stores beside each element it reads: the
// answer at element i of its own tensor, the factor at element i of the noise.
// Both therefore have to be packed at the input's shape and width.  The answer
// may be the input itself, whose element i is read before the store lands on it,
// but the noise may not share the input's block: for a positive element the
// factor is laid down first, and the answer reads that element afterwards.  Nor
// may answer and noise be one block -- the two stores then land in whichever
// order the branch took, while the run that hands them over fills the noise
// first and the answer last.
inline bool rrelu_with_noise_destinations_take(const Tensor& input,
                                               Tensor& noise,
                                               const Tensor& output) {
    const auto shape = static_cast<std::vector<int64_t>>(input.shape());
    const DType want = input.dtype();
    if (!destination_takes_flat(output, input.device(), want, shape, input) ||
        !destination_takes(noise, input.device(), want, shape) ||
        !noise.is_contiguous()) {
        return false;
    }
    return stored_apart(input, noise) && stored_apart(noise, output);
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
// _softmax / _log_softmax
// ---------------------------------------------------------------------------

Tensor _softmax_cpu(const Tensor& self, int64_t dim, bool half_to_float) {
    if (half_to_float) {
        TP_THROW(RuntimeError,
                 "softmax with half to float conversion is not supported on CPU");
    }
    return softmax_kernel(self, dim, self.dtype());
}

Tensor _log_softmax_cpu(const Tensor& self, int64_t dim, bool half_to_float) {
    if (half_to_float) {
        TP_THROW(RuntimeError,
                 "log_softmax with half to float conversion is not supported on CPU");
    }
    return log_softmax_kernel(self, dim, self.dtype());
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
    const DType want = softmax_backward_out_dtype(grad_output, input_dtype);
    if (softmax_backward_destination_takes(grad_output, output, want, grad_input)) {
        softmax_backward_data_core(grad_output, output, dim, false, &grad_input);
        return grad_input;
    }
    if (want != softmax_backward_walk_dtype(grad_output.dtype()) &&
        grad_input.dtype() == want) {
        // The walk runs wide and the answer comes narrow: let the destination take
        // that narrowing directly, rather than meeting a copy of a tensor the map
        // narrowed for it.
        return write_out(grad_input,
                         softmax_backward_data_core(grad_output, output, dim, false));
    }
    return write_out(grad_input,
                     _softmax_backward_data_cpu(grad_output, output, dim, input_dtype));
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
    const DType want = softmax_backward_out_dtype(grad_output, input_dtype);
    if (softmax_backward_destination_takes(grad_output, output, want, grad_input)) {
        softmax_backward_data_core(grad_output, output, dim, true, &grad_input);
        return grad_input;
    }
    if (want != softmax_backward_walk_dtype(grad_output.dtype()) &&
        grad_input.dtype() == want) {
        // The walk runs wide and the answer comes narrow: let the destination take
        // that narrowing directly, rather than meeting a copy of a tensor the map
        // narrowed for it.
        return write_out(grad_input,
                         softmax_backward_data_core(grad_output, output, dim, true));
    }
    return write_out(grad_input,
                     _log_softmax_backward_data_cpu(grad_output, output, dim,
                                                    input_dtype));
}

// ---------------------------------------------------------------------------
// _prelu_kernel
// ---------------------------------------------------------------------------

Tensor _prelu_kernel_cpu(const Tensor& self, const Tensor& weight_in) {
    if (isQuantizedType(self.dtype())) return quantized::prelu(self, weight_in);

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
// _prelu_kernel_backward
// ---------------------------------------------------------------------------

namespace {

// One fused pass over the broadcast of (self, weight, grad_output) writing
// both gradients:
//     grad_self   = x > 0 ? g : w * g
//     grad_weight = x > 0 ? 0 : x * g
// grad_weight keeps the broadcast geometry; folding it onto the weight shape
// belongs to the caller.  Sub-single-precision inputs evaluate in their math
// type so the product rounds once.
template <typename scalar_t>
void prelu_backward_loop(TensorIterator& iter) {
    using math_t = typename OpMathType<scalar_t>::type;
    iter.for_each([](char** data, const int64_t* strides, int64_t n) {
        char* gi = data[0];
        char* gw = data[1];
        const char* xs = data[2];
        const char* ws = data[3];
        const char* gs = data[4];
        for (int64_t i = 0; i < n; ++i) {
            const math_t x = static_cast<math_t>(
                *reinterpret_cast<const scalar_t*>(xs + i * strides[2]));
            const math_t w = static_cast<math_t>(
                *reinterpret_cast<const scalar_t*>(ws + i * strides[3]));
            const math_t g = static_cast<math_t>(
                *reinterpret_cast<const scalar_t*>(gs + i * strides[4]));
            const bool positive = x > math_t(0);
            *reinterpret_cast<scalar_t*>(gi + i * strides[0]) =
                static_cast<scalar_t>(positive ? g : w * g);
            *reinterpret_cast<scalar_t*>(gw + i * strides[1]) =
                static_cast<scalar_t>(positive ? math_t(0) : x * g);
        }
    });
}

}  // namespace

std::tuple<Tensor, Tensor> _prelu_kernel_backward_cpu(const Tensor& grad_output,
                                                      const Tensor& self,
                                                      const Tensor& weight) {
    TP_CHECK(self.dtype() == weight.dtype() &&
                 self.dtype() == grad_output.dtype(),
             "_prelu_kernel_backward: input, weight and grad_output must share "
             "one dtype");

    const std::vector<int64_t> out_shape = broadcast_shapes(self.sizes(), weight.sizes(), grad_output.sizes());
    Tensor grad_self = Tensor::empty(out_shape, self.dtype(), self.device());
    Tensor grad_weight = Tensor::empty(out_shape, weight.dtype(), weight.device());
    if (grad_self.numel() == 0) {
        return {grad_self, grad_weight};
    }

    TensorIterator iter = TensorIteratorConfig()
        .resize_outputs(false)
        .add_output(grad_self)
        .add_output(grad_weight)
        .add_const_input(self)
        .add_const_input(weight)
        .add_const_input(grad_output)
        .build();

    switch (self.dtype()) {
        case DType::Float32: prelu_backward_loop<float>(iter); break;
        case DType::Float64: prelu_backward_loop<double>(iter); break;
        case DType::Float16: prelu_backward_loop<Half>(iter); break;
        case DType::BFloat16: prelu_backward_loop<BFloat16>(iter); break;
        default:
            TP_THROW(TypeError, "_prelu_kernel_backward: unsupported dtype");
    }
    return {grad_self, grad_weight};
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
    const Tensor input_c = input.contiguous();
    const auto shape = static_cast<std::vector<int64_t>>(input_c.shape());
    const DType want = input_c.dtype();
    const int64_t n = input_c.numel();
    // The run stores element i of the answer beside element i of the saved
    // exponential, so it is offered to the two destinations together: either one
    // of them still needing a tensor of the op's own sends both answers through
    // the hand-over, since a single run cannot fill one and leave the other.
    Tensor answer, saved;
    Tensor* answer_to = &result;
    Tensor* saved_to = &buffer;
    if (n == 0 ||
        !destination_takes_flat(result, input_c.device(), want, shape, input_c) ||
        !destination_takes_flat(buffer, input_c.device(), want, shape, input_c)) {
        answer = Tensor::empty(shape, want, input_c.device());
        saved = Tensor::empty(shape, want, input_c.device());
        answer_to = &answer;
        saved_to = &saved;
    }
    if (n > 0) {
        switch (want) {
            case DType::Float32:
                log_sigmoid_forward_loop<float, float>(
                    input_c.data_ptr<float>(), answer_to->data_ptr<float>(),
                    saved_to->data_ptr<float>(), n);
                break;
            case DType::Float64:
                log_sigmoid_forward_loop<double, double>(
                    input_c.data_ptr<double>(), answer_to->data_ptr<double>(),
                    saved_to->data_ptr<double>(), n);
                break;
            case DType::Float16:
                log_sigmoid_forward_loop<Half, float>(
                    input_c.data_ptr<Half>(), answer_to->data_ptr<Half>(),
                    saved_to->data_ptr<Half>(), n);
                break;
            case DType::BFloat16:
                log_sigmoid_forward_loop<BFloat16, float>(
                    input_c.data_ptr<BFloat16>(),
                    answer_to->data_ptr<BFloat16>(),
                    saved_to->data_ptr<BFloat16>(), n);
                break;
            default:
                TP_THROW(TypeError, "log_sigmoid_forward: unsupported dtype");
        }
    }
    if (answer_to != &result) write_out(result, answer);
    if (saved_to != &buffer) write_out(buffer, saved);
    return {result, buffer};
}

Tensor& log_sigmoid_backward_out_cpu(const Tensor& grad_output,
                                     const Tensor& input, const Tensor& buffer,
                                     Tensor& grad_input) {
    const DType want = grad_output.dtype();
    const auto shape = static_cast<std::vector<int64_t>>(grad_output.shape());
    const int64_t n = grad_output.numel();
    if (n == 0) {
        write_out(grad_input, Tensor::empty(shape, want, grad_output.device()));
        return grad_input;
    }
    const Tensor gc = grad_output.contiguous();
    const Tensor ic = input.to(want).contiguous();
    const Tensor bc = buffer.to(want).contiguous();
    // The run reads the three materialised operands at the gradient's width and
    // stores element i in order, so a packed destination of that same width,
    // shape and device -- and none of those three reached by another step from
    // it -- takes the pass.  Anything else is answered into a tensor of the op's
    // own and handed over.
    Tensor result;
    Tensor* dest = &grad_input;
    if (!destination_takes_flat(grad_input, grad_output.device(), want, shape,
                                gc, ic) ||
        !read_while_write_is_safe(bc, grad_input)) {
        result = Tensor::empty(shape, want, grad_output.device());
        dest = &result;
    }
    switch (want) {
        case DType::Float32:
            log_sigmoid_backward_loop<float, float>(
                gc.data_ptr<float>(), ic.data_ptr<float>(),
                bc.data_ptr<float>(), dest->data_ptr<float>(), n);
            break;
        case DType::Float64:
            log_sigmoid_backward_loop<double, double>(
                gc.data_ptr<double>(), ic.data_ptr<double>(),
                bc.data_ptr<double>(), dest->data_ptr<double>(), n);
            break;
        case DType::Float16:
            log_sigmoid_backward_loop<Half, float>(
                gc.data_ptr<Half>(), ic.data_ptr<Half>(),
                bc.data_ptr<Half>(), dest->data_ptr<Half>(), n);
            break;
        case DType::BFloat16:
            log_sigmoid_backward_loop<BFloat16, float>(
                gc.data_ptr<BFloat16>(), ic.data_ptr<BFloat16>(),
                bc.data_ptr<BFloat16>(), dest->data_ptr<BFloat16>(), n);
            break;
        default:
            TP_THROW(TypeError, "log_sigmoid_backward: unsupported dtype");
    }
    if (dest != &grad_input) write_out(grad_input, result);
    return grad_input;
}

// ---------------------------------------------------------------------------
// rrelu_with_noise (+ out / inplace)
// ---------------------------------------------------------------------------

Tensor& rrelu_with_noise_out_cpu(const Tensor& self, Tensor& noise,
                                 const Scalar& lower, const Scalar& upper, bool training,
                                 std::optional<Generator> generator,
                                 Tensor& output) {
    TP_CHECK(self.shape() == noise.shape(),
             "noise tensor shape must match self tensor shape. Got self.shape = ",
             self.shape(), " noise.shape = ", noise.shape());
    if (!training) {
        // Eval mode has no randomness: the expected slope is the midpoint, and
        // the map that applies it writes the caller's tensor where it can.
        const double negative = (lower.toDouble() + upper.toDouble()) / 2.0;
        return leaky_relu_out_cpu(self, Scalar(negative), output);
    }
    const Tensor input_c = self.contiguous();
    if (rrelu_with_noise_destinations_take(input_c, noise, output)) {
        rrelu_with_noise_train_cpu(input_c, noise, lower, upper, generator,
                                   &output);
        return output;
    }
    Tensor result =
        rrelu_with_noise_core(self, noise, lower, upper, training, generator);
    const auto target = static_cast<std::vector<int64_t>>(result.shape());
    if (static_cast<std::vector<int64_t>>(output.shape()) != target) {
        output.resize_(target);
    }
    output.copy_(result);
    return output;
}

// Functional form: training draws a slope from U(lower, upper) for every
// non-positive element and records it in noise (1 elsewhere); evaluation
// applies the midpoint slope.
Tensor rrelu_with_noise_cpu(const Tensor& self, Tensor& noise, const Scalar& lower,
                            const Scalar& upper, bool training,
                            std::optional<Generator> generator) {
    TP_CHECK(self.shape() == noise.shape(),
             "noise tensor shape must match self tensor shape. Got self.shape = ",
             self.shape(), " noise.shape = ", noise.shape());
    return rrelu_with_noise_core(self, noise, lower, upper, training, generator);
}

Tensor& rrelu_with_noise__cpu(Tensor& self, Tensor& noise, const Scalar& lower,
                              const Scalar& upper, bool training,
                              std::optional<Generator> generator) {
    // The in-place form is the same run with the input as its own destination,
    // which is one of the pair the draw is allowed to walk.
    return rrelu_with_noise_out_cpu(self, noise, lower, upper, training,
                                    generator, self);
}

TENSORPLAY_LIBRARY_IMPL(CPU, ActivationMoreOps) {
    m.impl("_softmax", _softmax_cpu);
    m.impl("_softmax_backward_data", _softmax_backward_data_cpu);
    m.impl("_softmax_backward_data.out", _softmax_backward_data_out_cpu);
    m.impl("_log_softmax", _log_softmax_cpu);
    m.impl("_log_softmax_backward_data", _log_softmax_backward_data_cpu);
    m.impl("_log_softmax_backward_data.out", _log_softmax_backward_data_out_cpu);
    m.impl("_prelu_kernel", _prelu_kernel_cpu);
    m.impl("_prelu_kernel_backward", _prelu_kernel_backward_cpu);
    m.impl("log_sigmoid_forward", log_sigmoid_forward_cpu);
    m.impl("log_sigmoid_forward.output", log_sigmoid_forward_out_cpu);
    m.impl("log_sigmoid_backward.grad_input", log_sigmoid_backward_out_cpu);
    m.impl("rrelu_with_noise.out", rrelu_with_noise_out_cpu);
    m.impl("rrelu_with_noise", rrelu_with_noise_cpu);
    m.impl("rrelu_with_noise_", rrelu_with_noise__cpu);
}

}  // namespace cpu
}  // namespace tensorplay

// Pointwise CUDA kernels: activation family.
#include "PointwiseCommon.cuh"

namespace tensorplay {
namespace cuda {

Tensor relu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, ReluFunctor()); }
Tensor gelu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, GeluFunctor()); }
Tensor silu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SiluFunctor()); }
template<typename Functor>
Tensor activation_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, Functor functor);
Tensor silu_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self) {
    return activation_backward_kernel_cuda(grad_output, self, SiluBackwardFunctor());
}

// ---------------------------------------------------------------------------
//     (GeluCUDAKernelImpl / GeluBackwardCUDAKernelImpl)
// ---------------------------------------------------------------------------

template<typename Functor>
Tensor activation_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, Functor functor) {
    if (grad_output.shape() != self.shape()) TP_THROW(RuntimeError, "CUDA activation backward: shape mismatch");
    DType out_dtype = grad_output.dtype();
    if (!isFloatingType(out_dtype)) TP_THROW(TypeError, "CUDA activation backward: expected floating point dtype");
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), out_dtype, grad_output.device());
    if (grad_output.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(grad_output)
        .add_input(self)
        .build();

    switch (out_dtype) {
        case DType::Float16:
            gpu_kernel(iter, [functor] __host__ __device__(Half dy, Half x) -> Half {
                return static_cast<Half>(functor(static_cast<float>(dy),
                                                 static_cast<float>(x)));
            });
            break;
        case DType::BFloat16:
            gpu_kernel(iter, [functor] __host__ __device__(BFloat16 dy,
                                                  BFloat16 x) -> BFloat16 {
                return static_cast<BFloat16>(functor(static_cast<float>(dy),
                                                     static_cast<float>(x)));
            });
            break;
        case DType::Float32:
            gpu_kernel(iter, [functor] __host__ __device__(float dy, float x) -> float {
                return functor(dy, x);
            });
            break;
        case DType::Float64:
            gpu_kernel(iter, [functor] __host__ __device__(double dy, double x) -> double {
                return functor(dy, x);
            });
            break;
        default:
            TP_THROW(TypeError, "CUDA activation backward: Unsupported dtype");
    }
    return result;
}

struct GeluTanhFunctor {
    template<typename T> __device__ T operator()(T x) const {
        const T kBeta = static_cast<T>(1.41421356237309504880) * static_cast<T>(1.12837916709551257390) * static_cast<T>(0.5);
        const T kKappa = static_cast<T>(0.044715);
        T x_cube = x * x * x;
        T inner = kBeta * (x + kKappa * x_cube);
        return static_cast<T>(0.5) * x * (static_cast<T>(1) + ::tanh(inner));
    }
};
struct HardtanhFunctor {
    double min_val_, max_val_;
    HardtanhFunctor(double lo, double hi) : min_val_(lo), max_val_(hi) {}
    template<typename T> __device__ T operator()(T x) const {
        T lo = static_cast<T>(min_val_), hi = static_cast<T>(max_val_);
        return x < lo ? lo : (x > hi ? hi : x);
    }
};
struct HardtanhBackwardFunctor {
    double min_val_, max_val_;
    HardtanhBackwardFunctor(double lo, double hi) : min_val_(lo), max_val_(hi) {}
    template<typename T> __device__ T operator()(T dy, T x) const {
        return (x <= static_cast<T>(min_val_) || x >= static_cast<T>(max_val_)) ? static_cast<T>(0) : dy;
    }
};
struct HardswishFunctor {
    template<typename T> __device__ T operator()(T x) const {
        T v = x + static_cast<T>(3);
        v = v < static_cast<T>(0) ? static_cast<T>(0) : (v > static_cast<T>(6) ? static_cast<T>(6) : v);
        return x * v / static_cast<T>(6);
    }
};
struct HardswishBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        return x <= static_cast<T>(-3) ? static_cast<T>(0)
             : x >= static_cast<T>(3)  ? dy
             : dy * (x / static_cast<T>(6) + static_cast<T>(0.5));
    }
};
struct HardsigmoidFunctor {
    template<typename T> __device__ T operator()(T x) const {
        T v = x + static_cast<T>(3);
        v = v < static_cast<T>(0) ? static_cast<T>(0) : (v > static_cast<T>(6) ? static_cast<T>(6) : v);
        return v / static_cast<T>(6);
    }
};
struct HardsigmoidBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        return (x <= static_cast<T>(-3) || x >= static_cast<T>(3)) ? static_cast<T>(0)
                                                                   : dy * (x / static_cast<T>(6) + static_cast<T>(0.5));
    }
};
struct LeakyReluFunctor {
    double negative_slope_;
    LeakyReluFunctor(double s) : negative_slope_(s) {}
    template<typename T> __device__ T operator()(T x) const {
        return x > static_cast<T>(0) ? x : static_cast<T>(negative_slope_) * x;
    }
};
struct LeakyReluBackwardFunctor {
    double negative_slope_;
    LeakyReluBackwardFunctor(double s) : negative_slope_(s) {}
    template<typename T> __device__ T operator()(T dy, T x) const {
        return x > static_cast<T>(0) ? dy : static_cast<T>(negative_slope_) * dy;
    }
};
struct EluFunctor {
    double negcoef_, poscoef_, negiptcoef_;
    EluFunctor(double alpha, double scale, double input_scale)
        : negcoef_(alpha * scale), poscoef_(scale), negiptcoef_(input_scale) {}
    template<typename T> __device__ T operator()(T a) const {
        //   a < 0 ? ::expm1(a*input_scale)*negcoef : a*poscoef
        return a < static_cast<T>(0)
            ? ::expm1(a * static_cast<T>(negiptcoef_)) * static_cast<T>(negcoef_)
            : a * static_cast<T>(poscoef_);
    }
};
struct EluBackwardFunctor {
    double negcoef_, poscoef_, negiptcoef_;
    bool is_result_;
    EluBackwardFunctor(double alpha, double scale, double input_scale, bool is_result)
        : negcoef_(alpha * scale), poscoef_(scale), negiptcoef_(input_scale), is_result_(is_result) {}
    template<typename T> __device__ T operator()(T dy, T b) const {
        //   is_result: b <= 0 ? dy*negiptcoef*(b+negcoef) : dy*poscoef
        //   else:      b <= 0 ? dy*negiptcoef*negcoef*::exp(b*negiptcoef) : dy*poscoef
        return b <= static_cast<T>(0)
            ? (is_result_
                  ? dy * static_cast<T>(negiptcoef_) * (b + static_cast<T>(negcoef_))
                  : dy * static_cast<T>(negiptcoef_) * static_cast<T>(negcoef_) * ::exp(b * static_cast<T>(negiptcoef_)))
            : dy * static_cast<T>(poscoef_);
    }
};
struct MishFunctor {
    template<typename T> __device__ T operator()(T x) const {
        T sp = ::log1p(::exp(x));
        return x * ::tanh(sp);
    }
};
struct MishBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        T sp = ::log1p(::exp(x));
        T tanh_sp = ::tanh(sp);
        T sech2 = static_cast<T>(1) - tanh_sp * tanh_sp;
        T gsp = static_cast<T>(1) / (static_cast<T>(1) + ::exp(-x));
        return dy * (tanh_sp + x * sech2 * gsp);
    }
};
struct SeluFunctor {
    template<typename T> __device__ T operator()(T x) const {
        constexpr double lambda_ = 1.0507009873554804934193349852946;
        constexpr double alpha_ = 1.6732632423543772848170429916717;
        return x > static_cast<T>(0) ? static_cast<T>(lambda_) * x
                                     : static_cast<T>(alpha_ * lambda_) * ::expm1(x);
    }
};
struct CeluFunctor {
    double alpha_;
    CeluFunctor(double a) : alpha_(a) {}
    template<typename T> __device__ T operator()(T x) const {
        return x > static_cast<T>(0) ? x : static_cast<T>(alpha_) * ::expm1(x / static_cast<T>(alpha_));
    }
};
struct SoftplusFunctor {
    double beta_, threshold_;
    SoftplusFunctor(double beta, double threshold) : beta_(beta), threshold_(threshold) {}
    template<typename T> __device__ T operator()(T a) const {
        //   beta*a > threshold ? a : ::log1p(::exp(beta*a)) / beta
        T beta_in = static_cast<T>(beta_);
        return a * beta_in > static_cast<T>(threshold_)
            ? a
            : ::log1p(::exp(a * beta_in)) / beta_in;
    }
};
struct SoftplusBackwardFunctor {
    double beta_, threshold_;
    SoftplusBackwardFunctor(double beta, double threshold) : beta_(beta), threshold_(threshold) {}
    template<typename T> __device__ T operator()(T dy, T a) const {
        //   beta*a > threshold ? dy : dy * sigmoid(beta*a)
        T beta_in = static_cast<T>(beta_);
        return a * beta_in > static_cast<T>(threshold_)
            ? dy
            : dy / (static_cast<T>(1) + ::exp(-a * beta_in));
    }
};

struct GeluBackwardNoneFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        //   kAlpha = M_SQRT1_2; kBeta = M_2_SQRTPI*M_SQRT1_2*0.5
        //   cdf = 0.5*(1+::erf(x*kAlpha)); pdf = kBeta*::exp(-0.5*x*x)
        constexpr T kAlpha = static_cast<T>(0.70710678118654752440);
        constexpr T kBeta = static_cast<T>(1.12837916709551257390) * static_cast<T>(0.70710678118654752440) * static_cast<T>(0.5);
        T cdf = static_cast<T>(0.5) * (static_cast<T>(1) + ::erf(x * kAlpha));
        T pdf = kBeta * ::exp(x * x * static_cast<T>(-0.5));
        return dy * (cdf + x * pdf);
    }
};
struct GeluBackwardTanhFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        constexpr T kBeta = static_cast<T>(1.41421356237309504880) * static_cast<T>(1.12837916709551257390) * static_cast<T>(0.5);
        constexpr T kKappa = static_cast<T>(0.044715);
        T x_sq = x * x;
        T x_cube = x_sq * x;
        T inner = kBeta * (x + kKappa * x_cube);
        T tanh_inner = ::tanh(inner);
        T left = static_cast<T>(0.5) * x;
        T right = static_cast<T>(1) + tanh_inner;
        T left_derivative = static_cast<T>(0.5) * right;
        T tanh_derivative = static_cast<T>(1) - tanh_inner * tanh_inner;
        T inner_derivative = kBeta * (static_cast<T>(1) + static_cast<T>(3) * kKappa * x_sq);
        T right_derivative = left * tanh_derivative * inner_derivative;
        return dy * (left_derivative + right_derivative);
    }
};

Tensor gelu_kernel_cuda_v2(const Tensor& self, const std::string& approximate) {
    if (approximate == "tanh") return unary_float_op_kernel_v2(self, GeluTanhFunctor());
    else if (approximate != "none") TP_THROW(ValueError, "approximate argument must be either none or tanh, but got " + approximate);
    return unary_float_op_kernel_v2(self, GeluFunctor());
}
Tensor gelu_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, const std::string& approximate) {
    if (approximate == "tanh") return activation_backward_kernel_cuda(grad_output, self, GeluBackwardTanhFunctor());
    else if (approximate != "none") TP_THROW(ValueError, "approximate argument must be either none or tanh, but got " + approximate);
    return activation_backward_kernel_cuda(grad_output, self, GeluBackwardNoneFunctor());
}
Tensor hardtanh_kernel_cuda(const Tensor& self, Scalar min_val, Scalar max_val) {
    return unary_float_op_kernel_v2(self, HardtanhFunctor(min_val.toDouble(), max_val.toDouble()));
}
Tensor hardtanh_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, Scalar min_val, Scalar max_val) {
    return activation_backward_kernel_cuda(grad_output, self, HardtanhBackwardFunctor(min_val.toDouble(), max_val.toDouble()));
}
Tensor relu6_kernel_cuda(const Tensor& self) { return hardtanh_kernel_cuda(self, Scalar(0.0), Scalar(6.0)); }
Tensor hardswish_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, HardswishFunctor()); }
Tensor hardswish_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self) {
    return activation_backward_kernel_cuda(grad_output, self, HardswishBackwardFunctor());
}
Tensor hardsigmoid_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, HardsigmoidFunctor()); }
Tensor hardsigmoid_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self) {
    return activation_backward_kernel_cuda(grad_output, self, HardsigmoidBackwardFunctor());
}
Tensor leaky_relu_kernel_cuda(const Tensor& self, Scalar negative_slope) {
    return unary_float_op_kernel_v2(self, LeakyReluFunctor(negative_slope.toDouble()));
}
Tensor leaky_relu_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, Scalar negative_slope, bool self_is_result) {
    (void)self_is_result;
    return activation_backward_kernel_cuda(grad_output, self, LeakyReluBackwardFunctor(negative_slope.toDouble()));
}
Tensor elu_kernel_cuda(const Tensor& self, Scalar alpha, Scalar scale, Scalar input_scale) {
    return unary_float_op_kernel_v2(self, EluFunctor(alpha.toDouble(), scale.toDouble(), input_scale.toDouble()));
}
Tensor elu_backward_kernel_cuda(const Tensor& grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, const Tensor& self_or_result) {
    return activation_backward_kernel_cuda(grad_output, self_or_result,
        EluBackwardFunctor(alpha.toDouble(), scale.toDouble(), input_scale.toDouble(), is_result));
}
Tensor mish_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, MishFunctor()); }
Tensor mish_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self) {
    return activation_backward_kernel_cuda(grad_output, self, MishBackwardFunctor());
}
Tensor selu_kernel_cuda(const Tensor& self) { return unary_float_op_kernel_v2(self, SeluFunctor()); }
Tensor celu_kernel_cuda(const Tensor& self, Scalar alpha) { return unary_float_op_kernel_v2(self, CeluFunctor(alpha.toDouble())); }
Tensor softplus_kernel_cuda(const Tensor& self, Scalar beta, Scalar threshold) {
    return unary_float_op_kernel_v2(self, SoftplusFunctor(beta.toDouble(), threshold.toDouble()));
}
Tensor softplus_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, Scalar beta, Scalar threshold) {
    return activation_backward_kernel_cuda(grad_output, self, SoftplusBackwardFunctor(beta.toDouble(), threshold.toDouble()));
}

//   out = min(x, 0) - ::log1p(::exp(-|x|))
struct LogSigmoidFunctor {
    template<typename T> __device__ T operator()(T x) const {
        T z = x < static_cast<T>(0) ? x : static_cast<T>(0);
        T neg_abs = x < static_cast<T>(0) ? x : -x;
        return z - ::log1p(::exp(neg_abs));
    }
};
// branch-split so ::exp() never overflows.
struct LogSigmoidBackwardFunctor {
    template<typename T> __device__ T operator()(T dy, T x) const {
        if (x >= static_cast<T>(0)) {
            T e = ::exp(-x);
            return dy * (e / (static_cast<T>(1) + e));
        }
        return dy / (static_cast<T>(1) + ::exp(x));
    }
};
Tensor log_sigmoid_kernel_cuda(const Tensor& self) {
    return unary_float_op_kernel_v2(self, LogSigmoidFunctor());
}
Tensor log_sigmoid_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self) {
    return activation_backward_kernel_cuda(grad_output, self, LogSigmoidBackwardFunctor());
}
// out-variant: the saved buffer only feeds the CPU loop's stable form; the
// CUDA elementwise formula recomputes the same expression from x directly.
Tensor& log_sigmoid_backward_out_cuda(const Tensor& grad_output,
                                      const Tensor& self, const Tensor& buffer,
                                      Tensor& grad_input) {
    (void)buffer;
    grad_input = activation_backward_kernel_cuda(grad_output, self,
                                                 LogSigmoidBackwardFunctor());
    return grad_input;
}

// the caller-provided noise; eval is leaky_relu with slope (lower+upper)/2.
template<typename Functor>
Tensor binary_float_op_kernel_v2(const Tensor& self, const Tensor& other, Functor functor);


// Forward declaration: first call sites (rrelu_with_noise) precede the
// definition below, and nvcc's two-phase lookup needs the template declared.
template<typename Functor>
Tensor binary_float_op_kernel_v2(const Tensor& self, const Tensor& other, Functor functor);

Tensor rrelu_with_noise_kernel_cuda(const Tensor& self, const Tensor& noise, Scalar lower, Scalar upper, bool training) {
    return binary_float_op_kernel_v2(self, noise,
        RreluWithNoiseFunctor(lower.toDouble(), upper.toDouble(), training));
}
// inplace out-variant: recompute and write back through the same functor.
Tensor& rrelu_with_noise__cuda(Tensor& self, Tensor& noise, Scalar lower,
                               Scalar upper, bool training) {
    Tensor result = binary_float_op_kernel_v2(self, noise,
        RreluWithNoiseFunctor(lower.toDouble(), upper.toDouble(), training));
    self.copy_(result);
    return self;
}
// out-variant of the forward: the noise buffer is filled on the fly and
// returned alongside the result.
Tensor rrelu_with_noise_out_cuda(const Tensor& self, Tensor& noise, Scalar lower,
                                 Scalar upper, bool training) {
    noise = binary_float_op_kernel_v2(self, noise,
        RreluWithNoiseFunctor(lower.toDouble(), upper.toDouble(), training));
    return noise;
}
// log_sigmoid forward with its saved buffer: log_sigmoid(x) = -softplus(-x);
// the buffer caches ::exp(-|x|), the stable remainder of the softplus
// evaluation the backward reuses elementwise.  Composed from the elementwise
// kernels dispatched in PointwiseKernels.cu.
Tensor abs_kernel_cuda(const Tensor& self);
Tensor neg_kernel_cuda(const Tensor& self);
Tensor exp_kernel_cuda(const Tensor& self);
Tensor log_kernel_cuda(const Tensor& self);
std::tuple<Tensor, Tensor> log_sigmoid_forward_components_cuda(const Tensor& self) {
    const Scalar one(1.0);
    Tensor b = exp_kernel_cuda(neg_kernel_cuda(abs_kernel_cuda(self)));  // ::exp(-|x|)
    Tensor one_plus_b = b + one;
    Tensor log_b = log_kernel_cuda(b);
    Tensor log_one_plus_b = log_kernel_cuda(one_plus_b);
    Tensor pos_branch = log_b - log_one_plus_b;        // ::log(b) - ::log(1+b)
    Tensor neg_branch = self + log_b;                  // x + ::log(b), x < 0
    Tensor output = ops::where(self.lt(Scalar(0.0)), neg_branch, pos_branch);
    return {output, b};
}
// out-variants: run the value kernel, then transfer into the caller's buffer.
Tensor& gelu_out_cuda(const Tensor& self, const std::string& approximate,
                      Tensor& out) {
    out = gelu_kernel_cuda_v2(self, approximate);
    return out;
}
Tensor& gelu_backward_grad_input_cuda(const Tensor& grad_output,
                                      const Tensor& self,
                                      const std::string& approximate,
                                      Tensor& grad_input) {
    grad_input = gelu_backward_kernel_cuda(grad_output, self, approximate);
    return grad_input;
}
Tensor& glu_backward_grad_input_cuda(const Tensor& grad_output, const Tensor& self,
                                     int64_t dim, Tensor& grad_input) {
    grad_input = glu_backward_cuda(grad_output, self, dim);
    return grad_input;
}
std::tuple<Tensor, Tensor> log_sigmoid_forward_out_cuda(const Tensor& self,
                                                        Tensor& output,
                                                        Tensor& buffer) {
    auto [o, b] = log_sigmoid_forward_components_cuda(self);
    output = std::move(o);
    buffer = std::move(b);
    return std::make_tuple(output, buffer);
}
Tensor rrelu_with_noise_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, const Tensor& noise, Scalar lower, Scalar upper, bool training, bool self_is_result) {
    // Training uses the saved per-element noise; evaluation uses the mean
    // slope (lower + upper) / 2.
    if (training) {
        if (grad_output.shape() != noise.shape())
            TP_THROW(RuntimeError, "rrelu_with_noise_backward: shape mismatch");
        return binary_float_op_kernel_v2(
            grad_output, noise, RreluWithNoiseTrainBackwardFunctor());
    }
    (void)self_is_result; // result >= 0 iff self >= 0 for a positive slope.
    const double slope = (lower.toDouble() + upper.toDouble()) / 2.0;
    return binary_float_op_kernel_v2(grad_output, self, RreluWithNoiseEvalBackwardFunctor(slope));
}

TENSORPLAY_LIBRARY_IMPL(CUDA, PointwiseKernels) {
    m.impl("relu", relu_kernel_cuda);
    m.impl("gelu", gelu_kernel_cuda_v2);
    m.impl("gelu_backward", gelu_backward_kernel_cuda);
    m.impl("silu", silu_kernel_cuda);
    m.impl("silu_backward", silu_backward_kernel_cuda);
    m.impl("hardtanh", hardtanh_kernel_cuda);
    m.impl("hardtanh_backward", hardtanh_backward_kernel_cuda);
    m.impl("relu6", relu6_kernel_cuda);
    m.impl("hardswish", hardswish_kernel_cuda);
    m.impl("hardswish_backward", hardswish_backward_kernel_cuda);
    m.impl("hardsigmoid", hardsigmoid_kernel_cuda);
    m.impl("hardsigmoid_backward", hardsigmoid_backward_kernel_cuda);
    m.impl("leaky_relu", leaky_relu_kernel_cuda);
    m.impl("leaky_relu_backward", leaky_relu_backward_kernel_cuda);
    m.impl("elu", elu_kernel_cuda);
    m.impl("elu_backward", elu_backward_kernel_cuda);
    m.impl("mish", mish_kernel_cuda);
    m.impl("mish_backward", mish_backward_kernel_cuda);
    m.impl("selu", selu_kernel_cuda);
    m.impl("celu", celu_kernel_cuda);
    m.impl("softplus", softplus_kernel_cuda);
    m.impl("softplus_backward", softplus_backward_kernel_cuda);
    m.impl("log_sigmoid", log_sigmoid_kernel_cuda);
    m.impl("log_sigmoid_backward", log_sigmoid_backward_kernel_cuda);
    m.impl("log_sigmoid_backward.grad_input", log_sigmoid_backward_out_cuda);
    m.impl("log_sigmoid_forward", log_sigmoid_forward_components_cuda);
    m.impl("log_sigmoid_forward.output", log_sigmoid_forward_out_cuda);
    m.impl("rrelu_with_noise", rrelu_with_noise_kernel_cuda);
    m.impl("rrelu_with_noise.out", rrelu_with_noise_out_cuda);
    m.impl("rrelu_with_noise_", rrelu_with_noise__cuda);
    m.impl("rrelu_with_noise_backward", rrelu_with_noise_backward_kernel_cuda);

    m.impl("gelu.out", gelu_out_cuda);
    m.impl("gelu_backward.grad_input", gelu_backward_grad_input_cuda);
    m.impl("glu_backward.grad_input", glu_backward_grad_input_cuda);
}

} // namespace cuda
} // namespace tensorplay

// Pointwise CUDA kernels: binary math family.
#include "PointwiseCommon.cuh"

namespace tensorplay {
namespace cuda {


Tensor abs_kernel_cuda(const Tensor& self);
Tensor neg_kernel_cuda(const Tensor& self);
Tensor sqrt_kernel_cuda(const Tensor& self);
Tensor rsqrt_kernel_cuda(const Tensor& self);
Tensor square_kernel_cuda(const Tensor& self);
Tensor pow_kernel_cuda(const Tensor& self, const Tensor& other);

// clamp.Tensor: each optional bound is a broadcastable input evaluated per
// element. NaNs in the value or either present bound take precedence.
template <typename T>
void clamp_tensor_loop(TensorIterator& iter, bool has_min, bool has_max) {
    gpu_kernel(iter, [has_min, has_max] __host__ __device__(T value, T lower, T upper) -> T {
        T result = has_min && value < lower ? lower : value;
        result = has_max && upper < result ? upper : result;
        if constexpr (std::numeric_limits<T>::has_quiet_NaN) {
            if (has_max && upper != upper) result = upper;
            if (has_min && lower != lower) result = lower;
            if (value != value) result = value;
        }
        return result;
    });
}

Tensor clamp_tensor_cuda(const Tensor& self, const std::optional<Tensor>& min,
                         const std::optional<Tensor>& max) {
    if (!min.has_value() && !max.has_value()) {
        return self;
    }
    DType common_dtype = self.dtype();
    if (min.has_value()) common_dtype = promoteTypes(common_dtype, min->dtype());
    if (max.has_value()) common_dtype = promoteTypes(common_dtype, max->dtype());
    if (isComplexType(common_dtype)) {
        TP_THROW(RuntimeError, "clamp is not implemented for complex tensors");
    }
    std::vector<int64_t> out_shape = static_cast<std::vector<int64_t>>(self.shape());
    if (min.has_value()) {
        out_shape = broadcast_shapes(out_shape,
            static_cast<std::vector<int64_t>>(min->shape()));
    }
    if (max.has_value()) {
        out_shape = broadcast_shapes(out_shape,
            static_cast<std::vector<int64_t>>(max->shape()));
    }
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    if (result.numel() == 0) return result;
    Tensor a = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor lo = a;
    Tensor hi = a;
    if (min.has_value()) {
        lo = min->dtype() == common_dtype ? *min : min->to(common_dtype);
    }
    if (max.has_value()) {
        hi = max->dtype() == common_dtype ? *max : max->to(common_dtype);
    }
    const bool has_min = min.has_value();
    const bool has_max = max.has_value();
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_const_input(a)
        .add_const_input(lo)
        .add_const_input(hi)
        .build();

#define CLAMP_T_CASE(ctype, name) \
    case DType::name: clamp_tensor_loop<ctype>(iter, has_min, has_max); break;
    switch (common_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(CLAMP_T_CASE)
        default: TP_THROW(NotImplementedError, "CUDA clamp.Tensor: unsupported dtype");
    }
    #undef CLAMP_T_CASE
    return result;
}

Tensor& clamp_tensor__cuda(Tensor& self, const std::optional<Tensor>& min,
                           const std::optional<Tensor>& max) {
    NoGradGuard __tp_nograd;
    self.copy_(clamp_tensor_cuda(self, min, max));
    return self;
}

Tensor& clamp_tensor_out_cuda(const Tensor& self, const std::optional<Tensor>& min,
                              const std::optional<Tensor>& max, Tensor& out) {
    out.copy_(clamp_tensor_cuda(self, min, max));
    return out;
}


Tensor clamp_kernel_cuda(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (self.numel() == 0) return result;
    const bool has_min = min.has_value();
    const bool has_max = max.has_value();
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(self)
        .build();

    #define CLAMP_CASE(ctype, name) \
    case DType::name: { \
        const ctype min_val = has_min ? min->to<ctype>() : ctype(0); \
        const ctype max_val = has_max ? max->to<ctype>() : ctype(0); \
        gpu_kernel(iter, [=] __host__ __device__(ctype value) -> ctype { \
            if (has_min && value < min_val) value = min_val; \
            if (has_max && value > max_val) value = max_val; \
            return value; \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(CLAMP_CASE)
        default: TP_THROW(TypeError, "CUDA clamp: Unsupported dtype");
    }
    #undef CLAMP_CASE
    return result;
}

Tensor clamp_backward_kernel_cuda(const Tensor& grad_output, const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(grad_output.shape()), grad_output.dtype(), grad_output.device());
    int64_t n = grad_output.numel();
    if (n == 0) return result;

    TensorIterator iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .check_all_same_dtype(true)
        .resize_outputs(false)
        .add_output(result)
        .add_const_input(self)
        .add_const_input(grad_output)
        .build();
    const bool has_min = min.has_value();
    const bool has_max = max.has_value();

    #define CLAMP_BW_CASE(ctype, name) \
    case DType::name: { \
        ctype min_val = min.has_value() ? min->to<ctype>() : ctype(0); \
        ctype max_val = max.has_value() ? max->to<ctype>() : ctype(0); \
        gpu_kernel(iter, [=] __host__ __device__(ctype input_value, ctype grad_value) -> ctype { \
            if ((has_min && input_value < min_val) || \
                (has_max && input_value > max_val)) { \
                return ctype(0); \
            } \
            return grad_value; \
        }); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(CLAMP_BW_CASE)
        default: TP_THROW(TypeError, "CUDA clamp_backward: Unsupported dtype");
    }
    #undef CLAMP_BW_CASE
    return result;
}

// --- Binary Ops ---

template<typename Functor>
Tensor binary_float_op_kernel_v2(const Tensor& self, const Tensor& other, Functor functor) {
    if (self.shape() != other.shape()) TP_THROW(RuntimeError, "CUDA binary op: broadcasting not supported");

    DType out_dtype = self.dtype();
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;

    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
    if (self.numel() == 0) return result;
    Tensor a = self.dtype() == out_dtype ? self : self.to(out_dtype);
    Tensor b = other.dtype() == out_dtype ? other : other.to(out_dtype);
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(a)
        .add_input(b)
        .build();

    switch (out_dtype) {
        case DType::Float16:
            gpu_kernel(iter, [functor] __host__ __device__(Half lhs, Half rhs) -> Half {
                return functor(lhs, rhs);
            });
            break;
        case DType::BFloat16:
            gpu_kernel(iter, [functor] __host__ __device__(BFloat16 lhs,
                                                  BFloat16 rhs) -> BFloat16 {
                return functor(lhs, rhs);
            });
            break;
        case DType::Float32:
            gpu_kernel(iter, [functor] __host__ __device__(float lhs, float rhs) -> float {
                return functor(lhs, rhs);
            });
            break;
        case DType::Float64:
            gpu_kernel(iter, [functor] __host__ __device__(double lhs, double rhs) -> double {
                return functor(lhs, rhs);
            });
            break;
        default:
            TP_THROW(TypeError, "CUDA binary op: Unsupported output dtype");
    }
    return result;
}

struct PowFunctor { template<typename T> __device__ T operator()(T a, T b) const { return ::pow(a, b); } };
struct PowScalarFunctor {
    double exponent;
    PowScalarFunctor(double e) : exponent(e) {}
    template<typename T> __device__ T operator()(T x) const { return ::pow(x, static_cast<T>(exponent)); }
};
struct PowBaseFunctor {
    double base;
    template<typename T> __device__ T operator()(T exponent) const {
        return ::pow(static_cast<T>(base), exponent);
    }
};
struct CxPowBase {
    tensorplay::complex<double> base;
    explicit CxPowBase(tensorplay::complex<double> value) : base(value) {}

    template <typename T>
    __device__ tensorplay::complex<T> operator()(
        tensorplay::complex<T> exponent) const {
        const tensorplay::complex<T> base_value(
            static_cast<T>(base.real()), static_cast<T>(base.imag()));
        return tensorplay_complex_math::pow(base_value, exponent);
    }
};
Tensor pow_kernel_cuda(const Tensor& self, const Tensor& other) {
    if (isComplexType(promoteTypes(self.dtype(), other.dtype()))) {
        DType rd = promoteTypes(self.dtype(), other.dtype());
        std::vector<int64_t> out_shape = broadcast_shapes(
            static_cast<std::vector<int64_t>>(self.shape()),
            static_cast<std::vector<int64_t>>(other.shape()));
        Tensor result = Tensor::empty(out_shape, rd, self.device());
        if (result.numel() == 0) return result;
        Tensor a = self.to(rd);
        Tensor b = other.to(rd);
        TensorIterator iter = TensorIteratorConfig()
            .check_all_same_dtype(true)
            .add_output(result)
            .add_const_input(a)
            .add_const_input(b)
            .build();
        switch (rd) {
            case DType::ComplexHalf:
                gpu_kernel(iter, [] __host__ __device__(
                    tensorplay::complex<Half> base,
                    tensorplay::complex<Half> exponent) {
                    const auto b = static_cast<tensorplay::complex<float>>(base);
                    const auto e = static_cast<tensorplay::complex<float>>(exponent);
                    return static_cast<tensorplay::complex<Half>>(
                        tensorplay_complex_math::pow(b, e));
                });
                break;
            case DType::ComplexFloat:
                gpu_kernel(iter, [] __host__ __device__(
                    tensorplay::complex<float> base,
                    tensorplay::complex<float> exponent) {
                    return tensorplay_complex_math::pow(base, exponent);
                });
                break;
            case DType::ComplexDouble:
                gpu_kernel(iter, [] __host__ __device__(
                    tensorplay::complex<double> base,
                    tensorplay::complex<double> exponent) {
                    return tensorplay_complex_math::pow(base, exponent);
                });
                break;
            case DType::BComplex32:
                gpu_kernel(iter, [] __host__ __device__(
                    tensorplay::complex<BFloat16> base,
                    tensorplay::complex<BFloat16> exponent) {
                    const auto b = static_cast<tensorplay::complex<float>>(base);
                    const auto e = static_cast<tensorplay::complex<float>>(exponent);
                    return static_cast<tensorplay::complex<BFloat16>>(
                        tensorplay_complex_math::pow(b, e));
                });
                break;
            default:
                TP_THROW(NotImplementedError, "CUDA pow: unsupported complex dtype");
        }
        return result;
    }
    return binary_float_op_kernel_v2(self, other, PowFunctor());
}
Tensor pow_scalar_kernel_cuda(const Tensor& self, Scalar exponent) {
    if (!isComplexType(self.dtype()) && !exponent.isComplex() &&
        isIntegralType(self.dtype()) && !exponent.isFloatingPoint() &&
        exponent.to<int64_t>() < 0) {
        TP_THROW(RuntimeError, "Integers to negative integer powers are not allowed.");
    }
    if (isComplexType(self.dtype()) || exponent.isComplex()) {
        DType rd = isComplexType(self.dtype())
            ? self.dtype()
            : (isFloatingType(self.dtype()) ? toComplexType(self.dtype())
                                            : DType::ComplexFloat);
        Tensor base = self.to(rd);
        if (exponent.isFloatingPoint() && !exponent.isComplex()) {
            double ev = exponent.toDouble();
            if (ev == 0.5) return sqrt_kernel_cuda(base);
            if (ev == -0.5) return rsqrt_kernel_cuda(base);
            if (ev == 2.0) return square_kernel_cuda(base);
            return complex_math_kernel_cuda(base, CxPowScalar{ev});
        }
        return complex_math_kernel_cuda(
            base, CxPowScalar{exponent.to<tensorplay::complex<double>>()});
    }
    return unary_float_op_kernel_v2(self, PowScalarFunctor(exponent.toDouble()));
}
Tensor pow_scalar_tensor_kernel_cuda(Scalar base, const Tensor& exponent) {
    const DType result_dtype = ops::result_type(base, exponent);
    if (!base.isComplex() && base.toDouble() == 1.0) {
        return Tensor::ones(static_cast<std::vector<int64_t>>(exponent.shape()),
                            result_dtype, exponent.device());
    }
    Tensor exponent_cast = exponent.dtype() == result_dtype
        ? exponent : exponent.to(result_dtype);
    if (isComplexType(result_dtype)) {
        const auto base_value = base.to<tensorplay::complex<double>>();
        return complex_math_kernel_cuda(
            exponent_cast, CxPowBase{base_value});
    }
    return unary_float_op_kernel_v2(
        exponent_cast, PowBaseFunctor{base.toDouble()});
}
Tensor atan2_kernel_cuda(const Tensor& self, const Tensor& other) {
    if (self.shape() != other.shape()) TP_THROW(RuntimeError, "CUDA binary op: broadcasting not supported");

    DType out_dtype = self.dtype();
    if (isIntegralType(out_dtype)) out_dtype = DType::Float32;

    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), out_dtype, self.device());
    if (self.numel() == 0) return result;
    Tensor a = self.dtype() == out_dtype ? self : self.to(out_dtype);
    Tensor b = other.dtype() == out_dtype ? other : other.to(out_dtype);
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_input(a)
        .add_input(b)
        .build();

    switch (out_dtype) {
        case DType::Float16:
        case DType::BFloat16:
            opmath_gpu_kernel_with_scalars<float, float, float>(
                iter, [] __host__ __device__(float lhs, float rhs) -> float {
                    return ::atan2(lhs, rhs);
                });
            break;
        case DType::Float32:
            gpu_kernel_with_scalars(iter, [] __host__ __device__(float lhs,
                                                                  float rhs) -> float {
                return ::atan2(lhs, rhs);
            });
            break;
        case DType::Float64:
            gpu_kernel_with_scalars(iter, [] __host__ __device__(double lhs,
                                                                  double rhs) -> double {
                return ::atan2(lhs, rhs);
            });
            break;
        default: TP_THROW(TypeError, "CUDA atan2: Unsupported output dtype");
    }
    return result;
}

// --- Lerp ---
template <typename math_t>
__device__ bool lerp_weight_small(math_t value) {
    if constexpr (std::is_same_v<math_t, float>) {
        return ::fabsf(value) < 0.5f;
    } else {
        return ::fabs(value) < 0.5;
    }
}

template <typename scalar_t, typename math_t>
void lerp_tensor_loop(TensorIterator& iter) {
    gpu_kernel(iter, [] __host__ __device__(scalar_t start, scalar_t finish,
                                   scalar_t weight) -> scalar_t {
        const math_t s = static_cast<math_t>(start);
        const math_t e = static_cast<math_t>(finish);
        const math_t w = static_cast<math_t>(weight);
        const math_t value = lerp_weight_small(w)
            ? s + w * (e - s)
            : e - (e - s) * (static_cast<math_t>(1) - w);
        return static_cast<scalar_t>(value);
    });
}

template <typename math_t>
__device__ bool complex_lerp_weight_small(const math_t& value) {
    const auto real = value.real();
    const auto imag = value.imag();
    return real * real + imag * imag < 0.25;
}

template <typename scalar_t, typename math_t>
void complex_lerp_tensor_loop(TensorIterator& iter) {
    gpu_kernel(iter, [] __host__ __device__(scalar_t start, scalar_t finish,
                                   scalar_t weight) -> scalar_t {
        const math_t s = static_cast<math_t>(start);
        const math_t e = static_cast<math_t>(finish);
        const math_t w = static_cast<math_t>(weight);
        const math_t value = complex_lerp_weight_small(w)
            ? s + w * (e - s)
            : e - (e - s) * (math_t(1) - w);
        return static_cast<scalar_t>(value);
    });
}

Tensor lerp_scalar_kernel_cuda(const Tensor& self, const Tensor& end, Scalar weight) {
    if (self.shape() != end.shape()) TP_THROW(RuntimeError, "CUDA lerp: broadcasting not supported");
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    if (self.numel() == 0) return result;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_const_input(self)
        .add_const_input(end)
        .build();

    switch (self.dtype()) {
        case DType::Float32: {
            const float weight_value = weight.to<float>();
            gpu_kernel(iter, [weight_value] __host__ __device__(float start, float finish) -> float {
                return (fabsf(weight_value) < 0.5f)
                    ? start + weight_value * (finish - start)
                    : finish - (finish - start) * (1.0f - weight_value);
            });
            break;
        }
        case DType::Float64: {
            const double weight_value = weight.to<double>();
            gpu_kernel(iter, [weight_value] __host__ __device__(double start, double finish) -> double {
                return (fabs(weight_value) < 0.5)
                    ? start + weight_value * (finish - start)
                    : finish - (finish - start) * (1.0 - weight_value);
            });
            break;
        }
        case DType::Float16: {
            const float weight_value = weight.to<float>();
            gpu_kernel(iter, [weight_value] __host__ __device__(Half start, Half finish) -> Half {
                const float s = static_cast<float>(start);
                const float e = static_cast<float>(finish);
                const float value = (fabsf(weight_value) < 0.5f)
                    ? s + weight_value * (e - s)
                    : e - (e - s) * (1.0f - weight_value);
                return static_cast<Half>(value);
            });
            break;
        }
        case DType::BFloat16: {
            const float weight_value = weight.to<float>();
            gpu_kernel(iter, [weight_value] __host__ __device__(BFloat16 start,
                                                        BFloat16 finish) -> BFloat16 {
                const float s = static_cast<float>(start);
                const float e = static_cast<float>(finish);
                const float value = (fabsf(weight_value) < 0.5f)
                    ? s + weight_value * (e - s)
                    : e - (e - s) * (1.0f - weight_value);
                return static_cast<BFloat16>(value);
            });
            break;
        }
        default: TP_THROW(NotImplementedError, "CUDA lerp: unsupported dtype");
    }
    return result;
}

Tensor lerp_tensor_kernel_cuda(const Tensor& self, const Tensor& end, const Tensor& weight) {
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self.shape()),
        static_cast<std::vector<int64_t>>(end.shape()),
        static_cast<std::vector<int64_t>>(weight.shape()));
    DType common_dtype = promoteTypes(self.dtype(), end.dtype());
    common_dtype = promoteTypes(common_dtype, weight.dtype());
    if (isIntegralType(common_dtype)) common_dtype = DType::Float32;
    Tensor s = self.dtype() == common_dtype ? self : self.to(common_dtype);
    Tensor e = end.dtype() == common_dtype ? end : end.to(common_dtype);
    Tensor w = weight.dtype() == common_dtype ? weight : weight.to(common_dtype);
    Tensor result = Tensor::empty(out_shape, common_dtype, self.device());
    if (result.numel() == 0) return result;

    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(result)
        .add_const_input(s)
        .add_const_input(e)
        .add_const_input(w)
        .build();

    switch (common_dtype) {
        case DType::Float32: lerp_tensor_loop<float, float>(iter); break;
        case DType::Float64: lerp_tensor_loop<double, double>(iter); break;
        case DType::Float16: lerp_tensor_loop<Half, float>(iter); break;
        case DType::BFloat16: lerp_tensor_loop<BFloat16, float>(iter); break;
        case DType::ComplexHalf:
            complex_lerp_tensor_loop<tensorplay::complex<Half>,
                                     tensorplay::complex<float>>(iter);
            break;
        case DType::ComplexFloat:
            complex_lerp_tensor_loop<tensorplay::complex<float>,
                                     tensorplay::complex<float>>(iter);
            break;
        case DType::ComplexDouble:
            complex_lerp_tensor_loop<tensorplay::complex<double>,
                                     tensorplay::complex<double>>(iter);
            break;
        case DType::BComplex32:
            complex_lerp_tensor_loop<tensorplay::complex<BFloat16>,
                                     tensorplay::complex<float>>(iter);
            break;
        default: TP_THROW(NotImplementedError, "CUDA lerp: unsupported dtype");
    }
    return result;
}

Tensor& lerp_scalar_inplace_kernel_cuda(Tensor& self, const Tensor& end, Scalar weight) {
    self.copy_(lerp_scalar_kernel_cuda(self, end, weight));
    return self;
}

Tensor& lerp_tensor_inplace_kernel_cuda(Tensor& self, const Tensor& end, const Tensor& weight) {
    self.copy_(lerp_tensor_kernel_cuda(self, end, weight));
    return self;
}

Tensor& abs_inplace_kernel_cuda(Tensor& self) {
    self.copy_(abs_kernel_cuda(self));
    return self;
}

Tensor& neg_inplace_kernel_cuda(Tensor& self) {
    self.copy_(neg_kernel_cuda(self));
    return self;
}

Tensor& sqrt_inplace_kernel_cuda(Tensor& self) {
    self.copy_(sqrt_kernel_cuda(self));
    return self;
}

Tensor& rsqrt_inplace_kernel_cuda(Tensor& self) {
    self.copy_(rsqrt_kernel_cuda(self));
    return self;
}

// --- Masked Select ---
template <typename T>
__global__ void masked_select_gather_kernel(
    int64_t n, const T* input, const bool* mask,
    const int64_t* positions, T* output) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        if (mask[i]) {
            output[positions[i] - 1] = input[i];
        }
    }
}

Tensor masked_select_kernel_cuda(const Tensor& self, const Tensor& mask) {
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "CUDA masked_select: mask must be bool");
    }
    if (mask.device() != self.device()) {
        TP_THROW(DeviceMismatchError,
                 "CUDA masked_select: self and mask must be on the same device");
    }

    Tensor mask_temp = mask.dim() == 0 ? mask.unsqueeze(0) : mask;
    Tensor self_temp = self.dim() == 0 ? self.unsqueeze(0) : self;
    const std::vector<int64_t> select_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(self_temp.shape()),
        static_cast<std::vector<int64_t>>(mask_temp.shape()));
    Tensor mask_c = mask_temp.expand(select_shape).contiguous();
    Tensor self_c = self_temp.expand(select_shape).contiguous();
    const int64_t n = self_c.numel();
    if (n == 0) return Tensor::empty({0}, self.dtype(), self.device());
    TP_CHECK(n <= static_cast<int64_t>(std::numeric_limits<int>::max()),
             "CUDA masked_select: input is too large for device scan");

    Tensor mask_flat = mask_c.reshape({n});
    Tensor flags = Tensor::empty({n}, DType::Int64, self.device());
    Tensor positions = Tensor::empty({n}, DType::Int64, self.device());
    TensorIterator flag_iter = TensorIteratorConfig()
        .resize_outputs(false)
        .check_all_same_dtype(false)
        .add_output(flags)
        .add_const_input(mask_flat)
        .build();
    gpu_kernel(flag_iter, [] __host__ __device__(bool value) -> int64_t {
        return value ? int64_t(1) : int64_t(0);
    });

    const cudaStream_t stream = getCurrentCUDAStream().stream();
    size_t scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, scan_bytes, flags.data_ptr<int64_t>(),
        positions.data_ptr<int64_t>(), static_cast<int>(n), stream));
    Tensor scan_storage = Tensor::empty(
        {static_cast<int64_t>(scan_bytes == 0 ? 1 : scan_bytes)},
        DType::UInt8, self.device());
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        scan_storage.data_ptr(), scan_bytes, flags.data_ptr<int64_t>(),
        positions.data_ptr<int64_t>(), static_cast<int>(n), stream));

    int64_t count = 0;
    CUDA_CHECK(cudaMemcpyAsync(
        &count, positions.data_ptr<int64_t>() + n - 1, sizeof(int64_t),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    Tensor result = Tensor::empty({count}, self.dtype(), self.device());
    if (count > 0) {
        dim3 block(256);
        dim3 grid((n + 255) / 256);
#define SEL_CASE(ctype, name) \
        case DType::name: { \
            masked_select_gather_kernel<ctype><<<grid, block, 0, stream>>>( \
                n, self_c.data_ptr<ctype>(), mask_c.data_ptr<bool>(), \
                positions.data_ptr<int64_t>(), result.data_ptr<ctype>()); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(SEL_CASE)
            TENSORPLAY_FORALL_FP8_TYPES(SEL_CASE)
            case DType::ComplexHalf:
                masked_select_gather_kernel<tensorplay::complex<Half>><<<
                    grid, block, 0, stream>>>(
                    n, static_cast<const tensorplay::complex<Half>*>(self_c.data_ptr()),
                    mask_c.data_ptr<bool>(), positions.data_ptr<int64_t>(),
                    static_cast<tensorplay::complex<Half>*>(result.data_ptr()));
                break;
            case DType::ComplexFloat:
                masked_select_gather_kernel<tensorplay::complex<float>><<<
                    grid, block, 0, stream>>>(
                    n, static_cast<const tensorplay::complex<float>*>(self_c.data_ptr()),
                    mask_c.data_ptr<bool>(), positions.data_ptr<int64_t>(),
                    static_cast<tensorplay::complex<float>*>(result.data_ptr()));
                break;
            case DType::ComplexDouble:
                masked_select_gather_kernel<tensorplay::complex<double>><<<
                    grid, block, 0, stream>>>(
                    n, static_cast<const tensorplay::complex<double>*>(self_c.data_ptr()),
                    mask_c.data_ptr<bool>(), positions.data_ptr<int64_t>(),
                    static_cast<tensorplay::complex<double>*>(result.data_ptr()));
                break;
            case DType::BComplex32:
                masked_select_gather_kernel<tensorplay::complex<BFloat16>><<<
                    grid, block, 0, stream>>>(
                    n, static_cast<const tensorplay::complex<BFloat16>*>(self_c.data_ptr()),
                    mask_c.data_ptr<bool>(), positions.data_ptr<int64_t>(),
                    static_cast<tensorplay::complex<BFloat16>*>(result.data_ptr()));
                break;
            default: TP_THROW(TypeError, "CUDA masked_select: unsupported dtype");
        }
#undef SEL_CASE
    }

    CUDA_CHECK(cudaGetLastError());
    return result;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, PointwiseKernels) {
    m.impl("clamp", clamp_kernel_cuda);
    m.impl("clamp_backward", clamp_backward_kernel_cuda);

        m.impl("clamp.Tensor", clamp_tensor_cuda);
    m.impl("clamp_.Tensor", clamp_tensor__cuda);
    m.impl("clamp.Tensor_out", clamp_tensor_out_cuda);
    m.impl("clip.Tensor", clamp_tensor_cuda);
    m.impl("clip_.Tensor", clamp_tensor__cuda);
    m.impl("clip.Tensor_out", clamp_tensor_out_cuda);

    m.impl("pow.Tensor_Tensor", pow_kernel_cuda);
    m.impl("pow.Tensor_Scalar", pow_scalar_kernel_cuda);
    m.impl("pow.Scalar", pow_scalar_tensor_kernel_cuda);
    m.impl("atan2", atan2_kernel_cuda);
    m.impl("arctan2", atan2_kernel_cuda);

    m.impl("lerp", lerp_scalar_kernel_cuda);
    m.impl("lerp.Tensor", lerp_tensor_kernel_cuda);
    m.impl("lerp_.Scalar", lerp_scalar_inplace_kernel_cuda);
    m.impl("lerp_.Tensor", lerp_tensor_inplace_kernel_cuda);
    m.impl("abs_", abs_inplace_kernel_cuda);
    m.impl("neg_", neg_inplace_kernel_cuda);
    m.impl("sqrt_", sqrt_inplace_kernel_cuda);
    m.impl("rsqrt_", rsqrt_inplace_kernel_cuda);
    m.impl("masked_select", masked_select_kernel_cuda);
}

// The activation TU instantiates this template for the rrelu backward
// functors; the definition lives here, so export both instantiations.
template Tensor binary_float_op_kernel_v2<RreluWithNoiseTrainBackwardFunctor>(
    const Tensor&, const Tensor&, RreluWithNoiseTrainBackwardFunctor);
template Tensor binary_float_op_kernel_v2<RreluWithNoiseEvalBackwardFunctor>(
    const Tensor&, const Tensor&, RreluWithNoiseEvalBackwardFunctor);
template Tensor binary_float_op_kernel_v2<RreluWithNoiseFunctor>(
    const Tensor&, const Tensor&, RreluWithNoiseFunctor);

} // namespace cuda
} // namespace tensorplay

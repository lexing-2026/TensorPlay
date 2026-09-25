#include "QuantKernels.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Quantizer.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

void check_real_dtype(const Tensor& self, const char* op) {
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError,
                 std::string(op) + ": expected a floating point tensor, got " +
                     toString(self.dtype()));
    }
}

void check_fake_quant_range(int64_t zero_point, int64_t quant_min,
                            int64_t quant_max) {
    if (quant_min > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize(): quant_min must be <= quant_max");
    }
    if (zero_point < quant_min || zero_point > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize(): zero_point must be between quant_min and "
                 "quant_max");
    }
}

template <typename T>
__global__ void fake_quant_per_tensor_kernel(
    int64_t numel, const T* __restrict__ input, T* __restrict__ output,
    bool* __restrict__ mask, float scale, int64_t zero_point,
    int64_t quant_min, int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const float inv_scale = 1.0f / scale;
    const float raw = nearbyintf(static_cast<float>(input[i]) * inv_scale) +
                      static_cast<float>(zero_point);
    const int64_t q = static_cast<int64_t>(fminf(
        static_cast<float>(quant_max),
        fmaxf(static_cast<float>(quant_min), raw)));
    output[i] = static_cast<T>((static_cast<float>(q) -
                                static_cast<float>(zero_point)) * scale);
    mask[i] = raw >= static_cast<float>(quant_min) &&
              raw <= static_cast<float>(quant_max);
}

__global__ void fake_quant_per_tensor_kernel_double(
    int64_t numel, const double* __restrict__ input,
    double* __restrict__ output, bool* __restrict__ mask, double scale,
    int64_t zero_point, int64_t quant_min, int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const double inv_scale = 1.0 / scale;
    const double raw = nearbyint(input[i] * inv_scale) +
                       static_cast<double>(zero_point);
    const int64_t q = static_cast<int64_t>(fmin(
        static_cast<double>(quant_max),
        fmax(static_cast<double>(quant_min), raw)));
    output[i] = (static_cast<double>(q) - static_cast<double>(zero_point)) *
                scale;
    mask[i] = raw >= static_cast<double>(quant_min) &&
              raw <= static_cast<double>(quant_max);
}

void launch_fake_quant_per_tensor(const Tensor& input, Tensor& out,
                                  Tensor& mask, double scale,
                                  int64_t zero_point, int64_t quant_min,
                                  int64_t quant_max) {
    const int64_t numel = input.numel();
    if (numel == 0) return;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    switch (input.dtype()) {
        case DType::Float32:
            fake_quant_per_tensor_kernel<float><<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<float>(), out.data_ptr<float>(),
                mask.data_ptr<bool>(), static_cast<float>(scale), zero_point,
                quant_min, quant_max);
            break;
        case DType::Float64:
            fake_quant_per_tensor_kernel_double<<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<double>(), out.data_ptr<double>(),
                mask.data_ptr<bool>(), scale, zero_point, quant_min,
                quant_max);
            break;
        case DType::Float16:
            fake_quant_per_tensor_kernel<Half><<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<Half>(), out.data_ptr<Half>(),
                mask.data_ptr<bool>(), static_cast<float>(scale), zero_point,
                quant_min, quant_max);
            break;
        case DType::BFloat16:
            fake_quant_per_tensor_kernel<BFloat16><<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<BFloat16>(), out.data_ptr<BFloat16>(),
                mask.data_ptr<bool>(), static_cast<float>(scale), zero_point,
                quant_min, quant_max);
            break;
        default:
            TP_THROW(TypeError, "fake_quantize_per_tensor_affine(): "
                                "unsupported input dtype");
    }
    checkCuda(cudaGetLastError(), "CUDA fake_quantize_per_tensor kernel");
}

template <typename T>
__global__ void fake_quant_tensor_qparams_kernel(
    int64_t numel, const T* __restrict__ input, T* __restrict__ output,
    bool* __restrict__ mask, const float* __restrict__ scale,
    const int32_t* __restrict__ zero_point,
    const int64_t* __restrict__ fake_quant_enabled, int64_t quant_min,
    int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    if (*fake_quant_enabled == 0) {
        output[i] = input[i];
        mask[i] = true;
        return;
    }
    const float inv_scale = 1.0f / (*scale);
    const float raw = nearbyintf(static_cast<float>(input[i]) * inv_scale) +
                      static_cast<float>(*zero_point);
    const int64_t q = static_cast<int64_t>(fminf(
        static_cast<float>(quant_max),
        fmaxf(static_cast<float>(quant_min), raw)));
    output[i] = static_cast<T>((static_cast<float>(q) -
                                static_cast<float>(*zero_point)) * (*scale));
    mask[i] = raw >= static_cast<float>(quant_min) &&
              raw <= static_cast<float>(quant_max);
}

template <typename T>
__global__ void fake_quant_tensor_qparams_kernel_floatzp(
    int64_t numel, const T* __restrict__ input, T* __restrict__ output,
    bool* __restrict__ mask, const float* __restrict__ scale,
    const float* __restrict__ zero_point,
    const int64_t* __restrict__ fake_quant_enabled, int64_t quant_min,
    int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    if (*fake_quant_enabled == 0) {
        output[i] = input[i];
        mask[i] = true;
        return;
    }
    const float inv_scale = 1.0f / (*scale);
    // A floating zero point folds the shift into the rounding itself.
    const float raw = nearbyintf(static_cast<float>(input[i]) * inv_scale +
                                 (*zero_point));
    const int64_t q = static_cast<int64_t>(fminf(
        static_cast<float>(quant_max),
        fmaxf(static_cast<float>(quant_min), raw)));
    output[i] = static_cast<T>((static_cast<float>(q) - (*zero_point)) *
                               (*scale));
    mask[i] = raw >= static_cast<float>(quant_min) &&
              raw <= static_cast<float>(quant_max);
}

template <typename T, typename ZPT>
__global__ void fake_quant_tensor_qparams_kernel_double_typed(
    int64_t numel, const double* __restrict__ input,
    double* __restrict__ output, bool* __restrict__ mask,
    const double* __restrict__ scale, const ZPT* __restrict__ zero_point,
    const int64_t* __restrict__ fake_quant_enabled, int64_t quant_min,
    int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    if (*fake_quant_enabled == 0) {
        output[i] = input[i];
        mask[i] = true;
        return;
    }
    const double inv_scale = 1.0 / (*scale);
    const double zpv = static_cast<double>(*zero_point);
    const double raw = nearbyint(input[i] * inv_scale) + zpv;
    const int64_t q = static_cast<int64_t>(fmin(
        static_cast<double>(quant_max),
        fmax(static_cast<double>(quant_min), raw)));
    output[i] = (static_cast<double>(q) - zpv) * (*scale);
    mask[i] = raw >= static_cast<double>(quant_min) &&
              raw <= static_cast<double>(quant_max);
}

void launch_fake_quant_tensor_qparams(const Tensor& input, Tensor& out,
                                      Tensor& mask, const Tensor& scale,
                                      const Tensor& zero_point,
                                      const Tensor& fake_quant_enabled,
                                      int64_t quant_min, int64_t quant_max) {
    const int64_t numel = input.numel();
    if (numel == 0) return;
    Tensor sc = scale.to(DType::Float32).contiguous();
    const bool zp_float = !isIntegralType(zero_point.dtype());
    Tensor zpi = zp_float ? zero_point.to(DType::Float32).contiguous()
                          : zero_point.to(DType::Int32).contiguous();
    Tensor fq = fake_quant_enabled.to(DType::Int64).contiguous();
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (input.dtype() == DType::Float64) {
        Tensor sc64 = scale.to(DType::Float64).contiguous();
        Tensor zpi64 = zp_float
                           ? zero_point.to(DType::Float64).contiguous()
                           : zero_point.to(DType::Int64).contiguous();
        if (zp_float) {
            fake_quant_tensor_qparams_kernel_double_typed<
                double, double><<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<double>(), out.data_ptr<double>(),
                mask.data_ptr<bool>(), sc64.data_ptr<double>(),
                zpi64.data_ptr<double>(), fq.data_ptr<int64_t>(), quant_min,
                quant_max);
        } else {
            fake_quant_tensor_qparams_kernel_double_typed<
                double, int64_t><<<blocks, threads, 0, stream>>>(
                numel, input.data_ptr<double>(), out.data_ptr<double>(),
                mask.data_ptr<bool>(), sc64.data_ptr<double>(),
                zpi64.data_ptr<int64_t>(), fq.data_ptr<int64_t>(), quant_min,
                quant_max);
        }
        checkCuda(cudaGetLastError(), "CUDA fake_quantize tensor_qparams kernel");
        return;
    }
    switch (input.dtype()) {
        case DType::Float32:
            if (zp_float) {
                fake_quant_tensor_qparams_kernel_floatzp<float>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<float>(), out.data_ptr<float>(),
                        mask.data_ptr<bool>(), sc.data_ptr<float>(),
                        zpi.data_ptr<float>(), fq.data_ptr<int64_t>(),
                        quant_min, quant_max);
            } else {
                fake_quant_tensor_qparams_kernel<float>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<float>(), out.data_ptr<float>(),
                        mask.data_ptr<bool>(), sc.data_ptr<float>(),
                        zpi.data_ptr<int32_t>(), fq.data_ptr<int64_t>(),
                        quant_min, quant_max);
            }
            break;
        case DType::Float16:
            if (zp_float) {
                fake_quant_tensor_qparams_kernel_floatzp<Half>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<Half>(), out.data_ptr<Half>(),
                        mask.data_ptr<bool>(), sc.data_ptr<float>(),
                        zpi.data_ptr<float>(), fq.data_ptr<int64_t>(),
                        quant_min, quant_max);
            } else {
                fake_quant_tensor_qparams_kernel<Half>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<Half>(), out.data_ptr<Half>(),
                        mask.data_ptr<bool>(), sc.data_ptr<float>(),
                        zpi.data_ptr<int32_t>(), fq.data_ptr<int64_t>(),
                        quant_min, quant_max);
            }
            break;
        case DType::BFloat16:
            if (zp_float) {
                fake_quant_tensor_qparams_kernel_floatzp<BFloat16>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<BFloat16>(),
                        out.data_ptr<BFloat16>(), mask.data_ptr<bool>(),
                        sc.data_ptr<float>(), zpi.data_ptr<float>(),
                        fq.data_ptr<int64_t>(), quant_min, quant_max);
            } else {
                fake_quant_tensor_qparams_kernel<BFloat16>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<BFloat16>(),
                        out.data_ptr<BFloat16>(), mask.data_ptr<bool>(),
                        sc.data_ptr<float>(), zpi.data_ptr<int32_t>(),
                        fq.data_ptr<int64_t>(), quant_min, quant_max);
            }
            break;
        default:
            TP_THROW(TypeError, "fake_quantize_per_tensor_affine(): "
                                "unsupported input dtype");
    }
    checkCuda(cudaGetLastError(), "CUDA fake_quantize tensor_qparams kernel");
}

template <typename T>
__global__ void masked_grad_kernel(int64_t numel, const T* __restrict__ grad,
                                   const bool* __restrict__ mask,
                                   T* __restrict__ out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    out[i] = mask[i] ? grad[i] : static_cast<T>(0);
}

Tensor masked_grad_cuda(const Tensor& grad, const Tensor& mask) {
    Tensor out = Tensor::empty(grad.shape(), grad.dtype(), grad.device());
    const int64_t numel = grad.numel();
    if (numel == 0) return out;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    switch (grad.dtype()) {
        case DType::Float32:
            masked_grad_kernel<float><<<blocks, threads, 0, stream>>>(
                numel, grad.data_ptr<float>(), mask.data_ptr<bool>(),
                out.data_ptr<float>());
            break;
        case DType::Float64:
            masked_grad_kernel<double><<<blocks, threads, 0, stream>>>(
                numel, grad.data_ptr<double>(), mask.data_ptr<bool>(),
                out.data_ptr<double>());
            break;
        case DType::Float16:
            masked_grad_kernel<Half><<<blocks, threads, 0, stream>>>(
                numel, grad.data_ptr<Half>(), mask.data_ptr<bool>(),
                out.data_ptr<Half>());
            break;
        case DType::BFloat16:
            masked_grad_kernel<BFloat16><<<blocks, threads, 0, stream>>>(
                numel, grad.data_ptr<BFloat16>(), mask.data_ptr<bool>(),
                out.data_ptr<BFloat16>());
            break;
        default:
            TP_THROW(TypeError, "fake_quantize backward: expected a "
                                "floating point gradient");
    }
    checkCuda(cudaGetLastError(), "CUDA fake_quantize backward kernel");
    return out;
}

// Learnable-qparams backward: dX is a straight-through inside the
// representable range and zero outside; dScale and dZeroPoint collect one
// contribution per element, scaled by grad_factor.
template <typename T>
__global__ void learnable_backward_kernel(
    int64_t numel, const T* __restrict__ x, const T* __restrict__ dy,
    T* __restrict__ dx, T* __restrict__ dscale, T* __restrict__ dzp,
    float scale, int64_t zero_point, int64_t quant_min, int64_t quant_max,
    float grad_factor) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const float inv_scale = 1.0f / scale;
    const float dscale_small = static_cast<float>(quant_min - zero_point);
    const float dscale_big = static_cast<float>(quant_max - zero_point);
    const float xf = static_cast<float>(x[i]);
    const float dyf = static_cast<float>(dy[i]);
    const int64_t xq = static_cast<int64_t>(
                           nearbyintf(xf * inv_scale)) + zero_point;
    dx[i] = static_cast<T>(dyf * (xq >= quant_min && xq <= quant_max));
    const float xfq = static_cast<float>(
        (std::min<int64_t>(std::max<int64_t>(xq, quant_min), quant_max) -
         zero_point) * scale);
    if (xq < quant_min || xq > quant_max) {
        dscale[i] = static_cast<T>(
            (dyf * ((xq < quant_min) ? dscale_small : dscale_big)) *
            grad_factor);
        dzp[i] = static_cast<T>(dyf * (-1.0f) * scale * grad_factor);
    } else {
        dscale[i] = static_cast<T>(dyf * (xfq - xf) * inv_scale *
                                   grad_factor);
        dzp[i] = static_cast<T>(0);
    }
}

std::tuple<Tensor, Tensor, Tensor> learnable_backward_cuda(
    const Tensor& grad, const Tensor& x, double scale_val,
    int64_t zero_point_val, int64_t quant_min, int64_t quant_max,
    double grad_factor, DType out_dtype) {
    const int64_t numel = x.numel();
    Tensor dx = Tensor::empty(x.shape(), out_dtype, x.device());
    Tensor dscale_vec = Tensor::empty(x.shape(), out_dtype, x.device());
    Tensor dzp_vec = Tensor::empty(x.shape(), out_dtype, x.device());
    if (numel == 0) {
        return {std::move(dx), std::move(dscale_vec.sum().reshape({1})),
                std::move(dzp_vec.sum().reshape({1}))};
    }
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (out_dtype == DType::Float64) {
        learnable_backward_kernel<double><<<blocks, threads, 0, stream>>>(
            numel, x.data_ptr<double>(), grad.data_ptr<double>(),
            dx.data_ptr<double>(), dscale_vec.data_ptr<double>(),
            dzp_vec.data_ptr<double>(), static_cast<float>(scale_val),
            zero_point_val, quant_min, quant_max,
            static_cast<float>(grad_factor));
    } else {
        learnable_backward_kernel<float><<<blocks, threads, 0, stream>>>(
            numel, x.data_ptr<float>(), grad.data_ptr<float>(),
            dx.data_ptr<float>(), dscale_vec.data_ptr<float>(),
            dzp_vec.data_ptr<float>(), static_cast<float>(scale_val),
            zero_point_val, quant_min, quant_max,
            static_cast<float>(grad_factor));
    }
    checkCuda(cudaGetLastError(), "CUDA learnable backward kernel");
    return {std::move(dx), std::move(dscale_vec.sum().reshape({1})),
            std::move(dzp_vec.sum().reshape({1}))};
}

std::tuple<Tensor, Tensor, DType> promote_learnable_pair(const Tensor& grad,
                                                         const Tensor& x) {
    check_real_dtype(x, "fake_quantize backward");
    if (!isFloatingType(grad.dtype())) {
        TP_THROW(TypeError,
                 "fake_quantize backward: expected a floating point grad");
    }
    DType compute = x.dtype();
    if (compute == DType::Float16 || compute == DType::BFloat16) {
        compute = DType::Float32;
    }
    Tensor xc = (x.dtype() == compute) ? x : x.to(compute);
    Tensor gc = (grad.dtype() == compute) ? grad : grad.to(compute);
    xc = xc.is_contiguous() ? xc : xc.contiguous();
    gc = gc.is_contiguous() ? gc : gc.contiguous();
    if (xc.numel() != gc.numel()) {
        TP_THROW(ValueError,
                 "fake_quantize backward: X and dY must have the same number "
                 "of elements");
    }
    return {gc, xc, compute};
}

template <typename T>
__global__ void learnable_backward_per_channel_kernel(
    int64_t numel, const T* __restrict__ x, const T* __restrict__ dy,
    T* __restrict__ dx, T* __restrict__ dscale, T* __restrict__ dzp,
    int64_t stride_on_axis, int64_t channels, const float* __restrict__ scales,
    const float* __restrict__ zero_points, int64_t quant_min,
    int64_t quant_max, float grad_factor) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t c = (i / stride_on_axis) % channels;
    const float inv_scale = 1.0f / scales[c];
    const float zpf = zero_points[c];
    const float dscale_small = static_cast<float>(quant_min) - zpf;
    const float dscale_big = static_cast<float>(quant_max) - zpf;
    const float xf = static_cast<float>(x[i]);
    const float dyf = static_cast<float>(dy[i]);
    const int64_t xq = static_cast<int64_t>(nearbyintf(xf * inv_scale)) +
                       static_cast<int64_t>(zpf);
    dx[i] = static_cast<T>(dyf * (xq >= quant_min && xq <= quant_max));
    const float xfq = static_cast<float>(
        (std::min<int64_t>(std::max<int64_t>(xq, quant_min), quant_max)) -
        static_cast<int64_t>(zpf)) * scales[c];
    if (xq < quant_min || xq > quant_max) {
        dscale[i] = static_cast<T>(
            (dyf * ((xq < quant_min) ? dscale_small : dscale_big)) *
            grad_factor);
        dzp[i] = static_cast<T>(dyf * (-1.0f) * scales[c] * grad_factor);
    } else {
        dscale[i] = static_cast<T>(dyf * (xfq - xf) * inv_scale *
                                   grad_factor);
        dzp[i] = static_cast<T>(0);
    }
}

} // namespace

std::tuple<Tensor, Tensor> fake_quantize_per_tensor_affine_cachemask_cuda(
    const Tensor& self, double scale, int64_t zero_point, int64_t quant_min,
    int64_t quant_max) {
    check_real_dtype(self, "fake_quantize_per_tensor_affine");
    check_fake_quant_range(zero_point, quant_min, quant_max);
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor out = Tensor::empty(self.shape(), self.dtype(), self.device());
    Tensor mask = Tensor::empty(self.shape(), DType::Bool, self.device());
    launch_fake_quant_per_tensor(input, out, mask, scale, zero_point,
                                 quant_min, quant_max);
    return {std::move(out), std::move(mask)};
}

Tensor fake_quantize_per_tensor_affine_cuda(const Tensor& self, double scale,
                                            int64_t zero_point,
                                            int64_t quant_min,
                                            int64_t quant_max) {
    return std::get<0>(fake_quantize_per_tensor_affine_cachemask_cuda(
        self, scale, zero_point, quant_min, quant_max));
}

std::tuple<Tensor, Tensor>
_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    const Tensor& fake_quant_enabled, int64_t quant_min, int64_t quant_max) {
    check_real_dtype(self, "fake_quantize_per_tensor_affine");
    if (quant_min > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize(): quant_min must be <= quant_max");
    }
    TP_CHECK(scale.numel() == 1 && zero_point.numel() == 1 &&
                 fake_quant_enabled.numel() == 1,
             "fake_quantize(): scale, zero_point and the fake-quant flag "
             "must be 1-element tensors");
    Tensor out = Tensor::empty(self.shape(), self.dtype(), self.device());
    Tensor mask = Tensor::empty(self.shape(), DType::Bool, self.device());
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    const int64_t numel = input.numel();
    if (numel == 0) return {std::move(out), std::move(mask)};
    launch_fake_quant_tensor_qparams(input, out, mask, scale, zero_point,
                                     fake_quant_enabled, quant_min,
                                     quant_max);
    return {std::move(out), std::move(mask)};
}

Tensor fake_quantize_per_tensor_affine_tensor_qparams_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t quant_min, int64_t quant_max) {
    Tensor enabled = Tensor::full({1}, Scalar(static_cast<int64_t>(1)),
                                  DType::Int64, self.device());
    return std::get<0>(
        _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cuda(
            self, scale, zero_point, enabled, quant_min, quant_max));
}

Tensor fake_quantize_per_tensor_affine_cachemask_backward_cuda(
    const Tensor& grad, const Tensor& mask) {
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "fake_quantize backward: mask must be Bool");
    }
    if (mask.numel() != grad.numel()) {
        TP_THROW(ValueError,
                 "fake_quantize backward: mask and grad must have the same "
                 "number of elements");
    }
    if (grad.numel() == 0) return grad;
    const Tensor gc = grad.is_contiguous() ? grad : grad.contiguous();
    const Tensor mc = mask.is_contiguous() ? mask : mask.contiguous();
    return masked_grad_cuda(gc, mc);
}

Tensor _fake_quantize_learnable_per_tensor_affine_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t quant_min, int64_t quant_max, double grad_factor) {
    (void)grad_factor;
    check_real_dtype(self, "fake_quantize_per_tensor_affine");
    TP_CHECK(scale.numel() == 1 && zero_point.numel() == 1,
             "fake_quantize(): scale and zero_point must be 1-element "
             "tensors");
    const double scale_val = scale.item().toDouble();
    double zp_fp = std::nearbyint(zero_point.item().toDouble());
    zp_fp = std::min(static_cast<double>(quant_max),
                     std::max(static_cast<double>(quant_min), zp_fp));
    return fake_quantize_per_tensor_affine_cuda(
        self, scale_val, static_cast<int64_t>(zp_fp), quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor>
_fake_quantize_learnable_per_tensor_affine_backward_cuda(
    const Tensor& grad, const Tensor& self, const Tensor& scale,
    const Tensor& zero_point, int64_t quant_min, int64_t quant_max,
    double grad_factor) {
    TP_CHECK(scale.numel() == 1 && zero_point.numel() == 1,
             "fake_quantize backward: scale and zero_point must be "
             "1-element tensors");
    double zp_fp = zero_point.item().toDouble() + 0.5;
    zp_fp = std::min(static_cast<double>(quant_max),
                     std::max(static_cast<double>(quant_min), zp_fp));
    if (quant_min > 0 || quant_max < 0) {
        TP_THROW(ValueError,
                 "fake_quantize backward: the quantization range must "
                 "include 0");
    }
    const int64_t zero_point_val = static_cast<int64_t>(zp_fp);
    if (zero_point_val < quant_min || zero_point_val > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize backward: zero_point out of the quantized "
                 "range");
    }
    if (self.numel() == 0) {
        return {self, scale, zero_point};
    }
    auto promoted = promote_learnable_pair(grad, self);
    return learnable_backward_cuda(
        std::get<0>(promoted), std::get<1>(promoted),
        scale.item().toDouble(), zero_point_val, quant_min, quant_max,
        grad_factor, std::get<2>(promoted));
}

// ---------------------------------------------------------------------------
// Per-channel fake quantization: scale/zero_point arrays are indexed by the
// channel of each element under the axis-major layout.
// ---------------------------------------------------------------------------

namespace {

template <typename T>
__global__ void fake_quant_per_channel_kernel(
    int64_t numel, const T* __restrict__ input, T* __restrict__ output,
    bool* __restrict__ mask, int64_t stride_on_axis, int64_t channels,
    const float* __restrict__ scales,
    const int64_t* __restrict__ zero_points, int64_t quant_min,
    int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t c = (i / stride_on_axis) % channels;
    const float inv_scale = 1.0f / scales[c];
    const float raw = nearbyintf(static_cast<float>(input[i]) * inv_scale) +
                      static_cast<float>(zero_points[c]);
    const int64_t q = static_cast<int64_t>(fminf(
        static_cast<float>(quant_max),
        fmaxf(static_cast<float>(quant_min), raw)));
    output[i] = static_cast<T>((static_cast<float>(q) -
                                static_cast<float>(zero_points[c])) *
                               scales[c]);
    mask[i] = raw >= static_cast<float>(quant_min) &&
              raw <= static_cast<float>(quant_max);
}

template <typename T>
__global__ void fake_quant_per_channel_kernel_floatzp(
    int64_t numel, const T* __restrict__ input, T* __restrict__ output,
    bool* __restrict__ mask, int64_t stride_on_axis, int64_t channels,
    const float* __restrict__ scales, const float* __restrict__ zero_points,
    int64_t quant_min, int64_t quant_max) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t c = (i / stride_on_axis) % channels;
    const float inv_scale = 1.0f / scales[c];
    const float raw = nearbyintf(static_cast<float>(input[i]) * inv_scale +
                                 zero_points[c]);
    const int64_t q = static_cast<int64_t>(fminf(
        static_cast<float>(quant_max),
        fmaxf(static_cast<float>(quant_min), raw)));
    output[i] = static_cast<T>((static_cast<float>(q) - zero_points[c]) *
                               scales[c]);
    mask[i] = raw >= static_cast<float>(quant_min) &&
              raw <= static_cast<float>(quant_max);
}

} // namespace

std::tuple<Tensor, Tensor> fake_quantize_per_channel_affine_cachemask_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max) {
    check_real_dtype(self, "fake_quantize_per_channel_affine");
    TP_CHECK(scale.dim() == 1 && zero_point.dim() == 1,
             "fake_quantize(): scale and zero_point must be 1-D tensors");
    TP_CHECK(scale.numel() == zero_point.numel(),
             "fake_quantize(): scale and zero_point must have the same "
             "size");
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError, "fake_quantize(): axis out of range");
    }
    TP_CHECK(scale.numel() == self.size(axis),
             "fake_quantize(): scale size must match the quantized "
             "dimension");
    if (quant_min > quant_max) {
        TP_THROW(ValueError,
                 "fake_quantize(): quant_min must be <= quant_max");
    }
    if (isIntegralType(zero_point.dtype())) {
        Tensor zpc = zero_point.to(DType::Int64).contiguous();
        auto mm = Tensor::aminmax(zpc, {}, false);
        const int64_t min_zp = std::get<0>(mm).item().to<int64_t>();
        const int64_t max_zp = std::get<1>(mm).item().to<int64_t>();
        if (min_zp < quant_min || max_zp > quant_max) {
            TP_THROW(ValueError,
                     "fake_quantize(): zero_point out of the quantized "
                     "range");
        }
    }

    Tensor out = Tensor::empty(self.shape(), self.dtype(), self.device());
    Tensor mask = Tensor::empty(self.shape(), DType::Bool, self.device());
    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor sc = scale.to(DType::Float32).contiguous();
    const bool zp_float = !isIntegralType(zero_point.dtype());
    Tensor zpf = zp_float ? zero_point.to(DType::Float32).contiguous()
                          : zero_point.to(DType::Int64).contiguous();
    int64_t stride_on_axis = 1;
    for (int64_t d = axis + 1; d < input.dim(); ++d) {
        stride_on_axis *= input.size(d);
    }
    const int64_t channels = scale.numel();
    const int64_t numel = input.numel();
    if (numel == 0) return {std::move(out), std::move(mask)};
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    switch (input.dtype()) {
        case DType::Float32:
            if (zp_float) {
                fake_quant_per_channel_kernel_floatzp<float>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<float>(), out.data_ptr<float>(),
                        mask.data_ptr<bool>(), stride_on_axis, channels,
                        sc.data_ptr<float>(), zpf.data_ptr<float>(),
                        quant_min, quant_max);
            } else {
                fake_quant_per_channel_kernel<float>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<float>(), out.data_ptr<float>(),
                        mask.data_ptr<bool>(), stride_on_axis, channels,
                        sc.data_ptr<float>(), zpf.data_ptr<int64_t>(),
                        quant_min, quant_max);
            }
            break;
        case DType::Float16:
            if (zp_float) {
                fake_quant_per_channel_kernel_floatzp<Half>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<Half>(), out.data_ptr<Half>(),
                        mask.data_ptr<bool>(), stride_on_axis, channels,
                        sc.data_ptr<float>(), zpf.data_ptr<float>(),
                        quant_min, quant_max);
            } else {
                fake_quant_per_channel_kernel<Half>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<Half>(), out.data_ptr<Half>(),
                        mask.data_ptr<bool>(), stride_on_axis, channels,
                        sc.data_ptr<float>(), zpf.data_ptr<int64_t>(),
                        quant_min, quant_max);
            }
            break;
        case DType::BFloat16:
            if (zp_float) {
                fake_quant_per_channel_kernel_floatzp<BFloat16>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<BFloat16>(),
                        out.data_ptr<BFloat16>(), mask.data_ptr<bool>(),
                        stride_on_axis, channels, sc.data_ptr<float>(),
                        zpf.data_ptr<float>(), quant_min, quant_max);
            } else {
                fake_quant_per_channel_kernel<BFloat16>
                    <<<blocks, threads, 0, stream>>>(
                        numel, input.data_ptr<BFloat16>(),
                        out.data_ptr<BFloat16>(), mask.data_ptr<bool>(),
                        stride_on_axis, channels, sc.data_ptr<float>(),
                        zpf.data_ptr<int64_t>(), quant_min, quant_max);
            }
            break;
        default:
            TP_THROW(TypeError, "fake_quantize_per_channel_affine(): "
                                "unsupported input dtype");
    }
    checkCuda(cudaGetLastError(), "CUDA fake_quantize_per_channel kernel");
    return {std::move(out), std::move(mask)};
}

Tensor fake_quantize_per_channel_affine_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max) {
    return std::get<0>(fake_quantize_per_channel_affine_cachemask_cuda(
        self, scale, zero_point, axis, quant_min, quant_max));
}

Tensor fake_quantize_per_channel_affine_cachemask_backward_cuda(
    const Tensor& grad, const Tensor& mask) {
    return fake_quantize_per_tensor_affine_cachemask_backward_cuda(grad,
                                                                   mask);
}

Tensor _fake_quantize_learnable_per_channel_affine_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
    (void)grad_factor;
    Tensor zp = zero_point.to(DType::Float32)
                    .round()
                    .clamp(Scalar(static_cast<int64_t>(quant_min)),
                           Scalar(static_cast<int64_t>(quant_max)))
                    .to(DType::Int64);
    return fake_quantize_per_channel_affine_cuda(
        self, scale.to(DType::Float32), zp, axis, quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor>
_fake_quantize_learnable_per_channel_affine_backward_cuda(
    const Tensor& grad, const Tensor& self, const Tensor& scale,
    const Tensor& zero_point, int64_t axis, int64_t quant_min,
    int64_t quant_max, double grad_factor) {
    TP_CHECK(scale.dim() == 1 && zero_point.dim() == 1 &&
                 scale.numel() == zero_point.numel(),
             "fake_quantize backward: scale and zero_point must be 1-D "
             "tensors of the same size");
    if (quant_min > 0 || quant_max < 0) {
        TP_THROW(ValueError,
                 "fake_quantize backward: the quantization range must "
                 "include 0");
    }
    if (axis < 0) axis += self.dim();
    if (axis < 0 || axis >= self.dim()) {
        TP_THROW(ValueError, "fake_quantize backward: axis out of range");
    }
    TP_CHECK(scale.numel() == self.size(axis),
             "fake_quantize backward: scale size must match the quantized "
             "dimension");
    if (self.numel() == 0) {
        return {self, scale, zero_point};
    }

    auto promoted = promote_learnable_pair(grad, self);
    const Tensor& x = std::get<1>(promoted);
    const DType compute = std::get<2>(promoted);

    Tensor zp_f = zero_point.to(DType::Float32)
                      .round()
                      .clamp(Scalar(static_cast<int64_t>(quant_min)),
                             Scalar(static_cast<int64_t>(quant_max)))
                      .contiguous();
    Tensor sc = scale.to(DType::Float32).contiguous();
    int64_t stride_on_axis = 1;
    for (int64_t d = axis + 1; d < x.dim(); ++d) stride_on_axis *= x.size(d);
    const int64_t channels = scale.numel();
    const int64_t numel = x.numel();

    Tensor dx = Tensor::empty(x.shape(), compute, x.device());
    Tensor dscale_vec = Tensor::empty(x.shape(), compute, x.device());
    Tensor dzp_vec = Tensor::empty(x.shape(), compute, x.device());
    if (numel > 0) {
        const cudaStream_t stream = getCurrentCUDAStream().stream();
        const int threads = 256;
        const int blocks = static_cast<int>((numel + threads - 1) / threads);
        if (compute == DType::Float64) {
            TP_THROW(TypeError,
                     "fake_quantize backward: Float64 per-channel backward "
                     "is not supported on this device");
        }
        learnable_backward_per_channel_kernel<float>
            <<<blocks, threads, 0, stream>>>(
                numel, x.data_ptr<float>(),
                std::get<0>(promoted).data_ptr<float>(),
                dx.data_ptr<float>(), dscale_vec.data_ptr<float>(),
                dzp_vec.data_ptr<float>(), stride_on_axis, channels,
                sc.data_ptr<float>(), zp_f.data_ptr<float>(), quant_min,
                quant_max, static_cast<float>(grad_factor));
        checkCuda(cudaGetLastError(),
                  "CUDA learnable per-channel backward kernel");
    }
    std::vector<int64_t> reduce_dims;
    for (int64_t d = 0; d < x.dim(); ++d) {
        if (d != axis) reduce_dims.push_back(d);
    }
    Tensor dscale = dscale_vec.sum(reduce_dims, false);
    Tensor dzp = dzp_vec.sum(reduce_dims, false);
    return {std::move(dx), std::move(dscale), std::move(dzp)};
}



TENSORPLAY_LIBRARY_IMPL(CUDA, FakeQuantKernels) {
    m.impl("fake_quantize_per_tensor_affine",
           fake_quantize_per_tensor_affine_cuda);
    m.impl("fake_quantize_per_tensor_affine.tensor_qparams",
           fake_quantize_per_tensor_affine_tensor_qparams_cuda);
    m.impl("fake_quantize_per_tensor_affine_cachemask",
           fake_quantize_per_tensor_affine_cachemask_cuda);
    m.impl("_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
           _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cuda);
    m.impl("fake_quantize_per_tensor_affine_cachemask_backward",
           fake_quantize_per_tensor_affine_cachemask_backward_cuda);
    m.impl("_fake_quantize_learnable_per_tensor_affine",
           _fake_quantize_learnable_per_tensor_affine_cuda);
    m.impl("_fake_quantize_learnable_per_tensor_affine_backward",
           _fake_quantize_learnable_per_tensor_affine_backward_cuda);
    m.impl("fake_quantize_per_channel_affine",
           fake_quantize_per_channel_affine_cuda);
    m.impl("fake_quantize_per_channel_affine_cachemask",
           fake_quantize_per_channel_affine_cachemask_cuda);
    m.impl("fake_quantize_per_channel_affine_cachemask_backward",
           fake_quantize_per_channel_affine_cachemask_backward_cuda);
    m.impl("_fake_quantize_learnable_per_channel_affine",
           _fake_quantize_learnable_per_channel_affine_cuda);
    m.impl("_fake_quantize_learnable_per_channel_affine_backward",
           _fake_quantize_learnable_per_channel_affine_backward_cuda);
}

}
}

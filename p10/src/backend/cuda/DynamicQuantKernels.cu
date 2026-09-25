#include "QuantKernels.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Quantizer.h"
#include "Utils.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

std::tuple<Tensor, Tensor> fake_quantize_per_channel_affine_cachemask_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    int64_t axis, int64_t quant_min, int64_t quant_max);

std::tuple<Tensor, Tensor>
_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cuda(
    const Tensor& self, const Tensor& scale, const Tensor& zero_point,
    const Tensor& fake_quant_enabled, int64_t quant_min, int64_t quant_max);

namespace {

constexpr double kSmallScaleThreshold = 6.1e-5;

struct QParams {
    double scale;
    int64_t zero_point;
};

// Host-side qparams derivation from a real range [min, max] over the grid
// [qmin, qmax]: widen to contain 0, repair degenerate/too-small scales,
// then nudge the zero point into the grid with round-half-even.
QParams choose_qparams_host(double min, double max, int64_t qmin,
                            int64_t qmax, bool preserve_sparsity) {
    TP_CHECK(min <= max, "choose qparams: min must be <= max");
    if (min < 0 && max > 0 && preserve_sparsity) {
        const int64_t symmetric_qmin = -((qmax - qmin) / 2 + 1);
        const int64_t symmetric_qmax = (qmax - qmin) / 2;
        const double max_scale = std::max(
            std::fabs(min / static_cast<double>(symmetric_qmin)),
            std::fabs(max / static_cast<double>(symmetric_qmax)));
        min = max_scale * static_cast<double>(symmetric_qmin);
        max = max_scale * static_cast<double>(symmetric_qmax);
    }
    min = std::min(min, 0.0);
    max = std::max(max, 0.0);
    TP_CHECK(qmin < qmax, "choose qparams: qmin must be < qmax");
    double scale = (max - min) / static_cast<double>(qmax - qmin);
    if (static_cast<float>(scale) == 0.0f ||
        std::isinf(1.0f / static_cast<float>(scale))) {
        scale = 0.1;
    }
    if (scale < kSmallScaleThreshold) {
        const double org_scale = scale;
        scale = kSmallScaleThreshold;
        if (min == 0.0) {
            max = kSmallScaleThreshold * static_cast<double>(qmax - qmin);
        } else if (max == 0.0) {
            min = -kSmallScaleThreshold * static_cast<double>(qmax - qmin);
        } else {
            const double amplifier = kSmallScaleThreshold / org_scale;
            min *= amplifier;
            max *= amplifier;
        }
    }
    const double zero_point_from_min =
        static_cast<double>(qmin) - min / scale;
    const double zero_point_from_max =
        static_cast<double>(qmax) - max / scale;
    const double zero_point_from_min_error =
        std::abs(static_cast<double>(qmin)) - std::abs(min / scale);
    const double zero_point_from_max_error =
        std::abs(static_cast<double>(qmax)) - std::abs(max / scale);
    double initial_zero_point =
        zero_point_from_min_error < zero_point_from_max_error
            ? zero_point_from_min
            : zero_point_from_max;
    if (min < 0 && max > 0 && preserve_sparsity) {
        initial_zero_point = static_cast<double>(qmin + qmax) / 2.0;
    }
    int64_t nudged_zero_point = 0;
    if (initial_zero_point < static_cast<double>(qmin)) {
        nudged_zero_point = qmin;
    } else if (initial_zero_point > static_cast<double>(qmax)) {
        nudged_zero_point = qmax;
    } else {
        nudged_zero_point =
            static_cast<int64_t>(std::nearbyint(initial_zero_point));
    }
    return {scale, nudged_zero_point};
}

void check_real_dtype(const Tensor& self, const char* op) {
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError,
                 std::string(op) + ": expected a floating point tensor, got " +
                     toString(self.dtype()));
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Dynamic quantization and fused moving-average observer + fake quant.
// ---------------------------------------------------------------------------

namespace {

template <typename T>
__global__ void dynamic_quantize_kernel(int64_t numel,
                                        const float* __restrict__ input,
                                        T* __restrict__ output, float scale,
                                        int64_t zero_point, int64_t qmin,
                                        int64_t qmax) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const float q = nearbyintf(input[i] / scale) +
                    static_cast<float>(zero_point);
    output[i] = static_cast<T>(fminf(static_cast<float>(qmax),
                                     fmaxf(static_cast<float>(qmin), q)));
}

std::pair<double, double> tensor_min_max(const Tensor& self) {
    const Tensor input = self.contiguous();
    auto mm = Tensor::aminmax(input.reshape({input.numel()}), {}, false);
    return {std::get<0>(mm).item().toDouble(),
            std::get<1>(mm).item().toDouble()};
}

// Device-side moving average of the running min/max state.
__global__ void moving_average_minmax_kernel(
    int64_t size, const float* __restrict__ x_min,
    const float* __restrict__ x_max, float* __restrict__ running_min,
    float* __restrict__ running_max, float averaging_const) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size) return;
    running_min[i] = ::isinf(running_min[i])
                         ? x_min[i]
                         : running_min[i] +
                               averaging_const * (x_min[i] - running_min[i]);
    running_max[i] = ::isinf(running_max[i])
                         ? x_max[i]
                         : running_max[i] +
                               averaging_const * (x_max[i] - running_max[i]);
}

// Derives qparams from the running range on device; entries are only
// written while the fake-quant flag is on.
__global__ void choose_qparams_kernel(const int64_t* __restrict__ fake_on,
                                      const float* __restrict__ x_min,
                                      const float* __restrict__ x_max,
                                      int64_t qmin, int64_t qmax,
                                      bool preserve_sparsity, int64_t size,
                                      float* __restrict__ scale,
                                      int64_t* __restrict__ zero_point) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size || *fake_on == 0) return;
    float min_val = x_min[i];
    float max_val = x_max[i];
    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
        const int64_t symmetric_qmin = -((qmax - qmin) / 2 + 1);
        const int64_t symmetric_qmax = (qmax - qmin) / 2;
        const double max_scale = fmax(
            fabs(min_val / static_cast<double>(symmetric_qmin)),
            fabs(max_val / static_cast<double>(symmetric_qmax)));
        min_val = static_cast<float>(max_scale *
                                     static_cast<double>(symmetric_qmin));
        max_val = static_cast<float>(max_scale *
                                     static_cast<double>(symmetric_qmax));
    }
    min_val = fminf(min_val, 0.0f);
    max_val = fmaxf(max_val, 0.0f);
    float sc = static_cast<float>(
        (static_cast<double>(max_val) - static_cast<double>(min_val)) /
        static_cast<double>(qmax - qmin));
    if (sc == 0.0f || ::isinf(1.0f / sc)) sc = 0.1f;
    if (sc < static_cast<float>(kSmallScaleThreshold)) {
        const float org = sc;
        sc = static_cast<float>(kSmallScaleThreshold);
        if (min_val == 0.0f) {
            max_val = sc * static_cast<float>(qmax - qmin);
        } else if (max_val == 0.0f) {
            min_val = -sc * static_cast<float>(qmax - qmin);
        } else {
            const float amplifier = sc / org;
            min_val *= amplifier;
            max_val *= amplifier;
        }
    }
    const double zp_from_min =
        static_cast<double>(qmin) -
        static_cast<double>(min_val) / static_cast<double>(sc);
    const double zp_from_max =
        static_cast<double>(qmax) -
        static_cast<double>(max_val) / static_cast<double>(sc);
    const double err_min = std::abs(static_cast<double>(qmin)) -
                           std::abs(static_cast<double>(min_val) /
                                    static_cast<double>(sc));
    const double err_max = std::abs(static_cast<double>(qmax)) -
                           std::abs(static_cast<double>(max_val) /
                                    static_cast<double>(sc));
    double initial_zp = err_min < err_max ? zp_from_min : zp_from_max;
    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
        initial_zp = static_cast<double>(qmin + qmax) / 2.0;
    }
    int64_t nudged = 0;
    if (initial_zp < static_cast<double>(qmin)) {
        nudged = qmin;
    } else if (initial_zp > static_cast<double>(qmax)) {
        nudged = qmax;
    } else {
        nudged = static_cast<int64_t>(nearbyint(initial_zp));
    }
    scale[i] = sc;
    zero_point[i] = nudged;
}

void fill_or_resize_f32(Tensor& t, int64_t size, float value,
                        const Device& device) {
    if (t.numel() == 0) {
        t = Tensor::full({size}, Scalar(value), DType::Float32, device);
        return;
    }
    if (t.numel() != size) {
        t.resize_({size});
        t.fill_(Scalar(value));
        return;
    }
    t.fill_(Scalar(value));
}

} // namespace

Tensor quantize_per_tensor_dynamic_cuda(const Tensor& self, DType dtype,
                                        bool reduce_range) {
    if (dtype != DType::QInt8 && dtype != DType::QUInt8 &&
        dtype != DType::Float16) {
        TP_THROW(TypeError,
                 "quantize_per_tensor_dynamic(): only QInt8, QUInt8 and "
                 "Float16 outputs are supported");
    }
    check_real_dtype(self, "quantize_per_tensor_dynamic");
    if (dtype == DType::Float16) {
        return self.contiguous().to(DType::Float16);
    }
    const auto mm = tensor_min_max(self);
    int64_t qmin = (dtype == DType::QInt8) ? -128 : 0;
    int64_t qmax = (dtype == DType::QInt8) ? 127 : 255;
    if (reduce_range) {
        qmin /= 2;
        qmax /= 2;
    }
    const QParams qp = choose_qparams_host(mm.first, mm.second, qmin, qmax,
                                           /*preserve_sparsity=*/false);

    const Tensor input = self.contiguous();
    const Tensor input_f =
        (input.dtype() == DType::Float32) ? input : input.to(DType::Float32);
    Tensor out = Tensor::empty(self.shape(), dtype, self.device());
    const int64_t numel = input.numel();
    if (numel == 0) return out;
    const cudaStream_t stream = getCurrentCUDAStream().stream();
    const int threads = 256;
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (dtype == DType::QInt8) {
        dynamic_quantize_kernel<int8_t><<<blocks, threads, 0, stream>>>(
            numel, input_f.data_ptr<float>(), out.data_ptr<int8_t>(),
            static_cast<float>(qp.scale), qp.zero_point, qmin, qmax);
    } else {
        dynamic_quantize_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
            numel, input_f.data_ptr<float>(), out.data_ptr<uint8_t>(),
            static_cast<float>(qp.scale), qp.zero_point, qmin, qmax);
    }
    checkCuda(cudaGetLastError(), "CUDA quantize_per_tensor_dynamic kernel");
    out.impl()->set_quantizer(
        make_per_tensor_affine_quantizer(qp.scale, qp.zero_point, dtype));
    return out;
}

std::tuple<double, int64_t> _choose_qparams_per_tensor_cuda(
    const Tensor& self, bool reduce_range) {
    check_real_dtype(self, "_choose_qparams_per_tensor");
    const auto mm = tensor_min_max(self);
    int64_t qmin = 0;
    int64_t qmax = 255;
    if (reduce_range) {
        qmin /= 2;
        qmax /= 2;
    }
    const QParams qp =
        choose_qparams_host(mm.first, mm.second, qmin, qmax, false);
    return {qp.scale, qp.zero_point};
}

std::tuple<Tensor, Tensor> _fused_moving_avg_obs_fq_helper_cuda(
    const Tensor& self, const Tensor& observer_on,
    const Tensor& fake_quant_on, Tensor& running_min, Tensor& running_max,
    Tensor& scale, Tensor& zero_point, double averaging_const,
    int64_t quant_min, int64_t quant_max, int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant) {
    if (ch_axis >= self.dim()) {
        TP_THROW(ValueError,
                 "fused_moving_avg_obs_fake_quant(): ch_axis must be < "
                 "self.dim()");
    }
    check_real_dtype(self, "fused_moving_avg_obs_fake_quant");
    const bool observe = observer_on.item().to<int64_t>() != 0;
    const bool fake_on = fake_quant_on.item().to<int64_t>() != 0;

    if (per_row_fake_quant) {
        Tensor y = self;
        if (self.dim() != 2) {
            std::vector<int64_t> dims(self.dim());
            for (int64_t d = 0; d < self.dim(); ++d) dims[d] = d;
            dims[ch_axis] = 0;
            dims[0] = ch_axis;
            y = self.permute(dims).flatten(1);
        }
        const int64_t size = self.size(ch_axis);
        if (running_min.numel() == 0) {
            fill_or_resize_f32(running_min, size,
                               std::numeric_limits<float>::infinity(),
                               self.device());
            fill_or_resize_f32(running_max, size,
                               -std::numeric_limits<float>::infinity(),
                               self.device());
            scale.resize_({size});
            zero_point.resize_({size});
        }
        if (observe) {
            auto mm = Tensor::aminmax(y.contiguous(), {1}, false);
            Tensor mn = std::get<0>(mm).to(DType::Float32).contiguous();
            Tensor mx = std::get<1>(mm).to(DType::Float32).contiguous();
            const cudaStream_t stream = getCurrentCUDAStream().stream();
            const int threads = 256;
            const int blocks = static_cast<int>((size + threads - 1) / threads);
            moving_average_minmax_kernel<<<blocks, threads, 0, stream>>>(
                size, mn.data_ptr<float>(), mx.data_ptr<float>(),
                running_min.data_ptr<float>(), running_max.data_ptr<float>(),
                static_cast<float>(averaging_const));
            checkCuda(cudaGetLastError(), "CUDA moving average kernel");
        }
        if (!fake_on) {
            Tensor mask = Tensor::full_like(self, 1, DType::Bool);
            return {self.clone(), std::move(mask)};
        }
        Tensor sc = Tensor::empty({size}, DType::Float32, self.device());
        Tensor zp = Tensor::empty({size}, DType::Int64, self.device());
        {
            const cudaStream_t stream = getCurrentCUDAStream().stream();
            const int threads = 256;
            const int blocks =
                static_cast<int>((size + threads - 1) / threads);
            Tensor fq = fake_quant_on.to(DType::Int64).contiguous();
            choose_qparams_kernel<<<blocks, threads, 0, stream>>>(
                fq.data_ptr<int64_t>(), running_min.data_ptr<float>(),
                running_max.data_ptr<float>(), quant_min, quant_max,
                symmetric_quant, size, sc.data_ptr<float>(),
                zp.data_ptr<int64_t>());
            checkCuda(cudaGetLastError(), "CUDA choose qparams kernel");
        }
        scale.copy_(sc);
        zero_point.copy_(zp);
        return fake_quantize_per_channel_affine_cachemask_cuda(
            self, sc, zp, ch_axis, quant_min, quant_max);
    }

    if (observe) {
        auto mm = Tensor::aminmax(self.reshape({self.numel()}), {}, false);
        Tensor mn = std::get<0>(mm).to(DType::Float32).contiguous();
        Tensor mx = std::get<1>(mm).to(DType::Float32).contiguous();
        const cudaStream_t stream = getCurrentCUDAStream().stream();
        moving_average_minmax_kernel<<<1, 1, 0, stream>>>(
            1, mn.data_ptr<float>(), mx.data_ptr<float>(),
            running_min.data_ptr<float>(), running_max.data_ptr<float>(),
            static_cast<float>(averaging_const));
        checkCuda(cudaGetLastError(), "CUDA moving average kernel");
    }
    if (!fake_on) {
        Tensor mask = Tensor::full_like(self, 1, DType::Bool);
        return {self.clone(), std::move(mask)};
    }
    const double mn = running_min.item().toDouble();
    const double mx = running_max.item().toDouble();
    const QParams qp =
        choose_qparams_host(mn, mx, quant_min, quant_max, symmetric_quant);
    Tensor sc = Tensor::full({1}, Scalar(static_cast<float>(qp.scale)),
                             DType::Float32, self.device());
    Tensor zp = Tensor::full({1}, Scalar(qp.zero_point), DType::Int64,
                             self.device());
    scale.copy_(sc);
    zero_point.copy_(zp);
    Tensor enabled = Tensor::full({1}, Scalar(static_cast<int64_t>(1)),
                                  DType::Int64, self.device());
    return _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_cuda(
        self, sc, zp, enabled, quant_min, quant_max);
}

Tensor fused_moving_avg_obs_fake_quant_cuda(
    const Tensor& self, const Tensor& observer_on,
    const Tensor& fake_quant_on, Tensor& running_min, Tensor& running_max,
    Tensor& scale, Tensor& zero_point, double averaging_const,
    int64_t quant_min, int64_t quant_max, int64_t ch_axis,
    bool per_row_fake_quant, bool symmetric_quant) {
    if (self.numel() == 0) {
        return self.clone();
    }
    return std::get<0>(_fused_moving_avg_obs_fq_helper_cuda(
        self, observer_on, fake_quant_on, running_min, running_max, scale,
        zero_point, averaging_const, quant_min, quant_max, ch_axis,
        per_row_fake_quant, symmetric_quant));
}


TENSORPLAY_LIBRARY_IMPL(CUDA, DynamicQuantKernels) {
    m.impl("quantize_per_tensor_dynamic", quantize_per_tensor_dynamic_cuda);
    m.impl("_choose_qparams_per_tensor", _choose_qparams_per_tensor_cuda);
    m.impl("fused_moving_avg_obs_fake_quant",
           fused_moving_avg_obs_fake_quant_cuda);
    m.impl("_fused_moving_avg_obs_fq_helper",
           _fused_moving_avg_obs_fq_helper_cuda);
}

}
}

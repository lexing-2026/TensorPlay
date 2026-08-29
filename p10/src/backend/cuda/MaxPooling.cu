// CUDA 1-D max pooling uses dilated pooling kernels. This is a real CUDA
// kernel for the no-grad fast

#include "Tensor.h"
#include "CUDARuntime.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "GradMode.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace tensorplay::cuda {

namespace ops = tensorplay::tpx::ops;

namespace {

int64_t div_rtn(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

int64_t pooling_output_shape(int64_t input, int64_t kernel, int64_t padding,
                             int64_t stride, int64_t dilation, bool ceil_mode) {
    if (stride <= 0) TP_THROW(RuntimeError, "stride must be greater than zero");
    if (kernel <= 0) TP_THROW(RuntimeError, "kernel_size must be greater than zero");
    if (dilation <= 0) TP_THROW(RuntimeError, "dilation must be greater than zero");
    int64_t output = div_rtn(
        input + 2 * padding - dilation * (kernel - 1) - 1 +
            (ceil_mode ? stride - 1 : 0),
        stride) + 1;
    if (ceil_mode && (output - 1) * stride >= input + padding) --output;
    return output;
}

void check_max_pool1d_args(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (self.dim() != 2 && self.dim() != 3) {
        TP_THROW(RuntimeError, "max_pool1d() expected 2D or 3D input tensor");
    }
    if (kernel_size.size() != 1 || (!stride.empty() && stride.size() != 1) ||
        padding.size() != 1 || dilation.size() != 1) {
        TP_THROW(RuntimeError, "max_pool1d() arguments must contain one int");
    }
    const int64_t step = stride.empty() ? kernel_size[0] : stride[0];
    if ((self.dim() == 2 && (self.size(0) == 0 || self.size(1) == 0)) ||
        (self.dim() == 3 && (self.size(1) == 0 || self.size(2) == 0))) {
        TP_THROW(RuntimeError, "max_pool1d(): input dimensions must be non-zero");
    }
    if (kernel_size[0] <= 0) {
        TP_THROW(RuntimeError, "max_pool1d() kernel_size must be greater than zero");
    }
    if (step <= 0) {
        TP_THROW(RuntimeError, "max_pool1d() stride must be greater than zero");
    }
    if (padding[0] < 0) {
        TP_THROW(RuntimeError, "max_pool1d() padding must be non-negative");
    }
    if (padding[0] > kernel_size[0] / 2) {
        TP_THROW(RuntimeError,
                 "max_pool1d() padding should be at most half of kernel size");
    }
    if (dilation[0] <= 0) {
        TP_THROW(RuntimeError, "max_pool1d() dilation must be greater than zero");
    }
    if (pooling_output_shape(self.size(-1), kernel_size[0], padding[0], step,
                             dilation[0], ceil_mode) <= 0) {
        TP_THROW(RuntimeError, "max_pool1d() Invalid computed output size");
    }
}

template <typename scalar_t>
struct max_pool_accum {
    using type = scalar_t;
};

template <>
struct max_pool_accum<tensorplay::Half> {
    using type = float;
};

template <>
struct max_pool_accum<tensorplay::BFloat16> {
    using type = float;
};

template <typename scalar_t>
__global__ void max_pool1d_kernel(
    const scalar_t* __restrict__ input, scalar_t* __restrict__ output,
    int64_t total, int64_t iw, int64_t ow, int64_t kw, int64_t sj,
    int64_t pj, int64_t dj) {
    using acc_t = typename max_pool_accum<scalar_t>::type;
    const int64_t grid_stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total; index += grid_stride) {
        const int64_t oj = index % ow;
        const int64_t plane = index / ow;
        const int64_t start = oj * sj - pj;
        acc_t max_value = -std::numeric_limits<acc_t>::infinity();
        for (int64_t kj = 0; kj < kw; ++kj) {
            const int64_t ij = start + kj * dj;
            if (ij < 0 || ij >= iw) continue;
            const acc_t value = static_cast<acc_t>(input[plane * iw + ij]);
            if ((value > max_value) || std::isnan(value)) max_value = value;
        }
        output[index] = static_cast<scalar_t>(max_value);
    }
}

template <typename scalar_t>
void launch_max_pool1d(const Tensor& input, Tensor& output, int64_t total,
                       int64_t iw, int64_t ow, int64_t kw, int64_t sj,
                       int64_t pj, int64_t dj) {
    constexpr int threads = 256;
    const int64_t blocks = std::min<int64_t>((total + threads - 1) / threads,
                                             65535);
    max_pool1d_kernel<scalar_t><<<blocks, threads, 0,
                                  getCurrentCUDAStream().stream()>>>(
        input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), total, iw, ow,
        kw, sj, pj, dj);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, "max_pool1d CUDA: ", cudaGetErrorString(error));
    }
}

} // namespace

Tensor max_pool1d_native_cuda(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    check_max_pool1d_args(self, kernel_size, stride, padding, dilation, ceil_mode);

    // argmax.  The no-grad path below avoids the hidden index allocation.
    if (self.requires_grad() && tensorplay::GradMode::is_enabled()) {
        return std::get<0>(ops::max_pool1d_with_indices(
            self, kernel_size, stride, padding, dilation, ceil_mode));
    }

    const int64_t nb = self.dim() == 3 ? self.size(-3) : 1;
    const int64_t nc = self.size(-2);
    const int64_t iw = self.size(-1);
    const int64_t kw = kernel_size[0];
    const int64_t sj = stride.empty() ? kw : stride[0];
    const int64_t pj = padding[0];
    const int64_t dj = dilation[0];
    const int64_t ow = pooling_output_shape(iw, kw, pj, sj, dj, ceil_mode);
    if (ow <= 0) TP_THROW(RuntimeError, "max_pool1d() Invalid computed output size");

    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor output = Tensor::empty({nb, nc, ow}, self.dtype(), self.device());
    const int64_t total = nb * nc * ow;
    if (total == 0) return self.dim() == 2 ? ops::squeeze(output, 0) : output;
    switch (self.dtype()) {
        case DType::Float32:
            launch_max_pool1d<float>(input, output, total, iw, ow, kw, sj, pj, dj);
            break;
        case DType::Float64:
            launch_max_pool1d<double>(input, output, total, iw, ow, kw, sj, pj, dj);
            break;
        case DType::Float16:
            launch_max_pool1d<tensorplay::Half>(input, output, total, iw, ow, kw,
                                                sj, pj, dj);
            break;
        case DType::BFloat16:
            launch_max_pool1d<tensorplay::BFloat16>(input, output, total, iw, ow,
                                                    kw, sj, pj, dj);
            break;
        default:
            return std::get<0>(ops::max_pool1d_with_indices(
                self, kernel_size, stride, padding, dilation, ceil_mode));
    }
    return self.dim() == 2 ? ops::squeeze(output, 0) : output;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeMaxPooling) {
    m.impl("max_pool1d", max_pool1d_native_cuda);
}

} // namespace tensorplay::cuda

#include "QuantNative.h"
// Native quantized concatenation for the CUDA backend.
//
// When every input shares the output affine grid the codes are copied
// verbatim (byte-exact concat).  Otherwise each code is requantized inside
// the kernel from its own affine parameters to the output grid.  The
// relu-fused variant clamps the concatenated codes against the output zero
// point in the integer domain.

#include "QuantCudaKernels.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Quantizer.h"
#include "QuantUtils.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

__global__ void quantized_cat_copy_kernel(
    const int8_t* __restrict__ src, int8_t* __restrict__ dst, int64_t total,
    int64_t seg_inner, int64_t concat_inner, int64_t offset_inner) {
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t o = i / seg_inner;
    const int64_t k = i - o * seg_inner;
    dst[o * concat_inner + offset_inner + k] = src[i];
}

__global__ void quantized_cat_requantize_kernel(
    const int8_t* __restrict__ src, int8_t* __restrict__ dst, int64_t total,
    int64_t seg_inner, int64_t concat_inner, int64_t offset_inner,
    double in_scale, double in_zero_point, double inv_out_scale,
    double out_zero_point) {
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int64_t o = i / seg_inner;
    const int64_t k = i - o * seg_inner;
    const double x = (static_cast<double>(src[i]) - in_zero_point) * in_scale;
    dst[o * concat_inner + offset_inner + k] =
        quant::requantize_value(x, inv_out_scale,
                                static_cast<int64_t>(out_zero_point));
}

__global__ void quantized_cat_relu_kernel(int64_t numel,
                                          int8_t* __restrict__ codes,
                                          int8_t zero_point) {
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    if (codes[i] < zero_point) codes[i] = zero_point;
}

} // namespace

namespace {

Tensor quantized_cat_impl_cuda(const std::vector<Tensor>& tensors,
                               int64_t dim, std::optional<double> scale,
                               std::optional<int64_t> zero_point,
                               bool relu_fused) {
    if (tensors.empty()) {
        TP_THROW(ValueError,
                 "quantized_cat(): expected a non-empty tensor list");
    }
    for (const Tensor& t : tensors) {
        if (t.dtype() != DType::QInt8) {
            TP_THROW(TypeError,
                     "quantized_cat(): only per-tensor QInt8 quantization is "
                     "supported");
        }
        if (t.dim() != tensors[0].dim()) {
            TP_THROW(ValueError,
                     "quantized_cat(): tensors must have the same number of "
                     "dimensions");
        }
    }
    if (dim < 0) dim += tensors[0].dim();
    if (dim < 0 || dim >= tensors[0].dim()) {
        TP_THROW(ValueError, "quantized_cat(): dim out of range");
    }
    for (const Tensor& t : tensors) {
        for (int64_t d = 0; d < t.dim(); ++d) {
            if (d != dim && t.size(d) != tensors[0].size(d)) {
                TP_THROW(ValueError,
                         "quantized_cat(): sizes must match on the "
                         "non-concatenated dimensions");
            }
        }
    }

    const double out_scale =
        scale.has_value() ? *scale : quantized::q_scale(tensors[0]);
    const int64_t out_zp =
        zero_point.has_value() ? *zero_point : quantized::q_zero_point(tensors[0]);
    if (!(out_scale > 0.0)) {
        TP_THROW(ValueError, "quantized_cat(): scale must be positive");
    }

    std::vector<int64_t> out_shape =
        static_cast<std::vector<int64_t>>(tensors[0].shape());
    int64_t concat_size = 0;
    for (const Tensor& t : tensors) concat_size += t.size(dim);
    out_shape[dim] = concat_size;

    Tensor out = Tensor::empty(out_shape, DType::QInt8, tensors[0].device());
    int8_t* po = out.data_ptr<int8_t>();

    int64_t outer = 1;
    for (int64_t d = 0; d < dim; ++d) outer *= out_shape[d];
    int64_t inner = 1;
    for (int64_t d = dim + 1; d < out.dim(); ++d) inner *= out_shape[d];
    const int64_t concat_inner = concat_size * inner;

    bool same_qparams = true;
    for (const Tensor& t : tensors) {
        if (quantized::q_scale(t) != out_scale ||
            quantized::q_zero_point(t) != out_zp) {
            same_qparams = false;
            break;
        }
    }

    const double inv_out = 1.0 / out_scale;
    const int threads = 256;
    int64_t offset = 0;
    for (const Tensor& t : tensors) {
        const Tensor tc = t.contiguous();
        const int8_t* pi = tc.data_ptr<int8_t>();
        const int64_t seg = t.size(dim);
        const int64_t seg_inner = seg * inner;
        const int64_t offset_inner = offset * inner;
        const int64_t total = outer * seg_inner;
        const int blocks =
            static_cast<int>((total + threads - 1) / threads);
        if (same_qparams) {
            quantized_cat_copy_kernel<<<blocks, threads, 0,
                                        getCurrentCUDAStream().stream()>>>(
                pi, po, total, seg_inner, concat_inner, offset_inner);
        } else {
            const double in_scale = quantized::q_scale(t);
            const double in_zp =
                static_cast<double>(quantized::q_zero_point(t));
            quantized_cat_requantize_kernel<<<blocks, threads, 0,
                                              getCurrentCUDAStream().stream()>>>(
                pi, po, total, seg_inner, concat_inner, offset_inner,
                in_scale, in_zp, inv_out, static_cast<double>(out_zp));
        }
        offset += seg;
    }
    checkCuda(cudaGetLastError(), "CUDA quantized_cat kernels");

    if (relu_fused) {
        const int8_t q_zero = static_cast<int8_t>(out_zp);
        const int64_t numel = out.numel();
        const int blocks =
            static_cast<int>((numel + threads - 1) / threads);
        quantized_cat_relu_kernel<<<blocks, threads, 0,
                                    getCurrentCUDAStream().stream()>>>(
            numel, po, q_zero);
        checkCuda(cudaGetLastError(), "CUDA quantized_cat_relu kernel");
    }
    out.impl()->set_quantizer(make_per_tensor_affine_quantizer(
        out_scale, out_zp, DType::QInt8));
    return out;
}

} // namespace

Tensor quantized_cat_cuda(const std::vector<Tensor>& tensors, int64_t dim,
                          std::optional<double> scale,
                          std::optional<int64_t> zero_point) {
    return quant::cat_cuda(tensors, dim, scale, zero_point, false);
}

Tensor quantized_cat_relu_cuda(const std::vector<Tensor>& tensors,
                               int64_t dim, std::optional<double> scale,
                               std::optional<int64_t> zero_point) {
    return quant::cat_cuda(tensors, dim, scale, zero_point, true);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantCatKernels) {
}

} // namespace cuda
} // namespace tensorplay

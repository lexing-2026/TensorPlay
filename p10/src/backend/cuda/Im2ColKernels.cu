#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "ConvIm2colKernels.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace tensorplay {
namespace cuda {

Tensor im2col_cuda(const Tensor& self, const std::vector<int64_t>& kernel_size,
                   const std::vector<int64_t>& dilation,
                   const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& stride) {
    if (kernel_size.size() != 2 || dilation.size() != 2 || padding.size() != 2 ||
        stride.size() != 2)
        TP_THROW(ValueError, "im2col: expected 2-element kernel_size/dilation/padding/stride");
    Tensor input = self.is_contiguous() ? self : self.contiguous();
    const bool batched = input.dim() == 4;
    if (!batched && input.dim() != 3)
        TP_THROW(ValueError, "im2col: expected 3D (unbatched) or 4D input");
    const bool lowp = input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16;
    Tensor work = lowp ? input.to(DType::Float32) : input;

    const int64_t b = batched ? 1 : 0;
    const int64_t N = batched ? work.size(0) : 1;
    const int64_t C = work.size(b);
    const int64_t H = work.size(b + 1);
    const int64_t W = work.size(b + 2);
    const int64_t OH = (H + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1;
    const int64_t OW = (W + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1;
    if (OH <= 0 || OW <= 0) TP_THROW(RuntimeError, "im2col: calculated shape is too small");

    const int64_t CP = C * kernel_size[0] * kernel_size[1];
    const int64_t L = OH * OW;
    Tensor out = Tensor::empty({N, CP, L}, work.dtype(), work.device());

    const int64_t total = N * CP * L;
    dim3 threads(256, 1, 1);
    dim3 grid(cuda_blocks(total, 256), 1, 1);
    if (work.dtype() == DType::Float64) {
        im2col_kernel<double><<<grid, threads, 0, getCurrentCUDAStream().stream()>>>(
            work.data_ptr<double>(), out.data_ptr<double>(), N, C, H, W, kernel_size[0],
            kernel_size[1], padding[0], padding[1], stride[0], stride[1], dilation[0],
            dilation[1], OH, OW);
    } else {
        im2col_kernel<float><<<grid, threads, 0, getCurrentCUDAStream().stream()>>>(
            work.data_ptr<float>(), out.data_ptr<float>(), N, C, H, W, kernel_size[0],
            kernel_size[1], padding[0], padding[1], stride[0], stride[1], dilation[0],
            dilation[1], OH, OW);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        TP_THROW(RuntimeError, std::string("im2col CUDA: ") + cudaGetErrorString(err));
    if (!batched) out = out.squeeze(0);
    return lowp ? out.to(input.dtype()) : out;
}

Tensor col2im_cuda(const Tensor& self, const std::vector<int64_t>& output_size,
                   const std::vector<int64_t>& kernel_size,
                   const std::vector<int64_t>& dilation,
                   const std::vector<int64_t>& padding,
                   const std::vector<int64_t>& stride) {
    if (output_size.size() != 2)
        TP_THROW(ValueError, "col2im: output_size must have 2 elements");
    Tensor input = self.is_contiguous() ? self : self.contiguous();
    const bool batched = input.dim() == 3;
    if (!batched && input.dim() != 2)
        TP_THROW(ValueError, "col2im: expected 2D (unbatched) or 3D input");
    const bool lowp = input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16;
    Tensor work = lowp ? input.to(DType::Float32) : input;

    const int64_t H = output_size[0], W = output_size[1];
    const int64_t OH = (H + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1;
    const int64_t OW = (W + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1;
    const int64_t CP = work.size(work.dim() - 2);
    const int64_t L = work.size(work.dim() - 1);
    if (CP % (kernel_size[0] * kernel_size[1]) != 0 || L != OH * OW)
        TP_THROW(RuntimeError, "col2im: input shape does not match kernel/output parameters");
    const int64_t C = CP / (kernel_size[0] * kernel_size[1]);
    const int64_t N = batched ? work.size(0) : 1;

    Tensor out = Tensor::empty({N, C, H, W}, work.dtype(), work.device());
    const int64_t frame = C * H * W;
    dim3 threads(128, 1, 1);
    dim3 grid(cuda_blocks(frame, 128), 1, static_cast<unsigned>(N));
    if (work.dtype() == DType::Float64) {
        col2im_kernel<double><<<grid, threads, 0, getCurrentCUDAStream().stream()>>>(
            work.data_ptr<double>(), out.data_ptr<double>(), C, H, W, kernel_size[0],
            kernel_size[1], padding[0], padding[1], stride[0], stride[1], dilation[0],
            dilation[1], OH, OW);
    } else {
        col2im_kernel<float><<<grid, threads, 0, getCurrentCUDAStream().stream()>>>(
            work.data_ptr<float>(), out.data_ptr<float>(), C, H, W, kernel_size[0],
            kernel_size[1], padding[0], padding[1], stride[0], stride[1], dilation[0],
            dilation[1], OH, OW);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        TP_THROW(RuntimeError, std::string("col2im CUDA: ") + cudaGetErrorString(err));
    if (!batched) out = out.squeeze(0);
    return lowp ? out.to(input.dtype()) : out;
}

Tensor im2col_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& input_size,
                            const std::vector<int64_t>& kernel_size,
                            const std::vector<int64_t>& dilation,
                            const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& stride) {
    std::vector<int64_t> output_size = {input_size[input_size.size() - 2],
                                        input_size[input_size.size() - 1]};
    return col2im_cuda(grad_output, output_size, kernel_size, dilation, padding, stride);
}

Tensor col2im_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& input_size,
                            const std::vector<int64_t>& output_size,
                            const std::vector<int64_t>& kernel_size,
                            const std::vector<int64_t>& dilation,
                            const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& stride) {
    (void)input_size;
    (void)output_size;
    return im2col_cuda(grad_output, kernel_size, dilation, padding, stride);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, Im2ColKernels) {
    m.impl("im2col", im2col_cuda);
    m.impl("im2col_backward", im2col_backward_cuda);
    m.impl("col2im", col2im_cuda);
    m.impl("col2im_backward", col2im_backward_cuda);
}

}
}

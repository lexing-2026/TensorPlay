#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


Tensor interop_slow_conv_transpose3d_cuda(
        const Tensor& input, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& output_padding,
        const std::vector<int64_t>& dilation) {
    (void)kernel_size;
    return ops::conv_transpose3d(input, weight, bias, stride, padding,
                                 output_padding, 1, dilation);
}


Tensor& interop_slow_conv_transpose3d_out_cuda(
        const Tensor& input, const Tensor& weight,
        const std::vector<int64_t>& kernel_size,
        const std::optional<Tensor>& bias, const std::vector<int64_t>& stride,
        const std::vector<int64_t>& padding,
        const std::vector<int64_t>& output_padding,
        const std::vector<int64_t>& dilation, Tensor& out) {
    (void)kernel_size;
    write_out(out, ops::conv_transpose3d(input, weight, bias, stride, padding,
                                         output_padding, 1, dilation));
    return out;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, NaiveConvolutionTranspose3dKernels) {
    m.impl("slow_conv_transpose3d", interop_slow_conv_transpose3d_cuda);
    m.impl("slow_conv_transpose3d.out", interop_slow_conv_transpose3d_out_cuda);
}

} // namespace cuda
} // namespace tensorplay

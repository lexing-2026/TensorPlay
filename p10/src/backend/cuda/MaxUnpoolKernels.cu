// max_unpool2d / max_unpool3d CUDA kernels.
//
// element. Forward scatters each value into the zero canvas at its flat
// backward gathers grad_output at those same positions. Out-of-range indices
#include "Tensor.h"
#include "Dispatcher.h"
#include "Context.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Half.h"
#include "BFloat16.h"
#include "LinearAlgebraNames.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kThreads = 256;

template <typename T>
__global__ void max_unpool_fwd_kernel(const int64_t numel, const T* __restrict__ input,
                                      const int64_t* __restrict__ indices,
                                      const int64_t input_image, const int64_t output_image,
                                      T* __restrict__ output) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t plane = i / input_image;
    const int64_t maxp = indices[i];
    assert(maxp >= 0 && maxp < output_image);
    output[plane * output_image + maxp] = input[i];
}

template <typename T>
__global__ void max_unpool_bwd_kernel(const int64_t numel, const T* __restrict__ grad_output,
                                      const int64_t* __restrict__ indices,
                                      const int64_t input_image, const int64_t output_image,
                                      T* __restrict__ grad_input) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    const int64_t plane = i / input_image;
    const int64_t maxp = indices[i];
    assert(maxp >= 0 && maxp < output_image);
    grad_input[i] = grad_output[plane * output_image + maxp];
}

inline int64_t grid_blocks(int64_t n, int threads) {
    return (n + threads - 1) / threads;
}

void unpool_check_indices(const Tensor& self, const Tensor& indices, const char* name) {
    if (indices.dtype() != DType::Int64)
        TP_THROW(RuntimeError, std::string("elements in indices should be type int64 but got: ") +
                     pretty_dtype_name(indices.dtype()));
    if (self.shape() != indices.shape())
        TP_THROW(RuntimeError, std::string("Expected shape of indices to be same as that of the input tensor (") +
                     self.shape().toString() + ") but got indices tensor with shape: (" +
                     indices.shape().toString() + ")");
    for (int64_t i = 1; i < self.dim(); ++i) {
        if (self.size(i) <= 0)
            TP_THROW(RuntimeError, std::string(name) +
                         ": Expected input to have non-zero size for non-batch dimensions, but got " +
                         self.shape().toString() + " with dimension " + std::to_string(i) + " being empty.");
    }
}

std::vector<int64_t> unpool_output_shape(const Tensor& self, int64_t spatial_dims,
                                         const std::vector<int64_t>& output_size) {
    std::vector<int64_t> shape;
    for (int64_t d = 0; d < self.dim() - spatial_dims; ++d) shape.push_back(self.size(d));
    for (int64_t s : output_size) shape.push_back(s);
    return shape;
}

// batch/channel dims as the (indices) input; spatial dims checked separately
// against output_size.
void unpool_check_grad(const Tensor& grad_output, const Tensor& indices,
                       int64_t spatial_dims) {
    if (grad_output.dim() != indices.dim())
        TP_THROW(RuntimeError, "gradOutput and input Tensors should have same number of dimensions and also the same number of channels/slices");
    for (int64_t d = 0; d < grad_output.dim() - spatial_dims; ++d) {
        if (grad_output.size(d) != indices.size(d))
            TP_THROW(RuntimeError, "gradOutput and input Tensors should have same number of dimensions and also the same number of channels/slices");
    }
}

int64_t trailing_product(const Tensor& t, int64_t spatial_dims) {
    int64_t v = 1;
    for (int64_t d = 0; d < spatial_dims; ++d) v *= t.size(t.dim() - spatial_dims + d);
    return v;
}

template <typename T>
static Tensor max_unpool_forward_cuda_impl(const Tensor& self, const Tensor& indices,
                                           const std::vector<int64_t>& output_size,
                                           int64_t spatial_dims) {
    const Tensor ic = self.contiguous();
    const Tensor xc = indices.contiguous();
    Tensor output = Tensor::zeros(unpool_output_shape(ic, spatial_dims, output_size),
                                  ic.dtype(), ic.device());
    const int64_t numel = ic.numel();
    if (numel == 0) return output;
    const int64_t input_image = trailing_product(ic, spatial_dims);
    int64_t output_image = 1;
    for (int64_t s : output_size) output_image *= s;
    max_unpool_fwd_kernel<T><<<grid_blocks(numel, kThreads), kThreads, 0,
                               getCurrentCUDAStream().stream()>>>(
        numel, ic.data_ptr<T>(), xc.data_ptr<int64_t>(), input_image, output_image,
        output.data_ptr<T>());
    return output;
}

template <typename T>
static Tensor max_unpool_backward_cuda_impl(const Tensor& grad_output, const Tensor& indices,
                                            int64_t spatial_dims) {
    const Tensor goc = grad_output.contiguous();
    const Tensor xc = indices.contiguous();
    Tensor grad_input = Tensor::zeros(xc.shape(), goc.dtype(), goc.device());
    const int64_t numel = xc.numel();
    if (numel == 0) return grad_input;
    const int64_t input_image = trailing_product(xc, spatial_dims);
    const int64_t output_image = trailing_product(goc, spatial_dims);
    max_unpool_bwd_kernel<T><<<grid_blocks(numel, kThreads), kThreads, 0,
                               getCurrentCUDAStream().stream()>>>(
        numel, goc.data_ptr<T>(), xc.data_ptr<int64_t>(), input_image, output_image,
        grad_input.data_ptr<T>());
    return grad_input;
}

#define TP_MUP_CUDA_DISPATCH(ctype, name, ...) \
    case DType::name: { __VA_ARGS__; break; }

}  // namespace

Tensor max_unpool2d_cuda(const Tensor& self, const Tensor& indices,
                         const std::vector<int64_t>& output_size) {
    if (output_size.size() != 2)
        TP_THROW(RuntimeError, "There should be exactly two elements (height, width) in output_size, but got ",
                 output_size.size(), " elements.");
    if (self.dim() != 3 && self.dim() != 4)
        TP_THROW(RuntimeError, "Input to max_unpooling2d should be a 3d or 4d Tensor, but got a tensor with ",
                 self.dim(), " dimensions.");
    unpool_check_indices(self, indices, "max_unpooling2d_forward_out_cuda()");
    if (output_size[0] < 0 || output_size[1] < 0)
        TP_THROW(RuntimeError, "max_unpooling2d(): output_size must contain non-negative spatial dimensions, but got output_size=(",
                 output_size[0], ", ", output_size[1], ")");
    switch (self.dtype()) {
        TP_MUP_CUDA_DISPATCH(float, Float32,
            return max_unpool_forward_cuda_impl<float>(self, indices, output_size, 2))
        TP_MUP_CUDA_DISPATCH(double, Float64,
            return max_unpool_forward_cuda_impl<double>(self, indices, output_size, 2))
        TP_MUP_CUDA_DISPATCH(tensorplay::Half, Float16,
            return max_unpool_forward_cuda_impl<tensorplay::Half>(self, indices, output_size, 2))
        TP_MUP_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            return max_unpool_forward_cuda_impl<tensorplay::BFloat16>(self, indices, output_size, 2))
        default: TP_THROW(TypeError, "max_unpool2d: unsupported dtype");
    }
}

Tensor max_unpool2d_backward_cuda(const Tensor& grad_output, const Tensor& indices,
                                  const std::vector<int64_t>& output_size) {
    if (output_size.size() != 2)
        TP_THROW(RuntimeError, "There should be exactly two elements (height, width) in output_size, but got ",
                 output_size.size(), " elements.");
    if (grad_output.dim() != 3 && grad_output.dim() != 4)
        TP_THROW(RuntimeError, "MaxUnpool2d_backward: expect grad_output to be 3d or 4d tensor.");
    if (indices.dtype() != DType::Int64)
        TP_THROW(RuntimeError, "elements in indices should be type int64 but got: ",
                 pretty_dtype_name(indices.dtype()));
    if (grad_output.size(-2) != output_size[0] || grad_output.size(-1) != output_size[1])
        TP_THROW(RuntimeError, "Inconsistent gradOutput size. oH= ", output_size[0],
                 ", oW= ", output_size[1], ". gradOutput: ",
                 grad_output.size(-2), "x", grad_output.size(-1));
    unpool_check_grad(grad_output, indices, 2);
    switch (grad_output.dtype()) {
        TP_MUP_CUDA_DISPATCH(float, Float32,
            return max_unpool_backward_cuda_impl<float>(grad_output, indices, 2))
        TP_MUP_CUDA_DISPATCH(double, Float64,
            return max_unpool_backward_cuda_impl<double>(grad_output, indices, 2))
        TP_MUP_CUDA_DISPATCH(tensorplay::Half, Float16,
            return max_unpool_backward_cuda_impl<tensorplay::Half>(grad_output, indices, 2))
        TP_MUP_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            return max_unpool_backward_cuda_impl<tensorplay::BFloat16>(grad_output, indices, 2))
        default: TP_THROW(TypeError, "max_unpool2d_backward: unsupported dtype");
    }
}

Tensor max_unpool3d_cuda(const Tensor& self, const Tensor& indices,
                         const std::vector<int64_t>& output_size,
                         const std::vector<int64_t>& stride,
                         const std::vector<int64_t>& padding) {
    if (output_size.size() != 3)
        TP_THROW(RuntimeError, "There should be exactly three elements (depth, height, width) in output_size, but got ",
                 output_size.size(), " elements.");
    if (stride.size() != 3)
        TP_THROW(RuntimeError, "There should be exactly three elements (depth, height, width) in stride, but got ",
                 stride.size(), " elements.");
    if (padding.size() != 3)
        TP_THROW(RuntimeError, "There should be exactly three elements (depth, height, width) in padding, but got ",
                 padding.size(), " elements.");
    if (self.dim() != 4 && self.dim() != 5)
        TP_THROW(RuntimeError, "Input to max_unpooling3d should be a 4d or 5d Tensor, but got a tensor with ",
                 self.dim(), " dimensions.");
    unpool_check_indices(self, indices, "max_unpooling3d_forward_out_cuda()");
    if (stride[0] <= 0 || stride[1] <= 0 || stride[2] <= 0)
        TP_THROW(RuntimeError, "strides should be greater than zero, but got stride: (",
                 stride[0], ", ", stride[1], ", ", stride[2], ")");
    if (output_size[0] < 0 || output_size[1] < 0 || output_size[2] < 0)
        TP_THROW(RuntimeError, "max_unpooling3d(): output_size must contain non-negative spatial dimensions, but got output_size=(",
                 output_size[0], ", ", output_size[1], ", ", output_size[2], ")");
    switch (self.dtype()) {
        TP_MUP_CUDA_DISPATCH(float, Float32,
            return max_unpool_forward_cuda_impl<float>(self, indices, output_size, 3))
        TP_MUP_CUDA_DISPATCH(double, Float64,
            return max_unpool_forward_cuda_impl<double>(self, indices, output_size, 3))
        TP_MUP_CUDA_DISPATCH(tensorplay::Half, Float16,
            return max_unpool_forward_cuda_impl<tensorplay::Half>(self, indices, output_size, 3))
        TP_MUP_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            return max_unpool_forward_cuda_impl<tensorplay::BFloat16>(self, indices, output_size, 3))
        default: TP_THROW(TypeError, "max_unpool3d: unsupported dtype");
    }
}

Tensor max_unpool3d_backward_cuda(const Tensor& grad_output, const Tensor& indices,
                                  const std::vector<int64_t>& output_size) {
    if (output_size.size() != 3)
        TP_THROW(RuntimeError, "There should be exactly three elements (depth, height, width) in output_size, but got ",
                 output_size.size(), " elements.");
    if (grad_output.dim() != 4 && grad_output.dim() != 5)
        TP_THROW(RuntimeError, "MaxUnpool3d_backward: expect grad_output to be 4d or 5d tensor.");
    if (indices.dtype() != DType::Int64)
        TP_THROW(RuntimeError, "elements in indices should be type int64 but got: ",
                 pretty_dtype_name(indices.dtype()));
    if (grad_output.size(-3) != output_size[0] || grad_output.size(-2) != output_size[1] ||
        grad_output.size(-1) != output_size[2])
        TP_THROW(RuntimeError, "Inconsistent gradOutput size. oT= ", output_size[0],
                 ", oH= ", output_size[1], ", oW= ", output_size[2], ". gradOutput: ",
                 grad_output.size(-3), "x", grad_output.size(-2), "x", grad_output.size(-1));
    unpool_check_grad(grad_output, indices, 3);
    switch (grad_output.dtype()) {
        TP_MUP_CUDA_DISPATCH(float, Float32,
            return max_unpool_backward_cuda_impl<float>(grad_output, indices, 3))
        TP_MUP_CUDA_DISPATCH(double, Float64,
            return max_unpool_backward_cuda_impl<double>(grad_output, indices, 3))
        TP_MUP_CUDA_DISPATCH(tensorplay::Half, Float16,
            return max_unpool_backward_cuda_impl<tensorplay::Half>(grad_output, indices, 3))
        TP_MUP_CUDA_DISPATCH(tensorplay::BFloat16, BFloat16,
            return max_unpool_backward_cuda_impl<tensorplay::BFloat16>(grad_output, indices, 3))
        default: TP_THROW(TypeError, "max_unpool3d_backward: unsupported dtype");
    }
}

Tensor& interop_max_unpool2d_out_cuda(const Tensor& self, const Tensor& indices,
              const std::vector<int64_t>& output_size, Tensor& out) {
        out = max_unpool2d_cuda(self, indices, output_size);
        return out;
    
}

Tensor& interop_max_unpool3d_out_cuda(const Tensor& self, const Tensor& indices,
              const std::vector<int64_t>& output_size,
              const std::vector<int64_t>& stride,
              const std::vector<int64_t>& padding, Tensor& out) {
        out = max_unpool3d_cuda(self, indices, output_size, stride, padding);
        return out;
    
}

TENSORPLAY_LIBRARY_IMPL(CUDA, MaxUnpoolKernels) {
    m.impl("max_unpool2d", max_unpool2d_cuda);
    m.impl("max_unpool2d_backward", max_unpool2d_backward_cuda);
    m.impl("max_unpool3d", max_unpool3d_cuda);
    m.impl("max_unpool3d_backward", max_unpool3d_backward_cuda);

    m.impl("max_unpool2d.out", interop_max_unpool2d_out_cuda);
    m.impl("max_unpool3d.out", interop_max_unpool3d_out_cuda);
}

}  // namespace cuda
}  // namespace tensorplay

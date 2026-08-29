// max_unpool2d / max_unpool3d CPU kernels.
//
// cpu/MaxUnpoolKernel.cpp (scatter/gather): the forward scatters pooled values
// into a zero canvas at the flat in-plane int64 indices recorded by
// max_pool*_with_indices; the backward gathers grad_output at those same
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "Half.h"
#include "BFloat16.h"
#include "LinearAlgebraNames.h"
#include <vector>
#include <string>
#include <atomic>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

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

template <typename T>
static void max_unpool_forward_core(const Tensor& output, const Tensor& input,
                                    const Tensor& indices, int64_t spatial_dims,
                                    const std::vector<int64_t>& output_size) {
    const int64_t numel = input.numel();
    if (numel == 0) return;
    int64_t input_image = 1;
    for (int64_t d = 0; d < spatial_dims; ++d) input_image *= input.size(input.dim() - spatial_dims + d);
    int64_t output_image = 1;
    for (int64_t s : output_size) output_image *= s;

    const T* in = input.data_ptr<T>();
    const int64_t* idx = indices.data_ptr<int64_t>();
    T* out = output.data_ptr<T>();

    std::atomic<int64_t> error_index{-1};
    parallel_for(0, numel, GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const int64_t plane = i / input_image;
            const int64_t maxp = idx[i];
            if (maxp < 0 || maxp >= output_image) {
                error_index.store(maxp, std::memory_order_relaxed);
            } else {
                out[plane * output_image + maxp] = in[i];
            }
        }
    });
    const int64_t err = error_index.load(std::memory_order_relaxed);
    if (err >= 0) {
        std::string sizes;
        for (size_t d = 0; d < output_size.size(); ++d) {
            if (d) sizes += "x";
            sizes += std::to_string(output_size[d]);
        }
        TP_THROW(RuntimeError, "Found an invalid max index: ", err,
                 " (output volumes are of size ", sizes, ")");
    }
}

template <typename T>
static void max_unpool_backward_core(const Tensor& grad_input, const Tensor& grad_output,
                                     const Tensor& indices, int64_t spatial_dims) {
    const int64_t numel = grad_input.numel();
    if (numel == 0) return;
    int64_t input_image = 1;
    for (int64_t d = 0; d < spatial_dims; ++d)
        input_image *= grad_input.size(grad_input.dim() - spatial_dims + d);
    int64_t output_image = 1;
    for (int64_t d = 0; d < spatial_dims; ++d)
        output_image *= grad_output.size(grad_output.dim() - spatial_dims + d);

    const T* go = grad_output.data_ptr<T>();
    const int64_t* idx = indices.data_ptr<int64_t>();
    T* gi = grad_input.data_ptr<T>();

    std::atomic<int64_t> error_index{-1};
    parallel_for(0, numel, GRAIN_SIZE, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const int64_t plane = i / input_image;
            const int64_t maxp = idx[i];
            if (maxp < 0 || maxp >= output_image) {
                error_index.store(maxp, std::memory_order_relaxed);
            } else {
                gi[i] = go[plane * output_image + maxp];
            }
        }
    });
    const int64_t err = error_index.load(std::memory_order_relaxed);
    if (err >= 0) {
        if (spatial_dims == 3) {
            TP_THROW(RuntimeError, "invalid max index ", err,
                     ", odepth= ", grad_output.size(-3),
                     ", owidth= ", grad_output.size(-1),
                     ", oheight= ", grad_output.size(-2));
        }
        TP_THROW(RuntimeError, "invalid max index ", err,
                 ", owidth= ", grad_output.size(-1),
                 ", oheight= ", grad_output.size(-2));
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

#define TP_MUP_DISPATCH(ctype, name, ...) \
    case DType::name: { __VA_ARGS__; break; }

}  // namespace

Tensor max_unpool2d_cpu(const Tensor& self, const Tensor& indices,
                        const std::vector<int64_t>& output_size) {
    if (output_size.size() != 2)
        TP_THROW(RuntimeError, "There should be exactly two elements (height, width) in output_size, but got ",
                 output_size.size(), " elements.");
    if (self.dim() != 3 && self.dim() != 4)
        TP_THROW(RuntimeError, "Input to max_unpooling2d should be a 3d or 4d Tensor, but got a tensor with ",
                 self.dim(), " dimensions.");
    unpool_check_indices(self, indices, "max_unpooling2d_forward_out_cpu()");
    if (output_size[0] < 0 || output_size[1] < 0)
        TP_THROW(RuntimeError, "max_unpooling2d(): output_size must contain non-negative spatial dimensions, but got output_size=(",
                 output_size[0], ", ", output_size[1], ")");

    const Tensor ic = self.contiguous();
    const Tensor xc = indices.contiguous();
    Tensor output = Tensor::zeros(unpool_output_shape(ic, 2, output_size), ic.dtype(), ic.device());
    switch (ic.dtype()) {
        TP_MUP_DISPATCH(float, Float32,
            max_unpool_forward_core<float>(output, ic, xc, 2, output_size))
        TP_MUP_DISPATCH(double, Float64,
            max_unpool_forward_core<double>(output, ic, xc, 2, output_size))
        TP_MUP_DISPATCH(Half, Float16,
            max_unpool_forward_core<Half>(output, ic, xc, 2, output_size))
        TP_MUP_DISPATCH(BFloat16, BFloat16,
            max_unpool_forward_core<BFloat16>(output, ic, xc, 2, output_size))
        default: TP_THROW(TypeError, "max_unpool2d: unsupported dtype");
    }
    return output;
}

Tensor max_unpool2d_backward_cpu(const Tensor& grad_output, const Tensor& indices,
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

    const Tensor goc = grad_output.contiguous();
    const Tensor xc = indices.contiguous();
    Tensor grad_input = Tensor::zeros(xc.shape(), goc.dtype(), goc.device());
    switch (goc.dtype()) {
        TP_MUP_DISPATCH(float, Float32,
            max_unpool_backward_core<float>(grad_input, goc, xc, 2))
        TP_MUP_DISPATCH(double, Float64,
            max_unpool_backward_core<double>(grad_input, goc, xc, 2))
        TP_MUP_DISPATCH(Half, Float16,
            max_unpool_backward_core<Half>(grad_input, goc, xc, 2))
        TP_MUP_DISPATCH(BFloat16, BFloat16,
            max_unpool_backward_core<BFloat16>(grad_input, goc, xc, 2))
        default: TP_THROW(TypeError, "max_unpool2d_backward: unsupported dtype");
    }
    return grad_input;
}

Tensor max_unpool3d_cpu(const Tensor& self, const Tensor& indices,
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
    unpool_check_indices(self, indices, "max_unpooling3d_forward_out_cpu()");
    if (stride[0] <= 0 || stride[1] <= 0 || stride[2] <= 0)
        TP_THROW(RuntimeError, "strides should be greater than zero, but got stride: (",
                 stride[0], ", ", stride[1], ", ", stride[2], ")");
    if (output_size[0] < 0 || output_size[1] < 0 || output_size[2] < 0)
        TP_THROW(RuntimeError, "max_unpooling3d(): output_size must contain non-negative spatial dimensions, but got output_size=(",
                 output_size[0], ", ", output_size[1], ", ", output_size[2], ")");

    const Tensor ic = self.contiguous();
    const Tensor xc = indices.contiguous();
    Tensor output = Tensor::zeros(unpool_output_shape(ic, 3, output_size), ic.dtype(), ic.device());
    switch (ic.dtype()) {
        TP_MUP_DISPATCH(float, Float32,
            max_unpool_forward_core<float>(output, ic, xc, 3, output_size))
        TP_MUP_DISPATCH(double, Float64,
            max_unpool_forward_core<double>(output, ic, xc, 3, output_size))
        TP_MUP_DISPATCH(Half, Float16,
            max_unpool_forward_core<Half>(output, ic, xc, 3, output_size))
        TP_MUP_DISPATCH(BFloat16, BFloat16,
            max_unpool_forward_core<BFloat16>(output, ic, xc, 3, output_size))
        default: TP_THROW(TypeError, "max_unpool3d: unsupported dtype");
    }
    return output;
}

Tensor max_unpool3d_backward_cpu(const Tensor& grad_output, const Tensor& indices,
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

    const Tensor goc = grad_output.contiguous();
    const Tensor xc = indices.contiguous();
    Tensor grad_input = Tensor::zeros(xc.shape(), goc.dtype(), goc.device());
    switch (goc.dtype()) {
        TP_MUP_DISPATCH(float, Float32,
            max_unpool_backward_core<float>(grad_input, goc, xc, 3))
        TP_MUP_DISPATCH(double, Float64,
            max_unpool_backward_core<double>(grad_input, goc, xc, 3))
        TP_MUP_DISPATCH(Half, Float16,
            max_unpool_backward_core<Half>(grad_input, goc, xc, 3))
        TP_MUP_DISPATCH(BFloat16, BFloat16,
            max_unpool_backward_core<BFloat16>(grad_input, goc, xc, 3))
        default: TP_THROW(TypeError, "max_unpool3d_backward: unsupported dtype");
    }
    return grad_input;
}

TENSORPLAY_LIBRARY_IMPL(CPU, MaxUnpoolKernels) {
    m.impl("max_unpool2d", max_unpool2d_cpu);
    m.impl("max_unpool2d_backward", max_unpool2d_backward_cpu);
    m.impl("max_unpool3d", max_unpool3d_cpu);
    m.impl("max_unpool3d_backward", max_unpool3d_backward_cpu);
}

}  // namespace cpu
}  // namespace tensorplay

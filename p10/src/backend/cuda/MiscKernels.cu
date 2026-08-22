// Misc CUDA kernels: meshgrid / roll / diff / masked_fill / one_hot / glu.
//
// These mirror the CPU composites in cpu/MiscKernels.cpp; every primitive
// invoked (slice/view/expand/cat/sigmoid/where/eq) is itself dispatched to the
// device backend, matching ATen where these ops are composite functions:
//   aten/src/ATen/native/TensorShape.cpp        meshgrid()
//   aten/src/ATen/native/TensorTransformations.{h,cpp}  roll()/roll_common()
//   aten/src/ATen/native/ReduceOps.cpp          diff()/diff_helper()
//   aten/src/ATen/native/Onehot.cpp             one_hot()
//   aten/src/ATen/native/GatedLinearUnit.cpp    glu()/glu_backward()

#include "Tensor.h"
#include "Dispatcher.h"
#include "Utils.h"

namespace tensorplay {
namespace cuda {

// Defined in PointwiseKernels.cu.
Tensor eq_kernel_cuda(const Tensor& self, const Tensor& other);


// Defined in PointwiseKernels.cu.
Tensor where_cuda(const Tensor& condition, const Tensor& self, const Tensor& other);

namespace {

inline int64_t wrap_dim_local(int64_t dim, int64_t ndim) {
    const int64_t min_ = -ndim;
    const int64_t max_ = ndim - 1;
    if (dim < min_ || dim > max_) {
        TP_THROW(IndexError,
                 "Dimension out of range (expected to be in range of [" +
                     std::to_string(min_) + ", " + std::to_string(max_) + "], but got " +
                     std::to_string(dim) + ")");
    }
    if (dim < 0) dim += ndim;
    return dim;
}

} // anonymous namespace

static Tensor diff_helper(const Tensor& self, int64_t n, int64_t dim) {
    // ATen ReduceOps.cpp diff_helper.
    Tensor result = self;
    n = n > self.size(dim) ? self.size(dim) : n;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t out_len = result.size(dim) - 1;
        result = result.slice(dim, 1, out_len + 1) - result.slice(dim, 0, out_len);
    }
    return result;
}

Tensor diff_cuda(const Tensor& self, int64_t n, int64_t dim, const Tensor& prepend, const Tensor& append) {
    const int64_t d = wrap_dim_local(dim, self.dim());
    const bool has_prepend = prepend.defined();
    const bool has_append = append.defined();
    if ((!has_prepend && !has_append) || n == 0) return diff_helper(self, n, d);
    std::vector<Tensor> pieces;
    if (has_prepend) pieces.push_back(prepend);
    pieces.push_back(self);
    if (has_append) pieces.push_back(append);
    return diff_helper(Tensor::cat(pieces, d), n, d);
}

Tensor one_hot_cuda(const Tensor& self, int64_t num_classes) {
    // ATen Onehot.cpp functional branch: eq(self.unsqueeze(-1), arange)
    if (self.dtype() != DType::Int64) {
        TP_THROW(RuntimeError, "one_hot is only applicable to index tensor of type LongTensor.");
    }
    if (num_classes == -1) {
        if (self.numel() == 0) {
            TP_THROW(RuntimeError, "Can not infer total number of classes from empty tensor.");
        }
        num_classes = self.max().item().to<int64_t>() + 1;
    }
    Tensor index = Tensor::arange(Scalar(static_cast<int64_t>(0)), Scalar(num_classes),
                                  Scalar(static_cast<int64_t>(1)), DType::Int64, self.device());
    auto sizes = static_cast<std::vector<int64_t>>(self.shape());
    sizes.push_back(1);
    return eq_kernel_cuda(self.view(sizes), index).to(DType::Int64);
}

Tensor glu_cuda(const Tensor& self, int64_t dim) {
    // ATen GatedLinearUnit.cpp / cpu Activation.cpp glu_kernel.
    if (self.dim() == 0) TP_THROW(RuntimeError, "glu does not support 0-dimensional tensors");
    const int64_t d = wrap_dim_local(dim, self.dim());
    const int64_t nIn = self.size(d);
    if (nIn % 2 != 0) {
        TP_THROW(RuntimeError, "Halving dimension must be even, but dimension " + std::to_string(d) +
                                   " is size " + std::to_string(nIn));
    }
    const int64_t half = nIn / 2;
    Tensor firstHalf = self.slice(d, 0, half);
    Tensor secondHalf = self.slice(d, half, nIn);
    return firstHalf * secondHalf.sigmoid();
}

Tensor glu_backward_cuda(const Tensor& grad_output, const Tensor& self, int64_t dim) {
    // ATen GatedLinearUnit.cpp glu_backward_cpu_out semantics.
    if (self.dim() == 0) TP_THROW(RuntimeError, "glu does not support 0-dimensional tensors");
    const int64_t d = wrap_dim_local(dim, self.dim());
    const int64_t nIn = self.size(d);
    if (nIn % 2 != 0) {
        TP_THROW(RuntimeError, "Halving dimension must be even, but dimension " + std::to_string(d) +
                                   " is size " + std::to_string(nIn));
    }
    const int64_t inputSize = nIn / 2;
    Tensor firstHalf = self.slice(d, 0, inputSize);
    Tensor secondHalf = self.slice(d, inputSize, nIn);
    Tensor sig_second = secondHalf.sigmoid();
    Tensor grad_input_first = grad_output * sig_second;
    Tensor grad_input_second = grad_output * firstHalf * sig_second * (1 - sig_second);
    return Tensor::cat({grad_input_first, grad_input_second}, d);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, MiscKernels) {
    m.impl("diff", diff_cuda);
    m.impl("one_hot", one_hot_cuda);
    m.impl("glu", glu_cuda);
    m.impl("glu_backward", glu_backward_cuda);
}

} // namespace cuda
} // namespace tensorplay

// Misc kernels: meshgrid / roll / diff / masked_fill / one_hot / glu.
//
// Each function is a faithful port of the corresponding ATen composite:
//   third_party/pytorch/aten/src/ATen/native/TensorShape.cpp  meshgrid()
//   third_party/pytorch/aten/src/ATen/native/TensorTransformations.cpp
//     roll() (single-dim narrow+cat) and TensorTransformations.h roll_common()
//   third_party/pytorch/aten/src/ATen/native/ReduceOps.cpp
//     diff() / diff_helper()
//   third_party/pytorch/aten/src/ATen/native/Onehot.cpp        one_hot()
//   third_party/pytorch/aten/src/ATen/native/GatedLinearUnit.cpp glu()
//     and cpu/Activation.cpp glu_kernel (first * sigmoid(second))
// ATen's ``narrow(dim, start, length)`` is expressed with the dispatched
#// Tensor::slice(dim, start, start + length), which has identical semantics.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Utils.h"
#include <algorithm>
#include <numeric>

namespace tensorplay {
namespace cpu {

// Declared in ComparisonKernels.cpp; reused for masked_fill's broadcast
// select and one_hot's equality.  Lives at namespace scope so it resolves to
// the ComparisonKernels definition (an anonymous-namespace declaration would
// be a distinct, undefined symbol).
Tensor where_cpu(const Tensor& condition, const Tensor& self, const Tensor& other);
Tensor eq_tensor_kernel(const Tensor& self, const Tensor& other);

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

// ---------------------------------------------------------------------------
// meshgrid — ATen native/TensorShape.cpp meshgrid(tensors, indexing)
// ---------------------------------------------------------------------------
std::vector<Tensor> meshgrid_cpu(const std::vector<Tensor>& tensors, const std::string& indexing) {
    const int64_t size = static_cast<int64_t>(tensors.size());
    if (size <= 0) TP_THROW(RuntimeError, "meshgrid expects a non-empty TensorList");

    for (int64_t i = 0; i < size - 1; ++i) {
        if (tensors[i].dtype() != tensors[i + 1].dtype()) {
            TP_THROW(RuntimeError, "meshgrid expects all tensors to have the same dtype");
        }
        if (!(tensors[i].device() == tensors[i + 1].device())) {
            TP_THROW(RuntimeError, "meshgrid expects all tensors to have the same device");
        }
    }

    // Whether or not to swap the first two tensors ("xy" semantics).
    bool swap_first_and_second_tensors = false;
    std::vector<Tensor> tensor_refs(tensors.begin(), tensors.end());
    if (indexing == "xy") {
        swap_first_and_second_tensors = size >= 2;
        if (swap_first_and_second_tensors) {
            std::swap(tensor_refs[0], tensor_refs[1]);
        }
    } else {
        if (indexing != "ij") {
            TP_THROW(RuntimeError, "torch.meshgrid: indexing must be one of \"xy\" or \"ij\", but received: " + indexing);
        }
    }

    std::vector<int64_t> shape(size);
    for (int64_t i = 0; i < size; ++i) {
        if (tensor_refs[i].dim() > 1) {
            TP_THROW(RuntimeError, "torch.meshgrid: Expected 0D or 1D tensor in the tensor list but got: " +
                                       std::to_string(tensor_refs[i].dim()));
        }
        shape[i] = tensor_refs[i].numel(); // treat 0D tensors as 1D
    }

    std::vector<Tensor> grids;
    grids.reserve(size);
    std::vector<int64_t> view_shape(size, 1);
    for (int64_t i = 0; i < size; ++i) {
        view_shape[i] = -1; // select this dimension to infer
        grids.push_back(tensor_refs[i].view(view_shape).expand(shape));
        view_shape[i] = 1; // restore to previous value
    }

    if (swap_first_and_second_tensors) {
        std::swap(grids[0], grids[1]);
    }
    return grids;
}

// ---------------------------------------------------------------------------
// roll — ATen native/TensorTransformations.cpp roll() + roll_common()
// ---------------------------------------------------------------------------
Tensor roll_cpu(const Tensor& self, std::vector<int64_t> shifts, std::vector<int64_t> dims) {
    if (dims.empty()) {
        TP_THROW_IF(shifts.size() != 1, RuntimeError, "`shifts` required");
        dims.push_back(0);
    }
    TP_THROW_IF(shifts.size() != dims.size(), RuntimeError,
                "shifts and dimensions must align. shifts: " + std::to_string(shifts.size()) +
                ", dims:" + std::to_string(dims.size()));

    Tensor result = self;
    for (size_t i = 0; i < dims.size(); ++i) {
        // Single-dim roll from ATen roll(): narrow+cat.
        const int64_t dim = wrap_dim_local(dims[i], result.dim());
        const int64_t size = result.size(dim);
        if (result.numel() == 0 || size == 0) continue;
        const int64_t shift = shifts[i];
        const int64_t start = ((size - shift) % size + size) % size; // C++ % correction
        Tensor t0 = result.slice(dim, start, size);
        Tensor t1 = result.slice(dim, 0, start);
        result = Tensor::cat({t0, t1}, dim);
    }
    return result;
}

// ---------------------------------------------------------------------------
// diff — ATen native/ReduceOps.cpp diff()/diff_helper()
// ---------------------------------------------------------------------------
static Tensor diff_helper(const Tensor& self, int64_t n, int64_t dim) {
    // ATen diff_helper: repeated narrow(dim,1,out_len) - narrow(dim,0,out_len)
    Tensor result = self;
    n = n > self.size(dim) ? self.size(dim) : n;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t out_len = result.size(dim) - 1;
        result = result.slice(dim, 1, out_len + 1) - result.slice(dim, 0, out_len);
    }
    return result;
}

Tensor diff_cpu(const Tensor& self, int64_t n, int64_t dim, const std::optional<Tensor>& prepend_opt, const std::optional<Tensor>& append_opt) {
    const int64_t d = wrap_dim_local(dim, self.dim());
    // ATen diff(): concatenate prepend/append first when present.
    const Tensor prepend = prepend_opt.value_or(Tensor());
    const Tensor append = append_opt.value_or(Tensor());
    const bool has_prepend = prepend.defined();
    const bool has_append = append.defined();
    if ((!has_prepend && !has_append) || n == 0) {
        return diff_helper(self, n, d);
    }
    std::vector<Tensor> pieces;
    if (has_prepend) pieces.push_back(prepend);
    pieces.push_back(self);
    if (has_append) pieces.push_back(append);
    Tensor a = Tensor::cat(pieces, d);
    return diff_helper(a, n, d);
}

// ---------------------------------------------------------------------------
// masked_fill — ATen broadcasts mask against self and selects; expressed here
// through the dispatched where op (same semantics, see
// aten/src/ATen/native/TensorAdvancedIndexing.cpp masked_fill_impl).
// ---------------------------------------------------------------------------
Tensor masked_fill_cpu(const Tensor& self, const Tensor& mask, Scalar value) {
    Tensor filled = Tensor::full(
        static_cast<std::vector<int64_t>>(self.shape()), value, self.dtype(), self.device());
    return where_cpu(mask, filled, self);
}

Tensor& masked_fill__cpu(Tensor& self, const Tensor& mask, Scalar value) {
    Tensor filled = Tensor::full(
        static_cast<std::vector<int64_t>>(self.shape()), value, self.dtype(), self.device());
    self.copy_(where_cpu(mask, filled, self));
    return self;
}

// ---------------------------------------------------------------------------
// one_hot — ATen native/Onehot.cpp one_hot(): CPU path validates class values
// then scatters ones; the eq-vs-arange formulation from the same file
// (the "functional version" branch) produces the identical result without
// requiring a scatter dispatch.
// ---------------------------------------------------------------------------
Tensor one_hot_cpu(const Tensor& self, int64_t num_classes) {
    if (self.dtype() != DType::Int64) {
        TP_THROW(RuntimeError, "one_hot is only applicable to index tensor of type LongTensor.");
    }

    if (self.numel() == 0) {
        if (num_classes <= 0) {
            TP_THROW(RuntimeError, "Can not infer total number of classes from empty tensor.");
        }
        auto shape = static_cast<std::vector<int64_t>>(self.shape());
        shape.push_back(num_classes);
        return Tensor::empty(shape, self.dtype(), self.device());
    }

    auto [self_min, self_max] = [&]() -> std::pair<Scalar, Scalar> {
        Tensor mn = self.min();
        Tensor mx = self.max();
        return {mn.item(), mx.item()};
    }();
    if (self_min.to<int64_t>() < 0) {
        TP_THROW(RuntimeError, "Class values must be non-negative.");
    }
    if (num_classes == -1) {
        num_classes = self_max.to<int64_t>() + 1;
    } else if (num_classes <= self_max.to<int64_t>()) {
        TP_THROW(RuntimeError, "Class values must be smaller than num_classes.");
    }

    // Onehot.cpp functional branch: eq(self.unsqueeze(-1), arange(num_classes))
    Tensor index = Tensor::arange(Scalar(static_cast<int64_t>(0)), Scalar(num_classes),
                                  Scalar(static_cast<int64_t>(1)), DType::Int64, self.device());
    auto sizes = static_cast<std::vector<int64_t>>(self.shape());
    sizes.push_back(1);
    Tensor eq = eq_tensor_kernel(self.view(sizes), index).to(DType::Int64);
    return eq;
}

// ---------------------------------------------------------------------------
// glu — ATen native/GatedLinearUnit.cpp + cpu/Activation.cpp glu_kernel:
//   out = first_half * sigmoid(second_half)
// ---------------------------------------------------------------------------
Tensor glu_cpu(const Tensor& self, int64_t dim) {
    TP_THROW_IF(self.dim() == 0, RuntimeError, "glu does not support 0-dimensional tensors");
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

Tensor glu_backward_cpu(const Tensor& grad_output, const Tensor& self, int64_t dim) {
    // GatedLinearUnit.cpp glu_backward_cpu_out:
    //   grad_first = grad * sigmoid(second)
    //   grad_second = grad * first * sigmoid(second) * (1 - sigmoid(second))
    TP_THROW_IF(self.dim() == 0, RuntimeError, "glu does not support 0-dimensional tensors");
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

TENSORPLAY_LIBRARY_IMPL(CPU, MiscKernels) {
    m.impl("diff", diff_cpu);
    m.impl("masked_fill", masked_fill_cpu);
    m.impl("masked_fill_", masked_fill__cpu);
    m.impl("one_hot", one_hot_cpu);
    m.impl("glu", glu_cpu);
    m.impl("glu_backward", glu_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay

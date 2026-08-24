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
#include "Generator.h"
#include "DistributionsHelper.h"
#include <algorithm>
#include <numeric>
#include <tuple>

namespace tensorplay {
namespace cpu {

// Declared in ComparisonKernels.cpp; reused for masked_fill's broadcast
// select and one_hot's equality.  Lives at namespace scope so it resolves to
// the ComparisonKernels definition (an anonymous-namespace declaration would
// be a distinct, undefined symbol).
Tensor where_cpu(const Tensor& condition, const Tensor& self, const Tensor& other);
Tensor eq_tensor_kernel(const Tensor& self, const Tensor& other);

// Defined below the registration table.
Tensor& resize__cpu(Tensor& self, const std::vector<int64_t>& size);
std::tuple<Tensor, Tensor> native_dropout_cpu(const Tensor& input, double p);
std::tuple<Tensor, Tensor> native_alpha_dropout_cpu(const Tensor& input, double p);
Tensor alpha_dropout_backward_cpu(const Tensor& grad, const Tensor& mask, double p);
std::tuple<Tensor, Tensor> native_feature_dropout_cpu(const Tensor& input, double p);
Tensor feature_dropout_backward_cpu(const Tensor& grad, const Tensor& mask, double p);

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
// ---------------------------------------------------------------------------
// roll — ATen native/TensorTransformations.cpp roll() + roll_common()
// ---------------------------------------------------------------------------
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
    m.impl("one_hot", one_hot_cpu);
    m.impl("glu", glu_cpu);
    m.impl("glu_backward", glu_backward_cpu);
    m.impl("resize_", resize__cpu);
    m.impl("native_dropout", native_dropout_cpu);
    m.impl("native_alpha_dropout", native_alpha_dropout_cpu);
    m.impl("_alpha_dropout_backward", alpha_dropout_backward_cpu);
    m.impl("native_feature_dropout", native_feature_dropout_cpu);
    m.impl("_feature_dropout_backward", feature_dropout_backward_cpu);
}

// resize_ grows the storage in place (preserving the old contents) and then
// adopts contiguous strides; shrinking only changes the logical shape, like
// ATen. The metadata half is TensorImpl::set_sizes_contiguous.
Tensor& resize__cpu(Tensor& self, const std::vector<int64_t>& size) {
    auto* impl = self.unsafeGetTensorImpl().get();
    int64_t new_numel = 1;
    for (int64_t s : size) {
        if (s < 0) {
            TP_THROW(ValueError, "resize_: negative sizes are not allowed");
        }
        new_numel *= s;
    }
    const size_t new_bytes = static_cast<size_t>(new_numel) * impl->itemsize();
    if (!impl->has_storage()) {
        if (new_bytes > 0) {
            impl->set_storage(
                Storage(new_bytes, getAllocator(impl->device().type()), impl->device()));
        }
    } else if (new_bytes > impl->storage().nbytes()) {
        // Throws when the storage wraps foreign memory (resizable=false),
        // mirroring torch's resize error surface for such storages.
        Storage storage = impl->storage();
        storage.set_nbytes(new_bytes);
    }
    impl->set_sizes_contiguous(size);
    return self;
}

// Fused dropout forward: one RNG pass produces both the scaled output and
// the bool mask consumed by native_dropout's generated backward node
// (grad * mask / (1 - p)). p == 1 is rejected here because its scale is
// undefined; F.dropout gates that case in Python.
std::tuple<Tensor, Tensor> native_dropout_cpu(const Tensor& input, double p) {
    if (p < 0 || p >= 1) {
        TP_THROW(ValueError, "native_dropout: p must be in [0, 1)");
    }
    Tensor mask(static_cast<std::vector<int64_t>>(input.shape()), DType::Bool,
                input.device());
    Tensor out(static_cast<std::vector<int64_t>>(input.shape()), input.dtype(),
               input.device());
    const int64_t n = input.numel();
    auto& gen = default_generator();
    uniform_real_distribution<double> uniform(0.0, 1.0);
    const double scale = 1.0 / (1.0 - p);

    switch (input.dtype()) {
        case DType::Float32: {
            const float* in = input.data_ptr<float>();
            float* o = out.data_ptr<float>();
            bool* m = mask.data_ptr<bool>();
            for (int64_t i = 0; i < n; ++i) {
                const bool keep = uniform(&gen) >= p;
                m[i] = keep;
                o[i] = keep ? static_cast<float>(in[i] * scale) : 0.0f;
            }
            break;
        }
        case DType::Float64: {
            const double* in = input.data_ptr<double>();
            double* o = out.data_ptr<double>();
            bool* m = mask.data_ptr<bool>();
            for (int64_t i = 0; i < n; ++i) {
                const bool keep = uniform(&gen) >= p;
                m[i] = keep;
                o[i] = keep ? in[i] * scale : 0.0;
            }
            break;
        }
        case DType::Float16:
        case DType::BFloat16: {
            if (input.dtype() == DType::Float16) {
                const Half* in = input.data_ptr<Half>();
                Half* o = out.data_ptr<Half>();
                bool* m = mask.data_ptr<bool>();
                for (int64_t i = 0; i < n; ++i) {
                    const bool keep = uniform(&gen) >= p;
                    m[i] = keep;
                    o[i] = static_cast<Half>(keep
                                                 ? static_cast<double>(in[i]) * scale
                                                 : 0.0);
                }
            } else {
                const BFloat16* in = input.data_ptr<BFloat16>();
                BFloat16* o = out.data_ptr<BFloat16>();
                bool* m = mask.data_ptr<bool>();
                for (int64_t i = 0; i < n; ++i) {
                    const bool keep = uniform(&gen) >= p;
                    m[i] = keep;
                    o[i] = static_cast<BFloat16>(keep
                                                     ? static_cast<double>(in[i]) * scale
                                                     : 0.0);
                }
            }
            break;
        }
        default:
            TP_THROW(NotImplementedError,
                     "dropout is only supported on floating point tensors");
    }
    return {std::move(out), std::move(mask)};
}

// ---------------------------------------------------------------------------
// Alpha / feature dropout — ATen _dropout_impl<feature, alpha> fused as
// (output, mask) pairs so the backward can reapply the saved mask. The
// Bernoulli noise reuses the registered bernoulli_ kernel; the affine math
// is expressed through dispatched mul/add so both backends share one path.
// ---------------------------------------------------------------------------

namespace {

constexpr double kAlphaDropoutAlpha = 1.7580993408473766;

double alpha_dropout_scale(double p) {
    return 1.0 / std::sqrt((kAlphaDropoutAlpha * kAlphaDropoutAlpha * p + 1.0) *
                           (1.0 - p));
}

Tensor bernoulli_mask(const Tensor& input, const std::vector<int64_t>& shape,
                      double keep_prob) {
    Tensor noise = Tensor::full(shape, keep_prob, DType::Float32,
                                input.device());
    noise.bernoulli_();
    return noise;
}

} // anonymous namespace

std::tuple<Tensor, Tensor> native_alpha_dropout_cpu(const Tensor& input, double p) {
    if (p < 0 || p >= 1) {
        TP_THROW(ValueError, "alpha_dropout: p must be in [0, 1)");
    }
    Tensor mask = bernoulli_mask(input,
                                 static_cast<std::vector<int64_t>>(input.shape()),
                                 1.0 - p);
    const double a = alpha_dropout_scale(p);
    // out = mask * (x * a + alpha * a) + alpha * a * (p - 1)
    Tensor out = mask.mul(input.mul(a).add(kAlphaDropoutAlpha * a))
                    .add(kAlphaDropoutAlpha * a * (p - 1.0));
    return {std::move(out), std::move(mask)};
}

Tensor alpha_dropout_backward_cpu(const Tensor& grad, const Tensor& mask,
                                  double p) {
    const double a = alpha_dropout_scale(p);
    return grad.mul(mask).mul(a);
}

std::tuple<Tensor, Tensor> native_feature_dropout_cpu(const Tensor& input, double p) {
    if (p < 0 || p >= 1) {
        TP_THROW(ValueError, "feature_dropout: p must be in [0, 1)");
    }
    if (input.dim() < 2) {
        TP_THROW(RuntimeError, "feature_dropout requires at least 2D input");
    }
    std::vector<int64_t> mask_shape =
        static_cast<std::vector<int64_t>>(input.shape());
    for (int64_t d = 2; d < input.dim(); ++d) mask_shape[d] = 1;
    Tensor mask = bernoulli_mask(input, mask_shape, 1.0 - p);
    Tensor out = input.mul(mask).div(1.0 - p);
    return {std::move(out), std::move(mask)};
}

Tensor feature_dropout_backward_cpu(const Tensor& grad, const Tensor& mask,
                                    double p) {
    return grad.mul(mask).div(1.0 - p);
}

} // namespace cpu
} // namespace tensorplay

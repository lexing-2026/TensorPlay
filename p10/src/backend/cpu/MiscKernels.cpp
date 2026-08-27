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
#include <cmath>
#include <limits>
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
Tensor trapezoid_cpu(const Tensor& y, const std::optional<Tensor>& x, Scalar dx, int64_t dim);
Tensor cumulative_trapezoid_cpu(const Tensor& y, const std::optional<Tensor>& x, Scalar dx, int64_t dim);
Tensor trapezoid_backward_cpu(const Tensor& grad, const std::optional<Tensor>& x,
                              const std::vector<int64_t>& ysizes, Scalar dx,
                              int64_t dim);
Tensor cumulative_trapezoid_backward_cpu(const Tensor& grad,
                                         const std::optional<Tensor>& x_opt,
                                         Scalar dx_s, int64_t dim);
Tensor cov_cpu(const Tensor& self, int64_t correction,
               const std::optional<Tensor>& fweights_opt,
               const std::optional<Tensor>& aweights_opt);
Tensor corrcoef_cpu(const Tensor& self);
Tensor cov_backward_cpu(const Tensor& grad, const Tensor& self, int64_t correction,
                        const std::optional<Tensor>& fweights_opt,
                        const std::optional<Tensor>& aweights_opt);
Tensor corrcoef_backward_cpu(const Tensor& grad, const Tensor& self);
Tensor quantile_kernel(const Tensor& self, const Tensor& q,
                       std::optional<int64_t> dim, bool keepdim,
                       std::string interpolation);
Tensor nanquantile_kernel(const Tensor& self, const Tensor& q,
                          std::optional<int64_t> dim, bool keepdim,
                          std::string interpolation);
std::tuple<Tensor, Tensor> histogram_bins_tensor_kernel(
    const Tensor& self, const Tensor& bins,
    const std::optional<Tensor>& weight, bool density);
std::tuple<Tensor, Tensor> histogram_bin_ct_kernel(
    const Tensor& self, int64_t bins,
    const std::optional<std::vector<double>>& range,
    const std::optional<Tensor>& weight, bool density);

namespace {

inline int64_t wrap_dim_local(int64_t dim, int64_t ndim) {
    const int64_t min_ = -ndim;
    const int64_t max_ = ndim - 1;
    if (dim < min_ || dim > max_) {
        TP_THROW(IndexError,
                 "Dimension out of range (expected to be in range of [" +
                     std::to_string(min_) + ", " + std::to_string(max_) +
                     "], but got " + std::to_string(dim) + ")");
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
    m.impl("trapezoid", trapezoid_cpu);
    m.impl("cumulative_trapezoid", cumulative_trapezoid_cpu);
    m.impl("_trapezoid_backward", trapezoid_backward_cpu);
    m.impl("_cumulative_trapezoid_backward", cumulative_trapezoid_backward_cpu);
    m.impl("cov", cov_cpu);
    m.impl("corrcoef", corrcoef_cpu);
    m.impl("_cov_backward", cov_backward_cpu);
    m.impl("_corrcoef_backward", corrcoef_backward_cpu);
    m.impl("quantile", quantile_kernel);
    m.impl("nanquantile", nanquantile_kernel);
    m.impl("histogram.bins_tensor", histogram_bins_tensor_kernel);
    m.impl("histogram.bin_ct", histogram_bin_ct_kernel);
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


// ---------------------------------------------------------------------------
// Trapezoid integration — ATen native Sum.cpp trapezoid/cumulative_trapezoid
// expressed as dispatcher composites (narrow/add/mul/sum|cumsum). x=None
// selects uniform spacing dx. Backward rebuilds the per-element weights:
//   sum form:   w = dx * [0.5, 1, ..., 1, 0.5]
//   x form:     w = 0.5 * ([x1-x0] ++ diff(x) ++ [x_{n-1}-x_{n-2}])
//   cumulative: grad_y[j] = w[j] * sum_{k >= j} grad[k]
// ---------------------------------------------------------------------------

namespace {

int64_t trapz_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, "Dimension out of range");
    }
    return dim;
}

// Uniform-spacing weight vector for a length-n axis:
//   n >= 2: [0.5dx, dx, ..., dx, 0.5dx]   (n == 1: [dx])
Tensor uniform_weights(int64_t n, double dx, const Tensor& like) {
    if (n < 2) {
        return Tensor::full({n}, dx, like.dtype(), like.device());
    }
    return Tensor::cat({Tensor::full({1}, 0.5 * dx, like.dtype(),
                                      like.device()),
                        Tensor::full({std::max<int64_t>(n - 2, 0)}, dx,
                                     like.dtype(), like.device()),
                        Tensor::full({1}, 0.5 * dx, like.dtype(),
                                     like.device())},
                       0);
}

// Coordinate-difference weights along `dim` of x:
//   w = 0.5 * ([seg_0] ++ (seg_i + seg_{i+1}) ++ [seg_{n-2}]), length n.
Tensor onesided_weights(const Tensor& x, int64_t d) {
    const int64_t n = x.size(d);
    Tensor segs = x.narrow(d, 1, n - 1).sub(x.narrow(d, 0, n - 1));
    if (n == 2) {
        return segs.mul(0.5);
    }
    Tensor inner = segs.narrow(d, 0, n - 2).add(segs.narrow(d, 1, n - 2));
    return Tensor::cat({segs.narrow(d, 0, 1), inner,
                        segs.narrow(d, n - 2, 1)}, d).mul(0.5);
}

} // anonymous namespace

Tensor trapezoid_cpu(const Tensor& y, const std::optional<Tensor>& x_opt, Scalar dx_s, int64_t dim) {
    const double dx = dx_s.toDouble();
    const Tensor x = x_opt.value_or(Tensor());
    const int64_t d = trapz_dim(dim, y.dim());
    const int64_t n = y.size(d);
    if (n < 2) TP_THROW(RuntimeError, "trapezoid(): requires at least 2 points");
    Tensor avg = y.narrow(d, 0, n - 1).add(y.narrow(d, 1, n - 1)).mul(0.5);
    if (x.defined()) {
        bool was_1d = x.dim() == 1;
        Tensor xb = was_1d && y.dim() > 1
            ? [&] {
                  std::vector<int64_t> view(y.dim(), 1);
                  view[d] = x.size(0);
                  return x.reshape(view).expand(
                      static_cast<std::vector<int64_t>>(y.shape()));
              }()
            : x;
        avg = avg.mul(xb.narrow(d, 1, n - 1).sub(xb.narrow(d, 0, n - 1)));
    } else {
        avg = avg.mul(dx);
    }
    return avg.sum(std::vector<int64_t>{d}, false);
}

Tensor cumulative_trapezoid_cpu(const Tensor& y, const std::optional<Tensor>& x_opt,
                                Scalar dx_s, int64_t dim) {
    const double dx = dx_s.toDouble();
    const Tensor x = x_opt.value_or(Tensor());
    const int64_t d = trapz_dim(dim, y.dim());
    const int64_t n = y.size(d);
    if (n < 2) TP_THROW(RuntimeError,
                        "cumulative_trapezoid(): requires at least 2 points");
    Tensor avg = y.narrow(d, 0, n - 1).add(y.narrow(d, 1, n - 1)).mul(0.5);
    if (x.defined()) {
        bool was_1d = x.dim() == 1;
        Tensor xb = was_1d && y.dim() > 1
            ? [&] {
                  std::vector<int64_t> view(y.dim(), 1);
                  view[d] = x.size(0);
                  return x.reshape(view).expand(
                      static_cast<std::vector<int64_t>>(y.shape()));
              }()
            : x;
        avg = avg.mul(xb.narrow(d, 1, n - 1).sub(xb.narrow(d, 0, n - 1)));
    } else {
        avg = avg.mul(dx);
    }
    return avg.cumsum(d);
}

namespace {

// grad (shape minus the reduced dim) times weights viewed along dim.
Tensor apply_sum_weights(const Tensor& grad, int64_t d, const Tensor& w1d) {
    std::vector<int64_t> view(grad.dim() + 1, 1);
    view[d] = w1d.numel();
    return grad.unsqueeze(d).mul(w1d.reshape(view));
}

} // anonymous namespace

Tensor trapezoid_backward_cpu(const Tensor& grad, const std::optional<Tensor>& x_opt,
                              const std::vector<int64_t>& ysizes, Scalar dx_s,
                              int64_t dim) {
    const double dx = dx_s.toDouble();
    const Tensor x = x_opt.value_or(Tensor());
    const int64_t ndim = static_cast<int64_t>(ysizes.size());
    const int64_t d = trapz_dim(dim, ndim);
    const int64_t n = ysizes[d];
    Tensor w1d = x.defined() ? onesided_weights(x.reshape(
                                   std::vector<int64_t>{x.numel()}), 0)
                             : uniform_weights(n, dx, grad);
    if (n == 1) return Tensor::zeros(ysizes, grad.dtype(), grad.device());
    return apply_sum_weights(grad, d, w1d.to(grad.dtype()));
}

Tensor cumulative_trapezoid_backward_cpu(const Tensor& grad, const std::optional<Tensor>& x_opt,
                                         Scalar dx_s, int64_t dim) {
    // Each output element k accumulates segments 0..k; y_j is the right end
    // of segment j-1 and the left end of segment j:
    //   g_y[j] = w[j]*suffix(j) + w[j-1]*suffix(j-1)
    const Tensor x = x_opt.value_or(Tensor());
    const double dx = dx_s.toDouble();
    const int64_t d = trapz_dim(dim, grad.dim());
    const int64_t m = grad.size(d);
    const std::vector<int64_t> dv{d};
    Tensor acc = Tensor::flip(Tensor::flip(grad, dv).cumsum(d), dv);
    Tensor seg_w;
    if (x.defined()) {
        seg_w = onesided_weights(x.reshape(std::vector<int64_t>{x.numel()}),
                                 0).to(grad.dtype());
    } else {
        seg_w = Tensor::full({m}, dx, grad.dtype(), grad.device());
    }
    std::vector<int64_t> wview(grad.dim(), 1);
    wview[d] = m;
    Tensor ws = acc.mul(seg_w.reshape(wview));
    auto tail_shape = static_cast<std::vector<int64_t>>(ws.shape());
    tail_shape[d] = 1;
    Tensor zero_tail = Tensor::zeros(tail_shape, grad.dtype(), grad.device());
    // point j gets segment j (as left end) and segment j-1 (as right end)
    Tensor term_a = Tensor::cat({ws, zero_tail}, d);
    Tensor term_b = Tensor::cat({zero_tail, ws}, d);
    return term_a.add(term_b).mul(0.5);
}

// ---------------------------------------------------------------------------
// cov / corrcoef — ATen native/Correlation.cpp verbatim port. Each row is a
// variable, each column an observation; fweights are frequencies (integral),
// aweights reliability weights (floating). Arithmetic stays in the input
// dtype exactly like upstream (no upcast). The 1-observation single-weight
// corner zeroes `in` through its aliasing view precisely as upstream does.
// Backwards are explicit helpers (_cov_backward / _corrcoef_backward) since
// tp has no composite-implicit-autograd; formulas derived from the closed
// form and validated against torch's composite autograd numerically.
// ---------------------------------------------------------------------------

namespace {

struct CovParts {
    Tensor in;        // centered (post-corner) matrix, always floating
    Tensor w;         // combined weights (undefined when none given)
    Tensor wsum;      // sum(w) or scalar Long num_observations
    Tensor fact;      // norm_factor 0-dim tensor (Long or float)
    int64_t num_observations;
    bool had_fw;
    bool had_aw;
};

// at::is_scalar_tensor_true for 0-dim tensors.
bool cov_scalar_true(const Tensor& t) {
    if (t.dtype() == DType::Bool) return t.item().to<bool>();
    if (isIntegralType(t.dtype(), /*includeBool=*/false)) {
        return t.item().to<int64_t>() != 0;
    }
    return t.item().toDouble() != 0.0;
}

Tensor cov_scalar_long(int64_t v, const Tensor& like) {
    return Tensor::full({}, v, DType::Int64, like.device());
}

CovParts cov_parts(const Tensor& self, int64_t correction,
                   const std::optional<Tensor>& fweights_opt,
                   const std::optional<Tensor>& aweights_opt) {
    constexpr int64_t OBSERVATIONS_DIM = 1;
    CovParts p;
    p.had_fw = fweights_opt.has_value();
    p.had_aw = aweights_opt.has_value();

    if (self.dim() > 2) {
        TP_THROW(RuntimeError,
                 "cov(): expected input to have two or fewer dimensions but got "
                 "an input with " + std::to_string(self.dim()) + " dimensions");
    }
    TP_CHECK_NOT_IMPLEMENTED(self.dtype() != DType::Bool,
                             "cov(): bool dtype is not supported for input");

    // View input tensor as 2D (variables, observations)
    Tensor in = self.dim() < 2 ? self.view({1, -1}) : self;
    p.num_observations = in.size(OBSERVATIONS_DIM);

    Tensor w;
    if (p.had_fw) {
        const Tensor& fwv = *fweights_opt;
        if (fwv.dim() > 1) {
            TP_THROW(RuntimeError,
                     "cov(): expected fweights to have one or fewer dimensions but got "
                     "fweights with " + std::to_string(fwv.dim()) + " dimensions");
        }
        TP_CHECK(isIntegralType(fwv.dtype(), /*includeBool=*/false),
                 "cov(): expected fweights to have integral dtype but got fweights with dtype ",
                 static_cast<int>(fwv.dtype()));
        TP_CHECK(fwv.numel() == p.num_observations,
                 "cov(): expected fweights to have the same numel as there are observations "
                 "in the input but got ", fwv.numel(), " != ", p.num_observations);
        TP_CHECK(p.num_observations == 0 ||
                     fwv.min().item().toDouble() >= 0.0,
                 "cov(): fweights cannot be negative");
        w = fwv;
    }

    if (p.had_aw) {
        const Tensor& aw = *aweights_opt;
        if (aw.dim() > 1) {
            TP_THROW(RuntimeError,
                     "cov(): expected aweights to have one or fewer dimensions but got "
                     "aweights with " + std::to_string(aw.dim()) + " dimensions");
        }
        TP_CHECK(isFloatingType(aw.dtype()),
                 "cov(): expected aweights to have floating point dtype but got "
                 "aweights with dtype ", static_cast<int>(aw.dtype()));
        TP_CHECK(aw.numel() == p.num_observations,
                 "cov(): expected aweights to have the same numel as there are observations "
                 "in the input but got ", aw.numel(), " != ", p.num_observations);
        TP_CHECK(p.num_observations == 0 || aw.min().item().toDouble() >= 0.0,
                 "cov(): aweights cannot be negative");
        // product of frequencies (fweights) and reliability weights (aweights)
        w = w.defined() ? w.mul(aw) : aw;
    }
    p.w = w;

    // Weighted average of the observations
    p.wsum = w.defined()
        ? w.sum()
        : cov_scalar_long(p.num_observations, in);

    TP_CHECK(!w.defined() || cov_scalar_true(p.wsum),
             "cov(): weights sum to zero, can't be normalized");

    const Tensor avg =
        (w.defined() ? in.mul(w) : in)
            .sum(std::vector<int64_t>{OBSERVATIONS_DIM})
            .div(p.wsum);

    // Normalization factor
    if (!w.defined()) {
        p.fact = cov_scalar_long(p.num_observations - correction, in);
    } else if (correction == 0) {
        p.fact = p.wsum;
    } else if (!p.had_aw) {
        p.fact = p.wsum.sub(Scalar(correction));
    } else if (!p.had_fw && p.num_observations == 1 && correction == 1) {
        // corner case that was causing rounding error and deviating from numpy
        p.fact = cov_scalar_long(0, in);
    } else {
        p.fact = p.wsum.sub(w.mul(*aweights_opt).sum().mul(correction).div(p.wsum));
    }

    if (p.fact.item().toDouble() <= 0.0) {
        TP_WARN("cov(): degrees of freedom is <= 0. Correction should be strictly "
                "less than the number of observations.");
        p.fact.zero_();
    }

    if (p.num_observations == 1 && p.had_fw != p.had_aw) {
        // algebraically the weighted avg == the input so the centered matrix
        // would be zero; upstream zeroes `in` through its aliasing view.
        in.zero_();
        if (isIntegralType(in.dtype(), false)) {
            in = in.to(DType::Float32);
        }
        p.in = in;
    } else {
        p.in = in.sub(avg.unsqueeze(1));
    }
    return p;
}

Tensor cov_matrix_from(const CovParts& p) {
    Tensor c = Tensor::mm(p.in, (p.w.defined() ? p.in.mul(p.w) : p.in).t());
    return c.div(p.fact);
}

// dL/dX given grad H wrt the (k,k) covariance matrix:
//   G_M    = ((H + H^T) M diag(w)) / fact
//   dL/dX  = G_M - rowsum(G_M) * (w / wsum)
// (unweighted: w/wsum collapses to 1/n). The avg term is expressed as
// rowsum.mul(w).div(wsum) so integral weights divide in the gradient dtype
// instead of being rounded through a Float32 promotion.
Tensor cov_apply_grad(const Tensor& H, const CovParts& p, const Tensor& like) {
    const int64_t n = p.num_observations;
    Tensor GM = H.add(H.t()).mm(p.in).div(p.fact);
    if (p.w.defined()) {
        const Tensor wrow = p.w.reshape({1, n});
        GM = GM.mul(wrow);
        Tensor rowsum = GM.sum(std::vector<int64_t>{1}, true);
        // rowsum is taken over the *weighted* G_M above
        return GM.sub(rowsum.mul(wrow).div(p.wsum))
            .reshape(static_cast<std::vector<int64_t>>(like.shape()));
    }
    Tensor rowsum = GM.sum(std::vector<int64_t>{1}, true);
    Tensor inv_n = Tensor::full({}, static_cast<int64_t>(n), DType::Int64,
                                GM.device());
    return GM.sub(rowsum.div(inv_n))
        .reshape(static_cast<std::vector<int64_t>>(like.shape()));
}

} // anonymous namespace

Tensor cov_cpu(const Tensor& self, int64_t correction,
               const std::optional<Tensor>& fweights_opt,
               const std::optional<Tensor>& aweights_opt) {
    CovParts p = cov_parts(self, correction, fweights_opt, aweights_opt);
    return cov_matrix_from(p).squeeze();
}

Tensor corrcoef_cpu(const Tensor& self) {
    if (self.dim() > 2) {
        TP_THROW(RuntimeError,
                 "corrcoef(): expected input to have two or fewer dimensions but got "
                 "an input with " + std::to_string(self.dim()) + " dimensions");
    }
    Tensor c = cov_cpu(self, 1, std::nullopt, std::nullopt);
    if (c.dim() == 0) {
        // scalar covariance: NaN if c in {nan, inf, 0}, 1 otherwise
        return c.div(c);
    }
    const Tensor d = c.diagonal();
    const Tensor stddev = d.sqrt();
    c = c.div(stddev.view({-1, 1}));
    c = c.div(stddev.view({1, -1}));
    // values may be not within [-1, 1] due to rounding; clip like NumPy
    return c.clamp(Scalar(-1.0), Scalar(1.0));
}

Tensor cov_backward_cpu(const Tensor& grad, const Tensor& self, int64_t correction,
                        const std::optional<Tensor>& fweights_opt,
                        const std::optional<Tensor>& aweights_opt) {
    CovParts p = cov_parts(self, correction, fweights_opt, aweights_opt);
    if (p.num_observations == 1 && p.had_fw != p.had_aw) {
        // forward zeroes the input through its aliasing view: the covariance
        // is identically zero regardless of self
        return Tensor::zeros(static_cast<std::vector<int64_t>>(self.shape()),
                             grad.dtype(), self.device());
    }
    const int64_t k = p.in.size(0);
    return cov_apply_grad(grad.reshape({k, k}), p, self);
}

Tensor corrcoef_backward_cpu(const Tensor& grad, const Tensor& self) {
    if (self.dim() > 2) {
        TP_THROW(RuntimeError,
                 "corrcoef(): expected input to have two or fewer dimensions but got "
                 "an input with " + std::to_string(self.dim()) + " dimensions");
    }
    CovParts p = cov_parts(self, 1, std::nullopt, std::nullopt);
    if (p.num_observations == 1 && p.had_fw != p.had_aw) {
        return Tensor::zeros(static_cast<std::vector<int64_t>>(self.shape()),
                             grad.dtype(), self.device());
    }
    Tensor C = cov_matrix_from(p);
    const int64_t k = C.size(0);
    if (k == 1 && C.size(1) == 1) {
        // forward returned c / c: locally constant where defined
        return Tensor::zeros(static_cast<std::vector<int64_t>>(self.shape()),
                             grad.dtype(), self.device());
    }
    Tensor H = grad.reshape({k, k});
    const Tensor s = C.diagonal().sqrt();
    const Tensor R = C.div(s.view({-1, 1})).div(s.view({1, -1}));
    // clip backward: keep entries strictly inside (-1, 1)
    Tensor Hp = Tensor::where(R.gt(-1.0), H, Scalar(0));
    Hp = Tensor::where(R.lt(1.0), Hp, Scalar(0));
    // R = D^-1 C D^-1 with D = diag(s):
    //   dL/dC = K + diag(-g_i / (2 s_i^2)),  K = Hp / (s s^T),
    //   g_i = sum_j K_ij C_ij + sum_j K_ji C_ij  (R is sensitive to s_i
    //   through both its row and its column; C symmetric keeps C_ij shared)
    Tensor K = Hp.div(s.view({-1, 1})).div(s.view({1, -1}));
    Tensor crow = K.mul(C).sum(std::vector<int64_t>{1}, false);
    Tensor ccol = K.transpose(0, 1).mul(C).sum(std::vector<int64_t>{1}, false);
    Tensor coef = crow.add(ccol).div(s.mul(2)).div(s);
    Tensor diagm = Tensor::eye(k, k, K.dtype(), K.device())
                       .mul(coef.reshape({1, k}));
    Tensor GC = K.sub(diagm);
    return cov_apply_grad(GC, p, self);
}


// ---------------------------------------------------------------------------
// quantile / nanquantile — ATen native/Sorting.cpp quantile_impl expressed as
// a dispatcher composite over sort/gather/lerp (upstream's CPU nth_element
// fast path is a pure optimization; the sort path is semantically identical,
// NaN sorts last to match the rank masking below).  One body serves both
// dense backends, like ATen's device-generic native function.  Upstream has
// no derivatives.yaml entry: quantile is non-differentiable.
// ---------------------------------------------------------------------------
namespace {

enum class QuantileInterp { Linear, Lower, Higher, Midpoint, Nearest };

QuantileInterp get_quantile_interpolation_mode(const std::string& interpolation) {
    if (interpolation == "linear") return QuantileInterp::Linear;
    if (interpolation == "lower") return QuantileInterp::Lower;
    if (interpolation == "higher") return QuantileInterp::Higher;
    if (interpolation == "midpoint") return QuantileInterp::Midpoint;
    if (interpolation == "nearest") return QuantileInterp::Nearest;
    TP_THROW(ValueError,
             "quantile() interpolation must be one of linear, lower, higher, "
             "midpoint or nearest, but got ", interpolation);
}

void quantile_checks(const Tensor& self, const Tensor& q) {
    if (self.numel() == 0) {
        TP_THROW(ValueError, "quantile() input tensor must be non-empty");
    }
    if (q.dim() > 1) {
        TP_THROW(ValueError, "quantile() q must be a scalar or 1D tensor");
    }
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(ValueError,
                 "quantile() input tensor must be either float or double dtype");
    }
    if (self.dtype() != q.dtype()) {
        TP_THROW(ValueError,
                 "quantile() q tensor must be same dtype as the input tensor");
    }
    if (self.device() != q.device()) {
        TP_THROW(ValueError,
                 "quantile() q tensor must be on the same device as the input tensor");
    }
}

std::vector<int64_t> quantile_out_shape(const std::optional<int64_t>& original_dim,
                                        const Tensor& self, const Tensor& q,
                                        bool keepdim, int64_t wrapped_dim) {
    std::vector<int64_t> out_shape;
    if (original_dim && self.dim() > 0) {
        out_shape = static_cast<std::vector<int64_t>>(self.shape());
        if (keepdim) {
            out_shape[wrapped_dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + wrapped_dim);
        }
    } else if (keepdim) {
        out_shape.assign(self.dim(), 1);
    }
    if (q.dim() > 0) {
        out_shape.insert(out_shape.begin(), q.numel());
    }
    return out_shape;
}

Tensor quantile_compute(const Tensor& self, const Tensor& q,
                        const std::optional<int64_t>& original_dim,
                        bool keepdim, QuantileInterp interpolation,
                        bool ignore_nan, int64_t wrapped_dim,
                        std::vector<int64_t> out_shape) {
    // Upstream range-checks q only on the CPU to avoid a device sync.
    if (self.device().is_cpu()) {
        Tensor q_in_range = Tensor::logical_and(q.ge(0), q.le(1)).all();
        if (!q_in_range.item<bool>()) {
            TP_THROW(ValueError, "quantile() q values must be in the range [0, 1]");
        }
    }

    // Flatten input if no dim provided, else move dim to reduce as the last dim.
    Tensor reduced;
    if (!original_dim) {
        reduced = self.flatten();
    } else if (wrapped_dim == self.dim() - 1) {
        reduced = self;
    } else {
        reduced = self.unsqueeze(-1).transpose(wrapped_dim, -1);
    }

    // Treat q as 1-D: pad out_shape with a leading q slot for the reshape below.
    if (q.dim() == 0) {
        out_shape.insert(out_shape.begin(), 1);
    }
    std::vector<int64_t> in_shape(out_shape.size());
    std::copy(out_shape.begin() + 1, out_shape.end(), in_shape.begin());
    in_shape[in_shape.size() - 1] = reduced.size(-1);
    reduced = reduced.reshape(in_shape);

    // Ranks use double (exact to 2^53) on both dense backends, matching
    // upstream's non-MPS path.  Upstream relies on NaN sorting last; tp's
    // sort places NaN first (ascending), so NaN handling is made explicit
    // below instead of depending on the sort's NaN order.
    Tensor ranks;
    if (ignore_nan) {
        // nanquantile: ranks span [0, k-1] over the non-NaN count k.  Replace
        // each NaN with +inf so it sorts to the tail under *either* NaN
        // ordering; ranks capped at k-1 < n-1 never gather a substituted
        // value.  The count must be computed before substitution.
        Tensor count = reduced.isnan().logical_not().sum({-1}, true);
        ranks = q.to(DType::Float64).mul(count.sub(1));
        ranks = ranks.masked_fill(ranks.lt(0), 0);
        if (count.lt(reduced.size(-1)).any().item<bool>()) {
            reduced = reduced.masked_fill(
                reduced.isnan(), std::numeric_limits<double>::infinity());
        }
    } else {
        // quantile: any NaN in the row makes every output NaN.  Detect it
        // first, then fill the whole row with NaN so a gather at any rank --
        // including the pinned last index below -- yields NaN under either
        // sort placement.  Rows without NaN are untouched (bit-exact path).
        Tensor has_nan = reduced.isnan().any({-1}, true);
        if (has_nan.any().item<bool>()) {
            std::vector<int64_t> in_shape =
                static_cast<std::vector<int64_t>>(reduced.shape());
            reduced = reduced.masked_fill(
                has_nan.expand(in_shape),
                std::numeric_limits<double>::quiet_NaN());
        }
        const double last_index = static_cast<double>(reduced.size(-1) - 1);
        std::vector<Tensor> tl = Tensor::broadcast_tensors(
            {q.to(DType::Float64).mul(last_index), has_nan});
        ranks = tl[0].masked_fill(tl[1], last_index);
    }

    if (interpolation == QuantileInterp::Lower) {
        ranks = ranks.floor();
    } else if (interpolation == QuantileInterp::Higher) {
        ranks = ranks.ceil();
    } else if (interpolation == QuantileInterp::Nearest) {
        ranks = ranks.round();
    }

    Tensor ranks_below = ranks.to(DType::Int64);
    const bool interpolate = interpolation == QuantileInterp::Linear ||
                             interpolation == QuantileInterp::Midpoint;

    Tensor weights, ranks_above;
    if (interpolate) {
        if (interpolation == QuantileInterp::Midpoint) {
            // Weight 0.5 in the *value* dtype, like upstream's
            // at::full_like(ranks, 0.5, self.options()).
            weights = Tensor::full_like(ranks, 0.5, self.dtype());
        } else {
            weights = ranks.sub(ranks_below).to(self.dtype());
        }
        ranks_above = ranks.ceil().to(DType::Int64);
    }

    Tensor sorted = std::get<0>(reduced.sort(-1, false));
    Tensor values_below = sorted.gather(-1, ranks_below);
    if (interpolate) {
        Tensor values_above = sorted.gather(-1, ranks_above);
        values_below = values_below.lerp(values_above, weights);
    }

    Tensor values = values_below;
    if (q.dim() == 0) {
        // Scalar q: drop the padded q dim.
        values = values.squeeze(-1);
    } else {
        // Move the quantiles (last dim after broadcast) to the front.
        values = values.unsqueeze(0).transpose(0, -1).squeeze(-1);
    }
    return values;
}

} // anonymous namespace

Tensor quantile_kernel(const Tensor& self, const Tensor& q,
                       std::optional<int64_t> dim, bool keepdim,
                       std::string interpolation) {
    quantile_checks(self, q);
    const QuantileInterp mode = get_quantile_interpolation_mode(interpolation);
    int64_t wrapped_dim = dim.has_value() ? dim.value() : 0;
    // at::maybe_wrap_dim(dim.value_or(0), ndim) with wrap_scalar=true: a
    // 0-dim tensor wraps against a virtual 1-D range.
    const int64_t ndim = self.dim() == 0 ? 1 : self.dim();
    if (wrapped_dim < 0) wrapped_dim += ndim;
    if (wrapped_dim < 0 || wrapped_dim >= ndim) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim.value(), ")");
    }
    auto out_shape = quantile_out_shape(dim, self, q, keepdim, wrapped_dim);
    return quantile_compute(self, q, dim, keepdim, mode, /*ignore_nan=*/false,
                            wrapped_dim, std::move(out_shape));
}

Tensor nanquantile_kernel(const Tensor& self, const Tensor& q,
                          std::optional<int64_t> dim, bool keepdim,
                          std::string interpolation) {
    quantile_checks(self, q);
    const QuantileInterp mode = get_quantile_interpolation_mode(interpolation);
    int64_t wrapped_dim = dim.has_value() ? dim.value() : 0;
    const int64_t ndim = self.dim() == 0 ? 1 : self.dim();
    if (wrapped_dim < 0) wrapped_dim += ndim;
    if (wrapped_dim < 0 || wrapped_dim >= ndim) {
        TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim.value(), ")");
    }
    auto out_shape = quantile_out_shape(dim, self, q, keepdim, wrapped_dim);
    return quantile_compute(self, q, dim, keepdim, mode, /*ignore_nan=*/true,
                            wrapped_dim, std::move(out_shape));
}

// ---------------------------------------------------------------------------
// histogram — ATen native/Histogram.cpp + cpu/HistogramKernel.cpp.  The 1-D
// entry points reshape to (M, 1) and reuse the histogramdd machinery; the
// outer edges come from `range` or aminmax (NaN propagates, all-NaN input
// raises), with the empty-range ±0.5 expansion.  Per-element bin mapping
// reproduces BINARY_SEARCH (std::upper_bound == searchsorted right=True);
// upstream documents LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH as producing the
// same classification for linspace edges, so both paths share one
// formulation.  Elements outside [edges[0], edges[-1]] (and NaN) are skipped
// upstream, which equals zeroing their weight; accumulation goes through the
// dispatched index_add so the same body registers for CPU and CUDA.
// ---------------------------------------------------------------------------
namespace {

void histogram_check_weight(const Tensor& self, const std::optional<Tensor>& weight) {
    if (weight && weight->dtype() != self.dtype()) {
        TP_THROW(ValueError,
                 "torch.histogramdd: if weight tensor is provided, input "
                 "tensor and weight tensor should have the same dtype");
    }
}

std::pair<double, double> histogram_outer_edges_1d(
        const Tensor& self, const std::optional<std::vector<double>>& range) {
    // Defaults for empty input match numpy.histogram, like upstream.
    double leftmost = 0.0, rightmost = 1.0;
    if (range) {
        if (range->size() != 2) {
            TP_THROW(ValueError,
                     "torch.histogramdd: for a 1-dimensional histogram range "
                     "should have 2 elements, but got ", range->size());
        }
        leftmost = (*range)[0];
        rightmost = (*range)[1];
    } else if (self.numel() > 0) {
        auto mm = Tensor::aminmax(self.reshape({self.numel()}), {}, false);
        leftmost = std::get<0>(mm).item<double>();
        rightmost = std::get<1>(mm).item<double>();
    }
    if (!std::isfinite(leftmost) || !std::isfinite(rightmost)) {
        TP_THROW(ValueError, "torch.histogramdd: dimension 0's range [",
                 leftmost, ", ", rightmost, "] is not finite");
    }
    if (leftmost > rightmost) {
        TP_THROW(ValueError,
                 "torch.histogramdd: min should not exceed max, but got min ",
                 leftmost, " max ", rightmost);
    }
    // Expand an empty range like numpy to avoid a zero bin width.
    if (leftmost == rightmost) {
        leftmost -= 0.5;
        rightmost += 0.5;
    }
    return {leftmost, rightmost};
}

Tensor histogram_bin_counts(const Tensor& self, const Tensor& edges,
                            const std::optional<Tensor>& weight, bool density) {
    const int64_t nb = edges.numel() - 1;
    if (nb < 1) {
        TP_THROW(ValueError, "torch.histogram(): bins must be > 0, but got ",
                 nb, " for dimension 0");
    }
    Tensor hist = Tensor::zeros({nb}, self.dtype(), self.device());
    // Empty input: every bin stays zero (numpy/torch parity).  Skip the
    // searchsorted/index_add path entirely -- besides having nothing to
    // count, zero-element tensors may carry no storage, and broadcasting a
    // size-0 dim against the size-1 edge slices would otherwise drive the
    // elementwise kernels over unallocated memory.  density still applies
    // below, so an empty density histogram is all-NaN exactly like torch.
    if (self.numel() > 0) {
        Tensor v = self.reshape({self.numel()});
        // In-range compares against the edge tensor in the input dtype (NaN
        // fails both comparisons, matching the upstream
        // `elt >= lo && elt <= hi` skip).
        Tensor in_range = Tensor::logical_and(
            v.ge(edges.narrow(0, 0, 1)), v.le(edges.narrow(0, nb, 1)));
        // searchsorted right=True == std::upper_bound over the edges;
        // pos = idx-1 is the upstream BINARY_SEARCH classification, clamped
        // so the rightmost bin includes its right edge.
        Tensor idx = Tensor::searchsorted(edges, v, false, true).sub(1)
                         .clamp(0, nb - 1);
        Tensor w = weight.has_value() ? weight->reshape({self.numel()})
                                      : Tensor::ones_like(v);
        // Skipped elements contribute zero weight == upstream's skip.
        w = w.mul(in_range.to(self.dtype()));
        hist = hist.index_add(0, idx, w);
    }
    if (density) {
        hist = hist.div(hist.sum());
        hist = hist.div(edges.narrow(0, 1, nb).sub(edges.narrow(0, 0, nb)));
    }
    return hist;
}

} // anonymous namespace

std::tuple<Tensor, Tensor> histogram_bins_tensor_kernel(
        const Tensor& self, const Tensor& bins,
        const std::optional<Tensor>& weight, bool density) {
    if (bins.dim() != 1) {
        TP_THROW(ValueError,
                 "torch.histogramdd: bins tensor should have one dimension, "
                 "but got ", bins.dim(), " dimensions in the bins tensor for "
                 "dimension 0");
    }
    if (bins.dtype() != self.dtype()) {
        TP_THROW(ValueError,
                 "torch.histogramdd: input tensor and bins tensors should "
                 "have the same dtype, but got input with dtype ",
                 toString(self.dtype()), " and bins with dtype ",
                 toString(bins.dtype()));
    }
    histogram_check_weight(self, weight);
    // Upstream copies the bins into an empty self.options() tensor.
    Tensor bin_edges = bins.clone();
    Tensor hist = histogram_bin_counts(self, bin_edges, weight, density);
    return {std::move(hist), std::move(bin_edges)};
}

std::tuple<Tensor, Tensor> histogram_bin_ct_kernel(
        const Tensor& self, int64_t bins,
        const std::optional<std::vector<double>>& range,
        const std::optional<Tensor>& weight, bool density) {
    if (bins < 1) {
        TP_THROW(ValueError, "torch.histogram(): bins must be > 0, but got ",
                 bins, " for dimension 0");
    }
    histogram_check_weight(self, weight);
    auto outer = histogram_outer_edges_1d(self, range);
    Tensor bin_edges = Tensor::linspace(Scalar(outer.first),
                                        Scalar(outer.second), bins + 1,
                                        self.dtype(), self.device());
    Tensor hist = histogram_bin_counts(self, bin_edges, weight, density);
    return {std::move(hist), std::move(bin_edges)};
}

// quantile/nanquantile/histogram bodies are device-generic composites over
// dispatched primitives (sort/gather/lerp/searchsorted/index_add/aminmax/
// linspace), so the same functions register for CUDA (Einsum.cpp precedent).
TENSORPLAY_LIBRARY_IMPL(CUDA, MiscKernelsQuantileHistogramComposites) {
    m.impl("quantile", quantile_kernel);
    m.impl("nanquantile", nanquantile_kernel);
    m.impl("histogram.bins_tensor", histogram_bins_tensor_kernel);
    m.impl("histogram.bin_ct", histogram_bin_ct_kernel);
}


} // namespace cpu
} // namespace tensorplay

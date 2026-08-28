// Shape & view alignment kernels (torch native-parity batch).
//
// Schemas live in config/native_functions.yaml; composites mirror the ATen
// implementations op-for-op:
//   aten/src/ATen/native/TensorShape.cpp   expand/repeat/tile/hstack/vstack/
//     dstack/row_stack/column_stack/tensor_split/flatten/unflatten/ravel/
//     moveaxis/swapaxes/swapdims/broadcast_to
//   aten/src/ATen/native/TensorTransformations.cpp  atleast_*
//   aten/src/ATen/native/Fill.cpp          fill()
// Every composite invokes its primitives through the generated Tensor members,
// so each inner call routes through the Dispatcher (device + autograd keys)
// exactly like an at:: call inside an ATen composite.  Only repeat() carries
// real device code (single-pass index-math gather, the same materialization
// upstream achieves via unfold + copy_); its CUDA twin lives in
// cuda/ShapeAlignKernels.cu.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "ShapeAlignKernels.h"
#include "tensorplay/ops/TensorRedispatchGenerated.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace tensorplay {

namespace shapeops {

using namespace tensorplay::parallel;

Tensor tpsa_atleast_1d(const Tensor& self);
Tensor tpsa_atleast_2d(const Tensor& self);
Tensor tpsa_atleast_3d(const Tensor& self);

namespace {

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    // c10::maybe_wrap_dim, including the zero-dim special case that keeps
    // flatten(0-dim tensor) legal.
    if (ndim == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(IndexError,
                     "Dimension out of range (expected to be in range of [-1, 0], but got ",
                     dim, ")");
        }
        return 0;
    }
    const int64_t min = -ndim;
    const int64_t max = ndim - 1;
    if (dim < min || dim > max) {
        TP_THROW(IndexError,
                 "Dimension out of range (expected to be in range of [", min,
                 ", ", max, "], but got ", dim, ")");
    }
    return dim < 0 ? dim + ndim : dim;
}

std::vector<Tensor> atleast_n_seq(const std::vector<Tensor>& tensors, int n) {
    std::vector<Tensor> result;
    result.reserve(tensors.size());
    for (const auto& t : tensors) {
        switch (n) {
            case 1: result.push_back(tpsa_atleast_1d(t)); break;
            case 2: result.push_back(tpsa_atleast_2d(t)); break;
            default: result.push_back(tpsa_atleast_3d(t)); break;
        }
    }
    return result;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// expand family -- ATen: broadcast_to == self.expand(size); expand_as expands
// to other.sizes().  -1 in `size` infers the existing dimension (torch
// ExpandUtils semantics); -1 in a leading, non-existing dimension errors.
// The stride computation used to live on the handwritten Tensor::expand
// member; it moved here when expand became a dispatcher op.
// ---------------------------------------------------------------------------

namespace {

// torch legacy type name used in expand()'s size-mismatch message
// ("expand(torch.FloatTensor{[2, 3]}, size=[3]): ...").
const char* torch_legacy_type_name(DType dt) {
    switch (dt) {
        case DType::Float32: return "torch.FloatTensor";
        case DType::Float64: return "torch.DoubleTensor";
        case DType::Float16: return "torch.HalfTensor";
        case DType::BFloat16: return "torch.BFloat16Tensor";
        case DType::Int64: return "torch.LongTensor";
        case DType::Int32: return "torch.IntTensor";
        case DType::Int16: return "torch.ShortTensor";
        case DType::Int8: return "torch.CharTensor";
        case DType::UInt8: return "torch.ByteTensor";
        case DType::UInt16: return "torch.UInt16Tensor";
        case DType::UInt32: return "torch.UInt32Tensor";
        case DType::UInt64: return "torch.UInt64Tensor";
        case DType::Bool: return "torch.BoolTensor";
        case DType::ComplexFloat: return "torch.ComplexFloatTensor";
        case DType::ComplexDouble: return "torch.ComplexDoubleTensor";
        case DType::ComplexHalf: return "torch.ComplexHalfTensor";
        default: return "torch.Tensor";
    }
}

std::string fmt_dim_list(const std::vector<int64_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    s += "]";
    return s;
}

Tensor expand_impl(const Tensor& self, const std::vector<int64_t>& size) {
    // TensorShape.cpp expand + ExpandUtils.cpp inferExpandGeometryImpl.
    if (!self.defined()) TP_THROW(RuntimeError, "Tensor not defined");
    const int64_t ndim = self.dim();
    const int64_t new_ndim = static_cast<int64_t>(size.size());

    if (new_ndim < ndim) {
        TP_THROW(RuntimeError, "expand(", torch_legacy_type_name(self.dtype()),
                 "{", fmt_dim_list(static_cast<std::vector<int64_t>>(self.shape())),
                 "}, size=", fmt_dim_list(size),
                 "): the number of sizes provided (", new_ndim,
                 ") must be greater or equal to the number of dimensions in the tensor (",
                 ndim, ")");
    }

    std::vector<int64_t> new_sizes(size);
    std::vector<int64_t> new_strides(new_ndim, 0);

    // 0-d inputs expand to any shape with all-zero strides (upstream returns
    // InferExpandGeometryResult(sizes, ndim) with zeroed strides).
    if (ndim == 0) {
        return self.as_strided(new_sizes, new_strides);
    }

    for (int64_t i = new_ndim - 1; i >= 0; --i) {
        const int64_t offset = new_ndim - 1 - i;
        const int64_t dim = ndim - 1 - offset;
        int64_t sz = (dim >= 0) ? self.size(dim) : 1;
        int64_t stride = (dim >= 0)
                             ? self.stride(dim)
                             : new_sizes[i + 1] * new_strides[i + 1];
        int64_t target = new_sizes[i];
        if (target == -1) {
            if (dim < 0) {
                TP_THROW(RuntimeError, "The expanded size of the tensor (", target,
                         ") isn't allowed in a leading, non-existing dimension ", i);
            }
            target = sz;
        }
        if (sz != target) {
            if (sz != 1) {
                TP_THROW(RuntimeError, "The expanded size of the tensor (", target,
                         ") must match the existing size (", sz,
                         ") at non-singleton dimension ", i,
                         ".  Target sizes: ", fmt_dim_list(size),
                         ".  Tensor sizes: ",
                         fmt_dim_list(static_cast<std::vector<int64_t>>(self.shape())));
            }
            sz = target;
            stride = 0;
        }
        new_sizes[i] = sz;
        new_strides[i] = stride;
    }
    return self.as_strided(new_sizes, new_strides);
}

} // anonymous namespace

Tensor tpsa_expand(const Tensor& self, const std::vector<int64_t>& size, bool /*implicit*/) {
    return expand_impl(self, size);
}

Tensor tpsa_expand_as(const Tensor& self, const Tensor& other) {
    return expand_impl(self, static_cast<std::vector<int64_t>>(other.shape()));
}

Tensor tpsa_broadcast_to(const Tensor& self, const std::vector<int64_t>& size) {
    return expand_impl(self, size);
}

// ---------------------------------------------------------------------------
// repeat / tile -- ATen TensorShape.cpp repeat(): repeats.len >= self.dim(),
// leading unit dims pad the source, zero repeat yields an empty target.
// The copy itself is a single-pass gather (see repeat_cpu below / .cu twin);
// tile() prepends ones to short reps and otherwise defers to repeat.
// ---------------------------------------------------------------------------

namespace {

void check_repeat_args(const Tensor& self, const std::vector<int64_t>& repeats,
                       std::vector<int64_t>& padded, std::vector<int64_t>& padded_strides,
                       std::vector<int64_t>& target, bool& zero) {
    const int64_t nd = self.dim();
    if (static_cast<int64_t>(repeats.size()) < nd) {
        TP_THROW(RuntimeError,
                 "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");
    }
    const int64_t out_nd = static_cast<int64_t>(repeats.size());
    padded.assign(out_nd, 1);
    padded_strides.assign(out_nd, 0);
    for (int64_t i = 0; i < nd; ++i) {
        padded[out_nd - nd + i] = self.size(i);
        padded_strides[out_nd - nd + i] = self.stride(i);
    }
    zero = false;
    target.resize(out_nd);
    for (int64_t i = 0; i < out_nd; ++i) {
        zero = zero || repeats[i] == 0;
        target[i] = padded[i] * repeats[i];
    }
    // Negative repeats surface through the output allocation exactly like
    // upstream (at::empty -> check_size_nonnegative, EmptyTensor.h).
    for (const int64_t x : target) {
        if (x < 0) {
            std::string sizes = "[";
            for (size_t i = 0; i < target.size(); ++i) {
                if (i) sizes += ", ";
                sizes += std::to_string(target[i]);
            }
            sizes += "]";
            TP_THROW(RuntimeError, "Trying to create tensor with negative dimension ",
                     x, ": ", sizes);
        }
    }
}

} // anonymous namespace

Tensor tpsa_repeat_cpu(const Tensor& self, const std::vector<int64_t>& repeats);

Tensor tpsa_tile(const Tensor& self, const std::vector<int64_t>& dims) {
    const int64_t diff = self.dim() - static_cast<int64_t>(dims.size());
    if (diff > 0) {
        std::vector<int64_t> new_reps(static_cast<size_t>(diff), 1);
        new_reps.insert(new_reps.end(), dims.begin(), dims.end());
        return self.repeat(new_reps);
    }
    return self.repeat(dims);
}

// ---------------------------------------------------------------------------
// stacking family -- ATen TensorShape.cpp hstack/vstack/dstack/row_stack/
// column_stack: promote inputs with atleast_Nd, then cat along the axis.
// ---------------------------------------------------------------------------

Tensor tpsa_hstack(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) TP_THROW(RuntimeError, "hstack expects a non-empty TensorList");
    auto rep = atleast_n_seq(tensors, 1);
    if (rep[0].dim() == 1) return Tensor::cat(rep, 0);
    return Tensor::cat(rep, 1);
}

Tensor& tpsa_hstack_out(const std::vector<Tensor>& tensors, Tensor& out) {
    out.copy_(tpsa_hstack(tensors));
    return out;
}

Tensor tpsa_vstack(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) TP_THROW(RuntimeError, "vstack expects a non-empty TensorList");
    auto rep = atleast_n_seq(tensors, 2);
    return Tensor::cat(rep, 0);
}

Tensor& tpsa_vstack_out(const std::vector<Tensor>& tensors, Tensor& out) {
    out.copy_(tpsa_vstack(tensors));
    return out;
}

Tensor tpsa_dstack(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) TP_THROW(RuntimeError, "dstack expects a non-empty TensorList");
    auto rep = atleast_n_seq(tensors, 3);
    return Tensor::cat(rep, 2);
}

Tensor& tpsa_dstack_out(const std::vector<Tensor>& tensors, Tensor& out) {
    out.copy_(tpsa_dstack(tensors));
    return out;
}

Tensor tpsa_row_stack(const std::vector<Tensor>& tensors) {
    // torch.row_stack is a documented alias of torch.vstack.
    return tpsa_vstack(tensors);
}

Tensor& tpsa_row_stack_out(const std::vector<Tensor>& tensors, Tensor& out) {
    return tpsa_vstack_out(tensors, out);
}

Tensor tpsa_column_stack(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) TP_THROW(RuntimeError, "column_stack expects a non-empty TensorList");
    // reshape_input_for_column_stack: 0-D/1-D inputs become (numel, 1).
    std::vector<Tensor> reshaped;
    reshaped.reserve(tensors.size());
    for (const auto& t : tensors) {
        reshaped.push_back(t.dim() <= 1 ? t.reshape({t.numel(), 1}) : t);
    }
    return tpsa_hstack(reshaped);
}

Tensor& tpsa_column_stack_out(const std::vector<Tensor>& tensors, Tensor& out) {
    if (tensors.empty()) TP_THROW(RuntimeError, "column_stack expects a non-empty TensorList");
    std::vector<Tensor> reshaped;
    reshaped.reserve(tensors.size());
    for (const auto& t : tensors) {
        reshaped.push_back(t.dim() <= 1 ? t.reshape({t.numel(), 1}) : t);
    }
    return tpsa_hstack_out(reshaped, out);
}

// ---------------------------------------------------------------------------
// tensor_split & friends -- ATen TensorShape.cpp tensor_split_sections /
// _tensor_split_indices; hsplit/vsplit/dsplit are fixed-dim aliases.
// ---------------------------------------------------------------------------

std::vector<Tensor> tpsa_tensor_split_sections(const Tensor& self, int64_t sections, int64_t dim) {
    if (self.dim() <= 0) {
        TP_THROW(RuntimeError,
                 "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ",
                 self.dim(), " dims");
    }
    const int64_t d = wrap_dim(dim, self.dim());
    if (sections <= 0) {
        TP_THROW(RuntimeError, "number of sections must be larger than 0, got ", sections);
    }
    const int64_t sz = self.size(d);
    const int64_t min_split = sz / sections;
    const int64_t one_extra = sz % sections;
    std::vector<Tensor> splits(static_cast<size_t>(sections));
    int64_t start = 0;
    for (int64_t i = 0; i < sections; ++i) {
        const int64_t len = min_split + (i < one_extra ? 1 : 0);
        splits[static_cast<size_t>(i)] = self.slice(d, start, start + len, 1);
        start += len;
    }
    return splits;
}

std::vector<Tensor> tensor_split_indices_impl(const Tensor& self,
                                              const std::vector<int64_t>& indices,
                                              int64_t d) {
    const int64_t num = static_cast<int64_t>(indices.size());
    std::vector<Tensor> splits(static_cast<size_t>(num) + 1);
    int64_t start = 0;
    for (int64_t i = 0; i < num; ++i) {
        splits[static_cast<size_t>(i)] = self.slice(d, start, indices[static_cast<size_t>(i)], 1);
        start = indices[static_cast<size_t>(i)];
    }
    splits[static_cast<size_t>(num)] = self.slice(d, start, self.size(d), 1);
    return splits;
}

std::vector<Tensor> tpsa_tensor_split_indices(const Tensor& self,
                                              const std::vector<int64_t>& indices,
                                              int64_t dim) {
    if (self.dim() <= 0) {
        TP_THROW(RuntimeError,
                 "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ",
                 self.dim(), " dims");
    }
    return tensor_split_indices_impl(self, indices, wrap_dim(dim, self.dim()));
}

std::vector<Tensor> tpsa_tensor_split_tensor(const Tensor& self,
                                             const Tensor& tensor_indices_or_sections,
                                             int64_t dim) {
    if (self.dim() <= 0) {
        TP_THROW(RuntimeError,
                 "tensor_split expected at least a 1-dimensional tensor, but got a tensor with ",
                 self.dim(), " dims");
    }
    if (!(tensor_indices_or_sections.device() == Device(DeviceType::CPU))) {
        TP_THROW(RuntimeError,
                 "tensor_split expected tensor_indices_or_sections to be on cpu");
    }
    if (tensor_indices_or_sections.dtype() != DType::Int64) {
        TP_THROW(RuntimeError,
                 "tensor_split expected tensor_indices_or_sections to have dtype of long");
    }
    const auto tc = tensor_indices_or_sections.contiguous();
    const int64_t* p = tc.data_ptr<int64_t>();
    std::vector<int64_t> indices(p, p + tc.numel());
    return tensor_split_indices_impl(self, indices, wrap_dim(dim, self.dim()));
}

// hsplit/vsplit/dsplit -- ATen TensorShape.cpp: dimension floor checks,
// hsplit falls back to dim 0 for 1-D inputs, and the sections variants
// demand divisibility along the split dimension.

std::vector<Tensor> tpsa_hsplit_int(const Tensor& self, int64_t sections) {
    if (self.dim() < 1) {
        TP_THROW(RuntimeError,
                 "torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with ",
                 self.dim(), " dimensions!");
    }
    const int64_t d = (self.dim() == 1) ? 0 : 1;
    if (sections == 0 || self.size(d) % sections != 0) {
        TP_THROW(RuntimeError,
                 "torch.hsplit attempted to split along dimension ", d,
                 ", but the size of the dimension ", self.size(d),
                 " is not divisible by the split_size ", sections, "!");
    }
    return tpsa_tensor_split_sections(self, sections, d);
}
std::vector<Tensor> tpsa_hsplit_array(const Tensor& self, const std::vector<int64_t>& indices) {
    if (self.dim() < 1) {
        TP_THROW(RuntimeError,
                 "torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with ",
                 self.dim(), " dimensions!");
    }
    return tensor_split_indices_impl(self, indices, (self.dim() == 1) ? 0 : 1);
}
std::vector<Tensor> tpsa_vsplit_int(const Tensor& self, int64_t sections) {
    if (self.dim() < 2) {
        TP_THROW(RuntimeError,
                 "torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with ",
                 self.dim(), " dimensions!");
    }
    if (sections == 0 || self.size(0) % sections != 0) {
        TP_THROW(RuntimeError,
                 "torch.vsplit attempted to split along dimension 0",
                 ", but the size of the dimension ", self.size(0),
                 " is not divisible by the split_size ", sections, "!");
    }
    return tpsa_tensor_split_sections(self, sections, 0);
}
std::vector<Tensor> tpsa_vsplit_array(const Tensor& self, const std::vector<int64_t>& indices) {
    if (self.dim() < 2) {
        TP_THROW(RuntimeError,
                 "torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with ",
                 self.dim(), " dimensions!");
    }
    return tensor_split_indices_impl(self, indices, 0);
}
std::vector<Tensor> tpsa_dsplit_int(const Tensor& self, int64_t sections) {
    if (self.dim() < 3) {
        TP_THROW(RuntimeError,
                 "torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with ",
                 self.dim(), " dimensions!");
    }
    if (sections == 0 || self.size(2) % sections != 0) {
        TP_THROW(RuntimeError,
                 "torch.dsplit attempted to split along dimension 2",
                 ", but the size of the dimension ", self.size(2),
                 " is not divisible by the split_size ", sections, "!");
    }
    return tpsa_tensor_split_sections(self, sections, 2);
}
std::vector<Tensor> tpsa_dsplit_array(const Tensor& self, const std::vector<int64_t>& indices) {
    if (self.dim() < 3) {
        TP_THROW(RuntimeError,
                 "torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with ",
                 self.dim(), " dimensions!");
    }
    return tensor_split_indices_impl(self, indices, 2);
}

// ---------------------------------------------------------------------------
// atleast_Nd -- ATen TensorTransformations.cpp.
// ---------------------------------------------------------------------------

Tensor tpsa_atleast_1d(const Tensor& self) {
    return self.dim() == 0 ? self.reshape({1}) : self;
}
std::vector<Tensor> tpsa_atleast_1d_seq(const std::vector<Tensor>& tensors) {
    return atleast_n_seq(tensors, 1);
}
Tensor tpsa_atleast_2d(const Tensor& self) {
    switch (self.dim()) {
        case 0: return self.reshape({1, 1});
        case 1: return self.unsqueeze(0);
        default: return self;
    }
}
std::vector<Tensor> tpsa_atleast_2d_seq(const std::vector<Tensor>& tensors) {
    return atleast_n_seq(tensors, 2);
}
Tensor tpsa_atleast_3d(const Tensor& self) {
    switch (self.dim()) {
        case 0: return self.reshape({1, 1, 1});
        case 1: return self.unsqueeze(0).unsqueeze(-1);
        case 2: return self.unsqueeze(-1);
        default: return self;
    }
}
std::vector<Tensor> tpsa_atleast_3d_seq(const std::vector<Tensor>& tensors) {
    return atleast_n_seq(tensors, 3);
}

// ---------------------------------------------------------------------------
// flatten / unflatten / ravel -- ATen TensorShape.cpp.
// ---------------------------------------------------------------------------

Tensor tpsa_flatten(const Tensor& self, int64_t start_dim, int64_t end_dim) {
    const int64_t nd = self.dim();
    start_dim = wrap_dim(start_dim, nd);
    end_dim = wrap_dim(end_dim, nd);
    if (start_dim > end_dim) {
        TP_THROW(RuntimeError, "flatten() has invalid args: start_dim cannot come after end_dim");
    }
    if (nd == 0) return self.reshape({1});
    if (start_dim == end_dim) return self;
    int64_t slice_numel = 1;
    for (int64_t i = start_dim; i <= end_dim; ++i) slice_numel *= self.size(i);
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(nd - (end_dim - start_dim)));
    for (int64_t i = 0; i < start_dim; ++i) shape.push_back(self.size(i));
    shape.push_back(slice_numel);
    for (int64_t i = end_dim + 1; i < nd; ++i) shape.push_back(self.size(i));
    return self.reshape(shape);
}

Tensor tpsa_unflatten(const Tensor& self, int64_t dim, const std::vector<int64_t>& sizes) {
    // TensorShape.cpp unflatten_impl + handle_unflatten_exception: infer_size
    // failures containing "is invalid for input of size" are rephrased; any
    // other exception is re-raised prefixed with "unflatten got an unexpected
    // error:".
    const int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (sizes.empty()) TP_THROW(RuntimeError, "unflatten: sizes must be non-empty");

    auto fmt = [&sizes]() {
        std::string s = "[";
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (i) s += ", ";
            s += std::to_string(sizes[i]);
        }
        s += "]";
        return s;
    };
    auto friendly_mismatch = [&]() {
        TP_THROW(RuntimeError, "unflatten: Provided sizes ", fmt(),
                 " don't multiply up to the size of dim ", dim, " (",
                 self.size(dim), ") in the input tensor");
    };
    auto unexpected = [&](const std::string& what) {
        TP_THROW(RuntimeError, "unflatten got an unexpected error:\n", what);
    };

    if (nd == 0) {
        // torch reaches self.size(dim) on the 0-d tensor inside infer_size.
        unexpected("Dimension specified as " + std::to_string(dim) +
                   " but tensor has no dimensions");
    }
    const int64_t target = self.size(dim);

    // at::infer_size_impl
    std::vector<int64_t> inferred(sizes);
    int64_t newsize = 1;
    int64_t infer_dim = -1;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == -1) {
            if (infer_dim != -1) unexpected("only one dimension can be inferred");
            infer_dim = static_cast<int64_t>(i);
        } else {
            if (sizes[i] <= -2) {
                unexpected("invalid shape dimension " + std::to_string(sizes[i]) +
                           " at index " + std::to_string(i) + " of shape " + fmt());
            }
            newsize *= sizes[i];
        }
    }
    if (infer_dim != -1) {
        if (!((newsize > 0 && target % newsize == 0) || target == newsize)) {
            friendly_mismatch();
        }
        if (newsize == 0) {
            unexpected("cannot reshape tensor of 0 elements into shape " + fmt() +
                       " because the unspecified dimension size -1 can be any "
                       "value and is ambiguous");
        }
        inferred[static_cast<size_t>(infer_dim)] = target / newsize;
    } else if (target != newsize) {
        friendly_mismatch();
    }

    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(nd - 1) + inferred.size());
    for (int64_t i = 0; i < dim; ++i) shape.push_back(self.size(i));
    shape.insert(shape.end(), inferred.begin(), inferred.end());
    for (int64_t i = dim + 1; i < nd; ++i) shape.push_back(self.size(i));
    return self.view(shape);
}

Tensor tpsa_ravel(const Tensor& self) {
    // ATen ravel(): contiguous().view(-1)
    return self.contiguous().view({-1});
}

// ---------------------------------------------------------------------------
// moveaxis / swapaxes / swapdims -- moveaxis is documented as movedim alias;
// swapaxes/swapdims are transpose aliases (numpy names).
// ---------------------------------------------------------------------------

Tensor tpsa_moveaxis_intlist(const Tensor& self, const std::vector<int64_t>& source,
                             const std::vector<int64_t>& destination) {
    return self.movedim(source, destination);
}

Tensor tpsa_moveaxis_int(const Tensor& self, int64_t source, int64_t destination) {
    return self.movedim(std::vector<int64_t>{source}, std::vector<int64_t>{destination});
}

Tensor tpsa_swapaxes(const Tensor& self, int64_t axis0, int64_t axis1) {
    return self.transpose(axis0, axis1);
}

Tensor tpsa_swapdims(const Tensor& self, int64_t dim0, int64_t dim1) {
    return self.transpose(dim0, dim1);
}

// ---------------------------------------------------------------------------
// argwhere / equal / allclose -- argwhere is nonzero's (nnz, ndim) layout;
// equal/allclose compose eq/all/isclose exactly like ATen's native impls.
// ---------------------------------------------------------------------------

Tensor tpsa_argwhere(const Tensor& self) {
    return detail::redispatch_nonzero_function(self);
}

bool tpsa_equal(const Tensor& self, const Tensor& other) {
    if (!(self.shape() == other.shape())) return false;
    if (self.dtype() != other.dtype()) return false;
    if (!(self.device() == other.device())) return false;
    if (self.numel() == 0) return true;
    const Tensor eq = self.eq(other);
    if (eq.numel() == 0) return true;
    const Tensor all = eq.all();
    return all.item().to<bool>();
}

bool tpsa_allclose(const Tensor& self, const Tensor& other, double rtol, double atol,
                   bool equal_nan) {
    const Tensor close = detail::redispatch_isclose_function(self, other, rtol, atol, equal_nan);
    if (close.numel() == 0) return true;
    return close.all().item().to<bool>();
}

// ---------------------------------------------------------------------------
// fill -- ATen Fill.cpp: empty_like(self).fill_(value); full_like preserves
// dtype/device identically.
// ---------------------------------------------------------------------------

Tensor tpsa_fill_scalar(const Tensor& self, Scalar value) {
    return Tensor::full_like(self, value, self.dtype(), self.device());
}

Tensor tpsa_fill_tensor(const Tensor& self, const Tensor& value) {
    if (value.numel() != 1) {
        TP_THROW(RuntimeError, "fill only supports a value tensor with one element");
    }
    return Tensor::full_like(self, value.item(), self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// repeat CPU kernel: single-pass gather.  For flat output index f decomposed
// along row-major target strides, source offset = sum(stride_i * (c_i % ps_i))
// where ps_i is the unit-padded source size -- prepended/expanded dims fold
// naturally via the modulo.
// ---------------------------------------------------------------------------

Tensor tpsa_repeat_cpu(const Tensor& self, const std::vector<int64_t>& repeats) {
    std::vector<int64_t> padded, padded_strides, target;
    bool zero = false;
    check_repeat_args(self, repeats, padded, padded_strides, target, zero);

    Tensor out = Tensor::empty(target, self.dtype(), self.device());
    const int64_t total = out.numel();
    if (zero || total == 0) return out;

    const int64_t out_nd = static_cast<int64_t>(target.size());
    std::vector<int64_t> tstrides(out_nd, 1);
    for (int64_t i = out_nd - 2; i >= 0; --i) tstrides[i] = tstrides[i + 1] * target[i + 1];

    parallel_for(0, total, GRAIN_SIZE, [&](int64_t lo, int64_t hi) {
#define TP_REPEAT_CASE(ctype, name)                                                     \
        case DType::name: {                                                             \
            const ctype* src = self.data_ptr<ctype>();                                  \
            ctype* dst = out.data_ptr<ctype>();                                         \
            for (int64_t f = lo; f < hi; ++f) {                                         \
                int64_t rem = f, off = 0;                                               \
                for (int64_t i = 0; i < out_nd; ++i) {                                  \
                    const int64_t c = rem / tstrides[i];                                \
                    rem %= tstrides[i];                                                 \
                    off += (c % padded[i]) * padded_strides[i];                         \
                }                                                                       \
                dst[f] = src[off];                                                      \
            }                                                                           \
            break;                                                                      \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_REPEAT_CASE)
            default: TP_THROW(TypeError, "repeat: unsupported dtype");
        }
#undef TP_REPEAT_CASE
    });
    return out;
}

} // namespace shapeops

TENSORPLAY_LIBRARY_IMPL(CPU, ShapeAlign) {
    // repeat is the only op in this batch with real per-device code (the
    // single-pass gather below); its CUDA twin overrides it in
    // cuda/ShapeAlignKernels.cu -- upstream's MPS: repeat_mps pattern.
    m.impl("repeat", shapeops::tpsa_repeat_cpu);
    // Everything else in this batch maps to CompositeExplicitAutograd or the
    // default CompositeImplicitAutograd in native_functions.yaml and is
    // registered once under the backend-neutral Composite key from
    // src/RegisterComposites.cpp; the dispatcher's composite fallthrough
    // serves CPU tensors from there.
}

} // namespace tensorplay

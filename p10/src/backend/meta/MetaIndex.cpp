// Shape-only kernels for indexed reads and writes on the meta device.
//
// An indexed operation selects or overwrites whole slices of a tensor under an
// index tensor.  The out-of-place answers follow their operand's geometry:
// gather answers in the index's shape, index_select keeps the operand's shape
// with the indexed axis resized to the index length, and the scatter family
// keeps the destination's shape.  The index values themselves are element data
// a meta tensor does not own, so only their geometry and type are checked and
// an out-of-range entry is left for the computing backends to report.

#include "MetaCommon.h"

#include "AdvancedIndex.h"
#include "DType.h"
#include "DTypeNames.h"
#include "Scalar.h"
#include "TypeProperties.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

void require_scalar_elements(const Tensor& tensor, const char* op) {
    if (!is_scalar_dtype(tensor.dtype())) {
        TP_THROW(TypeError, op, ": unsupported dtype");
    }
}

void require_meta_devices(const Tensor& a, const Tensor& b, const Tensor& c,
                          const char* message) {
    if (!a.device().is_meta() || !b.device().is_meta() ||
        !c.device().is_meta()) {
        TP_THROW(DeviceMismatchError, message);
    }
}

// ---------------------------------------------------------------------------
// Gather: one destination element per index element, read from the source
// position the index names along `dim`; the answer takes the index geometry.
// ---------------------------------------------------------------------------

Tensor gather_kernel(const Tensor& self, int64_t dim, const Tensor& index) {
    const int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError,
                 "Index must have same number of dimensions as input tensor");
    }
    for (int64_t i = 0; i < nd; ++i) {
        if (i != dim && index.size(i) > self.size(i)) {
            TP_THROW(IndexError, "Size does not match at dimension ", i,
                     " (input: ", self.size(i), ", index: ", index.size(i),
                     ")");
        }
    }
    require_scalar_elements(self, "gather");
    return empty_meta(shape_of(index), self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// Scatter family: every index entry names a destination slice along `dim`,
// and the answer keeps the destination geometry.  A source entry broadcasts
// over the index shape, so a single-element source fills every write.
// ---------------------------------------------------------------------------

void check_scatter_geometry(const Tensor& self, int64_t dim,
                            const Tensor& index, const Tensor& src,
                            const char* op) {
    const int64_t nd = self.dim();
    wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError,
                 "Index must have same number of dimensions as output tensor");
    }
    if (src.numel() == 1) return;
    const std::vector<int64_t> idx_shape = shape_of(index);
    if (broadcast_shapes(shape_of(src), idx_shape) != idx_shape) {
        TP_THROW(RuntimeError, op,
                 ": src shape must broadcast to the index shape");
    }
}

void check_scatter_value_geometry(const Tensor& self, int64_t dim,
                                  const Tensor& index) {
    const int64_t nd = self.dim();
    wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError,
                 "Index must have same number of dimensions as output tensor");
    }
}

Tensor scatter_src_kernel(const Tensor& self, int64_t dim,
                          const Tensor& index, const Tensor& src) {
    check_scatter_geometry(self, dim, index, src, "scatter");
    require_scalar_elements(self, "scatter");
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& scatter_src_inplace_kernel(Tensor& self, int64_t dim,
                                   const Tensor& index, const Tensor& src) {
    check_scatter_geometry(self, dim, index, src, "scatter_");
    require_scalar_elements(self, "scatter_");
    return self;
}

Tensor scatter_value_kernel(const Tensor& self, int64_t dim,
                            const Tensor& index, const Scalar& /*value*/) {
    check_scatter_value_geometry(self, dim, index);
    require_scalar_elements(self, "scatter");
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& scatter_value_inplace_kernel(Tensor& self, int64_t dim,
                                     const Tensor& index,
                                     const Scalar& /*value*/) {
    check_scatter_value_geometry(self, dim, index);
    require_scalar_elements(self, "scatter_");
    return self;
}

Tensor scatter_add_kernel(const Tensor& self, int64_t dim,
                          const Tensor& index, const Tensor& src) {
    check_scatter_geometry(self, dim, index, src, "scatter");
    require_scalar_elements(self, "scatter");
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& scatter_add_inplace_kernel(Tensor& self, int64_t dim,
                                   const Tensor& index, const Tensor& src) {
    check_scatter_geometry(self, dim, index, src, "scatter_");
    require_scalar_elements(self, "scatter_");
    return self;
}

// ---------------------------------------------------------------------------
// Scatter reduce: the source broadcasts to the index shape, and `dim` of the
// index may be narrower than the destination axis it writes into.
// ---------------------------------------------------------------------------

void check_reduce_name(const std::string& reduce) {
    if (reduce != "sum" && reduce != "prod" && reduce != "mean" &&
        reduce != "amin" && reduce != "amax") {
        TP_THROW(ValueError,
                 "reduce argument must be one of 'sum', 'prod', 'mean', "
                 "'amin', 'amax' but got: " + reduce);
    }
}

Tensor scatter_reduce_kernel(const Tensor& self, int64_t dim,
                             const Tensor& index, const Tensor& src,
                             const std::string& reduce,
                             bool /*include_self*/) {
    check_reduce_name(reduce);
    const int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() != nd) {
        TP_THROW(IndexError,
                 "index must have the same number of dimensions as self");
    }
    if (index.numel() != 0 && index.dtype() != DType::Int32 &&
        index.dtype() != DType::Int64) {
        TP_THROW(TypeError,
                 "scatter_reduce(): Expected dtype int32/int64 for index");
    }
    if (src.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "scatter_reduce(): Expected self.dtype to be equal to src.dtype");
    }
    require_meta_devices(
        self, index, src,
        "scatter_reduce: self, index, and src must be on the same device");
    if (index.numel() == 0) {
        return empty_meta(shape_of(self), self.dtype(), self.device());
    }
    if (src.dim() == 0) {
        if (nd != 1 || index.size(0) != 1) {
            TP_THROW(RuntimeError, "src/source shape must match the index shape");
        }
    } else {
        if (src.dim() != nd) {
            TP_THROW(IndexError,
                     "src/source must have the same number of dimensions as index");
        }
        for (int64_t i = 0; i < nd; ++i) {
            if (i != dim && index.size(i) > self.size(i)) {
                TP_THROW(RuntimeError,
                         "index shape must not exceed self shape outside the "
                         "reduced dimension");
            }
            if (index.size(i) > src.size(i)) {
                TP_THROW(RuntimeError,
                         "index shape must not exceed source shape");
            }
        }
    }
    require_scalar_elements(self, "scatter_reduce");
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// Index reduce: a vector index, a source of the destination's rank, and the
// source's selected axis holding exactly one entry per index value.
// ---------------------------------------------------------------------------

void check_index_reduce_name(const std::string& reduce) {
    if (reduce != "prod" && reduce != "mean" && reduce != "amax" &&
        reduce != "amin") {
        TP_THROW(ValueError,
                 "index_reduce(): Expected reduce to be one of prod, mean, "
                 "amax or amin but got ", reduce);
    }
}

Tensor index_reduce_kernel(const Tensor& self, int64_t dim,
                           const Tensor& index, const Tensor& source,
                           const std::string& reduce,
                           bool /*include_self*/) {
    check_index_reduce_name(reduce);
    const int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (source.dim() != nd) {
        TP_THROW(IndexError,
                 "index_reduce(): Index is supposed to be a vector");
    }
    if (index.dim() > 1) {
        TP_THROW(IndexError,
                 "index_reduce(): Index is supposed to be a vector, but got "
                 "dim: ", index.dim());
    }
    if (index.dtype() != DType::Int32 && index.dtype() != DType::Int64) {
        TP_THROW(TypeError,
                 "index_reduce(): Expected dtype int32/int64 for index");
    }
    if (source.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "index_reduce(): Expected self.dtype to be equal to "
                 "source.dtype");
    }
    require_meta_devices(self, index, source,
                         "index_reduce: self, index, and source must be on "
                         "the same device");
    for (int64_t i = 0; i < nd; ++i) {
        if (i == dim) continue;
        if (source.size(i) != self.size(i)) {
            TP_THROW(IndexError,
                     "index_reduce(): Expected source and self to have the "
                     "same size at dimension ", i);
        }
    }
    if (source.size(dim) != index.numel()) {
        TP_THROW(IndexError, "index_reduce(): Number of indices (",
                 index.numel(), ") should be equal to source.size(dim): (",
                 source.size(dim), "),");
    }
    require_scalar_elements(self, "index_reduce");
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& index_reduce_inplace_kernel(Tensor& self, int64_t dim,
                                    const Tensor& index,
                                    const Tensor& source,
                                    const std::string& reduce,
                                    bool include_self) {
    index_reduce_kernel(self, dim, index, source, reduce, include_self);
    return self;
}

// ---------------------------------------------------------------------------
// Index select / add / copy / fill: one dimension is addressed by a vector
// index; the index values are checked only for being a vector, because their
// range is element data.
// ---------------------------------------------------------------------------

Tensor index_select_kernel(const Tensor& self, int64_t dim,
                           const Tensor& index) {
    const int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    if (index.dim() > 1) {
        TP_THROW(IndexError, "index_select(): index should be a vector");
    }
    if (!isComplexType(self.dtype()) && !is_scalar_dtype(self.dtype())) {
        TP_THROW(TypeError, "index_select: unsupported dtype");
    }
    std::vector<int64_t> out = shape_of(self);
    out[static_cast<size_t>(dim)] = index.dim() == 0 ? 1 : index.size(0);
    return empty_meta(out, self.dtype(), self.device());
}

Tensor index_add_kernel(const Tensor& self, int64_t dim, const Tensor& index,
                        const Tensor& source) {
    const int64_t nd = self.dim();
    dim = wrap_dim(dim, nd);
    const int64_t n_idx = index.numel();
    if (n_idx == 0) {
        return empty_meta(shape_of(self), self.dtype(), self.device());
    }
    if (source.dim() != 0 && source.dim() != nd) {
        TP_THROW(RuntimeError,
                 "index_add: source must have same number of dims as input");
    }
    if (source.dim() != 0 && source.size(dim) != n_idx) {
        TP_THROW(RuntimeError,
                 "index_add: source size along dim must equal index length");
    }
    require_scalar_elements(self, "index_add");
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& index_add_inplace_kernel(Tensor& self, int64_t dim,
                                 const Tensor& index, const Tensor& source) {
    index_add_kernel(self, dim, index, source);
    return self;
}

Tensor index_copy_kernel(const Tensor& self, int64_t dim, const Tensor& index,
                         const Tensor& /*source*/) {
    wrap_dim(dim, self.dim());
    if (index.numel() != 0) {
        require_scalar_elements(self, "index_copy");
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& index_copy_inplace_kernel(Tensor& self, int64_t dim,
                                  const Tensor& index,
                                  const Tensor& source) {
    index_copy_kernel(self, dim, index, source);
    return self;
}

void check_index_fill_geometry(const Tensor& self, int64_t dim,
                               const Tensor& index) {
    wrap_dim(dim, self.dim());
    if (index.numel() != 0) {
        require_scalar_elements(self, "index_fill");
    }
}

Tensor index_fill_scalar_kernel(const Tensor& self, int64_t dim,
                                const Tensor& index, const Scalar& /*value*/) {
    check_index_fill_geometry(self, dim, index);
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor index_fill_tensor_kernel(const Tensor& self, int64_t dim,
                                const Tensor& index, const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "index_fill only supports a 0-dimensional value tensor, but "
                 "got tensor with ", value.dim(), " dimension(s).");
    }
    return index_fill_scalar_kernel(self, dim, index, Scalar());
}

Tensor& index_fill_scalar_inplace_kernel(Tensor& self, int64_t dim,
                                         const Tensor& index,
                                         const Scalar& value) {
    index_fill_scalar_kernel(self, dim, index, value);
    return self;
}

Tensor& index_fill_tensor_inplace_kernel(Tensor& self, int64_t dim,
                                         const Tensor& index,
                                         const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "index_fill_ only supports a 0-dimensional value tensor, but "
                 "got tensor with ", value.dim(), " dimension(s).");
    }
    check_index_fill_geometry(self, dim, index);
    return self;
}

// ---------------------------------------------------------------------------
// Masked fill: the mask broadcasts against the input, and the answer keeps the
// broadcast geometry.
// ---------------------------------------------------------------------------

Tensor masked_fill_kernel(const Tensor& self, const Tensor& mask,
                          const Scalar& /*value*/) {
    if (mask.dtype() != DType::Bool) {
        TP_THROW(TypeError, "masked_fill only supports boolean masks");
    }
    require_scalar_elements(self, "masked_fill");
    return empty_meta(broadcast2(self, mask), self.dtype(), self.device());
}

Tensor masked_fill_tensor_kernel(const Tensor& self, const Tensor& mask,
                                 const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "masked_fill only supports a 0-dimensional value tensor, but "
                 "got tensor with ", value.dim(), " dimension(s).");
    }
    return masked_fill_kernel(self, mask, Scalar());
}

Tensor& masked_fill_inplace_kernel(Tensor& self, const Tensor& mask,
                                   const Scalar& value) {
    masked_fill_kernel(self, mask, value);
    return self;
}

Tensor& masked_fill_tensor_inplace_kernel(Tensor& self, const Tensor& mask,
                                          const Tensor& value) {
    if (value.dim() != 0) {
        TP_THROW(RuntimeError,
                 "masked_fill_ only supports a 0-dimensional value tensor, "
                 "but got tensor with ", value.dim(), " dimension(s).");
    }
    masked_fill_kernel(self, mask, Scalar());
    return self;
}

// ---------------------------------------------------------------------------
// Masked scatter: the answer keeps the receiver's geometry and type; the
// source only has to agree with the receiver's type, and the mask has to
// broadcast against it.
// ---------------------------------------------------------------------------

Tensor masked_scatter_kernel(const Tensor& self, const Tensor& mask,
                             const Tensor& source) {
    if (source.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "masked_scatter: self and source must have the same dtype");
    }
    require_meta_devices(self, mask, source,
                         "masked_scatter: self, mask, and source must be on the "
                         "same device");
    if (broadcast2(self, mask) != shape_of(self)) {
        TP_THROW(RuntimeError,
                 "masked_scatter: the mask must be broadcastable to the "
                 "receiver's shape");
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

Tensor& masked_scatter_inplace_kernel(Tensor& self, const Tensor& mask,
                                      const Tensor& source) {
    masked_scatter_kernel(self, mask, source);
    return self;
}

// ---------------------------------------------------------------------------
// Flat gather: take reads the flattened operand at each position the index
// names and answers in the index's geometry.
// ---------------------------------------------------------------------------

Tensor take_kernel(const Tensor& self, const Tensor& index) {
    require_scalar_elements(self, "take");
    return empty_meta(shape_of(index), self.dtype(), self.device());
}

// ---------------------------------------------------------------------------
// Lookup tables: searchsorted and bucketize report one position per query
// element, in the query's geometry; a scalar query answers a 0-dim position.
// A sorter's entries are element data, so only its layout is checked here.
// ---------------------------------------------------------------------------

bool dims_matched_before_last_dim(const Tensor& boundaries,
                                  const Tensor& input) {
    if (boundaries.dim() != input.dim()) return false;
    for (int64_t d = 0; d + 1 < boundaries.dim(); ++d) {
        if (boundaries.size(d) != input.size(d)) return false;
    }
    return true;
}

void check_bucketization(const Tensor& boundaries, const Tensor& input,
                         const std::optional<std::string>& side, bool right,
                         const Tensor& sorter, bool out_int32) {
    if (side.has_value()) {
        const std::string& name = *side;
        TP_CHECK(name == "left" || name == "right",
                 "searchsorted(): side can only be 'left' or 'right' but got ",
                 name);
        TP_CHECK(!right || name == "right",
                 "searchsorted(): side and right can't be set to opposites, got "
                 "side of ", name, " while right was True");
    }
    TP_CHECK(boundaries.device() == input.device(),
             "searchsorted(): boundaries and input value tensors should have "
             "the same device");
    if (sorter.defined()) {
        TP_CHECK(sorter.device() == boundaries.device(),
                 "searchsorted(): sorter and boundary tensors should have the "
                 "same device");
        TP_CHECK(sorter.shape() == boundaries.shape(),
                 "searchsorted(): boundary and sorter must have the same size");
        TP_CHECK(sorter.dtype() == DType::Int64,
                 "searchsorted(): sorter must be a tensor of long dtype but "
                 "got dtype ", scalarTypeName(sorter.dtype()));
    }
    TP_CHECK(input.dim() > 0 ||
                 (input.dim() == 0 && input.numel() == 1 &&
                  boundaries.dim() == 1),
             "searchsorted(): input value can be a scalar only when boundaries "
             "tensor dimension is 1, but we got boundaries dim(",
             boundaries.dim(), ") and input value dim(", input.dim(),
             ") numel(", input.numel(), ")");
    TP_CHECK(boundaries.dim() != 0,
             "searchsorted(): boundaries tensor should have positive "
             "dimension, but got 0 dimension");
    TP_CHECK(boundaries.dim() == 1 ||
                 dims_matched_before_last_dim(boundaries, input),
             "searchsorted(): boundaries tensor should be 1 dimension or the "
             "first N-1 dimensions of boundaries tensor and input value tensor "
             "must match");
    if (out_int32) {
        TP_CHECK(boundaries.size(-1) <
                     std::numeric_limits<int32_t>::max(),
                 "searchsorted(): the size of boundaries' last dimension should "
                 "be less than INT_MAX, but we got ", boundaries.size(-1));
    }
}

DType bucket_index_dtype(bool out_int32) {
    return out_int32 ? DType::Int32 : DType::Int64;
}

Tensor searchsorted_tensor_kernel(const Tensor& sorted_sequence,
                                  const Tensor& self, bool out_int32,
                                  bool right,
                                  const std::optional<std::string>& side,
                                  const std::optional<Tensor>& sorter) {
    check_bucketization(sorted_sequence, self, side, right,
                        sorter.value_or(Tensor()), out_int32);
    return empty_meta(shape_of(self), bucket_index_dtype(out_int32),
                      self.device());
}

Tensor searchsorted_scalar_kernel(const Tensor& sorted_sequence,
                                  const Scalar& /*self*/, bool out_int32,
                                  bool right,
                                  const std::optional<std::string>& side,
                                  const std::optional<Tensor>& sorter) {
    // A scalar query is the degenerate 0-dim query the shared checks describe.
    const Tensor query = empty_meta({}, sorted_sequence.dtype(),
                                    sorted_sequence.device());
    check_bucketization(sorted_sequence, query, side, right,
                        sorter.value_or(Tensor()), out_int32);
    return empty_meta({}, bucket_index_dtype(out_int32),
                      sorted_sequence.device());
}

Tensor bucketize_tensor_kernel(const Tensor& self, const Tensor& boundaries,
                               bool out_int32, bool right) {
    if (boundaries.dim() != 1) {
        TP_THROW(RuntimeError,
                 "bucketize(): boundaries tensor must be 1 dimension, but got "
                 "dim(", boundaries.dim(), ")");
    }
    check_bucketization(boundaries, self, std::nullopt, right, Tensor(),
                        out_int32);
    return empty_meta(shape_of(self), bucket_index_dtype(out_int32),
                      self.device());
}

Tensor bucketize_scalar_kernel(const Scalar& /*self*/,
                               const Tensor& boundaries, bool out_int32,
                               bool right) {
    if (boundaries.dim() != 1) {
        TP_THROW(RuntimeError,
                 "bucketize(): boundaries tensor must be 1 dimension, but got "
                 "dim(", boundaries.dim(), ")");
    }
    if (out_int32) {
        TP_CHECK(boundaries.size(-1) <
                     std::numeric_limits<int32_t>::max(),
                 "searchsorted(): the size of boundaries' last dimension should "
                 "be less than INT_MAX, but we got ", boundaries.size(-1));
    }
    return empty_meta({}, bucket_index_dtype(out_int32),
                      boundaries.device());
}

// ---------------------------------------------------------------------------
// Whole-tensor indexed writes: the destination keeps its geometry, and the
// source has to broadcast to the region the indices name.  A boolean mask's
// selected count is element data, so a masked write checks the layout only.
// ---------------------------------------------------------------------------

Tensor& index_put_impl_kernel(
    Tensor& self, const std::vector<std::optional<Tensor>>& indices,
    const Tensor& values, bool /*accumulate*/, bool /*unsafe*/) {
    const int64_t nd = self.dim();
    if (static_cast<int64_t>(indices.size()) > nd) {
        TP_THROW(IndexError, "too many indices for tensor of dimension ", nd);
    }
    bool has_mask = false;
    int64_t position = 0;
    for (const auto& maybe : indices) {
        if (!maybe.has_value() || !maybe->defined()) {
            ++position;
            continue;
        }
        const Tensor& index = *maybe;
        const DType dtype = index.dtype();
        if (dtype != DType::Int64 && dtype != DType::Int32 &&
            dtype != DType::Bool && dtype != DType::UInt8) {
            TP_THROW(IndexError,
                     "tensors used as indices must be long, int, byte or bool "
                     "tensors");
        }
        if (dtype == DType::Bool || dtype == DType::UInt8) {
            has_mask = true;
            for (int64_t d = 0; d < index.dim(); ++d) {
                const int64_t source_dim = position + d;
                TP_CHECK_INDEX(source_dim < nd &&
                                   index.size(d) == self.size(source_dim),
                               "The shape of the mask does not match the "
                               "indexed tensor");
            }
            position += index.dim();
        } else {
            ++position;
        }
    }
    if (values.dtype() != self.dtype()) {
        TP_THROW(RuntimeError,
                 "Index put requires the source and destination dtypes match");
    }
    if (has_mask) return self;
    const std::vector<int64_t> target =
        shape_of(indexing::native::AdvancedIndex(self, indices).source);
    if (values.dim() > static_cast<int64_t>(target.size())) {
        TP_THROW(RuntimeError,
                 "shape mismatch: value tensor cannot be broadcast to indexing "
                 "result");
    }
    for (int64_t d = 0; d < values.dim(); ++d) {
        const size_t target_dim = target.size() - values.dim() + d;
        if (values.size(d) != 1 && values.size(d) != target[target_dim]) {
            TP_THROW(RuntimeError,
                     "shape mismatch: value tensor cannot be broadcast to "
                     "indexing result");
        }
    }
    return self;
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaIndex) {
    m.impl("gather", gather_kernel);
    m.impl("take", take_kernel);
    m.impl("searchsorted.Tensor", searchsorted_tensor_kernel);
    m.impl("searchsorted.Scalar", searchsorted_scalar_kernel);
    m.impl("bucketize.Tensor", bucketize_tensor_kernel);
    m.impl("bucketize.Scalar", bucketize_scalar_kernel);
    m.impl("_index_put_impl_", index_put_impl_kernel);

    m.impl("scatter.src", scatter_src_kernel);
    m.impl("scatter.value", scatter_value_kernel);
    m.impl("scatter_.src", scatter_src_inplace_kernel);
    m.impl("scatter_.value", scatter_value_inplace_kernel);
    m.impl("scatter_add", scatter_add_kernel);
    m.impl("scatter_add_", scatter_add_inplace_kernel);
    m.impl("scatter_reduce", scatter_reduce_kernel);

    m.impl("index_reduce", index_reduce_kernel);
    m.impl("index_reduce_", index_reduce_inplace_kernel);
    m.impl("index_select", index_select_kernel);
    m.impl("index_add", index_add_kernel);
    m.impl("index_add_", index_add_inplace_kernel);
    m.impl("index_copy", index_copy_kernel);
    m.impl("index_copy_", index_copy_inplace_kernel);
    m.impl("index_fill.Tensor", index_fill_tensor_kernel);
    m.impl("index_fill.Scalar", index_fill_scalar_kernel);
    m.impl("index_fill_.Tensor", index_fill_tensor_inplace_kernel);
    m.impl("index_fill_.Scalar", index_fill_scalar_inplace_kernel);

    m.impl("masked_fill", masked_fill_kernel);
    m.impl("masked_fill_", masked_fill_inplace_kernel);
    m.impl("masked_fill.Tensor", masked_fill_tensor_kernel);
    m.impl("masked_fill_.Tensor", masked_fill_tensor_inplace_kernel);

    m.impl("masked_scatter", masked_scatter_kernel);
    m.impl("masked_scatter_", masked_scatter_inplace_kernel);
}

} // namespace meta
} // namespace tensorplay

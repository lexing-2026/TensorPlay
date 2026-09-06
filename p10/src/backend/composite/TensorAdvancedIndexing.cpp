// Composite kernels for scattering and flat index writes.
//
//   put            flat index writes.
//   scatter with a reduction: the legacy spelling names its reduction "add"
//                  or "multiply" and always folds the destination's own value
//                  in, which is the include_self form of scatter_reduce; the
//                  scalar-source overloads broadcast the value across the
//                  index before scattering.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

Tensor put_native(const Tensor& self, const Tensor& index,
                  const Tensor& source, bool accumulate) {
    if (index.dtype() != DType::Int64) {
        TP_THROW(RuntimeError,
                 "put(): Expected a long tensor for index, but got ",
                 toString(index.dtype()));
    }
    if (self.dtype() != source.dtype()) {
        TP_THROW(RuntimeError,
                 "put(): expected self and source to have the same dtype");
    }
    if (source.numel() == 0 && index.numel() != 0) {
        TP_THROW(IndexError,
                 "put(): Expected source to have at least one element");
    }
    if (self.numel() == 0 && index.numel() != 0) {
        TP_THROW(IndexError, "put(): Tried to put elements into an empty tensor");
    }
    Tensor out = ops::clone(self, std::nullopt);
    if (index.numel() == 0) return out;
    const Tensor idx = ops::reshape(index, {-1});
    Tensor src = ops::reshape(source, {-1});
    if (src.numel() < idx.numel()) {
        // Legacy put_ semantics (kept by the alignment baseline): a shorter
        // source is cycled to cover the full index list.
        const int64_t reps =
            (idx.numel() + src.numel() - 1) / src.numel();
        src = ops::slice(ops::tile(src, {reps}), 0, 0, idx.numel());
    }
    Tensor flat = ops::view(out, {-1});
    const Tensor updated = accumulate ? ops::index_add(flat, 0, idx, src)
                                      : ops::index_copy(flat, 0, idx, src);
    ops::copy_(flat, updated);
    return out;
}

// Fixed-size variant of nonzero: the result always has `size` rows, taken
// from the leading matches and padded (or truncated) with fill_value rows.
Tensor nonzero_static_native(const Tensor& self, std::optional<int64_t> size,
                             int64_t fill_value) {
    if (self.dim() == 0) {
        TP_THROW(RuntimeError,
                 "nonzero_static(): not supported with 0-d tensors");
    }
    Tensor nz = ops::nonzero(self);
    if (!size.has_value()) {
        return nz;
    }
    const int64_t n = *size;
    if (n < 0) {
        TP_THROW(RuntimeError,
                 "nonzero_static(): size must be non-negative, got ", n);
    }
    const int64_t k = nz.size(0);
    if (k >= n) {
        return ops::narrow(nz, 0, 0, n);
    }
    const Tensor tail = ops::full({n - k, self.dim()}, Scalar(fill_value),
                                  DType::Int64, self.device());
    return ops::cat({nz, tail}, 0);
}

namespace {

// The legacy scatter reduction names, translated to the reduction spelling
// the general scatter_reduce kernels use.
std::string legacy_reduce_name(const std::string& reduce) {
    if (reduce == "add") return "sum";
    if (reduce == "multiply") return "prod";
    TP_THROW(RuntimeError,
             "scatter(): reduce argument must be either add or multiply, but "
             "got ", reduce);
}

// The scalar overloads scatter one repeated value; materializing it at the
// index geometry hands the reduction kernel the source layout it expects.
Tensor scalar_source_like(const Tensor& self, const Tensor& index, Scalar value) {
    return ops::full(static_cast<std::vector<int64_t>>(index.shape()), value,
                     self.dtype(), self.device());
}

// out= keeps the caller's buffer: resize only when the result does not fit,
// then write into the storage the caller handed over.
Tensor& write_out(Tensor& out, const Tensor& value) {
    if (!out.defined()) {
        out = value;
        return out;
    }
    if (out.dtype() != value.dtype()) {
        TP_THROW(TypeError,
                 "scatter output must have the same dtype as the input");
    }
    if (out.device() != value.device()) {
        TP_THROW(DeviceMismatchError,
                 "scatter output must be on the same device as the input");
    }
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

}  // namespace

Tensor scatter_reduce_variant_native(const Tensor& self, int64_t dim,
                                     const Tensor& index, const Tensor& src,
                                     std::string reduce) {
    return ops::scatter_reduce(self, dim, index, src, legacy_reduce_name(reduce),
                               /*include_self=*/true);
}

Tensor& scatter_reduce_variant_inplace_native(Tensor& self, int64_t dim,
                                              const Tensor& index,
                                              const Tensor& src,
                                              std::string reduce) {
    self.copy_(scatter_reduce_variant_native(self, dim, index, src, reduce));
    return self;
}

Tensor& scatter_reduce_variant_out_native(const Tensor& self, int64_t dim,
                                          const Tensor& index, const Tensor& src,
                                          std::string reduce, Tensor& out) {
    return write_out(out,
                     scatter_reduce_variant_native(self, dim, index, src, reduce));
}

Tensor scatter_value_reduce_native(const Tensor& self, int64_t dim,
                                   const Tensor& index, Scalar value,
                                   std::string reduce) {
    return scatter_reduce_variant_native(
        self, dim, index, scalar_source_like(self, index, value), reduce);
}

Tensor& scatter_value_reduce_inplace_native(Tensor& self, int64_t dim,
                                            const Tensor& index, Scalar value,
                                            std::string reduce) {
    self.copy_(scatter_value_reduce_native(self, dim, index, value, reduce));
    return self;
}

Tensor& scatter_value_reduce_out_native(const Tensor& self, int64_t dim,
                                        const Tensor& index, Scalar value,
                                        std::string reduce, Tensor& out) {
    return write_out(out,
                     scatter_value_reduce_native(self, dim, index, value, reduce));
}

Tensor& scatter_src_out_native(const Tensor& self, int64_t dim,
                               const Tensor& index, const Tensor& src,
                               Tensor& out) {
    return write_out(out, ops::scatter(self, dim, index, src));
}

Tensor& scatter_value_out_native(const Tensor& self, int64_t dim,
                                 const Tensor& index, Scalar value,
                                 Tensor& out) {
    return write_out(out, ops::scatter(self, dim, index, value));
}

TENSORPLAY_LIBRARY_IMPL(Composite, TensorAdvancedIndexingComposite) {
    m.impl("put", put_native);
    m.impl("nonzero_static", nonzero_static_native);
    m.impl("scatter.reduce", scatter_reduce_variant_native);
    m.impl("scatter_.reduce", scatter_reduce_variant_inplace_native);
    m.impl("scatter.reduce_out", scatter_reduce_variant_out_native);
    m.impl("scatter.value_reduce", scatter_value_reduce_native);
    m.impl("scatter_.value_reduce", scatter_value_reduce_inplace_native);
    m.impl("scatter.value_reduce_out", scatter_value_reduce_out_native);
    m.impl("scatter.src_out", scatter_src_out_native);
    m.impl("scatter.value_out", scatter_value_out_native);
}

} // namespace composite
} // namespace tensorplay

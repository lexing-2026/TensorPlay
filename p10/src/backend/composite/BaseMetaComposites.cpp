// Composite kernels for the trivial base family: dtype casts that forward to
// `to`, sparse layout accessors that forward to the underlying component
// handles, and small metadata reads/writes (set_data, output_nr, zerotensor
// probe, shallow-copy compatibility, scalar readout, inference lifting).
//
// Everything here is device-neutral: kernels either re-enter the dispatcher
// for a different op or only touch TensorImpl metadata, so one Composite
// registration serves every backend.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "TensorImpl.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <optional>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

// -----------------------------------------------------------------------------
// _cast_*: dtype casts with the same dtype alias behavior as `to`
// -----------------------------------------------------------------------------

Tensor _cast_Byte_native(const Tensor& self, bool non_blocking) {
    reject_active_transform(self, "_cast_Byte");
    return ops::to(self, DType::UInt8, non_blocking);
}

Tensor _cast_Char_native(const Tensor& self, bool non_blocking) {
    reject_active_transform(self, "_cast_Char");
    return ops::to(self, DType::Int8, non_blocking);
}

Tensor _cast_Short_native(const Tensor& self, bool non_blocking) {
    reject_active_transform(self, "_cast_Short");
    return ops::to(self, DType::Int16, non_blocking);
}

Tensor _cast_Int_native(const Tensor& self, bool non_blocking) {
    reject_active_transform(self, "_cast_Int");
    return ops::to(self, DType::Int32, non_blocking);
}

Tensor _cast_Long_native(const Tensor& self, bool non_blocking) {
    reject_active_transform(self, "_cast_Long");
    return ops::to(self, DType::Int64, non_blocking);
}

Tensor _cast_Half_native(const Tensor& self, bool non_blocking) {
    reject_active_transform(self, "_cast_Half");
    return ops::to(self, DType::Float16, non_blocking);
}

Tensor _cast_Float_native(const Tensor& self, bool non_blocking) {
    reject_active_transform(self, "_cast_Float");
    return ops::to(self, DType::Float32, non_blocking);
}

Tensor _cast_Double_native(const Tensor& self, bool non_blocking) {
    reject_active_transform(self, "_cast_Double");
    return ops::to(self, DType::Float64, non_blocking);
}

// -----------------------------------------------------------------------------
// Dense/sparse materialization
// -----------------------------------------------------------------------------

// _to_dense is the identity for dense inputs; sparse inputs go through the
// registered dense-conversion kernel.  The optional dtype re-casts the dense
// result; masked_grad only shapes gradient flow and has no effect here.
Tensor _to_dense_native(const Tensor& self, std::optional<DType> dtype,
                        std::optional<bool> masked_grad) {
    reject_active_transform(self, "_to_dense");
    (void)masked_grad;
    const bool sparse = self.unsafeGetTensorImpl()->is_sparse();
    if (!sparse && !dtype.has_value()) {
        return self;
    }
    Tensor dense = sparse ? ops::to_dense(self) : self;
    if (dtype.has_value()) {
        dense = ops::to(dense, *dtype);
    }
    return dense;
}

// _to_cpu migrates a list of tensors to CPU, passing CPU entries through by
// reference and copying anything that lives elsewhere.
std::vector<Tensor> _to_cpu_native(const std::vector<Tensor>& tensors) {
    for (const Tensor& tensor : tensors) {
        reject_active_transform(tensor, "_to_cpu");
    }
    std::vector<Tensor> out;
    out.reserve(tensors.size());
    for (const Tensor& tensor : tensors) {
        if (tensor.defined() &&
            !tensor.unsafeGetTensorImpl()->device().is_cpu()) {
            out.push_back(ops::to(tensor, Device(DeviceType::CPU),
                                  tensor.dtype()));
        } else {
            out.push_back(tensor);
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// Sparse layout accessors
// -----------------------------------------------------------------------------

Tensor indices_native(const Tensor& self) {
    reject_active_transform(self, "indices");
    return ops::_indices(self);
}

Tensor values_native(const Tensor& self) {
    reject_active_transform(self, "values");
    return ops::_values(self);
}

Tensor _indices_copy_native(const Tensor& self) {
    reject_active_transform(self, "_indices_copy");
    return ops::clone(ops::_indices(self), kContiguous);
}

Tensor _values_copy_native(const Tensor& self) {
    reject_active_transform(self, "_values_copy");
    return ops::clone(ops::_values(self), kContiguous);
}

Tensor crow_indices_copy_native(const Tensor& self) {
    reject_active_transform(self, "row_indices_copy");
    reject_active_transform(self, "crow_indices_copy");
    return ops::clone(ops::crow_indices(self), kContiguous);
}

Tensor col_indices_copy_native(const Tensor& self) {
    reject_active_transform(self, "col_indices_copy");
    return ops::clone(ops::col_indices(self), kContiguous);
}

int64_t _dimI_native(const Tensor& self) {
    reject_active_transform(self, "_dimI");
    return ops::sparse_dim(self);
}

int64_t _dimV_native(const Tensor& self) {
    reject_active_transform(self, "_dimV");
    return ops::dense_dim(self);
}

// Row-compressed layouts keep one compressed index slot and one plain index
// slot; this build has a single compressed representation, so the ccol/row
// spellings (the column-compressed reading) serve the same two slots.
namespace {

Tensor compressed_indices_slot(const Tensor& self, const char* name) {
    const auto impl = self.unsafeGetTensorImpl();
    if (!impl || !impl->is_sparse() ||
        impl->sparse_layout() != TensorImpl::kSparseCSRLayout) {
        TP_THROW(RuntimeError, name,
                 " expected a tensor with a sparse compressed layout");
    }
    auto compressed = impl->sparse_crow_impl();
    TP_CHECK(compressed != nullptr, name,
             " expected a tensor with a sparse compressed layout");
    return Tensor(std::move(compressed));
}

Tensor plain_indices_slot(const Tensor& self, const char* name) {
    const auto impl = self.unsafeGetTensorImpl();
    if (!impl || !impl->is_sparse() ||
        impl->sparse_layout() != TensorImpl::kSparseCSRLayout) {
        TP_THROW(RuntimeError, name,
                 " expected a tensor with a sparse compressed layout");
    }
    auto plain = impl->sparse_col_impl();
    TP_CHECK(plain != nullptr, name,
             " expected a tensor with a sparse compressed layout");
    return Tensor(std::move(plain));
}

} // namespace

Tensor ccol_indices_native(const Tensor& self) {
    reject_active_transform(self, "ccol_indices");
    return compressed_indices_slot(self, "ccol_indices");
}

Tensor row_indices_native(const Tensor& self) {
    reject_active_transform(self, "row_indices");
    return plain_indices_slot(self, "row_indices");
}

Tensor ccol_indices_copy_native(const Tensor& self) {
    reject_active_transform(self, "ccol_indices_copy");
    return ops::clone(ccol_indices_native(self), kContiguous);
}

Tensor row_indices_copy_native(const Tensor& self) {
    return ops::clone(row_indices_native(self), kContiguous);
}

// _coalesced_ flips the COO coalescing flag in place.  Row-compressed layouts
// are canonical by construction, so only the coalesced request is accepted.
Tensor& _coalesced__native(Tensor& self, bool coalesced) {
    reject_active_transform(self, "_coalesced_");
    const auto impl = self.unsafeGetTensorImpl();
    TP_CHECK(impl && impl->is_sparse(),
             "_coalesced_ expected a sparse tensor");
    if (impl->sparse_layout() == TensorImpl::kSparseCOOLayout) {
        impl->set_sparse_state(impl->sparse_indices_impl(),
                               impl->sparse_values_impl(),
                               impl->sparse_sizes(), coalesced);
    } else {
        TP_CHECK(coalesced,
                 "row-compressed layouts are always coalesced; marking them "
                 "uncoalesced is not allowed");
    }
    return self;
}

// -----------------------------------------------------------------------------
// Base metadata
// -----------------------------------------------------------------------------

// set_data rebinds self to share new_data's storage, layout and version
// counter.  The metadata copy carries no autograd history, so self comes back
// as a fresh leaf regardless of the graphs on either side.
void set_data_native(Tensor& self, const Tensor& new_data) {
    reject_active_transform(self, "set_data");
    TP_CHECK(new_data.defined(), "set_data expected a defined tensor");
    self = Tensor(
        std::make_shared<TensorImpl>(*new_data.unsafeGetTensorImpl()));
}

// Single-output materialization: this build does not number op outputs, so
// every tensor reports position 0.
int64_t output_nr_native(const Tensor& self) {
    (void)self;
    return 0;
}

// True only when there is no addressable payload: an undefined tensor or one
// whose storage holds zero bytes.
bool _is_zerotensor_native(const Tensor& self) {
    reject_active_transform(self, "_is_zerotensor");
    if (!self.defined()) {
        return false;
    }
    const auto impl = self.unsafeGetTensorImpl();
    return !impl->has_storage() || impl->storage().nbytes() == 0;
}

// Shallow copies are possible exactly when both tensors live on the same
// device type and use the same storage discipline (dense vs sparse).
bool _has_compatible_shallow_copy_type_native(const Tensor& self,
                                              const Tensor& from) {
    reject_active_transform(self, "_has_compatible_shallow_copy_type");
    const auto a = self.unsafeGetTensorImpl();
    const auto b = from.unsafeGetTensorImpl();
    if (!a || !b) {
        return false;
    }
    if (a->device().type() != b->device().type()) {
        return false;
    }
    return a->is_sparse() == b->is_sparse();
}

Scalar _local_scalar_dense_native(const Tensor& self) {
    reject_active_transform(self, "_local_scalar_dense");
    return ops::item(self);
}

// lift/lift_fresh are identities here: tensors are materialized eagerly and
// carry no inference-mode indirection to strip.
Tensor lift_native(const Tensor& self) {
    return self;
}

Tensor lift_fresh_native(const Tensor& self) {
    return self;
}

Tensor lift_fresh_copy_native(const Tensor& self) {
    reject_active_transform(self, "lift_fresh_copy");
    return ops::clone(self, kContiguous);
}

TENSORPLAY_LIBRARY_IMPL(Composite, BaseMetaComposites) {
    m.impl("_cast_Byte", _cast_Byte_native);
    m.impl("_cast_Char", _cast_Char_native);
    m.impl("_cast_Short", _cast_Short_native);
    m.impl("_cast_Int", _cast_Int_native);
    m.impl("_cast_Long", _cast_Long_native);
    m.impl("_cast_Half", _cast_Half_native);
    m.impl("_cast_Float", _cast_Float_native);
    m.impl("_cast_Double", _cast_Double_native);
    m.impl("_to_dense", _to_dense_native);
    m.impl("_to_cpu", _to_cpu_native);
    m.impl("indices", indices_native);
    m.impl("values", values_native);
    m.impl("_indices_copy", _indices_copy_native);
    m.impl("_values_copy", _values_copy_native);
    m.impl("crow_indices_copy", crow_indices_copy_native);
    m.impl("col_indices_copy", col_indices_copy_native);
    m.impl("_dimI", _dimI_native);
    m.impl("_dimV", _dimV_native);
    m.impl("ccol_indices", ccol_indices_native);
    m.impl("row_indices", row_indices_native);
    m.impl("ccol_indices_copy", ccol_indices_copy_native);
    m.impl("row_indices_copy", row_indices_copy_native);
    m.impl("_coalesced_", _coalesced__native);
    m.impl("set_data", set_data_native);
    m.impl("output_nr", output_nr_native);
    m.impl("_is_zerotensor", _is_zerotensor_native);
    m.impl("_has_compatible_shallow_copy_type",
           _has_compatible_shallow_copy_type_native);
    m.impl("_local_scalar_dense", _local_scalar_dense_native);
    m.impl("lift", lift_native);
    m.impl("lift_fresh", lift_fresh_native);
    m.impl("lift_fresh_copy", lift_fresh_copy_native);
}

} // namespace composite
} // namespace tensorplay

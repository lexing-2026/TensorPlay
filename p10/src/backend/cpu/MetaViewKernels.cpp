// Host kernels for the metadata ops whose answer depends on host storage or
// on the host sparse machinery: the scalar read behind item() and the COO
// coalescing pass.  The device-neutral half of this family (view/alias
// metadata, detach, storage offsets, memory pinning, sparse component
// accessors) lives in the backend-neutral composite registration, which every
// device shares.
//
// These kernels must never re-enter the dispatcher for their own op name: a
// generated Tensor method resolves through the dispatcher, so a kernel that
// called the same-named member would recurse without bound.

#include "Tensor.h"
#include "TensorImpl.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Storage.h"
#include "SparseKernels.h"

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace tensorplay {
namespace cpu {

// -----------------------------------------------------------------------------
// item
// -----------------------------------------------------------------------------

Scalar item_cpu(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (impl->is_sparse()) {
        TP_THROW(RuntimeError, "item() is not supported for sparse tensors");
    }
    if (impl->numel() != 1) {
        TP_THROW(ValueError, "item() only supported for 1-element tensors");
    }
    if (!impl->device().is_cpu()) {
        TP_THROW(RuntimeError, "item(): expected a CPU tensor but got ",
                 impl->device().toString());
    }

    switch (impl->dtype()) {
        case DType::Float32: return Scalar(static_cast<double>(*impl->data<float>()));
        case DType::Float64: return Scalar(*impl->data<double>());
        case DType::Float16: return Scalar(static_cast<float>(*impl->data<Half>()));
        case DType::BFloat16: return Scalar(static_cast<float>(*impl->data<BFloat16>()));
        case DType::Float8_e4m3fn: return Scalar(static_cast<float>(*impl->data<Float8_e4m3fn>()));
        case DType::Float8_e5m2: return Scalar(static_cast<float>(*impl->data<Float8_e5m2>()));
        case DType::Float8_e4m3fnuz: return Scalar(static_cast<float>(*impl->data<Float8_e4m3fnuz>()));
        case DType::Float8_e5m2fnuz: return Scalar(static_cast<float>(*impl->data<Float8_e5m2fnuz>()));
        case DType::Float8_e8m0fnu: return Scalar(static_cast<float>(*impl->data<Float8_e8m0fnu>()));
        case DType::Int8: return Scalar(static_cast<int64_t>(*impl->data<int8_t>()));
        case DType::Int16: return Scalar(static_cast<int64_t>(*impl->data<int16_t>()));
        case DType::Int32: return Scalar(static_cast<int64_t>(*impl->data<int32_t>()));
        case DType::Int64: return Scalar(*impl->data<int64_t>());
        case DType::UInt8: return Scalar(static_cast<uint64_t>(*impl->data<uint8_t>()));
        case DType::UInt16: return Scalar(static_cast<uint64_t>(*impl->data<uint16_t>()));
        case DType::UInt32: return Scalar(static_cast<uint64_t>(*impl->data<uint32_t>()));
        case DType::UInt64: return Scalar(*impl->data<uint64_t>());
        case DType::Bool: return Scalar(static_cast<bool>(*impl->data<bool>()));
        case DType::ComplexHalf: {
            const auto value = *impl->data<std::complex<Half>>();
            return Scalar(std::complex<float>(static_cast<float>(value.real()),
                                              static_cast<float>(value.imag())));
        }
        case DType::ComplexFloat: return Scalar(*impl->data<std::complex<float>>());
        case DType::ComplexDouble: return Scalar(*impl->data<std::complex<double>>());
        case DType::BComplex32: {
            const auto value = *impl->data<std::complex<BFloat16>>();
            return Scalar(std::complex<float>(static_cast<float>(value.real()),
                                              static_cast<float>(value.imag())));
        }
        default:
            TP_THROW(NotImplementedError, "item() not implemented for this dtype");
    }
}

// -----------------------------------------------------------------------------
// coalesce
// -----------------------------------------------------------------------------

Tensor coalesce_cpu(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError,
                 "coalesce() is only defined for sparse COO tensors");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (!impl->is_sparse() ||
        impl->sparse_layout() == TensorImpl::kSparseCSRLayout) {
        TP_THROW(RuntimeError,
                 "coalesce() is only defined for sparse COO tensors");
    }
    if (impl->is_coalesced()) {
        return self;
    }
    return coalesce_sparse_cpu(self);
}

Tensor _coalesce_cpu(const Tensor& self) {
    return coalesce_cpu(self);
}

TENSORPLAY_LIBRARY_IMPL(CPU, MetaViewOps) {
    m.impl("item", item_cpu);
    m.impl("coalesce", coalesce_cpu);
    m.impl("_coalesce", _coalesce_cpu);
}

}  // namespace cpu
}  // namespace tensorplay

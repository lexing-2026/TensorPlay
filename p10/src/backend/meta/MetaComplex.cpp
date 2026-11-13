// Shape-only kernels for the complex views and factories on the meta device.
//
// The view forms (view_as_real, view_as_complex) rewrite sizes, strides and
// the storage offset only: every complex element splits into two consecutive
// value-type scalars, so positions and strides double, and folding a pair
// back halves them.  The factory forms (complex, polar) answer the broadcast
// of their operands with the complex twin of the real operand type.  The
// conjugate family cannot apply its sign flip without element data, so a
// complex operand answers the materialized geometry instead of a view, while
// real operands -- whose conjugate is the identity -- answer views.
//
// The materializing views share the source storage and its version counter,
// so in-place bookkeeping treats them like any other view.

#include "MetaCommon.h"

#include "DType.h"
#include "Exception.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

bool is_factory_real_dtype(DType dtype) {
    return dtype == DType::Float16 || dtype == DType::Float32 ||
           dtype == DType::Float64 || dtype == DType::BFloat16;
}

void check_complex_factory_pair(const Tensor& a, const Tensor& b,
                                const char* name) {
    if (!is_factory_real_dtype(a.dtype()) || !is_factory_real_dtype(b.dtype())) {
        TP_THROW(NotImplementedError, name, " expects floating-point inputs");
    }
    if (a.dtype() != b.dtype()) {
        TP_THROW(RuntimeError, name, " expects inputs with the same dtype");
    }
    if (a.device() != b.device()) {
        TP_THROW(DeviceMismatchError, name, " expects inputs on the same device");
    }
}

Tensor real_kernel(const Tensor& self) {
    // A real tensor is its own real part; a complex one answers the value
    // type of its complex dtype.
    if (!isComplexType(self.dtype())) return self.clone();
    return empty_meta(shape_of(self), toRealValueType(self.dtype()), self.device());
}

Tensor imag_kernel(const Tensor& self) {
    // The imaginary part of a real tensor is zero everywhere, so the answer
    // keeps the input dtype with no entries to fill.
    if (!isComplexType(self.dtype())) {
        return Tensor::zeros(shape_of(self), self.dtype(), self.device());
    }
    return empty_meta(shape_of(self), toRealValueType(self.dtype()), self.device());
}

Tensor conj_kernel(const Tensor& self) {
    if (!isComplexType(self.dtype())) {
        // Real values are their own conjugates: answer a fresh view with the
        // input's layout (never the input impl itself, which would re-tag the
        // source's grad node).
        return self.as_strided(shape_of(self),
                               static_cast<std::vector<int64_t>>(self.strides()),
                               static_cast<int64_t>(self.storage_offset()));
    }
    return empty_meta(shape_of(self), self.dtype(), self.device());
}

// transpose(-2, -1) composed with conj(): ndim <= 1 is plain conj, a real
// operand is the transposed view, and a complex operand answers the
// transposed contiguous geometry.
Tensor adjoint_kernel(const Tensor& self) {
    if (self.dim() <= 1) return conj_kernel(self);
    std::vector<int64_t> sizes = shape_of(self);
    std::vector<int64_t> strides = self.strides();
    std::swap(sizes[sizes.size() - 2], sizes[sizes.size() - 1]);
    if (!isComplexType(self.dtype())) {
        std::swap(strides[strides.size() - 2], strides[strides.size() - 1]);
        return self.as_strided(sizes, strides,
                               static_cast<int64_t>(self.storage_offset()));
    }
    return empty_meta(sizes, self.dtype(), self.device());
}

Tensor complex_kernel(const Tensor& real, const Tensor& imag) {
    check_complex_factory_pair(real, imag, "complex");
    return empty_meta(broadcast2(real, imag),
                      toComplexType(real.dtype()), real.device());
}

Tensor polar_kernel(const Tensor& abs_, const Tensor& angle_) {
    check_complex_factory_pair(abs_, angle_, "polar");
    return empty_meta(broadcast2(abs_, angle_),
                      toComplexType(abs_.dtype()), abs_.device());
}

Tensor view_as_real_kernel(const Tensor& self) {
    if (!isComplexType(self.dtype())) {
        TP_THROW(RuntimeError,
                 "view_as_real is only supported for complex tensors, but got ",
                 toString(self.dtype()));
    }
    std::vector<int64_t> sizes = shape_of(self);
    std::vector<int64_t> strides = self.strides();
    // Every element splits into two consecutive scalars, so each stride and
    // the entry position double; refuse the rewrite when a doubled value
    // leaves the representable range.
    for (int64_t& stride : strides) {
        if (stride < 0 || stride > std::numeric_limits<int64_t>::max() / 2) {
            TP_THROW(RuntimeError, "view_as_real: stride overflow");
        }
        stride *= 2;
    }
    sizes.push_back(2);
    strides.push_back(1);
    const size_t offset = self.unsafeGetTensorImpl()->storage_offset();
    if (offset > std::numeric_limits<size_t>::max() / 2) {
        TP_THROW(RuntimeError, "view_as_real: storage offset overflow");
    }
    Tensor result(self.unsafeGetTensorImpl()->storage(), sizes, strides,
                  toRealValueType(self.dtype()), offset * 2);
    result.unsafeGetTensorImpl()->share_version_counter(*self.unsafeGetTensorImpl());
    return result;
}

Tensor view_as_complex_kernel(const Tensor& self) {
    if (self.dtype() != DType::Float16 && self.dtype() != DType::BFloat16 &&
        self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(RuntimeError,
                 "view_as_complex is only supported for half, bfloat16, float "
                 "and double tensors, but got ",
                 toString(self.dtype()));
    }
    if (self.dim() == 0 || self.size(self.dim() - 1) != 2) {
        TP_THROW(RuntimeError,
                 "view_as_complex: input tensor must have a last dimension of size 2");
    }
    if (self.stride(self.dim() - 1) != 1) {
        TP_THROW(RuntimeError, "view_as_complex: last dimension must have stride 1");
    }
    // Pairs are adjacent scalars, so a stride over pairs is half the stride
    // over scalars; an odd outer stride would split a pair.
    for (int64_t dim = 0; dim + 1 < self.dim(); ++dim) {
        if (self.size(dim) != 1 && (self.stride(dim) & 1) != 0) {
            TP_THROW(RuntimeError,
                     "view_as_complex: strides of all dimensions except the last "
                     "must be divisible by 2");
        }
    }
    const size_t offset = self.unsafeGetTensorImpl()->storage_offset();
    if ((offset & 1) != 0) {
        TP_THROW(RuntimeError,
                 "view_as_complex: storage offset must be divisible by 2");
    }
    std::vector<int64_t> sizes = shape_of(self);
    std::vector<int64_t> strides = self.strides();
    sizes.pop_back();
    strides.pop_back();
    for (int64_t& stride : strides) stride /= 2;
    Tensor result(self.unsafeGetTensorImpl()->storage(), sizes, strides,
                  toComplexType(self.dtype()), offset / 2);
    result.unsafeGetTensorImpl()->share_version_counter(*self.unsafeGetTensorImpl());
    return result;
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaComplex) {
    m.impl("real", real_kernel);
    m.impl("imag", imag_kernel);
    m.impl("conj", conj_kernel);
    m.impl("adjoint", adjoint_kernel);
    m.impl("complex", complex_kernel);
    m.impl("polar", polar_kernel);
    m.impl("view_as_real", view_as_real_kernel);
    m.impl("view_as_complex", view_as_complex_kernel);
}

} // namespace meta
} // namespace tensorplay

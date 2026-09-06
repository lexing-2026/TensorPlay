#include "CompositeCommon.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <optional>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

void check_logical_out(const Tensor& out) {
    if (out.dtype() != DType::Bool && out.dtype() != DType::UInt8) {
        TP_THROW(TypeError,
                 "logical reduction output must have Bool or UInt8 dtype");
    }
}

template <bool IsAll>
Tensor reduce_dims_default(const Tensor& self,
                           const std::optional<std::vector<int64_t>>& dim,
                           bool keepdim) {
    if (!dim.has_value()) {
        Tensor result = IsAll ? ops::all(self) : ops::any(self);
        if (keepdim) {
            return ops::expand(
                result, std::vector<int64_t>(static_cast<size_t>(self.dim()), 1),
                false);
        }
        return result;
    }

    if (dim->empty()) {
        if constexpr (IsAll) {
            if (self.dtype() == DType::UInt8) {
                return ops::ne(self, Scalar(int64_t(0)));
            }
        } else {
            if (self.dtype() == DType::UInt8) {
                return ops::ne(self, Scalar(int64_t(0)));
            }
        }
        return ops::_to_copy(self, std::optional<DType>(DType::Bool),
                             std::nullopt, std::nullopt, std::nullopt,
                             false, std::nullopt);
    }

    Tensor result = self;
    for (int64_t d : *dim) {
        if constexpr (IsAll) {
            result = ops::all(result, d, true);
        } else {
            result = ops::any(result, d, true);
        }
    }
    return keepdim ? result : ops::squeeze(result, *dim);
}

template <bool IsAll>
Tensor& reduce_out(const Tensor& self, int64_t dim, bool keepdim, Tensor& out) {
    check_logical_out(out);
    if (self.device() != out.device()) {
        TP_THROW(DeviceMismatchError,
                 "output must be on the same device as input");
    }
    Tensor result = IsAll ? ops::all(self, dim, keepdim)
                          : ops::any(self, dim, keepdim);
    ops::resize_(out, static_cast<std::vector<int64_t>>(result.shape()));
    return ops::copy_(out, result);
}

template <bool IsAll>
Tensor& reduce_dims_out(const Tensor& self,
                        const std::optional<std::vector<int64_t>>& dim,
                        bool keepdim, Tensor& out) {
    check_logical_out(out);
    if (self.device() != out.device()) {
        TP_THROW(DeviceMismatchError,
                 "output must be on the same device as input");
    }
    Tensor result = IsAll ? reduce_dims_default<true>(self, dim, keepdim)
                          : reduce_dims_default<false>(self, dim, keepdim);
    ops::resize_(out, static_cast<std::vector<int64_t>>(result.shape()));
    return ops::copy_(out, result);
}

} // namespace

Tensor all_dims_default(const Tensor& self,
                        std::optional<std::vector<int64_t>> dim,
                        bool keepdim) {
    return reduce_dims_default<true>(self, dim, keepdim);
}

Tensor any_dims_default(const Tensor& self,
                        std::optional<std::vector<int64_t>> dim,
                        bool keepdim) {
    return reduce_dims_default<false>(self, dim, keepdim);
}

Tensor& all_out_default(const Tensor& self, int64_t dim, bool keepdim,
                        Tensor& out) {
    return reduce_out<true>(self, dim, keepdim, out);
}

Tensor& any_out_default(const Tensor& self, int64_t dim, bool keepdim,
                        Tensor& out) {
    return reduce_out<false>(self, dim, keepdim, out);
}

Tensor& all_dims_out_default(
        const Tensor& self, std::optional<std::vector<int64_t>> dim,
        bool keepdim, Tensor& out) {
    return reduce_dims_out<true>(self, dim, keepdim, out);
}

Tensor& any_dims_out_default(
        const Tensor& self, std::optional<std::vector<int64_t>> dim,
        bool keepdim, Tensor& out) {
    return reduce_dims_out<false>(self, dim, keepdim, out);
}

Tensor& all_all_out_default(const Tensor& self, Tensor& out) {
    check_logical_out(out);
    if (self.device() != out.device()) {
        TP_THROW(DeviceMismatchError,
                 "output must be on the same device as input");
    }
    Tensor result = ops::all(self);
    ops::resize_(out, static_cast<std::vector<int64_t>>(result.shape()));
    return ops::copy_(out, result);
}

Tensor& any_all_out_default(const Tensor& self, Tensor& out) {
    check_logical_out(out);
    if (self.device() != out.device()) {
        TP_THROW(DeviceMismatchError,
                 "output must be on the same device as input");
    }
    Tensor result = ops::any(self);
    ops::resize_(out, static_cast<std::vector<int64_t>>(result.shape()));
    return ops::copy_(out, result);
}

TENSORPLAY_LIBRARY_IMPL(Composite, ReductionComposite) {
    m.impl("all.dims", all_dims_default);
    m.impl("any.dims", any_dims_default);
    m.impl("all.out", all_out_default);
    m.impl("any.out", any_out_default);
    m.impl("all.dims_out", all_dims_out_default);
    m.impl("any.dims_out", any_dims_out_default);
    m.impl("all.all_out", all_all_out_default);
    m.impl("any.all_out", any_all_out_default);
}

} // namespace composite
} // namespace tensorplay

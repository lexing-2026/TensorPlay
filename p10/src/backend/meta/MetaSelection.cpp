// Shape-only kernels for ordering selections on the meta device.
//
// A sort, an argsort or a topk reorders existing elements, so only the answer
// geometry and the answer types are observable: values keep the input dtype
// and positions are Int64, both repeating the input shape except along the
// selected axis, which holds the chosen extent for topk.  Which element lands
// where depends on element data, which a meta tensor does not own.

#include "MetaCommon.h"

#include "DType.h"
#include "TypeProperties.h"

#include <cstdint>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace meta {

namespace {

std::tuple<Tensor, Tensor> sort_kernel(const Tensor& self, int64_t dim,
                                       bool /*descending*/) {
    const int64_t nd = self.dim();
    if (nd == 0) {
        TP_THROW(RuntimeError, "sort: expects at least 1 dimension");
    }
    wrap_dim(dim, nd);
    if (!is_scalar_dtype(self.dtype())) {
        TP_THROW(TypeError, "sort: unsupported dtype");
    }
    const std::vector<int64_t> out = shape_of(self);
    return {empty_meta(out, self.dtype(), self.device()),
            empty_meta(out, DType::Int64, self.device())};
}

Tensor argsort_kernel(const Tensor& self, int64_t dim, bool descending) {
    return std::get<1>(sort_kernel(self, dim, descending));
}

std::tuple<Tensor, Tensor> topk_kernel(const Tensor& self, int64_t k,
                                       int64_t dim, bool /*largest*/,
                                       bool /*sorted*/, int64_t /*impl*/) {
    const int64_t nd = self.dim();
    if (nd == 0) {
        if (dim != 0 && dim != -1) {
            TP_THROW(RuntimeError, "topk: dimension out of range");
        }
    } else {
        if (dim < 0) dim += nd;
        if (dim < 0 || dim >= nd) {
            TP_THROW(RuntimeError, "topk: dimension out of range");
        }
    }
    const int64_t dim_size = nd == 0 ? 1 : self.size(dim);
    if (k < 0 || k > dim_size) {
        TP_THROW(RuntimeError, "topk: k must be in [0, dimension size]");
    }
    if (self.dtype() == DType::Bool || !is_scalar_dtype(self.dtype())) {
        TP_THROW(NotImplementedError, "topk: unsupported dtype");
    }
    std::vector<int64_t> out = shape_of(self);
    if (nd != 0) out[static_cast<size_t>(dim)] = k;
    return {empty_meta(out, self.dtype(), self.device()),
            empty_meta(out, DType::Int64, self.device())};
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Meta, MetaSelection) {
    m.impl("sort", sort_kernel);
    m.impl("argsort", argsort_kernel);
    m.impl("topk", topk_kernel);
}

} // namespace meta
} // namespace tensorplay

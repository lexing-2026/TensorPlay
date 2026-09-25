#include "ReduceKernels.cuh"

namespace tensorplay {
namespace cuda {

namespace {


Tensor renorm_cuda(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm) {
    if (p.isComplex()) {
        TP_THROW(TypeError, "renorm: p must be real-valued");
    }
    const double pd = p.toDouble();
    if (!(pd > 0.0)) {
        TP_THROW(ValueError, "renorm: norm order must be positive");
    }
    if (maxnorm.isComplex()) {
        TP_THROW(TypeError, "renorm: maxnorm must be real-valued");
    }
    const double max_norm = maxnorm.toDouble();
    if (!(max_norm >= 0.0)) {
        TP_THROW(ValueError, "renorm: maxnorm must be non-negative");
    }
    if (!isFloatingOrComplexType(self.dtype())) {
        TP_THROW(TypeError, "renorm: input must have a floating or complex dtype");
    }
    int64_t nd = self.dim();
    if (nd <= 1) {
        TP_THROW(RuntimeError, "renorm: input must have at least 2 dimensions");
    }
    dim = wrap_dim(dim, nd);
    std::vector<int64_t> reduce_dims;
    reduce_dims.reserve(static_cast<size_t>(nd - 1));
    for (int64_t axis = 0; axis < nd; ++axis) {
        if (axis != dim) reduce_dims.push_back(axis);
    }
    Tensor norms = ops::norm(self, reduce_dims, pd, true);
    Tensor factors = ops::where(
        ops::gt(norms, Scalar(max_norm)),
        ops::div(
            Tensor::full_like(norms, Scalar(max_norm), norms.dtype(), norms.device()),
            ops::add(norms, Scalar(1e-7))),
        Scalar(1.0));
    Tensor result = ops::mul(self, factors);
    return result.dtype() == self.dtype() ? result : result.to(self.dtype());
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, RenormKernels) {
    m.impl("renorm", renorm_cuda);
}

} // namespace cuda
} // namespace tensorplay

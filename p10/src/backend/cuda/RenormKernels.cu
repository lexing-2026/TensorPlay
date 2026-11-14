#include "ReduceKernels.cuh"
#include "CUDALoops.cuh"
#include "OutWrite.h"

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

// Destination form for real floating widths: one reduce for the slice norms,
// then a single elementwise pass that folds the cap-and-epsilon factor into
// the scale.  The pass walks contiguous storage and indexes the norms buffer
// per slice, so writing self in place reads each element exactly once.
// Everything else keeps the composite route.
namespace {

// norms keeps one value per position along `dim` (every other axis reduced
// and kept as a size-1 axis): its flat slot is the element's index along dim.
template <typename T>
__global__ void renorm_apply_kernel(int64_t numel, int64_t slice_span,
                                    int64_t inner, const T* self,
                                    const T* norms, T* out, double max_norm) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; i < numel; i += stride) {
        const T n = norms[(i % slice_span) / inner];
        const T limit = static_cast<T>(max_norm);
        const T f = n > limit ? static_cast<T>(max_norm / (static_cast<double>(n) + 1e-7))
                              : static_cast<T>(1);
        out[i] = self[i] * f;
    }
}

}  // namespace

Tensor& renorm_out_cuda(const Tensor& self, const Scalar& p, int64_t dim,
                        const Scalar& maxnorm, Tensor& out) {
    const DType dt = self.dtype();
    if ((dt == DType::Float32 || dt == DType::Float64) &&
        self.is_contiguous() && out.is_contiguous() &&
        destination_takes(out, self.device(), dt,
                          static_cast<std::vector<int64_t>>(self.shape())) &&
        read_while_write_is_safe(self, out)) {
        const double pd = p.toDouble();
        const double max_norm = maxnorm.toDouble();
        int64_t nd = self.dim();
        dim = wrap_dim(dim, nd);
        std::vector<int64_t> reduce_dims;
        reduce_dims.reserve(static_cast<size_t>(nd - 1));
        for (int64_t axis = 0; axis < nd; ++axis) {
            if (axis != dim) reduce_dims.push_back(axis);
        }
        Tensor norms = ops::norm(self, reduce_dims, pd, true);
        const std::vector<int64_t> sizes =
            static_cast<std::vector<int64_t>>(self.shape());
        int64_t slice_span = 1;
        for (int64_t axis = dim; axis < nd; ++axis) slice_span *= sizes[axis];
        int64_t inner = 1;
        for (int64_t axis = dim + 1; axis < nd; ++axis) inner *= sizes[axis];
        const int64_t numel = self.numel();
        if (numel > 0) {
            auto stream = getCurrentCUDAStream().stream();
            const int64_t threads = 256;
            const int64_t blocks =
                std::min<int64_t>((numel + threads - 1) / threads, 4096);
            if (dt == DType::Float64) {
                renorm_apply_kernel<double><<<static_cast<unsigned>(blocks),
                                              static_cast<unsigned>(threads), 0,
                                              stream>>>(
                    numel, slice_span, inner, self.data_ptr<double>(),
                    norms.data_ptr<double>(), out.data_ptr<double>(), max_norm);
            } else {
                renorm_apply_kernel<float><<<static_cast<unsigned>(blocks),
                                             static_cast<unsigned>(threads), 0,
                                             stream>>>(
                    numel, slice_span, inner, self.data_ptr<float>(),
                    norms.data_ptr<float>(), out.data_ptr<float>(), max_norm);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        return out;
    }
    return write_out(out, renorm_cuda(self, p, dim, maxnorm));
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, RenormKernels) {
    m.impl("renorm", renorm_cuda);
    m.impl("renorm.out", renorm_out_cuda);
}

} // namespace cuda
} // namespace tensorplay

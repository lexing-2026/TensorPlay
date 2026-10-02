#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


// ---------------------------------------------------------------------------
// nonzero_static: first `size` nonzero coordinates, padded with fill_value
// rows beyond the actual count.
// ---------------------------------------------------------------------------

Tensor interop_nonzero_static_cuda(const Tensor& self, int64_t size,
                                   int64_t fill_value) {
    Tensor nz = ops::nonzero(self);
    const int64_t ndim = self.dim();
    const int64_t cap = size;
    Tensor result = ops::full({cap, ndim}, Scalar(fill_value), DType::Int64,
                              self.device());
    const int64_t copy_n = std::min<int64_t>(cap, nz.size(0));
    if (copy_n > 0) {
        result.narrow(0, 0, copy_n).copy_(nz.narrow(0, 0, copy_n));
    }
    return result;
}


Tensor& interop_nonzero_static_out_cuda(const Tensor& self, int64_t size,
                                        int64_t fill_value, Tensor& out) {
    Tensor nz = ops::nonzero(self);
    const int64_t cap = size;
    Tensor result = ops::full({cap, self.dim()}, Scalar(fill_value),
                              DType::Int64, self.device());
    const int64_t copy_n = std::min<int64_t>(cap, nz.size(0));
    if (copy_n > 0) {
        result.narrow(0, 0, copy_n).copy_(nz.narrow(0, 0, copy_n));
    }
    write_out(out, result);
    return out;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, NonzeroKernels) {
    m.impl("nonzero_static", interop_nonzero_static_cuda);
    m.impl("nonzero_static.out", interop_nonzero_static_out_cuda);
}

} // namespace cuda
} // namespace tensorplay

// Tensor factory native implementations.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TypePromotion.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <optional>

namespace tensorplay::cuda {

namespace ops = tensorplay::tpx::ops;

Tensor vander_native_cuda(const Tensor& x, std::optional<int64_t> N,
                          bool increasing) {
    if (x.dim() != 1) {
        TP_THROW(RuntimeError, "x must be a one-dimensional tensor.");
    }
    const int64_t columns = N.value_or(x.size(0));
    if (columns < 0) TP_THROW(RuntimeError, "N must be non-negative.");

    // Integer inputs count in Long.  The first column is all ones and each
    // later one multiplies in x once more, so the powers are a running
    // product along the row -- built from differentiable ops, nothing is
    // written into place.
    const DType dtype = promoteTypes(x.dtype(), DType::Int64);
    const std::optional<Device> device(x.device());
    Tensor result = ops::ones({x.size(0), std::min<int64_t>(columns, 1)}, dtype,
                              device);
    if (columns > 1) {
        Tensor powers = ops::cumprod(
            ops::expand(ops::unsqueeze(x, 1), {x.size(0), columns - 1}, false),
            1, dtype);
        result = ops::cat({result, powers}, 1);
    }
    return increasing ? result : ops::flip(result, {1});
}

Tensor linalg_vander_native_cuda(const Tensor& x, std::optional<int64_t> N) {
    return vander_native_cuda(x, N, true);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeTensorFactories) {
    m.impl("vander", vander_native_cuda);
    m.impl("linalg_vander", linalg_vander_native_cuda);
}

} // namespace tensorplay::cuda

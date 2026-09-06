// Tensor factory native implementations.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TypePromotion.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <optional>

namespace tensorplay::cpu {

namespace ops = tensorplay::tpx::ops;

Tensor vander_native_cpu(const Tensor& x, std::optional<int64_t> N,
                         bool increasing) {
    if (x.dim() != 1) {
        TP_THROW(RuntimeError, "x must be a one-dimensional tensor.");
    }
    const int64_t columns = N.value_or(x.size(0));
    if (columns < 0) TP_THROW(RuntimeError, "N must be non-negative.");

    // integer tensors to Long.
    const DType dtype = promoteTypes(x.dtype(), DType::Int64);
    // TensorPlay's low-level fill kernel is contiguous-only; constructing the
    // trying to fill a strided select view.
    Tensor result = ops::full({x.size(0), columns}, Scalar(1), dtype,
                              std::optional<Device>(x.device()), false, false);
    if (columns > 1) {
        Tensor tail = ops::slice(result, 1, 1, std::nullopt, 1);
        Tensor powers = ops::expand(
            ops::unsqueeze(x, 1), {x.size(0), columns - 1}, false);
        ops::copy_(tail, powers, false);
        ops::copy_(tail, ops::cumprod(tail, 1, std::nullopt), false);
    }
    return increasing ? result : ops::flip(result, {1});
}

Tensor linalg_vander_native_cpu(const Tensor& x, std::optional<int64_t> N) {
    return vander_native_cpu(x, N, true);
}

TENSORPLAY_LIBRARY_IMPL(CPU, NativeTensorFactories) {
    m.impl("vander", vander_native_cpu);
    m.impl("linalg_vander", linalg_vander_native_cpu);
}

} // namespace tensorplay::cpu

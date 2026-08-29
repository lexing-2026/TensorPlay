// Linear-algebra native wrappers.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

namespace tensorplay::cuda {

namespace ops = tensorplay::tpx::ops;

Tensor ger_native_cuda(const Tensor& self, const Tensor& vec2) {
    return ops::outer(self, vec2);
}

Tensor kron_native_cuda(const Tensor& self, const Tensor& other) {
    const int64_t maxdim = std::max(self.dim(), other.dim());
    const int64_t pad_self = maxdim - self.dim();
    const int64_t pad_other = maxdim - other.dim();
    std::vector<int64_t> a_shape(2 * maxdim);
    std::vector<int64_t> b_shape(2 * maxdim);
    std::vector<int64_t> result_shape(maxdim);
    for (int64_t i = 0; i < maxdim; ++i) {
        a_shape[2 * i] = i >= pad_self ? self.size(i - pad_self) : 1;
        a_shape[2 * i + 1] = 1;
        b_shape[2 * i] = 1;
        b_shape[2 * i + 1] = i >= pad_other ? other.size(i - pad_other) : 1;
        result_shape[i] = a_shape[2 * i] * b_shape[2 * i + 1];
    }
    return ops::view(
        ops::mul(ops::view(self, a_shape), ops::view(other, b_shape)),
        result_shape);
}

Tensor matrix_power_native_cuda(const Tensor& self, int64_t n) {
    if (self.dim() < 2 || self.size(-2) != self.size(-1)) {
        TP_THROW(RuntimeError, "matrix_power(): expected a square matrix");
    }
    const int64_t order = self.size(-1);
    if (n == 0) {
        Tensor result = ops::clone(self, 0);
        Tensor identity = ops::eye(order, order, self.dtype(),
                                   std::optional<Device>(self.device()), false);
        if (self.dim() > 2) {
            const std::vector<int64_t> shape =
                static_cast<std::vector<int64_t>>(self.shape());
            identity = ops::clone(ops::expand(identity, shape, false), 0);
        }
        return ops::copy_(result, identity, false);
    }
    if (n == 1) return ops::clone(self, 0);
    if (n == std::numeric_limits<int64_t>::min()) {
        TP_THROW(RuntimeError, "matrix_power(): exponent is too small");
    }
    Tensor base = n < 0 ? ops::linalg_inv(self) : self;
    if (n == -1) return base;
    n = std::abs(n);
    if (n == 2) return ops::matmul(base, base);
    if (n == 3) return ops::matmul(ops::matmul(base, base), base);

    Tensor z;
    Tensor result;
    while (n > 0) {
        const int64_t bit = n % 2;
        n /= 2;
        z = z.defined() ? ops::matmul(z, z) : base;
        if (bit == 1) {
            result = result.defined() ? ops::matmul(result, z) : z;
        }
    }
    return result;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeLinearAlgebra) {
    m.impl("ger", ger_native_cuda);
    m.impl("kron", kron_native_cuda);
    m.impl("matrix_power", matrix_power_native_cuda);
}

} // namespace tensorplay::cuda

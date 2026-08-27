// User-side kernel implementations for the sample myext extension.
#include "OpsGenerated.h"

namespace tp_custom {
namespace myext {
namespace impl {

Tensor scale_add_cpu(const Tensor& self, double factor, Scalar bias) {
    Tensor out = self.mul(factor);
    if (bias.toDouble() != 0.0) {
        out = out.add(bias.toDouble());
    }
    return out;
}

Tensor& scale_add__cpu(Tensor& self, double factor) {
    Tensor tmp = self.mul(factor);
    self.copy_(tmp);
    return self;
}

Tensor sum_dims_cpu(const Tensor& self, const std::vector<int64_t>& dims) {
    return self.sum(dims);
}

} // namespace impl
} // namespace myext
} // namespace tp_custom

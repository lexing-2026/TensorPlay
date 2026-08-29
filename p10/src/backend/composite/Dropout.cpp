// Composite kernels: the dropout family.
// the no-op/zero fast paths wrap the fused native_* primitives (which reject
// p == 1 themselves).

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

void check_dropout_p(double p) {
    if (p < 0 || p > 1) {
        TP_THROW(RuntimeError,
                 "dropout probability has to be between 0 and 1, but got ", p);
    }
}

Tensor zero_like_mul(const Tensor& input) {
    return ops::mul(input, ops::zeros({}, input.dtype(), input.device()));
}

} // anonymous namespace

Tensor dropout_native(const Tensor& input, double p, bool train) {
    check_dropout_p(p);
    if (p == 0 || !train || input.numel() == 0) return input;
    if (p == 1) return zero_like_mul(input);
    return std::get<0>(ops::native_dropout(input, p));
}

Tensor alpha_dropout_native(const Tensor& input, double p, bool train) {
    check_dropout_p(p);
    if (p == 0 || !train || input.numel() == 0) return input;
    if (p == 1) return zero_like_mul(input);
    return std::get<0>(ops::native_alpha_dropout(input, p));
}

Tensor feature_dropout_native(const Tensor& input, double p, bool train) {
    check_dropout_p(p);
    if (p == 0 || !train || input.numel() == 0) return input;
    if (p == 1) return zero_like_mul(input);
    return std::get<0>(ops::native_feature_dropout(input, p));
}

// No fused native kernel exists for the feature+alpha combination, so it is
// bernoulli noise, alpha affine rescaling.
Tensor feature_alpha_dropout_native(const Tensor& input, double p, bool train) {
    check_dropout_p(p);
    if (p == 0 || !train || input.numel() == 0) return input;
    if (p == 1) return zero_like_mul(input);
    if (input.dim() < 2) {
        TP_THROW(RuntimeError,
                 "Feature dropout requires at least 2 dimensions in the input");
    }
    constexpr double alpha = 1.7580993408473766;
    const double a = 1.0 / std::sqrt((alpha * alpha * p + 1.0) * (1.0 - p));
    std::vector<int64_t> noise_shape(input.dim(), 1);
    noise_shape[0] = input.size(0);
    noise_shape[1] = input.size(1);
    Tensor noise = ops::full(noise_shape, Scalar(1.0 - p), input.dtype(),
                             input.device());
    ops::bernoulli_(noise);
    // out = input * (noise * a) + (noise - 1) * (alpha * a) + alpha * a * p
    const Tensor b = ops::add(ops::mul(ops::sub(noise, Scalar(1)),
                                       Scalar(alpha * a)),
                              Scalar(alpha * a * p));
    return ops::add(ops::mul(input, ops::mul(noise, Scalar(a))), b);
}

namespace {

template <typename Fn>
Tensor& dropout_inplace(Tensor& self, double p, bool train, Fn&& out_of_place) {
    Tensor result = out_of_place(self, p, train);
    if (result.impl() != self.impl()) ops::copy_(self, result);
    return self;
}

} // anonymous namespace

Tensor& dropout__native(Tensor& self, double p, bool train) {
    return dropout_inplace(self, p, train, dropout_native);
}

Tensor& alpha_dropout__native(Tensor& self, double p, bool train) {
    return dropout_inplace(self, p, train, alpha_dropout_native);
}

Tensor& feature_dropout__native(Tensor& self, double p, bool train) {
    return dropout_inplace(self, p, train, feature_dropout_native);
}

Tensor& feature_alpha_dropout__native(Tensor& self, double p, bool train) {
    return dropout_inplace(self, p, train, feature_alpha_dropout_native);
}

TENSORPLAY_LIBRARY_IMPL(Composite, DropoutComposite) {
    m.impl("dropout", dropout_native);
    m.impl("dropout_", dropout__native);
    m.impl("alpha_dropout", alpha_dropout_native);
    m.impl("alpha_dropout_", alpha_dropout__native);
    m.impl("feature_dropout", feature_dropout_native);
    m.impl("feature_dropout_", feature_dropout__native);
    m.impl("feature_alpha_dropout", feature_alpha_dropout_native);
    m.impl("feature_alpha_dropout_", feature_alpha_dropout__native);
}

} // namespace composite
} // namespace tensorplay

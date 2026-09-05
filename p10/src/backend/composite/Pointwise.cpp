// Composite kernels: numerically-special elementwise ops -- xlogy, xlog1py,
// ldexp, fmax/fmin, float_power, mvlgamma, conj_physical and the negative_
// alias.
//
//   xlogy:      x * log(y), with the x == 0 branch collapsing every finite or
//               infinite y to 0; a NaN y still yields NaN.
//   xlog1py:    the same rule for x * log1p(y), singular at y == -1.
//   ldexp:      x * 2^y; the exponent is evaluated in the input dtype so the
//               power of two is exact.
//   fmax/fmin:  maximum/minimum with NaN treated as missing -- whenever one
//               operand is NaN the other is returned verbatim.
//   float_power: exponentiation evaluated in Float64, cast back to the
//               natural result type (integral inputs promote to Float32).
//   mvlgamma:   sum_{i=0..p-1} lgamma(x - i/2), elementwise.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

// The self == 0 shortcut collapses every y, but a NaN y still has to survive:
// the product carries it, so only the non-NaN positions take the zero.
Tensor xlogy_impl(const Tensor& self, const Tensor& other) {
    const Tensor product = ops::mul(self, ops::log(other));
    return ops::where(ops::isnan(other), product,
                      ops::where(ops::eq(self, Scalar(0)), Scalar(0), product));
}

Tensor xlog1py_impl(const Tensor& self, const Tensor& other) {
    const Tensor product = ops::mul(self, ops::log1p(other));
    return ops::where(ops::isnan(other), product,
                      ops::where(ops::eq(self, Scalar(0)), Scalar(0), product));
}

Tensor fmaxmin_impl(const Tensor& self, const Tensor& other, bool max) {
    const Tensor nan_self = ops::isnan(self);
    const Tensor nan_other = ops::isnan(other);
    const Tensor base = max ? ops::maximum(self, other)
                            : ops::minimum(self, other);
    return ops::where(nan_self, other,
                      ops::where(nan_other, self, base));
}

} // anonymous namespace

Tensor xlogy_native(const Tensor& self, const Tensor& other) {
    return xlogy_impl(self, other);
}

Tensor& xlogy__native(Tensor& self, const Tensor& other) {
    ops::copy_(self, xlogy_impl(self, other));
    return self;
}

Tensor xlog1py_native(const Tensor& self, const Tensor& other) {
    return xlog1py_impl(self, other);
}

Tensor& xlog1py__native(Tensor& self, const Tensor& other) {
    ops::copy_(self, ops::xlog1py(self, other));
    return self;
}

Tensor ldexp_native(const Tensor& self, const Tensor& other) {
    return ops::mul(self, ops::exp2(other.to(self.dtype())));
}

Tensor& ldexp__native(Tensor& self, const Tensor& other) {
    ops::copy_(self, ldexp_native(self, other));
    return self;
}

Tensor fmax_native(const Tensor& self, const Tensor& other) {
    return fmaxmin_impl(self, other, true);
}

Tensor fmin_native(const Tensor& self, const Tensor& other) {
    return fmaxmin_impl(self, other, false);
}

Tensor float_power_native(const Tensor& self, const Tensor& exponent) {
    const DType rt = ops::result_type(self, exponent);
    const DType out = (isFloatingType(rt) || isComplexType(rt)) ? rt
                                                                : DType::Float32;
    return ops::pow(self.to(DType::Float64), exponent.to(DType::Float64))
        .to(out);
}

Tensor mvlgamma_native(const Tensor& self, int64_t p) {
    if (p < 1) {
        TP_THROW(RuntimeError, "multigammaln requires p >= 1, but got ", p);
    }
    // sum_i lgamma(x - i/2) plus the pi normalization of the multivariate
    // gamma: p*(p-1)/4 * log(pi).
    Tensor acc = ops::lgamma(self);
    for (int64_t i = 1; i < p; ++i) {
        acc = ops::add(acc, ops::lgamma(ops::sub(self, Scalar(i / 2.0))));
    }
    const double pi_norm = static_cast<double>(p) * (p - 1) / 4.0 *
                           std::log(std::acos(-1.0));
    return ops::add(acc, Scalar(pi_norm));
}

Tensor conj_physical_native(const Tensor& self) {
    // Real dtypes are self-conjugate; skip the clone round trip.
    if (!isComplexType(self.dtype())) return self;
    return ops::clone(ops::conj(self), kContiguous);
}

Tensor& conj_physical__native(Tensor& self) {
    ops::copy_(self, ops::conj(self));
    return self;
}

Tensor& negative__native(Tensor& self) {
    return ops::neg_(self);
}

// ---------------------------------------------------------------------------
// Special functions: the CPU kernels hold the exact implementations; other
// devices use the backend-independent fallback until device kernels land.
// ---------------------------------------------------------------------------
} // namespace composite

namespace cpu {
std::tuple<Tensor, Tensor> frexp_cpu(const Tensor& self);
Tensor igamma_cpu(const Tensor& a, const Tensor& x);
Tensor igammac_cpu(const Tensor& a, const Tensor& x);
} // namespace cpu

namespace composite {

std::tuple<Tensor, Tensor> frexp_native(const Tensor& self) {
    const Tensor s = self.device().is_cpu() ? self : self.to(Device(DeviceType::CPU));
    auto r = cpu::frexp_cpu(s);
    return {std::get<0>(r).to(self.device()), std::get<1>(r).to(self.device())};
}

Tensor igamma_native(const Tensor& a, const Tensor& x) {
    if (a.device().is_cpu() && x.device().is_cpu()) {
        return cpu::igamma_cpu(a, x);
    }
    return cpu::igamma_cpu(a.to(Device(DeviceType::CPU)),
                           x.to(Device(DeviceType::CPU))).to(a.device());
}

Tensor igammac_native(const Tensor& a, const Tensor& x) {
    if (a.device().is_cpu() && x.device().is_cpu()) {
        return cpu::igammac_cpu(a, x);
    }
    return cpu::igammac_cpu(a.to(Device(DeviceType::CPU)),
                            x.to(Device(DeviceType::CPU))).to(a.device());
}

// Tensor repeats specify one repetition count per selected input element.
// The backend builds the flat source-index list used by index_select.
Tensor repeat_interleave_tensor_native(const Tensor& self, const Tensor& repeats,
                                       std::optional<int64_t> dim_opt,
                                       std::optional<int64_t> output_size) {
    Tensor input = self;
    int64_t dim = 0;
    if (!dim_opt.has_value()) {
        input = input.flatten();
    } else {
        dim = wrap_dim(dim_opt.value(), input.dim());
    }
    if (input.dim() == 0) {
        TP_THROW(RuntimeError,
                 "repeat_interleave(): dimension required for scalar repeats");
    }
    const int64_t d_size = input.size(dim);
    Tensor rep = repeats;
    if (rep.dim() == 0 || (rep.dim() == 1 && rep.numel() == 1)) {
        rep = rep.reshape({1}).expand({d_size});
    } else if (rep.dim() == 1) {
        if (rep.numel() != d_size) {
            TP_THROW(RuntimeError,
                     "repeats must have the same size as input along dim, but got repeats.size(0) = ",
                     rep.numel(), " and input.size(", dim, ") = ", d_size);
        }
    } else {
        TP_THROW(RuntimeError, "repeats must be 0-dim or 1-dim tensor");
    }
    Tensor index = ops::repeat_interleave(rep, output_size);
    if (index.dtype() != DType::Int32 && index.dtype() != DType::Int64) {
        index = index.to(DType::Int64);
    }
    return ops::index_select(input, dim, index);
}

// Scalar repeats add a repetition axis, expand it, materialize the values,
// then merge that axis back into the selected input dimension.
Tensor repeat_interleave_int_native(const Tensor& self, int64_t repeats,
                                    std::optional<int64_t> dim_opt,
                                    std::optional<int64_t> output_size) {
    Tensor input = dim_opt.has_value() ? self : self.flatten();
    const int64_t dim = wrap_dim(dim_opt.value_or(0), self.dim());
    if (repeats < 0) {
        TP_THROW(RuntimeError, "repeats can not be negative");
    }

    input = input.unsqueeze(dim + 1);
    std::vector<int64_t> expand_shape =
        static_cast<std::vector<int64_t>>(input.shape());
    expand_shape[dim + 1] = repeats;
    input = input.expand(expand_shape);

    const int64_t calculated_size = repeats * expand_shape[dim];
    if (output_size.has_value() && *output_size != calculated_size) {
        TP_THROW(RuntimeError, "allocated size does not match required size");
    }
    return input.clone(kContiguous).flatten(dim, dim + 1);
}

TENSORPLAY_LIBRARY_IMPL(Composite, PointwiseComposite) {
    m.impl("xlogy", xlogy_native);
    m.impl("xlogy_", xlogy__native);
    m.impl("xlog1py", xlog1py_native);
    m.impl("xlog1py_", xlog1py__native);
    m.impl("ldexp", ldexp_native);
    m.impl("ldexp_", ldexp__native);
    m.impl("fmax", fmax_native);
    m.impl("fmin", fmin_native);
    m.impl("float_power", float_power_native);
    m.impl("mvlgamma", mvlgamma_native);
    m.impl("conj_physical", conj_physical_native);
    m.impl("conj_physical_", conj_physical__native);
    m.impl("negative_", negative__native);
    m.impl("frexp", frexp_native);
    m.impl("igamma", igamma_native);
    m.impl("igammac", igammac_native);
    // arctan2/arctan2_ route to the native atan2 kernels; the composite
    // wrappers above are no longer registered for them.
    m.impl("repeat_interleave.self_Tensor", repeat_interleave_tensor_native);
    m.impl("repeat_interleave.self_int", repeat_interleave_int_native);
}

} // namespace composite
} // namespace tensorplay

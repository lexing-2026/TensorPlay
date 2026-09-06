// Backend-neutral out=, in-place and alias spellings that had no kernel.
//
// Each entry computes its value through the operator that already carries a
// native kernel on every device and then writes it where the schema says it
// belongs.  The out= wrappers keep the destination the caller supplied: the
// buffer is resized only when the produced value does not already fit, and the
// values are written into that storage, so a view of the destination observes
// the result and its address does not move.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

Tensor& write_out(Tensor& out, const Tensor& value) {
    if (!out.defined()) {
        out = value;
        return out;
    }
    const auto target = static_cast<std::vector<int64_t>>(value.shape());
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    out.copy_(value);
    return out;
}

}  // namespace

// ---------------------------------------------------------------- comparison

#define TP_COMPARE_OUT(name)                                                   \
    Tensor& name##_scalar_out_native(const Tensor& self, Scalar other,         \
                                     Tensor& out) {                            \
        return write_out(out, ops::name(self, other));                         \
    }                                                                          \
    Tensor& name##_tensor_out_native(const Tensor& self, const Tensor& other,  \
                                     Tensor& out) {                            \
        return write_out(out, ops::name(self, other));                         \
    }

TP_COMPARE_OUT(eq)
TP_COMPARE_OUT(ne)
TP_COMPARE_OUT(lt)
TP_COMPARE_OUT(le)
TP_COMPARE_OUT(gt)
TP_COMPARE_OUT(ge)
#undef TP_COMPARE_OUT

// ------------------------------------------------------------------- binary

Tensor& pow_tensor_tensor_out_native(const Tensor& self, const Tensor& exponent,
                                     Tensor& out) {
    return write_out(out, ops::pow(self, exponent));
}

Tensor& pow_scalar_out_native(Scalar self, const Tensor& exponent, Tensor& out) {
    return write_out(out, ops::pow(self, exponent));
}

Tensor& pow_tensor_scalar_out_native(const Tensor& self, Scalar exponent,
                                     Tensor& out) {
    return write_out(out, ops::pow(self, exponent));
}

Tensor& fmod_scalar_out_native(const Tensor& self, Scalar other, Tensor& out) {
    return write_out(out, ops::fmod(self, other));
}

Tensor& fmod_tensor_out_native(const Tensor& self, const Tensor& other,
                               Tensor& out) {
    return write_out(out, ops::fmod(self, other));
}

Tensor& remainder_scalar_out_native(const Tensor& self, Scalar other,
                                    Tensor& out) {
    return write_out(out, ops::remainder(self, other));
}

Tensor& remainder_tensor_out_native(const Tensor& self, const Tensor& other,
                                    Tensor& out) {
    return write_out(out, ops::remainder(self, other));
}

Tensor& arctan2_out_native(const Tensor& self, const Tensor& other,
                           Tensor& out) {
    return write_out(out, ops::atan2(self, other));
}

Tensor& copysign_out_native(const Tensor& self, const Tensor& other,
                            Tensor& out) {
    return write_out(out, ops::copysign(self, other));
}

Tensor& div_out_native(const Tensor& self, const Tensor& other, Tensor& out) {
    return write_out(out, ops::div(self, other));
}

Tensor& div_out_mode_native(const Tensor& self, const Tensor& other,
                            std::optional<std::string> rounding_mode,
                            Tensor& out) {
    return write_out(out, ops::div(self, other, rounding_mode));
}

Tensor& true_divide_out_native(const Tensor& self, const Tensor& other,
                               Tensor& out) {
    return write_out(out, ops::true_divide(self, other));
}

Tensor& mul_out_native(const Tensor& self, const Tensor& other, Tensor& out) {
    return write_out(out, ops::mul(self, other));
}

Tensor& sub_out_native(const Tensor& self, const Tensor& other, Scalar alpha,
                       Tensor& out) {
    return write_out(out, ops::sub(self, other, alpha));
}

Tensor& where_self_out_native(const Tensor& condition, const Tensor& self,
                              const Tensor& other, Tensor& out) {
    return write_out(out, ops::where(condition, self, other));
}

// ---------------------------------------------------------------------- isin

Tensor& isin_tensor_tensor_out_native(const Tensor& elements,
                                      const Tensor& test_elements,
                                      bool assume_unique, bool invert,
                                      Tensor& out) {
    return write_out(out,
                     ops::isin(elements, test_elements, assume_unique, invert));
}

Tensor& isin_tensor_scalar_out_native(const Tensor& elements,
                                      Scalar test_element, bool assume_unique,
                                      bool invert, Tensor& out) {
    return write_out(out,
                     ops::isin(elements, test_element, assume_unique, invert));
}

Tensor& isin_scalar_tensor_out_native(Scalar element,
                                      const Tensor& test_elements,
                                      bool assume_unique, bool invert,
                                      Tensor& out) {
    return write_out(out,
                     ops::isin(element, test_elements, assume_unique, invert));
}

// ------------------------------------------------------- indexing / scatter

Tensor index_fill_int_scalar_native(const Tensor& self, int64_t dim,
                                    const Tensor& index, Scalar value) {
    return ops::index_fill(self, dim, index, value);
}

Tensor& index_fill__int_scalar_native(Tensor& self, int64_t dim,
                                      const Tensor& index, Scalar value) {
    self.copy_(ops::index_fill(self, dim, index, value));
    return self;
}

Tensor index_fill_int_tensor_native(const Tensor& self, int64_t dim,
                                    const Tensor& index, const Tensor& value) {
    return ops::index_fill(self, dim, index, value);
}

Tensor& index_fill__int_tensor_native(Tensor& self, int64_t dim,
                                      const Tensor& index,
                                      const Tensor& value) {
    self.copy_(ops::index_fill(self, dim, index, value));
    return self;
}

Tensor& scatter_reduce__two_native(Tensor& self, int64_t dim,
                                   const Tensor& index, const Tensor& src,
                                   std::string reduce, bool include_self) {
    self.copy_(ops::scatter_reduce(self, dim, index, src, reduce, include_self));
    return self;
}

// -------------------------------------------------------------- alias copies

// The *_copy spellings answer the same value as their aliasing counterparts
// but own their storage, so a later write to either tensor cannot be observed
// through the other.
Tensor view_as_real_copy_native(const Tensor& self) {
    return ops::clone(ops::view_as_real(self), kContiguous);
}

Tensor view_as_complex_copy_native(const Tensor& self) {
    return ops::clone(ops::view_as_complex(self), kContiguous);
}

Tensor indices_copy_native(const Tensor& self) {
    return ops::clone(ops::indices(self), kContiguous);
}

Tensor values_copy_native(const Tensor& self) {
    return ops::clone(ops::values(self), kContiguous);
}

// ------------------------------------------------------------------- linalg

Tensor linalg_matmul_native(const Tensor& self, const Tensor& other) {
    return ops::matmul(self, other);
}

Tensor& linalg_matmul_out_native(const Tensor& self, const Tensor& other,
                                 Tensor& out) {
    return write_out(out, ops::matmul(self, other));
}

std::tuple<Tensor, Tensor> slogdet_out_native(const Tensor& self,
                                              Tensor& sign,
                                              Tensor& logabsdet) {
    auto parts = ops::linalg_slogdet(self);
    write_out(sign, std::get<0>(parts));
    write_out(logabsdet, std::get<1>(parts));
    return {sign, logabsdet};
}

}  // namespace composite

TENSORPLAY_LIBRARY_IMPL(Composite, OutAndAliasVariants) {
    m.impl("eq.Scalar_out", composite::eq_scalar_out_native);
    m.impl("eq.Tensor_out", composite::eq_tensor_out_native);
    m.impl("ne.Scalar_out", composite::ne_scalar_out_native);
    m.impl("ne.Tensor_out", composite::ne_tensor_out_native);
    m.impl("lt.Scalar_out", composite::lt_scalar_out_native);
    m.impl("lt.Tensor_out", composite::lt_tensor_out_native);
    m.impl("le.Scalar_out", composite::le_scalar_out_native);
    m.impl("le.Tensor_out", composite::le_tensor_out_native);
    m.impl("gt.Scalar_out", composite::gt_scalar_out_native);
    m.impl("gt.Tensor_out", composite::gt_tensor_out_native);
    m.impl("ge.Scalar_out", composite::ge_scalar_out_native);
    m.impl("ge.Tensor_out", composite::ge_tensor_out_native);

    m.impl("pow.Tensor_Tensor_out", composite::pow_tensor_tensor_out_native);
    m.impl("pow.Scalar_out", composite::pow_scalar_out_native);
    m.impl("pow.Tensor_Scalar_out", composite::pow_tensor_scalar_out_native);
    m.impl("fmod.Scalar_out", composite::fmod_scalar_out_native);
    m.impl("fmod.Tensor_out", composite::fmod_tensor_out_native);
    m.impl("remainder.Scalar_out", composite::remainder_scalar_out_native);
    m.impl("remainder.Tensor_out", composite::remainder_tensor_out_native);
    m.impl("arctan2.out", composite::arctan2_out_native);
    m.impl("copysign.out", composite::copysign_out_native);

    m.impl("div.out", composite::div_out_native);
    m.impl("div.out_mode", composite::div_out_mode_native);
    m.impl("divide.out", composite::div_out_native);
    m.impl("divide.out_mode", composite::div_out_mode_native);
    m.impl("true_divide.out", composite::true_divide_out_native);
    m.impl("mul.out", composite::mul_out_native);
    m.impl("multiply.out", composite::mul_out_native);
    m.impl("sub.out", composite::sub_out_native);
    m.impl("subtract.out", composite::sub_out_native);
    m.impl("where.self_out", composite::where_self_out_native);

    m.impl("isin.Tensor_Tensor_out", composite::isin_tensor_tensor_out_native);
    m.impl("isin.Tensor_Scalar_out", composite::isin_tensor_scalar_out_native);
    m.impl("isin.Scalar_Tensor_out", composite::isin_scalar_tensor_out_native);

    m.impl("index_fill.int_Scalar", composite::index_fill_int_scalar_native);
    m.impl("index_fill_.int_Scalar", composite::index_fill__int_scalar_native);
    m.impl("index_fill.int_Tensor", composite::index_fill_int_tensor_native);
    m.impl("index_fill_.int_Tensor", composite::index_fill__int_tensor_native);
    m.impl("scatter_reduce_.two", composite::scatter_reduce__two_native);

    m.impl("view_as_real_copy", composite::view_as_real_copy_native);
    m.impl("view_as_complex_copy", composite::view_as_complex_copy_native);
    m.impl("indices_copy", composite::indices_copy_native);
    m.impl("values_copy", composite::values_copy_native);

    m.impl("linalg_matmul", composite::linalg_matmul_native);
    m.impl("linalg_matmul.out", composite::linalg_matmul_out_native);
    m.impl("slogdet.out", composite::slogdet_out_native);
}

}  // namespace tensorplay

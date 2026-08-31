// Composite kernels: the dunder spellings of the bitwise binary family --
// and/or/xor and left/right shift, each in a Tensor and a Scalar overload,
// plus the corresponding in-place forms.
//
//   out-of-place: alias of the bitwise op; broadcast shapes, promote dtypes,
//                 result is a fresh tensor. Dtype constraints (integral and
//                 boolean only) and their error messages are enforced by the
//                 underlying bitwise kernels.
//   in-place:     compute with the promoted dtype of self and other, then
//                 write the result back through copy_, which casts into
//                 self's dtype; self keeps its shape and dtype.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

// ---------------------------------------------------------------------------
// __and__ / __iand__
// ---------------------------------------------------------------------------

Tensor and_tensor_native(const Tensor& self, const Tensor& other) {
    return ops::bitwise_and(self, other);
}

Tensor and_scalar_native(const Tensor& self, const Scalar& other) {
    return ops::bitwise_and(self, other);
}

Tensor& iand_tensor_native(Tensor& self, const Tensor& other) {
    ops::copy_(self, ops::bitwise_and(self, other));
    return self;
}

Tensor& iand_scalar_native(Tensor& self, const Scalar& other) {
    ops::copy_(self, ops::bitwise_and(self, other));
    return self;
}

// ---------------------------------------------------------------------------
// __or__ / __ior__
// ---------------------------------------------------------------------------

Tensor or_tensor_native(const Tensor& self, const Tensor& other) {
    return ops::bitwise_or(self, other);
}

Tensor or_scalar_native(const Tensor& self, const Scalar& other) {
    return ops::bitwise_or(self, other);
}

Tensor& ior_tensor_native(Tensor& self, const Tensor& other) {
    ops::copy_(self, ops::bitwise_or(self, other));
    return self;
}

Tensor& ior_scalar_native(Tensor& self, const Scalar& other) {
    ops::copy_(self, ops::bitwise_or(self, other));
    return self;
}

// ---------------------------------------------------------------------------
// __xor__ / __ixor__
// ---------------------------------------------------------------------------

Tensor xor_tensor_native(const Tensor& self, const Tensor& other) {
    return ops::bitwise_xor(self, other);
}

Tensor xor_scalar_native(const Tensor& self, const Scalar& other) {
    return ops::bitwise_xor(self, other);
}

Tensor& ixor_tensor_native(Tensor& self, const Tensor& other) {
    ops::copy_(self, ops::bitwise_xor(self, other));
    return self;
}

Tensor& ixor_scalar_native(Tensor& self, const Scalar& other) {
    ops::copy_(self, ops::bitwise_xor(self, other));
    return self;
}

// ---------------------------------------------------------------------------
// __lshift__ / __ilshift__
// ---------------------------------------------------------------------------

Tensor lshift_tensor_native(const Tensor& self, const Tensor& other) {
    return ops::bitwise_left_shift(self, other);
}

Tensor lshift_scalar_native(const Tensor& self, const Scalar& other) {
    return ops::bitwise_left_shift(self, other);
}

Tensor& ilshift_tensor_native(Tensor& self, const Tensor& other) {
    ops::copy_(self, ops::bitwise_left_shift(self, other));
    return self;
}

Tensor& ilshift_scalar_native(Tensor& self, const Scalar& other) {
    ops::copy_(self, ops::bitwise_left_shift(self, other));
    return self;
}

// ---------------------------------------------------------------------------
// __rshift__ / __irshift__
// ---------------------------------------------------------------------------

Tensor rshift_tensor_native(const Tensor& self, const Tensor& other) {
    return ops::bitwise_right_shift(self, other);
}

Tensor rshift_scalar_native(const Tensor& self, const Scalar& other) {
    return ops::bitwise_right_shift(self, other);
}

Tensor& irshift_tensor_native(Tensor& self, const Tensor& other) {
    ops::copy_(self, ops::bitwise_right_shift(self, other));
    return self;
}

Tensor& irshift_scalar_native(Tensor& self, const Scalar& other) {
    ops::copy_(self, ops::bitwise_right_shift(self, other));
    return self;
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(Composite, DunderBinaryComposite) {
    m.impl("__and__.Tensor", and_tensor_native);
    m.impl("__and__.Scalar", and_scalar_native);
    m.impl("__iand__.Tensor", iand_tensor_native);
    m.impl("__iand__.Scalar", iand_scalar_native);

    m.impl("__or__.Tensor", or_tensor_native);
    m.impl("__or__.Scalar", or_scalar_native);
    m.impl("__ior__.Tensor", ior_tensor_native);
    m.impl("__ior__.Scalar", ior_scalar_native);

    m.impl("__xor__.Tensor", xor_tensor_native);
    m.impl("__xor__.Scalar", xor_scalar_native);
    m.impl("__ixor__.Tensor", ixor_tensor_native);
    m.impl("__ixor__.Scalar", ixor_scalar_native);

    m.impl("__lshift__.Tensor", lshift_tensor_native);
    m.impl("__lshift__.Scalar", lshift_scalar_native);
    m.impl("__ilshift__.Tensor", ilshift_tensor_native);
    m.impl("__ilshift__.Scalar", ilshift_scalar_native);

    m.impl("__rshift__.Tensor", rshift_tensor_native);
    m.impl("__rshift__.Scalar", rshift_scalar_native);
    m.impl("__irshift__.Tensor", irshift_tensor_native);
    m.impl("__irshift__.Scalar", irshift_scalar_native);
}

} // namespace composite
} // namespace tensorplay

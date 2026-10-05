// Bitwise operator family - CPU kernels.
//
// The family covers bitwise_not, bitwise_and/or/xor, bitwise_left_shift and
// bitwise_right_shift. All of them are defined on integral and boolean data
// only; boolean operands apply the corresponding logical operation.
//
// Variants:
//   .Tensor          elementwise with broadcasting
//   .Scalar          constant operand folded into the tensor's dtype
//   .Scalar_Tensor   scalar on the left, tensor on the right; the scalar is
//                    materialized as a 0-dim tensor in the tensor's dtype
//                    (wrapped-number semantics: the tensor dtype wins)
//   *_out            write the broadcasted result into a caller-owned tensor
//
// Tensors carrying an active transform level (vmap) hold their payload in the
// transform wrapper, so every entry point rejects them instead of touching
// storage directly; batch rules live in the transform layer.
//
// The vectorized integral cores are tier-compiled (BitwiseKernelsImpl.cpp)
// and reached through the stubs declared in cpu/BitwiseKernels.h; this TU
// keeps the base-tier plumbing: checks, broadcasting, dtype promotion, the
// boolean special cases and the dispatcher registrations.

#include "Tensor.h"
#include "TensorImpl.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "DType.h"
#include "Utils.h"
#include "Exception.h"
#include "Parallel.h"
#include "TypePromotion.h"
#include "TypeProperties.h"
#include "cpu/BitwiseKernels.h"

#include <cstdint>
#include <vector>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

inline void bitwise_check_cpu(const Tensor& t, const char* name) {
    if (t.unsafeGetTensorImpl() && t.unsafeGetTensorImpl()->is_batched()) {
        TP_THROW(NotImplementedError, name,
                 " is not supported for tensors inside an active transform "
                 "(vmap/grad) layer");
    }
    DType d = t.dtype();
    if (d == DType::Bool || isIntegralType(d)) return;
    TP_THROW(TypeError, name, ": only integral and boolean types are supported");
}

// Boolean operands carry 0/1 in each byte and apply the matching logical
// operation; only and/or/xor reach a boolean tensor (shifts refuse them).
inline bool bitwise_bool_apply(int op, uint8_t a, uint8_t b) {
    switch (op) {
        case static_cast<int>(BitwiseOp::kAnd): return a & b;
        case static_cast<int>(BitwiseOp::kOr): return a | b;
        case static_cast<int>(BitwiseOp::kXor): return a ^ b;
        default:
            TP_THROW(TypeError, "bitwise: unsupported operation on bool");
    }
}

Tensor bitwise_binary_cpu(const Tensor& a_in, const Tensor& b_in,
                          BitwiseOp op, const char* name) {
    bitwise_check_cpu(a_in, name);
    bitwise_check_cpu(b_in, name);
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    // A zero-dim operand takes part by category only, as it does in every
    // other binary operation: it widens a tensor of a lower category but never
    // one of its own.
    DType dt = native::result_type(a_in, b_in);
    if (dt != DType::Bool && !isIntegralType(dt)) {
        TP_THROW(TypeError, name, ": only integral and boolean types are supported");
    }
    Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt)).expand(out_shape).contiguous();
    Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt)).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    int64_t n = out.numel();
    if (dt == DType::Bool) {
        const bool* ap = ac.data_ptr<bool>();
        const bool* bp = bc.data_ptr<bool>();
        bool* dp = out.data_ptr<bool>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i)
                dp[i] = bitwise_bool_apply(static_cast<int>(op),
                                           static_cast<uint8_t>(ap[i]),
                                           static_cast<uint8_t>(bp[i]));
        });
        return out;
    }
    bitwise_binary_stub(DeviceType::CPU, ac.data_ptr(), bc.data_ptr(),
                        out.data_ptr(), n, static_cast<int>(dt),
                        static_cast<int>(op));
    return out;
}

Tensor bitwise_scalar_cpu(const Tensor& self_in, Scalar other, BitwiseOp op,
                          const char* name) {
    // An integer scalar moves a boolean tensor to int64; a scalar of the
    // tensor's own category leaves its type alone.
    const DType dt = result_type(other, self_in.dtype());
    if (dt != self_in.dtype()) return bitwise_scalar_cpu(self_in.to(dt), other, op, name);
    bitwise_check_cpu(self_in, name);
    Tensor sc = self_in.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self_in.shape()),
                               self_in.dtype(), self_in.device());
    int64_t n = out.numel();
    if (self_in.dtype() == DType::Bool) {
        const bool* sp = sc.data_ptr<bool>();
        uint8_t o = other.to<bool>() ? 1 : 0;
        bool* dp = out.data_ptr<bool>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i)
                dp[i] = bitwise_bool_apply(static_cast<int>(op),
                                           static_cast<uint8_t>(sp[i]), o);
        });
        return out;
    }
    bitwise_scalar_stub(DeviceType::CPU, sc.data_ptr(), other.to<int64_t>(),
                        out.data_ptr(), n, static_cast<int>(self_in.dtype()),
                        static_cast<int>(op));
    return out;
}

template <bool kLeft>
Tensor bitwise_shift_scalar_cpu(const Tensor& self_in, Scalar other, const char* name) {
    const DType dt = result_type(other, self_in.dtype());
    if (dt != self_in.dtype()) return bitwise_shift_scalar_cpu<kLeft>(self_in.to(dt), other, name);
    bitwise_check_cpu(self_in, name);
    if (self_in.dtype() == DType::Bool) {
        TP_THROW(TypeError, name, ": unsupported dtype");
    }
    const int64_t shift = other.to<int64_t>();
    Tensor sc = self_in.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self_in.shape()),
                               self_in.dtype(), self_in.device());
    bitwise_scalar_stub(DeviceType::CPU, sc.data_ptr(), shift, out.data_ptr(),
                        out.numel(), static_cast<int>(self_in.dtype()),
                        static_cast<int>(kLeft ? BitwiseOp::kLshift
                                               : BitwiseOp::kRshift));
    return out;
}

Tensor bitwise_not_cpu(const Tensor& self) {
    bitwise_check_cpu(self, "bitwise_not");
    Tensor sc = self.contiguous();
    Tensor out = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()),
                               self.dtype(), self.device());
    int64_t n = out.numel();
    if (self.dtype() == DType::Bool) {
        const bool* sp = sc.data_ptr<bool>();
        bool* dp = out.data_ptr<bool>();
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) dp[i] = !sp[i];
        });
        return out;
    }
    bitwise_not_stub(DeviceType::CPU, sc.data_ptr(), out.data_ptr(), n,
                     static_cast<int>(self.dtype()));
    return out;
}

template <bool kLeft>
Tensor bitwise_shift_tensor_cpu(const Tensor& a_in, const Tensor& b_in, const char* name) {
    bitwise_check_cpu(a_in, name);
    bitwise_check_cpu(b_in, name);
    std::vector<int64_t> out_shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(a_in.shape()),
        static_cast<std::vector<int64_t>>(b_in.shape()));
    // A zero-dim operand takes part by category only, as it does in every
    // other binary operation: it widens a tensor of a lower category but never
    // one of its own.
    DType dt = native::result_type(a_in, b_in);
    if (dt != DType::Bool && !isIntegralType(dt)) {
        TP_THROW(TypeError, name, ": only integral and boolean types are supported");
    }
    Tensor ac = (a_in.dtype() == dt ? a_in : a_in.to(dt)).expand(out_shape).contiguous();
    Tensor bc = (b_in.dtype() == dt ? b_in : b_in.to(dt)).expand(out_shape).contiguous();
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    if (dt == DType::Bool) {
        TP_THROW(TypeError, name, ": unsupported dtype");
    }
    bitwise_binary_stub(DeviceType::CPU, ac.data_ptr(), bc.data_ptr(),
                        out.data_ptr(), out.numel(), static_cast<int>(dt),
                        static_cast<int>(kLeft ? BitwiseOp::kLshift
                                               : BitwiseOp::kRshift));
    return out;
}

// --- Named entry points registered with the dispatcher ----------------------

Tensor bitwise_and_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_binary_cpu(a, b, BitwiseOp::kAnd, "bitwise_and");
}
Tensor bitwise_or_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_binary_cpu(a, b, BitwiseOp::kOr, "bitwise_or");
}
Tensor bitwise_xor_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_binary_cpu(a, b, BitwiseOp::kXor, "bitwise_xor");
}
Tensor bitwise_and_scalar_cpu(const Tensor& a, const Scalar& b) {
    return bitwise_scalar_cpu(a, b, BitwiseOp::kAnd, "bitwise_and");
}
Tensor bitwise_or_scalar_cpu(const Tensor& a, const Scalar& b) {
    return bitwise_scalar_cpu(a, b, BitwiseOp::kOr, "bitwise_or");
}
Tensor bitwise_xor_scalar_cpu(const Tensor& a, const Scalar& b) {
    return bitwise_scalar_cpu(a, b, BitwiseOp::kXor, "bitwise_xor");
}
Tensor bitwise_lshift_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_shift_tensor_cpu<true>(a, b, "bitwise_left_shift");
}
Tensor bitwise_rshift_tensor_cpu(const Tensor& a, const Tensor& b) {
    return bitwise_shift_tensor_cpu<false>(a, b, "bitwise_right_shift");
}
Tensor bitwise_lshift_scalar_cpu(const Tensor& a, const Scalar& b) {
    return bitwise_shift_scalar_cpu<true>(a, b, "bitwise_left_shift");
}
Tensor bitwise_rshift_scalar_cpu(const Tensor& a, const Scalar& b) {
    return bitwise_shift_scalar_cpu<false>(a, b, "bitwise_right_shift");
}

// Scalar-first variants: materialize the scalar as a 0-dim tensor in the
// type the pair answers in, then run the plain tensor-tensor kernel.  A
// floating or complex scalar would move the result out of the integral
// domain, so it is refused up front.

inline void bitwise_scalar_check_cpu(Scalar self, const char* name) {
    if (self.isBoolean() || self.isIntegral()) return;
    TP_THROW(TypeError, name,
             ": only integral and boolean scalar operands are supported");
}

Tensor bitwise_and_scalar_tensor_cpu(const Scalar& self, const Tensor& other) {
    bitwise_check_cpu(other, "bitwise_and");
    bitwise_scalar_check_cpu(self, "bitwise_and");
    Tensor wrapped = Tensor::full({}, self, result_type(self, other.dtype()), other.device());
    return bitwise_binary_cpu(wrapped, other, BitwiseOp::kAnd, "bitwise_and");
}
Tensor bitwise_or_scalar_tensor_cpu(const Scalar& self, const Tensor& other) {
    bitwise_check_cpu(other, "bitwise_or");
    bitwise_scalar_check_cpu(self, "bitwise_or");
    Tensor wrapped = Tensor::full({}, self, result_type(self, other.dtype()), other.device());
    return bitwise_binary_cpu(wrapped, other, BitwiseOp::kOr, "bitwise_or");
}
Tensor bitwise_xor_scalar_tensor_cpu(const Scalar& self, const Tensor& other) {
    bitwise_check_cpu(other, "bitwise_xor");
    bitwise_scalar_check_cpu(self, "bitwise_xor");
    Tensor wrapped = Tensor::full({}, self, result_type(self, other.dtype()), other.device());
    return bitwise_binary_cpu(wrapped, other, BitwiseOp::kXor, "bitwise_xor");
}
Tensor bitwise_lshift_scalar_tensor_cpu(const Scalar& self, const Tensor& other) {
    bitwise_check_cpu(other, "bitwise_left_shift");
    bitwise_scalar_check_cpu(self, "bitwise_left_shift");
    Tensor wrapped = Tensor::full({}, self, result_type(self, other.dtype()), other.device());
    return bitwise_shift_tensor_cpu<true>(wrapped, other, "bitwise_left_shift");
}
Tensor bitwise_rshift_scalar_tensor_cpu(const Scalar& self, const Tensor& other) {
    bitwise_check_cpu(other, "bitwise_right_shift");
    bitwise_scalar_check_cpu(self, "bitwise_right_shift");
    Tensor wrapped = Tensor::full({}, self, result_type(self, other.dtype()), other.device());
    return bitwise_shift_tensor_cpu<false>(wrapped, other, "bitwise_right_shift");
}

// Out variants: compute into a fresh buffer, then transfer into the
// caller-owned tensor.  Matching shapes copy in place; otherwise the output
// adopts the result's metadata.

Tensor& bitwise_assign_out_cpu(Tensor& out, const Tensor& result) {
    if (static_cast<std::vector<int64_t>>(out.shape()) ==
        static_cast<std::vector<int64_t>>(result.shape())) {
        out.copy_(result);
    } else {
        out.unsafeGetTensorImpl()->copy_metadata_from(*result.unsafeGetTensorImpl());
    }
    return out;
}

Tensor& bitwise_and_tensor_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_and_tensor_cpu(a, b));
}
Tensor& bitwise_or_tensor_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_or_tensor_cpu(a, b));
}
Tensor& bitwise_xor_tensor_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_xor_tensor_cpu(a, b));
}
Tensor& bitwise_and_scalar_out_cpu(const Tensor& a, const Scalar& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_and_scalar_cpu(a, b));
}
Tensor& bitwise_or_scalar_out_cpu(const Tensor& a, const Scalar& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_or_scalar_cpu(a, b));
}
Tensor& bitwise_xor_scalar_out_cpu(const Tensor& a, const Scalar& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_xor_scalar_cpu(a, b));
}
Tensor& bitwise_lshift_tensor_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_lshift_tensor_cpu(a, b));
}
Tensor& bitwise_rshift_tensor_out_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_rshift_tensor_cpu(a, b));
}
Tensor& bitwise_lshift_scalar_out_cpu(const Tensor& a, const Scalar& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_lshift_scalar_cpu(a, b));
}
Tensor& bitwise_rshift_scalar_out_cpu(const Tensor& a, const Scalar& b, Tensor& out) {
    return bitwise_assign_out_cpu(out, bitwise_rshift_scalar_cpu(a, b));
}

} // anonymous namespace

DEFINE_DISPATCH(bitwise_binary_stub);
DEFINE_DISPATCH(bitwise_scalar_stub);
DEFINE_DISPATCH(bitwise_not_stub);

TENSORPLAY_LIBRARY_IMPL(CPU, BitwiseKernels) {
    m.impl("bitwise_not", bitwise_not_cpu);
    m.impl("bitwise_and.Tensor", bitwise_and_tensor_cpu);
    m.impl("bitwise_or.Tensor", bitwise_or_tensor_cpu);
    m.impl("bitwise_xor.Tensor", bitwise_xor_tensor_cpu);
    m.impl("bitwise_and.Scalar", bitwise_and_scalar_cpu);
    m.impl("bitwise_or.Scalar", bitwise_or_scalar_cpu);
    m.impl("bitwise_xor.Scalar", bitwise_xor_scalar_cpu);
    m.impl("bitwise_left_shift.Tensor", bitwise_lshift_tensor_cpu);
    m.impl("bitwise_right_shift.Tensor", bitwise_rshift_tensor_cpu);
    m.impl("bitwise_left_shift.Tensor_Scalar", bitwise_lshift_scalar_cpu);
    m.impl("bitwise_right_shift.Tensor_Scalar", bitwise_rshift_scalar_cpu);
    // Scalar-first variants
    m.impl("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor_cpu);
    m.impl("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor_cpu);
    m.impl("bitwise_xor.Scalar_Tensor", bitwise_xor_scalar_tensor_cpu);
    m.impl("bitwise_left_shift.Scalar_Tensor", bitwise_lshift_scalar_tensor_cpu);
    m.impl("bitwise_right_shift.Scalar_Tensor", bitwise_rshift_scalar_tensor_cpu);
    // Out variants
    m.impl("bitwise_and.Tensor_out", bitwise_and_tensor_out_cpu);
    m.impl("bitwise_or.Tensor_out", bitwise_or_tensor_out_cpu);
    m.impl("bitwise_xor.Tensor_out", bitwise_xor_tensor_out_cpu);
    m.impl("bitwise_left_shift.Tensor_out", bitwise_lshift_tensor_out_cpu);
    m.impl("bitwise_right_shift.Tensor_out", bitwise_rshift_tensor_out_cpu);
    m.impl("bitwise_and.Scalar_out", bitwise_and_scalar_out_cpu);
    m.impl("bitwise_or.Scalar_out", bitwise_or_scalar_out_cpu);
    m.impl("bitwise_xor.Scalar_out", bitwise_xor_scalar_out_cpu);
    m.impl("bitwise_left_shift.Tensor_Scalar_out", bitwise_lshift_scalar_out_cpu);
    m.impl("bitwise_right_shift.Tensor_Scalar_out", bitwise_rshift_scalar_out_cpu);
}

} // namespace cpu
} // namespace tensorplay

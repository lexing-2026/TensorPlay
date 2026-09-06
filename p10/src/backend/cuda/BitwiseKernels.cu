// Bitwise operator family - CUDA kernels.
//
// Elementwise operations over integral and boolean data; boolean
// operands apply the corresponding logical operation.  The .Scalar_Tensor
// variants materialize the leading scalar as a 0-dim tensor in the tensor's
// dtype (wrapped-number semantics: the tensor dtype wins); a floating or
// complex scalar would leave the integral domain and is refused up front.
// Out variants compute into a fresh buffer and transfer into the
// caller-owned tensor.
//
// Tensors carrying an active transform level (vmap) hold their payload in the
// transform wrapper, so every entry point rejects them instead of touching
// storage directly; batch rules live in the transform layer.

#include "Tensor.h"
#include "TensorImpl.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "DType.h"
#include "Utils.h"
#include "Exception.h"
#include "TypePromotion.h"
#include "CUDALoops.cuh"

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

inline std::vector<int64_t> shape_of(const Tensor& t) {
    return static_cast<std::vector<int64_t>>(t.shape());
}

#define TENSORPLAY_FORALL_INT_TYPES_CUDA(_) \
    _(uint8_t, UInt8)                       \
    _(int8_t, Int8)                         \
    _(int16_t, Int16)                       \
    _(int32_t, Int32)                       \
    _(int64_t, Int64)                       \
    _(uint16_t, UInt16)                     \
    _(uint32_t, UInt32)                     \
    _(uint64_t, UInt64)

struct BitwiseAnd {
    template <typename T>
    __device__ T operator()(T lhs, T rhs) const {
        return static_cast<T>(lhs & rhs);
    }
};

struct BitwiseOr {
    template <typename T>
    __device__ T operator()(T lhs, T rhs) const {
        return static_cast<T>(lhs | rhs);
    }
};

struct BitwiseXor {
    template <typename T>
    __device__ T operator()(T lhs, T rhs) const {
        return static_cast<T>(lhs ^ rhs);
    }
};

inline void bitwise_check_cuda(const Tensor& t, const char* name) {
    if (t.unsafeGetTensorImpl() && t.unsafeGetTensorImpl()->is_batched()) {
        TP_THROW(NotImplementedError, name,
                 " is not supported for tensors inside an active transform "
                 "(vmap/grad) layer");
    }
    DType d = t.dtype();
    if (d == DType::Bool || isIntegralType(d)) return;
    TP_THROW(TypeError, name, ": only integral and boolean types are supported");
}

template <typename T, bool kLeft>
__device__ __forceinline__ T bitwise_shift_value(T value, T shift) {
    using S = typename std::make_signed<T>::type;
    using U = typename std::make_unsigned<T>::type;
    constexpr U kBits = static_cast<U>(sizeof(T) * 8);
    const bool invalid = static_cast<S>(shift) < 0 || static_cast<U>(shift) >= kBits;
    if constexpr (kLeft) {
        if (invalid) return T(0);
        return static_cast<T>(static_cast<U>(value) << static_cast<U>(shift));
    }
    if (invalid) {
        if constexpr (std::is_signed_v<T>) return value < 0 ? T(-1) : T(0);
        return T(0);
    }
    return static_cast<T>(value >> static_cast<U>(shift));
}

template <typename Pred>
Tensor bitwise_binary_cuda(const Tensor& a_in, const Tensor& b_in, Pred pred, const char* name) {
    bitwise_check_cuda(a_in, name);
    bitwise_check_cuda(b_in, name);
    std::vector<int64_t> out_shape = broadcast_shapes(shape_of(a_in), shape_of(b_in));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (a_in.dtype() == DType::Bool && b_in.dtype() == DType::Bool) dt = DType::Bool;
    if (dt != DType::Bool && !isIntegralType(dt)) {
        TP_THROW(TypeError, name, ": only integral and boolean types are supported");
    }
    Tensor ac = a_in.dtype() == dt ? a_in : a_in.to(dt);
    Tensor bc = b_in.dtype() == dt ? b_in : b_in.to(dt);
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    if (out.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(out)
        .add_input(ac)
        .add_input(bc)
        .build();
    if (dt == DType::Bool) {
        gpu_kernel(iter, [pred] __device__ (bool lhs, bool rhs) -> bool {
            return pred(lhs, rhs);
        });
        return out;
    }
#define TP_BIT_BIN(ctype, name_) \
    case DType::name_: \
        gpu_kernel(iter, [pred] __device__ (ctype lhs, ctype rhs) -> ctype { \
            return pred(lhs, rhs); \
        }); \
        break;
    switch (dt) {
        TENSORPLAY_FORALL_INT_TYPES_CUDA(TP_BIT_BIN)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BIT_BIN
    return out;
}

template <typename Pred>
Tensor bitwise_scalar_cuda(const Tensor& self_in, Scalar other, Pred pred, const char* name) {
    bitwise_check_cuda(self_in, name);
    Tensor out = Tensor::empty(shape_of(self_in), self_in.dtype(), self_in.device());
    if (out.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(out)
        .add_input(self_in)
        .build();
    if (self_in.dtype() == DType::Bool) {
        const bool ov = other.to<bool>();
        gpu_kernel(iter, [pred, ov] __device__ (bool value) -> bool {
            return pred(value, ov);
        });
        return out;
    }
#define TP_BIT_SCALAR(ctype, name_) \
    case DType::name_: { \
        ctype ov = static_cast<ctype>(other.to<int64_t>()); \
        gpu_kernel(iter, [pred, ov] __device__ (ctype value) -> ctype { \
            return pred(value, ov); \
        }); \
        break; \
    }
    switch (self_in.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES_CUDA(TP_BIT_SCALAR)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_BIT_SCALAR
    return out;
}

Tensor bitwise_not_cuda(const Tensor& self) {
    bitwise_check_cuda(self, "bitwise_not");
    Tensor out = Tensor::empty(shape_of(self), self.dtype(), self.device());
    if (out.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(out)
        .add_input(self)
        .build();
    if (self.dtype() == DType::Bool) {
        gpu_kernel(iter, [] __device__ (bool value) -> bool {
            return !value;
        });
        return out;
    }
#define TP_BNOT(ctype, name_) \
    case DType::name_: \
        gpu_kernel(iter, [] __device__ (ctype value) -> ctype { \
            return static_cast<ctype>(~value); \
        }); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES_CUDA(TP_BNOT)
        default: TP_THROW(TypeError, "bitwise_not: unsupported dtype");
    }
#undef TP_BNOT
    return out;
}

template <bool kLeft>
Tensor bitwise_shift_tensor_cuda_impl(const Tensor& a_in, const Tensor& b_in, const char* name) {
    bitwise_check_cuda(a_in, name);
    bitwise_check_cuda(b_in, name);
    std::vector<int64_t> out_shape = broadcast_shapes(shape_of(a_in), shape_of(b_in));
    DType dt = promoteTypes(a_in.dtype(), b_in.dtype());
    if (dt != DType::Bool && !isIntegralType(dt)) {
        TP_THROW(TypeError, name, ": only integral and boolean types are supported");
    }
    Tensor ac = a_in.dtype() == dt ? a_in : a_in.to(dt);
    Tensor bc = b_in.dtype() == dt ? b_in : b_in.to(dt);
    Tensor out = Tensor::empty(out_shape, dt, a_in.device());
    if (out.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(out)
        .add_input(ac)
        .add_input(bc)
        .build();
#define TP_SHIFT_BIN(ctype, name_) \
    case DType::name_: { \
        gpu_kernel(iter, [] __device__ (ctype value, ctype shift) -> ctype { \
            return bitwise_shift_value<ctype, kLeft>(value, shift); \
        }); \
        break; \
    }
    switch (dt) {
        TENSORPLAY_FORALL_INT_TYPES_CUDA(TP_SHIFT_BIN)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_SHIFT_BIN
    return out;
}

template <bool kLeft>
Tensor bitwise_shift_scalar_cuda_impl(const Tensor& a_in, Scalar other, const char* name) {
    bitwise_check_cuda(a_in, name);
    const int64_t shift = other.to<int64_t>();
    Tensor out = Tensor::empty(shape_of(a_in), a_in.dtype(), a_in.device());
    if (out.numel() == 0) return out;
    TensorIterator iter = TensorIteratorConfig()
        .check_all_same_dtype(true)
        .add_output(out)
        .add_input(a_in)
        .build();
#define TP_SHIFT_SCALAR(ctype, name_) \
    case DType::name_: { \
        const ctype sh = static_cast<ctype>(shift); \
        gpu_kernel(iter, [sh] __device__ (ctype value) -> ctype { \
            return bitwise_shift_value<ctype, kLeft>(value, sh); \
        }); \
        break; \
    }
    switch (a_in.dtype()) {
        TENSORPLAY_FORALL_INT_TYPES_CUDA(TP_SHIFT_SCALAR)
        default: TP_THROW(TypeError, name, ": unsupported dtype");
    }
#undef TP_SHIFT_SCALAR
    return out;
}

// --- Named entry points registered with the dispatcher ----------------------

Tensor bitwise_and_tensor_cuda(const Tensor& a, const Tensor& b) {
    return bitwise_binary_cuda(a, b, BitwiseAnd(), "bitwise_and");
}
Tensor bitwise_or_tensor_cuda(const Tensor& a, const Tensor& b) {
    return bitwise_binary_cuda(a, b, BitwiseOr(), "bitwise_or");
}
Tensor bitwise_xor_tensor_cuda(const Tensor& a, const Tensor& b) {
    return bitwise_binary_cuda(a, b, BitwiseXor(), "bitwise_xor");
}
Tensor bitwise_and_scalar_cuda(const Tensor& a, Scalar b) {
    return bitwise_scalar_cuda(a, b, BitwiseAnd(), "bitwise_and");
}
Tensor bitwise_or_scalar_cuda(const Tensor& a, Scalar b) {
    return bitwise_scalar_cuda(a, b, BitwiseOr(), "bitwise_or");
}
Tensor bitwise_xor_scalar_cuda(const Tensor& a, Scalar b) {
    return bitwise_scalar_cuda(a, b, BitwiseXor(), "bitwise_xor");
}
Tensor bitwise_lshift_tensor_cuda(const Tensor& a, const Tensor& b) {
    return bitwise_shift_tensor_cuda_impl<true>(a, b, "bitwise_left_shift");
}
Tensor bitwise_rshift_tensor_cuda(const Tensor& a, const Tensor& b) {
    return bitwise_shift_tensor_cuda_impl<false>(a, b, "bitwise_right_shift");
}
Tensor bitwise_lshift_scalar_cuda(const Tensor& a, Scalar b) {
    return bitwise_shift_scalar_cuda_impl<true>(a, b, "bitwise_left_shift");
}
Tensor bitwise_rshift_scalar_cuda(const Tensor& a, Scalar b) {
    return bitwise_shift_scalar_cuda_impl<false>(a, b, "bitwise_right_shift");
}

// Scalar-first variants: materialize the scalar as a 0-dim tensor in the
// tensor's dtype, then run the plain tensor-tensor kernel.  A floating or
// complex scalar would move the result out of the integral domain, so it is
// refused up front.

inline void bitwise_scalar_check_cuda(Scalar self, const char* name) {
    if (self.isBoolean() || self.isIntegral()) return;
    TP_THROW(TypeError, name,
             ": only integral and boolean scalar operands are supported");
}

Tensor bitwise_and_scalar_tensor_cuda(Scalar self, const Tensor& other) {
    bitwise_check_cuda(other, "bitwise_and");
    bitwise_scalar_check_cuda(self, "bitwise_and");
    Tensor wrapped = Tensor::full({}, self, other.dtype(), other.device());
    return bitwise_binary_cuda(wrapped, other, BitwiseAnd(), "bitwise_and");
}
Tensor bitwise_or_scalar_tensor_cuda(Scalar self, const Tensor& other) {
    bitwise_check_cuda(other, "bitwise_or");
    bitwise_scalar_check_cuda(self, "bitwise_or");
    Tensor wrapped = Tensor::full({}, self, other.dtype(), other.device());
    return bitwise_binary_cuda(wrapped, other, BitwiseOr(), "bitwise_or");
}
Tensor bitwise_xor_scalar_tensor_cuda(Scalar self, const Tensor& other) {
    bitwise_check_cuda(other, "bitwise_xor");
    bitwise_scalar_check_cuda(self, "bitwise_xor");
    Tensor wrapped = Tensor::full({}, self, other.dtype(), other.device());
    return bitwise_binary_cuda(wrapped, other, BitwiseXor(), "bitwise_xor");
}
Tensor bitwise_lshift_scalar_tensor_cuda(Scalar self, const Tensor& other) {
    bitwise_check_cuda(other, "bitwise_left_shift");
    bitwise_scalar_check_cuda(self, "bitwise_left_shift");
    Tensor wrapped = Tensor::full({}, self, other.dtype(), other.device());
    return bitwise_shift_tensor_cuda_impl<true>(wrapped, other, "bitwise_left_shift");
}
Tensor bitwise_rshift_scalar_tensor_cuda(Scalar self, const Tensor& other) {
    bitwise_check_cuda(other, "bitwise_right_shift");
    bitwise_scalar_check_cuda(self, "bitwise_right_shift");
    Tensor wrapped = Tensor::full({}, self, other.dtype(), other.device());
    return bitwise_shift_tensor_cuda_impl<false>(wrapped, other, "bitwise_right_shift");
}

// Out variants: compute into a fresh buffer, then transfer into the
// caller-owned tensor.  Matching shapes copy in place; otherwise the output
// adopts the result's metadata.

Tensor& bitwise_assign_out_cuda(Tensor& out, const Tensor& result) {
    if (static_cast<std::vector<int64_t>>(out.shape()) ==
        static_cast<std::vector<int64_t>>(result.shape())) {
        out.copy_(result);
    } else {
        out.unsafeGetTensorImpl()->copy_metadata_from(*result.unsafeGetTensorImpl());
    }
    return out;
}

Tensor& bitwise_and_tensor_out_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cuda(out, bitwise_and_tensor_cuda(a, b));
}
Tensor& bitwise_or_tensor_out_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cuda(out, bitwise_or_tensor_cuda(a, b));
}
Tensor& bitwise_xor_tensor_out_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cuda(out, bitwise_xor_tensor_cuda(a, b));
}
Tensor& bitwise_and_scalar_out_cuda(const Tensor& a, Scalar b, Tensor& out) {
    return bitwise_assign_out_cuda(out, bitwise_and_scalar_cuda(a, b));
}
Tensor& bitwise_or_scalar_out_cuda(const Tensor& a, Scalar b, Tensor& out) {
    return bitwise_assign_out_cuda(out, bitwise_or_scalar_cuda(a, b));
}
Tensor& bitwise_xor_scalar_out_cuda(const Tensor& a, Scalar b, Tensor& out) {
    return bitwise_assign_out_cuda(out, bitwise_xor_scalar_cuda(a, b));
}
Tensor& bitwise_lshift_tensor_out_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cuda(out, bitwise_lshift_tensor_cuda(a, b));
}
Tensor& bitwise_rshift_tensor_out_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    return bitwise_assign_out_cuda(out, bitwise_rshift_tensor_cuda(a, b));
}
Tensor& bitwise_lshift_scalar_out_cuda(const Tensor& a, Scalar b, Tensor& out) {
    return bitwise_assign_out_cuda(out, bitwise_lshift_scalar_cuda(a, b));
}
Tensor& bitwise_rshift_scalar_out_cuda(const Tensor& a, Scalar b, Tensor& out) {
    return bitwise_assign_out_cuda(out, bitwise_rshift_scalar_cuda(a, b));
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, BitwiseKernels) {
    m.impl("bitwise_not", bitwise_not_cuda);
    m.impl("bitwise_and.Tensor", bitwise_and_tensor_cuda);
    m.impl("bitwise_or.Tensor", bitwise_or_tensor_cuda);
    m.impl("bitwise_xor.Tensor", bitwise_xor_tensor_cuda);
    m.impl("bitwise_and.Scalar", bitwise_and_scalar_cuda);
    m.impl("bitwise_or.Scalar", bitwise_or_scalar_cuda);
    m.impl("bitwise_xor.Scalar", bitwise_xor_scalar_cuda);
    m.impl("bitwise_left_shift.Tensor", bitwise_lshift_tensor_cuda);
    m.impl("bitwise_right_shift.Tensor", bitwise_rshift_tensor_cuda);
    m.impl("bitwise_left_shift.Tensor_Scalar", bitwise_lshift_scalar_cuda);
    m.impl("bitwise_right_shift.Tensor_Scalar", bitwise_rshift_scalar_cuda);
    // Scalar-first variants
    m.impl("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor_cuda);
    m.impl("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor_cuda);
    m.impl("bitwise_xor.Scalar_Tensor", bitwise_xor_scalar_tensor_cuda);
    m.impl("bitwise_left_shift.Scalar_Tensor", bitwise_lshift_scalar_tensor_cuda);
    m.impl("bitwise_right_shift.Scalar_Tensor", bitwise_rshift_scalar_tensor_cuda);
    // Out variants
    m.impl("bitwise_and.Tensor_out", bitwise_and_tensor_out_cuda);
    m.impl("bitwise_or.Tensor_out", bitwise_or_tensor_out_cuda);
    m.impl("bitwise_xor.Tensor_out", bitwise_xor_tensor_out_cuda);
    m.impl("bitwise_left_shift.Tensor_out", bitwise_lshift_tensor_out_cuda);
    m.impl("bitwise_right_shift.Tensor_out", bitwise_rshift_tensor_out_cuda);
    m.impl("bitwise_and.Scalar_out", bitwise_and_scalar_out_cuda);
    m.impl("bitwise_or.Scalar_out", bitwise_or_scalar_out_cuda);
    m.impl("bitwise_xor.Scalar_out", bitwise_xor_scalar_out_cuda);
    m.impl("bitwise_left_shift.Tensor_Scalar_out", bitwise_lshift_scalar_out_cuda);
    m.impl("bitwise_right_shift.Tensor_Scalar_out", bitwise_rshift_scalar_out_cuda);
}

} // namespace cuda
} // namespace tensorplay

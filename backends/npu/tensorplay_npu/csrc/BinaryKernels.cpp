// Elementwise binary arithmetic through the Ascend operator library: add,
// sub, mul and div with broadcasting, their in-place forms and the scalar
// forms.  Operands the device cannot compute in (integers, float64, a host
// operand that is not a 0-d scalar) run through the host fallback.

#include <optional>

#include "AscendOpApi.h"
#include "AscendTensor.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "Utils.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace npu {

namespace {

namespace sig = kernel_signatures;

enum class BinOp { Add, Sub, Mul, Div };

// Entry points, resolved once per process.
const OpApi& pair_api(BinOp op) {
    static const OpApi add("aclnnAdd");
    static const OpApi sub("aclnnSub");
    static const OpApi mul("aclnnMul");
    static const OpApi div("aclnnDiv");
    switch (op) {
        case BinOp::Add: return add;
        case BinOp::Sub: return sub;
        case BinOp::Mul: return mul;
        default: return div;
    }
}

const OpApi& pair_inplace_api(BinOp op) {
    static const OpApi add("aclnnInplaceAdd");
    static const OpApi sub("aclnnInplaceSub");
    static const OpApi mul("aclnnInplaceMul");
    static const OpApi div("aclnnInplaceDiv");
    switch (op) {
        case BinOp::Add: return add;
        case BinOp::Sub: return sub;
        case BinOp::Mul: return mul;
        default: return div;
    }
}

const OpApi& scalar_api(BinOp op) {
    static const OpApi adds("aclnnAdds");
    static const OpApi subs("aclnnSubs");
    static const OpApi muls("aclnnMuls");
    static const OpApi divs("aclnnDivs");
    switch (op) {
        case BinOp::Add: return adds;
        case BinOp::Sub: return subs;
        case BinOp::Mul: return muls;
        default: return divs;
    }
}

const OpApi& scalar_inplace_api(BinOp op) {
    static const OpApi adds("aclnnInplaceAdds");
    static const OpApi subs("aclnnInplaceSubs");
    static const OpApi muls("aclnnInplaceMuls");
    static const OpApi divs("aclnnInplaceDivs");
    switch (op) {
        case BinOp::Add: return adds;
        case BinOp::Sub: return subs;
        case BinOp::Mul: return muls;
        default: return divs;
    }
}

bool is_host_scalar(const Tensor& t) {
    return t.device().is_cpu() && t.numel() == 1 && t.dim() == 0;
}

void emit_pair(BinOp op, bool inplace, int index, const AclTensorRef& x, const AclTensorRef& y,
               const AclScalarRef& alpha, const AclTensorRef& out) {
    const OpApi& api = inplace ? pair_inplace_api(op) : pair_api(op);
    switch (op) {
        case BinOp::Add:
        case BinOp::Sub:
            if (inplace) {
                api.run(index, x, y, alpha);
            } else {
                api.run(index, x, y, alpha, out);
            }
            break;
        default:
            if (inplace) {
                api.run(index, x, y);
            } else {
                api.run(index, x, y, out);
            }
            break;
    }
}

void emit_scalar(BinOp op, bool inplace, int index, const AclTensorRef& x,
                 const AclScalarRef& value, const AclScalarRef& alpha, const AclTensorRef& out) {
    const OpApi& api = inplace ? scalar_inplace_api(op) : scalar_api(op);
    switch (op) {
        case BinOp::Add:
        case BinOp::Sub:
            if (inplace) {
                api.run(index, x, value, alpha);
            } else {
                api.run(index, x, value, alpha, out);
            }
            break;
        default:
            if (inplace) {
                api.run(index, x, value);
            } else {
                api.run(index, x, value, out);
            }
            break;
    }
}

// The native path; nullopt when the operands need the host.  ``out``, when
// given, receives the result in place (it must already have the result's
// shape and dtype).
std::optional<Tensor> binary(const Tensor& a, const Tensor& b, double alpha, BinOp op,
                             const std::optional<Tensor>& out) {
    const bool a_device = a.device().is_privateuse1();
    const bool b_device = b.device().is_privateuse1();
    if (!(a_device || is_host_scalar(a)) || !(b_device || is_host_scalar(b))) return std::nullopt;
    if (!a_device && !b_device) return std::nullopt;
    if (op == BinOp::Div && alpha != 1.0) return std::nullopt;
    const DType dtype = binary_result_dtype(a, b);
    if (!is_native_float(dtype)) return std::nullopt;
    const std::vector<int64_t> shape = broadcast_shapes(a.sizes(), b.sizes());
    if (shape.empty() || shape.size() > kMaxAclDims) return std::nullopt;
    const bool inplace = out.has_value();
    if (inplace) {
        if (!out->device().is_privateuse1() || out->dtype() != dtype ||
            out->sizes().vec() != shape || a.dtype() != dtype) {
            return std::nullopt;
        }
    }
    const bool both = a_device && b_device;
    const bool scalar_first = !a_device;
    if (scalar_first && op == BinOp::Div) return std::nullopt;

    // Every entry point the call would need, before any allocation.
    const OpApi* primary = nullptr;
    const OpApi* secondary = nullptr;
    if (both) {
        primary = &(inplace ? pair_inplace_api(op) : pair_api(op));
    } else if (!scalar_first) {
        primary = &(inplace ? scalar_inplace_api(op) : scalar_api(op));
    } else if (op == BinOp::Mul || (op == BinOp::Add && alpha == 1.0)) {
        primary = &(inplace ? scalar_inplace_api(op == BinOp::Add ? BinOp::Add : BinOp::Mul)
                            : scalar_api(op == BinOp::Add ? BinOp::Add : BinOp::Mul));
    } else {
        primary = &(inplace ? scalar_inplace_api(BinOp::Mul) : scalar_api(BinOp::Mul));
        secondary = &(inplace ? scalar_inplace_api(BinOp::Add) : scalar_api(BinOp::Add));
    }
    if (!primary->valid() || (secondary != nullptr && !secondary->valid())) return std::nullopt;

    const int index = index_of((a_device ? a : b).device());
    Tensor result = inplace ? *out : empty_on(shape, dtype, index);
    if (result.numel() == 0) return result;

    // The operators write dense buffers; a strided in-place target receives
    // the result through its view afterwards.
    const bool strided_target = inplace && !result.is_contiguous();
    Tensor target = strided_target ? empty_on(shape, dtype, index) : result;
    const bool direct = inplace && !strided_target;

    AclArgs args;
    if (both) {
        const Tensor x = compute_operand(a, dtype);
        const Tensor y = compute_operand(b, dtype);
        const AclTensorRef xr = args.tensor(x);
        const AclTensorRef yr = args.tensor(y);
        const AclTensorRef tr = direct ? xr : args.tensor(target);
        emit_pair(op, direct, index, xr, yr, args.scalar(alpha, dtype), tr);
        if (direct) return result;
    } else if (!scalar_first) {
        const double value = b.item().toDouble();
        const Tensor x = compute_operand(a, dtype);
        const AclTensorRef xr = args.tensor(x);
        const AclTensorRef tr = direct ? xr : args.tensor(target);
        const double scalar_value = op == BinOp::Mul ? alpha * value : value;
        emit_scalar(op, direct, index, xr, args.scalar(scalar_value, dtype),
                    args.scalar(alpha, dtype), tr);
        if (direct) return result;
    } else {
        const double value = a.item().toDouble();
        const Tensor x = compute_operand(b, dtype);
        const AclTensorRef xr = args.tensor(x);
        const AclTensorRef tr = direct ? xr : args.tensor(target);
        if (op == BinOp::Mul) {
            emit_scalar(BinOp::Mul, direct, index, xr, args.scalar(alpha * value, dtype),
                        args.scalar(alpha, dtype), tr);
        } else if (op == BinOp::Add && alpha == 1.0) {
            emit_scalar(BinOp::Add, direct, index, xr, args.scalar(value, dtype),
                        args.scalar(1.0, dtype), tr);
        } else {
            // value + alpha * x, or value - alpha * x: scale first, add after.
            const double scale = op == BinOp::Add ? alpha : -alpha;
            emit_scalar(BinOp::Mul, direct, index, xr, args.scalar(scale, dtype),
                        args.scalar(scale, dtype), tr);
            const AclTensorRef scaled = tr;
            emit_scalar(BinOp::Add, true, index, scaled, args.scalar(value, dtype),
                        args.scalar(1.0, dtype), AclTensorRef());
        }
        if (direct) return result;
    }
    if (strided_target) strided_copy(result, target);
    return result;
}

// self (op) alpha * value, one scalar entry point per op.
std::optional<Tensor> scalar_op(const Tensor& self, double value, double alpha, BinOp op,
                                const std::optional<Tensor>& out) {
    if (!self.device().is_privateuse1() || !is_native_float(self.dtype())) return std::nullopt;
    if (self.dim() > kMaxAclDims) return std::nullopt;
    const DType dtype = self.dtype();
    const bool inplace = out.has_value();
    if (inplace && (!out->device().is_privateuse1() || out->dtype() != dtype ||
                    out->sizes().vec() != self.sizes().vec())) {
        return std::nullopt;
    }
    const OpApi* api = &(inplace ? scalar_inplace_api(op) : scalar_api(op));
    if (!api->valid()) return std::nullopt;

    const int index = index_of(self.device());
    Tensor result = inplace ? *out : empty_on(self.sizes().vec(), dtype, index);
    if (result.numel() == 0) return result;
    const bool strided_target = inplace && !result.is_contiguous();
    Tensor target = strided_target ? empty_on(self.sizes().vec(), dtype, index) : result;
    const bool direct = inplace && !strided_target;

    AclArgs args;
    const Tensor x = compute_operand(self, dtype);
    const AclTensorRef xr = args.tensor(x);
    const AclTensorRef tr = direct ? xr : args.tensor(target);
    const double scalar_value = op == BinOp::Mul ? alpha * value : value;
    emit_scalar(op, direct, index, xr, args.scalar(scalar_value, dtype),
                args.scalar(alpha, dtype), tr);
    if (direct) return result;
    if (strided_target) strided_copy(result, target);
    return result;
}

// value - alpha * self: the device scales, the host constant adds.
std::optional<Tensor> rsub_scalar(const Tensor& self, double value, double alpha) {
    if (!self.device().is_privateuse1() || !is_native_float(self.dtype())) return std::nullopt;
    if (self.dim() > kMaxAclDims) return std::nullopt;
    const OpApi& scale = scalar_api(BinOp::Mul);
    const OpApi& shift = scalar_inplace_api(BinOp::Add);
    if (!scale.valid() || !shift.valid()) return std::nullopt;

    const DType dtype = self.dtype();
    const int index = index_of(self.device());
    Tensor out = empty_on(self.sizes().vec(), dtype, index);
    if (out.numel() == 0) return out;
    AclArgs args;
    const Tensor x = compute_operand(self, dtype);
    const AclTensorRef xr = args.tensor(x);
    const AclTensorRef outr = args.tensor(out);
    scale.run(index, xr, args.scalar(-alpha, dtype), outr);
    shift.run(index, outr, args.scalar(value, dtype), args.scalar(1.0, dtype));
    return out;
}

Tensor add_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    if (auto r = binary(self, other, alpha.toDouble(), BinOp::Add, std::nullopt)) return *r;
    return cpu_fallback::Kernel<"add.Tensor", sig::add__Tensor>::call(self, other, alpha);
}

Tensor sub_kernel(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    if (auto r = binary(self, other, alpha.toDouble(), BinOp::Sub, std::nullopt)) return *r;
    return cpu_fallback::Kernel<"sub.Tensor", sig::sub__Tensor>::call(self, other, alpha);
}

Tensor mul_kernel(const Tensor& self, const Tensor& other) {
    if (auto r = binary(self, other, 1.0, BinOp::Mul, std::nullopt)) return *r;
    return cpu_fallback::Kernel<"mul.Tensor", sig::mul__Tensor>::call(self, other);
}

Tensor div_kernel(const Tensor& self, const Tensor& other) {
    if (auto r = binary(self, other, 1.0, BinOp::Div, std::nullopt)) return *r;
    return cpu_fallback::Kernel<"div.Tensor", sig::div__Tensor>::call(self, other);
}

Tensor& add__kernel(Tensor& self, const Tensor& other, const Scalar& alpha) {
    if (binary(self, other, alpha.toDouble(), BinOp::Add, self)) return self;
    return cpu_fallback::Kernel<"add_.Tensor", sig::add___Tensor>::call(self, other, alpha);
}

Tensor& sub__kernel(Tensor& self, const Tensor& other, const Scalar& alpha) {
    if (binary(self, other, alpha.toDouble(), BinOp::Sub, self)) return self;
    return cpu_fallback::Kernel<"sub_.Tensor", sig::sub___Tensor>::call(self, other, alpha);
}

Tensor& mul__kernel(Tensor& self, const Tensor& other) {
    if (binary(self, other, 1.0, BinOp::Mul, self)) return self;
    return cpu_fallback::Kernel<"mul_.Tensor", sig::mul___Tensor>::call(self, other);
}

Tensor& div__kernel(Tensor& self, const Tensor& other) {
    if (binary(self, other, 1.0, BinOp::Div, self)) return self;
    return cpu_fallback::Kernel<"div_.Tensor", sig::div___Tensor>::call(self, other);
}

Tensor add_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    if (auto r = scalar_op(self, other.toDouble(), alpha.toDouble(), BinOp::Add, std::nullopt)) {
        return *r;
    }
    return cpu_fallback::Kernel<"add.Scalar", sig::add__Scalar>::call(self, other, alpha);
}

Tensor sub_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    if (auto r = scalar_op(self, other.toDouble(), alpha.toDouble(), BinOp::Sub, std::nullopt)) {
        return *r;
    }
    return cpu_fallback::Kernel<"sub.Scalar", sig::sub__Scalar>::call(self, other, alpha);
}

Tensor rsub_scalar_kernel(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    if (auto r = rsub_scalar(self, other.toDouble(), alpha.toDouble())) return *r;
    return cpu_fallback::Kernel<"rsub.Scalar", sig::rsub__Scalar>::call(self, other, alpha);
}

Tensor mul_scalar_kernel(const Tensor& self, const Scalar& other) {
    if (auto r = scalar_op(self, other.toDouble(), 1.0, BinOp::Mul, std::nullopt)) return *r;
    return cpu_fallback::Kernel<"mul.Scalar", sig::mul__Scalar>::call(self, other);
}

Tensor div_scalar_kernel(const Tensor& self, const Scalar& other) {
    if (auto r = scalar_op(self, other.toDouble(), 1.0, BinOp::Div, std::nullopt)) return *r;
    return cpu_fallback::Kernel<"div.Scalar", sig::div__Scalar>::call(self, other);
}

Tensor& add__scalar_kernel(Tensor& self, const Scalar& other, const Scalar& alpha) {
    if (scalar_op(self, other.toDouble(), alpha.toDouble(), BinOp::Add, self)) return self;
    return cpu_fallback::Kernel<"add_.Scalar", sig::add___Scalar>::call(self, other, alpha);
}

Tensor& mul__scalar_kernel(Tensor& self, const Scalar& other) {
    if (scalar_op(self, other.toDouble(), 1.0, BinOp::Mul, self)) return self;
    return cpu_fallback::Kernel<"mul_.Scalar", sig::mul___Scalar>::call(self, other);
}

Tensor& div__scalar_kernel(Tensor& self, const Scalar& other) {
    if (scalar_op(self, other.toDouble(), 1.0, BinOp::Div, self)) return self;
    return cpu_fallback::Kernel<"div_.Scalar", sig::div___Scalar>::call(self, other);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, NPUBinaryKernels) {
    m.impl("add.Tensor", static_cast<sig::add__Tensor*>(&add_kernel));
    m.impl("sub.Tensor", static_cast<sig::sub__Tensor*>(&sub_kernel));
    m.impl("mul.Tensor", static_cast<sig::mul__Tensor*>(&mul_kernel));
    m.impl("div.Tensor", static_cast<sig::div__Tensor*>(&div_kernel));
    m.impl("add_.Tensor", static_cast<sig::add___Tensor*>(&add__kernel));
    m.impl("sub_.Tensor", static_cast<sig::sub___Tensor*>(&sub__kernel));
    m.impl("mul_.Tensor", static_cast<sig::mul___Tensor*>(&mul__kernel));
    m.impl("div_.Tensor", static_cast<sig::div___Tensor*>(&div__kernel));
    m.impl("add.Scalar", static_cast<sig::add__Scalar*>(&add_scalar_kernel));
    m.impl("sub.Scalar", static_cast<sig::sub__Scalar*>(&sub_scalar_kernel));
    m.impl("rsub.Scalar", static_cast<sig::rsub__Scalar*>(&rsub_scalar_kernel));
    m.impl("mul.Scalar", static_cast<sig::mul__Scalar*>(&mul_scalar_kernel));
    m.impl("div.Scalar", static_cast<sig::div__Scalar*>(&div_scalar_kernel));
    m.impl("add_.Scalar", static_cast<sig::add___Scalar*>(&add__scalar_kernel));
    m.impl("mul_.Scalar", static_cast<sig::mul___Scalar*>(&mul__scalar_kernel));
    m.impl("div_.Scalar", static_cast<sig::div___Scalar*>(&div__scalar_kernel));
}

} // namespace npu
} // namespace tensorplay

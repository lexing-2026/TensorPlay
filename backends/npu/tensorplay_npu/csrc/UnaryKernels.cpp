// Elementwise unary math and activations through the Ascend operator
// library, with the backward kernels autograd calls for the activations.

#include <optional>
#include <string>

#include "AscendOpApi.h"
#include "AscendTensor.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace npu {

namespace {

namespace sig = kernel_signatures;

bool native_operand(const Tensor& t) {
    return t.device().is_privateuse1() && is_native_float(t.dtype()) &&
           t.dim() <= kMaxAclDims && !has_negative_stride(t);
}

// One dense pass from ``self`` into a fresh result; ``extra`` arguments sit
// between the input and the output, as the operator declares them.
template <typename... Extra>
Tensor unary(const Tensor& self, const OpApi& api, Extra... extra) {
    const int index = index_of(self.device());
    const Tensor x = compute_operand(self, self.dtype());
    Tensor out = empty_on(x.sizes().vec(), x.dtype(), index);
    if (out.numel() == 0) return out;
    AclArgs args;
    const AclTensorRef input = args.tensor(x);
    const AclTensorRef result = args.tensor(out);
    api.run(index, input, extra..., result);
    return out;
}

// ``grad`` flowed back through a single-input operator.
Tensor backward1(const Tensor& grad, const Tensor& a, const OpApi& api) {
    const int index = index_of(grad.device());
    const DType dtype = grad.dtype();
    const Tensor dy = compute_operand(grad, dtype);
    const Tensor da = compute_operand(a, dtype);
    Tensor dx = empty_on(dy.sizes().vec(), dtype, index);
    if (dx.numel() == 0) return dx;
    AclArgs args;
    const AclTensorRef dyr = args.tensor(dy);
    const AclTensorRef ar = args.tensor(da);
    const AclTensorRef dxr = args.tensor(dx);
    api.run(index, dyr, ar, dxr);
    return dx;
}

#define TP_NPU_UNARY(NAME, API, SYM)                                              \
    Tensor NAME##_kernel(const Tensor& self) {                                    \
        static const OpApi api(API);                                              \
        if (!native_operand(self) || !api.valid()) {                              \
            return cpu_fallback::Kernel<#SYM, sig::SYM>::call(self);              \
        }                                                                         \
        return unary(self, api);                                                  \
    }

TP_NPU_UNARY(exp, "aclnnExp", exp)
TP_NPU_UNARY(log, "aclnnLog", log)
TP_NPU_UNARY(sqrt, "aclnnSqrt", sqrt)
TP_NPU_UNARY(rsqrt, "aclnnRsqrt", rsqrt)
TP_NPU_UNARY(neg, "aclnnNeg", neg)
TP_NPU_UNARY(abs, "aclnnAbs", abs)
TP_NPU_UNARY(relu, "aclnnRelu", relu)
TP_NPU_UNARY(sigmoid, "aclnnSigmoid", sigmoid)
TP_NPU_UNARY(tanh, "aclnnTanh", tanh)
TP_NPU_UNARY(silu, "aclnnSilu", silu)
#undef TP_NPU_UNARY

Tensor& relu__kernel(Tensor& self) {
    static const OpApi inplace("aclnnInplaceRelu");
    static const OpApi plain("aclnnRelu");
    if (!native_operand(self)) return cpu_fallback::Kernel<"relu_", sig::relu_>::call(self);
    if (self.is_contiguous() && inplace.valid()) {
        AclArgs args;
        const AclTensorRef selfr = args.tensor(self);
        inplace.run(index_of(self.device()), selfr);
        return self;
    }
    if (!plain.valid()) return cpu_fallback::Kernel<"relu_", sig::relu_>::call(self);
    const Tensor result = unary(self, plain);
    strided_copy(self, result);
    return self;
}

Tensor gelu_kernel(const Tensor& self, const std::string& approximate) {
    static const OpApi api("aclnnGeluV2");
    if (!native_operand(self) || !api.valid()) {
        return cpu_fallback::Kernel<"gelu", sig::gelu>::call(self, approximate);
    }
    const int64_t mode = approximate == "tanh" ? 1 : 0;
    return unary(self, api, mode);
}

Tensor threshold_backward_kernel(const Tensor& grad_output, const Tensor& self,
                                 const Scalar& threshold) {
    static const OpApi api("aclnnThresholdBackward");
    if (!native_operand(grad_output) || !native_operand(self) || !api.valid() ||
        grad_output.dtype() != self.dtype() ||
        grad_output.sizes().vec() != self.sizes().vec()) {
        return cpu_fallback::Kernel<"threshold_backward", sig::threshold_backward>::call(
            grad_output, self, threshold);
    }
    const int index = index_of(grad_output.device());
    const DType dtype = grad_output.dtype();
    const Tensor dy = compute_operand(grad_output, dtype);
    const Tensor x = compute_operand(self, dtype);
    Tensor dx = empty_on(dy.sizes().vec(), dtype, index);
    if (dx.numel() == 0) return dx;
    AclArgs args;
    const AclTensorRef dyr = args.tensor(dy);
    const AclTensorRef xr = args.tensor(x);
    const AclTensorRef dxr = args.tensor(dx);
    api.run(index, dyr, xr, args.scalar(threshold, dtype), dxr);
    return dx;
}

Tensor sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& output) {
    static const OpApi api("aclnnSigmoidBackward");
    if (!native_operand(grad_output) || !native_operand(output) || !api.valid() ||
        grad_output.dtype() != output.dtype() ||
        grad_output.sizes().vec() != output.sizes().vec()) {
        return cpu_fallback::Kernel<"sigmoid_backward", sig::sigmoid_backward>::call(grad_output,
                                                                                     output);
    }
    return backward1(grad_output, output, api);
}

Tensor tanh_backward_kernel(const Tensor& grad_output, const Tensor& output) {
    static const OpApi api("aclnnTanhBackward");
    if (!native_operand(grad_output) || !native_operand(output) || !api.valid() ||
        grad_output.dtype() != output.dtype() ||
        grad_output.sizes().vec() != output.sizes().vec()) {
        return cpu_fallback::Kernel<"tanh_backward", sig::tanh_backward>::call(grad_output,
                                                                               output);
    }
    return backward1(grad_output, output, api);
}

Tensor gelu_backward_kernel(const Tensor& grad_output, const Tensor& self,
                            const std::string& approximate) {
    static const OpApi api("aclnnGeluBackwardV2");
    if (!native_operand(grad_output) || !native_operand(self) || !api.valid() ||
        grad_output.dtype() != self.dtype()) {
        return cpu_fallback::Kernel<"gelu_backward", sig::gelu_backward>::call(grad_output, self,
                                                                               approximate);
    }
    const int index = index_of(grad_output.device());
    const DType dtype = grad_output.dtype();
    const Tensor dy = compute_operand(grad_output, dtype);
    const Tensor x = compute_operand(self, dtype);
    Tensor dx = empty_on(dy.sizes().vec(), dtype, index);
    if (dx.numel() == 0) return dx;
    char mode[] = "none";
    if (approximate == "tanh") {
        mode[0] = 't';
        mode[1] = 'a';
        mode[2] = 'n';
        mode[3] = 'h';
        mode[4] = '\0';
    }
    AclArgs args;
    const AclTensorRef dyr = args.tensor(dy);
    const AclTensorRef xr = args.tensor(x);
    const AclTensorRef dxr = args.tensor(dx);
    api.run(index, dyr, xr, static_cast<char*>(mode), dxr);
    return dx;
}

Tensor silu_backward_kernel(const Tensor& grad_output, const Tensor& self) {
    static const OpApi api("aclnnSiluBackward");
    if (!native_operand(grad_output) || !native_operand(self) || !api.valid() ||
        grad_output.dtype() != self.dtype()) {
        return cpu_fallback::Kernel<"silu_backward", sig::silu_backward>::call(grad_output, self);
    }
    return backward1(grad_output, self, api);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, NPUUnaryKernels) {
    m.impl("exp", static_cast<sig::exp*>(&exp_kernel));
    m.impl("log", static_cast<sig::log*>(&log_kernel));
    m.impl("sqrt", static_cast<sig::sqrt*>(&sqrt_kernel));
    m.impl("rsqrt", static_cast<sig::rsqrt*>(&rsqrt_kernel));
    m.impl("neg", static_cast<sig::neg*>(&neg_kernel));
    m.impl("abs", static_cast<sig::abs*>(&abs_kernel));
    m.impl("relu", static_cast<sig::relu*>(&relu_kernel));
    m.impl("relu_", static_cast<sig::relu_*>(&relu__kernel));
    m.impl("sigmoid", static_cast<sig::sigmoid*>(&sigmoid_kernel));
    m.impl("tanh", static_cast<sig::tanh*>(&tanh_kernel));
    m.impl("silu", static_cast<sig::silu*>(&silu_kernel));
    m.impl("gelu", static_cast<sig::gelu*>(&gelu_kernel));
    m.impl("threshold_backward", static_cast<sig::threshold_backward*>(&threshold_backward_kernel));
    m.impl("sigmoid_backward", static_cast<sig::sigmoid_backward*>(&sigmoid_backward_kernel));
    m.impl("tanh_backward", static_cast<sig::tanh_backward*>(&tanh_backward_kernel));
    m.impl("gelu_backward", static_cast<sig::gelu_backward*>(&gelu_backward_kernel));
    m.impl("silu_backward", static_cast<sig::silu_backward*>(&silu_backward_kernel));
}

} // namespace npu
} // namespace tensorplay

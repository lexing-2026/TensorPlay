// Elementwise unary math and activations on CNNL, with the backward
// kernels autograd calls for the activations.

#include <optional>
#include <string>

#include "Cnnl.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace mlu {

namespace {

namespace sig = kernel_signatures;

bool native_operand(const Tensor& t) {
    return t.device().is_privateuse1() && is_native_float(t.dtype());
}

// Run ``launch(handle, x_desc, x, y_desc, y)`` from a dense copy of
// ``self`` into ``out`` (allocated when not given).
template <typename Launch>
Tensor unary(const Tensor& self, const std::optional<Tensor>& out, Launch launch) {
    const int index = index_of(self.device());
    const Tensor x = contiguous_on_device(self);
    Tensor y = out && out->is_contiguous() ? *out : empty_on(x.sizes().vec(), x.dtype(), index);
    if (y.numel() > 0) {
        DeviceGuard guard(index);
        TensorDesc x_desc(x), y_desc(y);
        launch(handle(index), x_desc.get(), x.data_ptr(), y_desc.get(), y.data_ptr());
    }
    if (out && !out->is_contiguous()) {
        strided_copy(*out, y);
        return *out;
    }
    return y;
}

class ActivationDesc {
public:
    ActivationDesc(cnnlActivationMode_t mode, float coef = 1.0f, bool approximate = false) {
        TP_CNNL_CHECK(cnnlCreateActivationDescriptor(&desc_));
        TP_CNNL_CHECK(cnnlSetActivationDescriptor_v6(desc_, mode, CNNL_ACTIVATION_HIGH_PRECISION,
                                                     CNNL_NOT_PROPAGATE_NAN, coef, 0, 0.0f, 0.0f,
                                                     false, approximate));
    }
    ~ActivationDesc() { cnnlDestroyActivationDescriptor(desc_); }
    ActivationDesc(const ActivationDesc&) = delete;
    ActivationDesc& operator=(const ActivationDesc&) = delete;
    operator cnnlActivationDescriptor_t() const { return desc_; }

private:
    cnnlActivationDescriptor_t desc_ = nullptr;
};

Tensor activation(const Tensor& self, cnnlActivationMode_t mode, const std::optional<Tensor>& out,
                  bool approximate = false) {
    ActivationDesc desc(mode, 1.0f, approximate);
    return unary(self, out, [&](cnnlHandle_t h, cnnlTensorDescriptor_t xd, void* x,
                                cnnlTensorDescriptor_t yd, void* y) {
        TP_CNNL_CHECK(cnnlActivationForward(h, desc, nullptr, xd, x, nullptr, yd, y));
    });
}

// grad_input from an activation's output (``y``) or input (``x``).
Tensor activation_backward(const Tensor& grad, const Tensor* y, const Tensor* x,
                           cnnlActivationMode_t mode, bool approximate = false) {
    const int index = index_of(grad.device());
    ActivationDesc desc(mode, 1.0f, approximate);
    const Tensor dy = contiguous_on_device(grad);
    const Tensor yc = y ? contiguous_on_device(*y) : Tensor();
    const Tensor xc = x ? contiguous_on_device(*x) : Tensor();
    Tensor dx = empty_on(dy.sizes().vec(), dy.dtype(), index);
    if (dx.numel() == 0) return dx;
    DeviceGuard guard(index);
    TensorDesc dy_desc(dy), dx_desc(dx);
    TensorDesc y_desc = y ? TensorDesc(yc) : TensorDesc();
    TensorDesc x_desc = x ? TensorDesc(xc) : TensorDesc();
    TP_CNNL_CHECK(cnnlActivationBackward(handle(index), desc, nullptr, y ? y_desc.get() : nullptr,
                                         y ? yc.data_ptr() : nullptr, dy_desc, dy.data_ptr(),
                                         x ? x_desc.get() : nullptr, x ? xc.data_ptr() : nullptr,
                                         nullptr, dx_desc, dx.data_ptr()));
    return dx;
}

#define TP_MLU_UNARY_PREFERENCE(NAME, CALL, OP, SYM)                                       \
    Tensor NAME##_kernel(const Tensor& self) {                                             \
        if (!native_operand(self)) return cpu_fallback::Kernel<OP, sig::SYM>::call(self);  \
        return unary(self, std::nullopt,                                                   \
                     [](cnnlHandle_t h, cnnlTensorDescriptor_t xd, void* x,                \
                        cnnlTensorDescriptor_t yd, void* y) {                              \
                         TP_CNNL_CHECK(CALL(h, CNNL_COMPUTATION_HIGH_PRECISION, xd, x, yd, y)); \
                     });                                                                   \
    }

TP_MLU_UNARY_PREFERENCE(exp, cnnlExp_v2, "exp", exp)
TP_MLU_UNARY_PREFERENCE(sqrt, cnnlSqrt_v2, "sqrt", sqrt)
TP_MLU_UNARY_PREFERENCE(rsqrt, cnnlRsqrt_v2, "rsqrt", rsqrt)
#undef TP_MLU_UNARY_PREFERENCE

Tensor log_kernel(const Tensor& self) {
    if (!native_operand(self)) return cpu_fallback::Kernel<"log", sig::log>::call(self);
    return unary(self, std::nullopt, [](cnnlHandle_t h, cnnlTensorDescriptor_t xd, void* x,
                                        cnnlTensorDescriptor_t yd, void* y) {
        TP_CNNL_CHECK(cnnlLog_v2(h, CNNL_COMPUTATION_HIGH_PRECISION, CNNL_LOG_E, xd, x, yd, y));
    });
}

Tensor neg_kernel(const Tensor& self) {
    if (!native_operand(self)) return cpu_fallback::Kernel<"neg", sig::neg>::call(self);
    return unary(self, std::nullopt, [](cnnlHandle_t h, cnnlTensorDescriptor_t xd, void* x,
                                        cnnlTensorDescriptor_t yd, void* y) {
        TP_CNNL_CHECK(cnnlNegTensor(h, xd, x, yd, y));
    });
}

Tensor abs_kernel(const Tensor& self) {
    if (!native_operand(self)) return cpu_fallback::Kernel<"abs", sig::abs>::call(self);
    return unary(self, std::nullopt, [](cnnlHandle_t h, cnnlTensorDescriptor_t xd, void* x,
                                        cnnlTensorDescriptor_t yd, void* y) {
        TP_CNNL_CHECK(cnnlAbs(h, xd, x, yd, y));
    });
}

Tensor relu_kernel(const Tensor& self) {
    if (!native_operand(self)) return cpu_fallback::Kernel<"relu", sig::relu>::call(self);
    return activation(self, CNNL_ACTIVATION_RELU, std::nullopt);
}

Tensor& relu__kernel(Tensor& self) {
    if (!native_operand(self)) return cpu_fallback::Kernel<"relu_", sig::relu_>::call(self);
    activation(self, CNNL_ACTIVATION_RELU, self);
    return self;
}

Tensor sigmoid_kernel(const Tensor& self) {
    if (!native_operand(self)) return cpu_fallback::Kernel<"sigmoid", sig::sigmoid>::call(self);
    return activation(self, CNNL_ACTIVATION_SIGMOID, std::nullopt);
}

Tensor tanh_kernel(const Tensor& self) {
    if (!native_operand(self)) return cpu_fallback::Kernel<"tanh", sig::tanh>::call(self);
    return activation(self, CNNL_ACTIVATION_TANH, std::nullopt);
}

Tensor silu_kernel(const Tensor& self) {
    if (!native_operand(self)) return cpu_fallback::Kernel<"silu", sig::silu>::call(self);
    return activation(self, CNNL_ACTIVATION_SILU, std::nullopt);
}

Tensor gelu_kernel(const Tensor& self, const std::string& approximate) {
    if (!native_operand(self)) {
        return cpu_fallback::Kernel<"gelu", sig::gelu>::call(self, approximate);
    }
    return activation(self, CNNL_ACTIVATION_GELU, std::nullopt, approximate == "tanh");
}

Tensor threshold_backward_kernel(const Tensor& grad_output, const Tensor& self,
                                 const Scalar& threshold) {
    if (!native_operand(grad_output) || !native_operand(self) ||
        grad_output.dtype() != self.dtype() ||
        grad_output.sizes().vec() != self.sizes().vec()) {
        return cpu_fallback::Kernel<"threshold_backward", sig::threshold_backward>::call(
            grad_output, self, threshold);
    }
    const int index = index_of(self.device());
    const Tensor x = contiguous_on_device(self);
    const Tensor dy = contiguous_on_device(grad_output);
    Tensor dx = empty_on(x.sizes().vec(), x.dtype(), index);
    if (dx.numel() == 0) return dx;
    alignas(8) unsigned char value[8] = {};
    switch (x.dtype()) {
        case DType::Float16: *reinterpret_cast<Half*>(value) = Half(threshold.to<float>()); break;
        case DType::BFloat16:
            *reinterpret_cast<BFloat16*>(value) = BFloat16(threshold.to<float>());
            break;
        default: *reinterpret_cast<float*>(value) = threshold.to<float>(); break;
    }
    DeviceGuard guard(index);
    TensorDesc x_desc(x), dy_desc(dy), dx_desc(dx);
    TP_CNNL_CHECK(cnnlThresholdBackward(handle(index), x_desc, x.data_ptr(), dy_desc,
                                        dy.data_ptr(), value, dx_desc, dx.data_ptr()));
    return dx;
}

Tensor sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& output) {
    if (!native_operand(grad_output) || !native_operand(output) ||
        grad_output.dtype() != output.dtype()) {
        return cpu_fallback::Kernel<"sigmoid_backward", sig::sigmoid_backward>::call(grad_output,
                                                                                     output);
    }
    return activation_backward(grad_output, &output, nullptr, CNNL_ACTIVATION_SIGMOID);
}

Tensor tanh_backward_kernel(const Tensor& grad_output, const Tensor& output) {
    if (!native_operand(grad_output) || !native_operand(output) ||
        grad_output.dtype() != output.dtype()) {
        return cpu_fallback::Kernel<"tanh_backward", sig::tanh_backward>::call(grad_output,
                                                                               output);
    }
    return activation_backward(grad_output, &output, nullptr, CNNL_ACTIVATION_TANH);
}

Tensor gelu_backward_kernel(const Tensor& grad_output, const Tensor& self,
                            const std::string& approximate) {
    if (!native_operand(grad_output) || !native_operand(self) ||
        grad_output.dtype() != self.dtype()) {
        return cpu_fallback::Kernel<"gelu_backward", sig::gelu_backward>::call(grad_output, self,
                                                                               approximate);
    }
    return activation_backward(grad_output, nullptr, &self, CNNL_ACTIVATION_GELU,
                               approximate == "tanh");
}

Tensor silu_backward_kernel(const Tensor& grad_output, const Tensor& self) {
    if (!native_operand(grad_output) || !native_operand(self) ||
        grad_output.dtype() != self.dtype()) {
        return cpu_fallback::Kernel<"silu_backward", sig::silu_backward>::call(grad_output, self);
    }
    return activation_backward(grad_output, nullptr, &self, CNNL_ACTIVATION_SILU);
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, MLUUnaryKernels) {
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

} // namespace mlu
} // namespace tensorplay

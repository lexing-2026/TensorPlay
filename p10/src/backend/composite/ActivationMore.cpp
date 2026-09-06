// Composite kernels for the extended activation family.  These are thin
// wrappers: the elementwise math already lives in dedicated CPU kernels, so
// each entry here either re-composes the clamp family through the generic
// clamp op (the bounded-side helpers are clamp with one bound omitted) or
// forwards to the functional kernel and lands the result in the
// caller-provided output tensor.
//
//   clamp_max/min(.out|.Tensor_out): clamp with only max/min set.
//   clamp_max_/.Tensor, clamp_min_/.Tensor: in-place flavors; the functional
//       result is computed first and then copied into self, so an input that
//       aliases the bound stays well-defined.
//   *_backward.grad_input / *.out: evaluate the registered functional op and
//       assign the result into the out/grad_input argument.
//   log_sigmoid.out: log_sigmoid output component of the forward pair; the
//       saved buffer is only consumed by the backward pass.
//   logcumsumexp.out: running log-sum-exp scan along dim, result written to
//       the caller's tensor.

#include "CompositeCommon.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

// clamp_max(self, max) == clamp(self, min=nullopt, max)
Tensor& clamp_max_out_native(const Tensor& self, Scalar max, Tensor& out) {
    out = ops::clamp(self, std::nullopt, max);
    return out;
}

Tensor& clamp_max_tensor_out_native(const Tensor& self, const Tensor& max,
                                    Tensor& out) {
    out = ops::clamp_max(self, max);
    return out;
}

Tensor& clamp_max__tensor_native(Tensor& self, const Tensor& max) {
    Tensor result = ops::clamp_max(self, max);
    ops::copy_(self, result);
    return self;
}

// clamp_min(self, min) == clamp(self, min, max=nullopt)
Tensor& clamp_min_out_native(const Tensor& self, Scalar min, Tensor& out) {
    out = ops::clamp(self, min, std::nullopt);
    return out;
}

Tensor& clamp_min_tensor_out_native(const Tensor& self, const Tensor& min,
                                    Tensor& out) {
    out = ops::clamp_min(self, min);
    return out;
}

Tensor& clamp_min__tensor_native(Tensor& self, const Tensor& min) {
    Tensor result = ops::clamp_min(self, min);
    ops::copy_(self, result);
    return self;
}

Tensor& gelu_backward_grad_input_native(const Tensor& grad_output,
                                        const Tensor& self,
                                        std::string approximate,
                                        Tensor& grad_input) {
    grad_input = ops::gelu_backward(grad_output, self, std::move(approximate));
    return grad_input;
}

Tensor& silu_backward_grad_input_native(const Tensor& grad_output,
                                        const Tensor& self, Tensor& grad_input) {
    grad_input = ops::silu_backward(grad_output, self);
    return grad_input;
}

Tensor& hardsigmoid_backward_grad_input_native(const Tensor& grad_output,
                                               const Tensor& self,
                                               Tensor& grad_input) {
    grad_input = ops::hardsigmoid_backward(grad_output, self);
    return grad_input;
}

Tensor& hardshrink_backward_grad_input_native(const Tensor& grad_out,
                                              const Tensor& self, Scalar lambd,
                                              Tensor& grad_input) {
    grad_input = ops::hardshrink_backward(grad_out, self, lambd);
    return grad_input;
}

Tensor& softshrink_backward_grad_input_native(const Tensor& grad_output,
                                              const Tensor& self, Scalar lambd,
                                              Tensor& grad_input) {
    grad_input = ops::softshrink_backward(grad_output, self, lambd);
    return grad_input;
}

Tensor& softplus_backward_grad_input_native(const Tensor& grad_output,
                                            const Tensor& self, Scalar beta,
                                            Scalar threshold,
                                            Tensor& grad_input) {
    grad_input = ops::softplus_backward(grad_output, self, beta, threshold);
    return grad_input;
}

Tensor& leaky_relu_backward_grad_input_native(const Tensor& grad_output,
                                              const Tensor& self,
                                              Scalar negative_slope,
                                              bool self_is_result,
                                              Tensor& grad_input) {
    grad_input = ops::leaky_relu_backward(grad_output, self, negative_slope,
                                          self_is_result);
    return grad_input;
}

Tensor& elu_backward_grad_input_native(const Tensor& grad_output, Scalar alpha,
                                       Scalar scale, Scalar input_scale,
                                       bool is_result,
                                       const Tensor& self_or_result,
                                       Tensor& grad_input) {
    grad_input = ops::elu_backward(grad_output, alpha, scale, input_scale,
                                   is_result, self_or_result);
    return grad_input;
}

Tensor& glu_backward_grad_input_native(const Tensor& grad_output,
                                       const Tensor& self, int64_t dim,
                                       Tensor& grad_input) {
    grad_input = ops::glu_backward(grad_output, self, dim);
    return grad_input;
}

// threshold_backward: positions with self <= threshold carry no gradient,
// elsewhere the incoming grad passes through (the functional kernel encodes
// this as threshold() with value = 0).
Tensor& threshold_backward_grad_input_native(const Tensor& grad_output,
                                             const Tensor& self,
                                             Scalar threshold,
                                             Tensor& grad_input) {
    grad_input = ops::threshold_backward(grad_output, self, threshold);
    return grad_input;
}

Tensor& hardsigmoid_out_native(const Tensor& self, Tensor& out) {
    out = ops::hardsigmoid(self);
    return out;
}

Tensor& hardswish_out_native(const Tensor& self, Tensor& out) {
    out = ops::hardswish(self);
    return out;
}

Tensor& log_sigmoid_out_native(const Tensor& self, Tensor& out) {
    out = std::get<0>(ops::log_sigmoid_forward(self));
    return out;
}

Tensor& logcumsumexp_out_native(const Tensor& self, int64_t dim, Tensor& out) {
    if (out.defined()) {
        if (out.dtype() != self.dtype()) {
            TP_THROW(RuntimeError, "logcumsumexp: output dtype must match input dtype");
        }
        if (out.device() != self.device()) {
            TP_THROW(RuntimeError, "logcumsumexp: output device must match input device");
        }
    }
    const Tensor value = ops::logcumsumexp(self, dim);
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

TENSORPLAY_LIBRARY_IMPL(Composite, ActivationMoreComposite) {
    m.impl("clamp_max.out", clamp_max_out_native);
    m.impl("clamp_max.Tensor_out", clamp_max_tensor_out_native);
    m.impl("clamp_max_.Tensor", clamp_max__tensor_native);
    m.impl("clamp_min.out", clamp_min_out_native);
    m.impl("clamp_min.Tensor_out", clamp_min_tensor_out_native);
    m.impl("clamp_min_.Tensor", clamp_min__tensor_native);
    m.impl("gelu_backward.grad_input", gelu_backward_grad_input_native);
    m.impl("silu_backward.grad_input", silu_backward_grad_input_native);
    m.impl("hardsigmoid_backward.grad_input",
           hardsigmoid_backward_grad_input_native);
    m.impl("hardshrink_backward.grad_input",
           hardshrink_backward_grad_input_native);
    m.impl("softshrink_backward.grad_input",
           softshrink_backward_grad_input_native);
    m.impl("softplus_backward.grad_input", softplus_backward_grad_input_native);
    m.impl("leaky_relu_backward.grad_input",
           leaky_relu_backward_grad_input_native);
    m.impl("elu_backward.grad_input", elu_backward_grad_input_native);
    m.impl("glu_backward.grad_input", glu_backward_grad_input_native);
    m.impl("threshold_backward.grad_input", threshold_backward_grad_input_native);
    m.impl("hardsigmoid.out", hardsigmoid_out_native);
    m.impl("hardswish.out", hardswish_out_native);
    m.impl("log_sigmoid.out", log_sigmoid_out_native);
    m.impl("logcumsumexp.out", logcumsumexp_out_native);
}

}  // namespace composite
}  // namespace tensorplay

// Reductions (sum, mean) and softmax / log-softmax with their backward
// kernels, through the Ascend operator library.

#include <vector>

#include "AscendOpApi.h"
#include "AscendTensor.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace npu {

namespace {

namespace sig = kernel_signatures;

const OpApi& sum_api() {
    static const OpApi api("aclnnReduceSum");
    return api;
}

const OpApi& mean_api() {
    static const OpApi api("aclnnMean");
    return api;
}

const OpApi& softmax_api() {
    static const OpApi api("aclnnSoftmax");
    return api;
}

const OpApi& log_softmax_api() {
    static const OpApi api("aclnnLogSoftmax");
    return api;
}

const OpApi& softmax_backward_api() {
    static const OpApi api("aclnnSoftmaxBackward");
    return api;
}

const OpApi& log_softmax_backward_api() {
    static const OpApi api("aclnnLogSoftmaxBackward");
    return api;
}

// The dtype a reduction produces: the requested one, or the input's with
// integer inputs widening to a 64-bit count.
DType reduce_out_dtype(const Tensor& self, DType requested) {
    if (requested != DType::Undefined) return requested;
    if (isIntegralType(self.dtype(), /*includeBool=*/true)) return DType::Int64;
    return self.dtype();
}

bool native_reduce(const Tensor& self, DType out_dtype, const OpApi& api) {
    return self.device().is_privateuse1() && self.dim() <= kMaxAclDims &&
           is_native_float(self.dtype()) && is_native_float(out_dtype) &&
           self.numel() > 0 && !has_negative_stride(self) && api.valid();
}

enum class ReduceMode { Sum, Mean };

// Reduce ``self`` over ``dims`` (all when empty) into ``out_dtype``; ``Mean``
// divides by the number of reduced elements.  The operator reads the
// operand and produces the requested type; the reduced axes stay size one
// for it and are viewed away afterwards.
Tensor reduced(const Tensor& self, DType out_dtype, std::vector<int64_t> dims, bool keepdim,
               ReduceMode mode) {
    const int index = index_of(self.device());
    const Tensor& x = self;
    const int64_t nd = x.dim();
    if (nd == 0) {
        // A single element: the reduction only converts the type.
        for (int64_t d : dims) {
            TP_CHECK(d == 0 || d == -1, "reduction: dimension ", d,
                     " out of range for a 0-d tensor");
        }
        Tensor out = empty_on({}, out_dtype, index);
        strided_copy(out, x);
        return keepdim ? out.view({1}) : out;
    }
    if (dims.empty()) {
        dims.resize(static_cast<size_t>(nd));
        for (int64_t i = 0; i < nd; ++i) dims[static_cast<size_t>(i)] = i;
    }
    std::vector<bool> reduced_axis(static_cast<size_t>(nd), false);
    for (int64_t d : dims) {
        if (d < 0) d += nd;
        TP_CHECK(d >= 0 && d < nd, "reduction: dimension ", d, " out of range for rank ", nd);
        reduced_axis[static_cast<size_t>(d)] = true;
    }
    std::vector<int64_t> axes;
    std::vector<int64_t> keep_shape = x.sizes().vec();
    std::vector<int64_t> final_shape;
    for (int64_t i = 0; i < nd; ++i) {
        if (reduced_axis[static_cast<size_t>(i)]) {
            axes.push_back(i);
            keep_shape[static_cast<size_t>(i)] = 1;
            if (keepdim) final_shape.push_back(1);
        } else {
            final_shape.push_back(x.sizes()[static_cast<size_t>(i)]);
        }
    }
    Tensor out = empty_on(keep_shape, out_dtype, index);
    aclDataType acl_out = ACL_DT_UNDEFINED;
    TP_CHECK(acl_dtype_of(out_dtype, &acl_out), "reduction: no device element type for the result");
    AclArgs args;
    const AclTensorRef xr = args.tensor(x);
    const AclIntArrayRef dimsr = args.ints(axes);
    const AclTensorRef outr = args.tensor(out);
    const OpApi& api = mode == ReduceMode::Mean ? mean_api() : sum_api();
    api.run(index, xr, dimsr, true, acl_out, outr);
    return keepdim ? out : out.view(final_shape);
}

Tensor sum_kernel(const Tensor& self, DType dtype) {
    const DType out = reduce_out_dtype(self, dtype);
    if (!native_reduce(self, out, sum_api())) {
        return cpu_fallback::Kernel<"sum", sig::sum>::call(self, dtype);
    }
    return reduced(self, out, {}, false, ReduceMode::Sum);
}

Tensor sum_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim,
                      DType dtype) {
    const DType out = reduce_out_dtype(self, dtype);
    if (!native_reduce(self, out, sum_api())) {
        return cpu_fallback::Kernel<"sum.dim_IntList", sig::sum__dim_IntList>::call(self, dim,
                                                                                   keepdim, dtype);
    }
    return reduced(self, out, dim, keepdim, ReduceMode::Sum);
}

Tensor mean_kernel(const Tensor& self, DType dtype) {
    const DType out = reduce_out_dtype(self, dtype);
    if (!native_reduce(self, out, mean_api())) {
        return cpu_fallback::Kernel<"mean", sig::mean>::call(self, dtype);
    }
    return reduced(self, out, {}, false, ReduceMode::Mean);
}

Tensor mean_dim_kernel(const Tensor& self, const std::vector<int64_t>& dim, bool keepdim,
                       DType dtype) {
    const DType out = reduce_out_dtype(self, dtype);
    // No axis named leaves the divisor one, which is a plain total.
    const ReduceMode mode = dim.empty() ? ReduceMode::Sum : ReduceMode::Mean;
    if (!native_reduce(self, out, mode == ReduceMode::Mean ? mean_api() : sum_api())) {
        return cpu_fallback::Kernel<"mean.dim", sig::mean__dim>::call(self, dim, keepdim, dtype);
    }
    return reduced(self, out, dim, keepdim, mode);
}

int64_t softmax_axis(const Tensor& self, int64_t dim) {
    const int64_t nd = self.dim() == 0 ? 1 : self.dim();
    const int64_t axis = dim < 0 ? dim + nd : dim;
    TP_CHECK(axis >= 0 && axis < nd, "softmax: dimension ", dim, " out of range for rank ",
             self.dim());
    return axis;
}

bool native_softmax(const Tensor& self) {
    return self.device().is_privateuse1() && is_native_float(self.dtype()) &&
           self.dim() <= kMaxAclDims && !has_negative_stride(self);
}

Tensor softmax_forward(const Tensor& self, int64_t dim, const OpApi& api) {
    const int index = index_of(self.device());
    const Tensor x = compute_operand(self, self.dtype());
    Tensor out = empty_on(x.sizes().vec(), x.dtype(), index);
    if (out.numel() == 0) return out;
    AclArgs args;
    const AclTensorRef xr = args.tensor(x);
    const AclTensorRef outr = args.tensor(out);
    api.run(index, xr, softmax_axis(self, dim), outr);
    return out;
}

Tensor softmax_kernel(const Tensor& self, int64_t dim, bool half_to_float) {
    if (half_to_float || !native_softmax(self) || !softmax_api().valid()) {
        return cpu_fallback::Kernel<"_softmax", sig::_softmax>::call(self, dim, half_to_float);
    }
    return softmax_forward(self, dim, softmax_api());
}

Tensor log_softmax_kernel(const Tensor& self, int64_t dim, bool half_to_float) {
    if (half_to_float || !native_softmax(self) || !log_softmax_api().valid()) {
        return cpu_fallback::Kernel<"_log_softmax", sig::_log_softmax>::call(self, dim,
                                                                             half_to_float);
    }
    return softmax_forward(self, dim, log_softmax_api());
}

bool native_softmax_backward(const Tensor& grad_output, const Tensor& output, DType input_dtype) {
    return grad_output.device().is_privateuse1() && output.device().is_privateuse1() &&
           is_native_float(grad_output.dtype()) && output.dtype() == grad_output.dtype() &&
           input_dtype == grad_output.dtype() &&
           grad_output.sizes().vec() == output.sizes().vec() &&
           grad_output.dim() <= kMaxAclDims && !has_negative_stride(grad_output) &&
           !has_negative_stride(output);
}

Tensor softmax_backward(const Tensor& grad_output, const Tensor& output, int64_t dim,
                        const OpApi& api) {
    const int index = index_of(grad_output.device());
    const Tensor dy = compute_operand(grad_output, grad_output.dtype());
    const Tensor y = compute_operand(output, grad_output.dtype());
    Tensor dx = empty_on(dy.sizes().vec(), dy.dtype(), index);
    if (dx.numel() == 0) return dx;
    AclArgs args;
    const AclTensorRef dyr = args.tensor(dy);
    const AclTensorRef yr = args.tensor(y);
    const AclTensorRef dxr = args.tensor(dx);
    api.run(index, dyr, yr, softmax_axis(grad_output, dim), dxr);
    return dx;
}

Tensor softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim,
                               DType input_dtype) {
    if (!native_softmax_backward(grad_output, output, input_dtype) ||
        !softmax_backward_api().valid()) {
        return cpu_fallback::Kernel<"_softmax_backward_data", sig::_softmax_backward_data>::call(
            grad_output, output, dim, input_dtype);
    }
    return softmax_backward(grad_output, output, dim, softmax_backward_api());
}

Tensor log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim,
                                   DType input_dtype) {
    if (!native_softmax_backward(grad_output, output, input_dtype) ||
        !log_softmax_backward_api().valid()) {
        return cpu_fallback::Kernel<"_log_softmax_backward_data",
                                    sig::_log_softmax_backward_data>::call(grad_output, output,
                                                                          dim, input_dtype);
    }
    return softmax_backward(grad_output, output, dim, log_softmax_backward_api());
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, NPUReduceKernels) {
    m.impl("sum", static_cast<sig::sum*>(&sum_kernel));
    m.impl("sum.dim_IntList", static_cast<sig::sum__dim_IntList*>(&sum_dim_kernel));
    m.impl("mean", static_cast<sig::mean*>(&mean_kernel));
    m.impl("mean.dim", static_cast<sig::mean__dim*>(&mean_dim_kernel));
    m.impl("_softmax", static_cast<sig::_softmax*>(&softmax_kernel));
    m.impl("_log_softmax", static_cast<sig::_log_softmax*>(&log_softmax_kernel));
    m.impl("_softmax_backward_data",
           static_cast<sig::_softmax_backward_data*>(&softmax_backward_kernel));
    m.impl("_log_softmax_backward_data",
           static_cast<sig::_log_softmax_backward_data*>(&log_softmax_backward_kernel));
}

} // namespace npu
} // namespace tensorplay

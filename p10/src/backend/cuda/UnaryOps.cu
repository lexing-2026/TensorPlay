// The delegated unary operation resolves to the CUDA vectorized kernel in
// PointwiseKernels.cu.

#include "Tensor.h"
#include "Dispatcher.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay::cuda {

namespace ops = tensorplay::tpx::ops;

Tensor absolute_native_cuda(const Tensor& self) { return ops::abs(self); }
Tensor& absolute__native_cuda(Tensor& self) { return ops::abs_(self); }

Tensor arccos_native_cuda(const Tensor& self) { return ops::acos(self); }
Tensor& arccos__native_cuda(Tensor& self) { return ops::acos_(self); }

Tensor arccosh_native_cuda(const Tensor& self) { return ops::acosh(self); }
Tensor& arccosh__native_cuda(Tensor& self) { return ops::acosh_(self); }

Tensor arcsin_native_cuda(const Tensor& self) { return ops::asin(self); }
Tensor& arcsin__native_cuda(Tensor& self) { return ops::asin_(self); }

Tensor arcsinh_native_cuda(const Tensor& self) { return ops::asinh(self); }
Tensor& arcsinh__native_cuda(Tensor& self) { return ops::asinh_(self); }

Tensor arctan_native_cuda(const Tensor& self) { return ops::atan(self); }
Tensor& arctan__native_cuda(Tensor& self) { return ops::atan_(self); }

Tensor arctanh_native_cuda(const Tensor& self) { return ops::atanh(self); }
Tensor& arctanh__native_cuda(Tensor& self) { return ops::atanh_(self); }

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeUnaryOps) {
    m.impl("absolute", absolute_native_cuda);
    m.impl("absolute_", absolute__native_cuda);
    m.impl("arccos", arccos_native_cuda);
    m.impl("arccos_", arccos__native_cuda);
    m.impl("arccosh", arccosh_native_cuda);
    m.impl("arccosh_", arccosh__native_cuda);
    m.impl("arcsin", arcsin_native_cuda);
    m.impl("arcsin_", arcsin__native_cuda);
    m.impl("arcsinh", arcsinh_native_cuda);
    m.impl("arcsinh_", arcsinh__native_cuda);
    m.impl("arctan", arctan_native_cuda);
    m.impl("arctan_", arctan__native_cuda);
    m.impl("arctanh", arctanh_native_cuda);
    m.impl("arctanh_", arctanh__native_cuda);
}

} // namespace tensorplay::cuda

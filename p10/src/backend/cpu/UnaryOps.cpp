// Unary aliases forward to the concrete primitive, whose CPU kernel lives in
// PointwiseKernels.cpp.

#include "Tensor.h"
#include "Dispatcher.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay::cpu {

namespace ops = tensorplay::tpx::ops;

Tensor absolute_native_cpu(const Tensor& self) { return ops::abs(self); }
Tensor& absolute__native_cpu(Tensor& self) { return ops::abs_(self); }

Tensor arccos_native_cpu(const Tensor& self) { return ops::acos(self); }
Tensor& arccos__native_cpu(Tensor& self) { return ops::acos_(self); }

Tensor arccosh_native_cpu(const Tensor& self) { return ops::acosh(self); }
Tensor& arccosh__native_cpu(Tensor& self) { return ops::acosh_(self); }

Tensor arcsin_native_cpu(const Tensor& self) { return ops::asin(self); }
Tensor& arcsin__native_cpu(Tensor& self) { return ops::asin_(self); }

Tensor arcsinh_native_cpu(const Tensor& self) { return ops::asinh(self); }
Tensor& arcsinh__native_cpu(Tensor& self) { return ops::asinh_(self); }

Tensor arctan_native_cpu(const Tensor& self) { return ops::atan(self); }
Tensor& arctan__native_cpu(Tensor& self) { return ops::atan_(self); }

Tensor arctanh_native_cpu(const Tensor& self) { return ops::atanh(self); }
Tensor& arctanh__native_cpu(Tensor& self) { return ops::atanh_(self); }

TENSORPLAY_LIBRARY_IMPL(CPU, NativeUnaryOps) {
    m.impl("absolute", absolute_native_cpu);
    m.impl("absolute_", absolute__native_cpu);
    m.impl("arccos", arccos_native_cpu);
    m.impl("arccos_", arccos__native_cpu);
    m.impl("arccosh", arccosh_native_cpu);
    m.impl("arccosh_", arccosh__native_cpu);
    m.impl("arcsin", arcsin_native_cpu);
    m.impl("arcsin_", arcsin__native_cpu);
    m.impl("arcsinh", arcsinh_native_cpu);
    m.impl("arcsinh_", arcsinh__native_cpu);
    m.impl("arctan", arctan_native_cpu);
    m.impl("arctan_", arctan__native_cpu);
    m.impl("arctanh", arctanh_native_cpu);
    m.impl("arctanh_", arctanh__native_cpu);
}

} // namespace tensorplay::cpu

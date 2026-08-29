// Composite kernels: isin (3 overloads) and is_nonzero.
// single-element truthiness check.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "TypePromotion.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <complex>
#include <cstdint>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

void isin_dtype_check(DType dt) {
    if (dt == DType::Bool || isComplexType(dt)) {
        TP_THROW(NotImplementedError,
                 "isin is not supported for bool/complex dtypes");
    }
}

} // anonymous namespace

Tensor isin_tensor_tensor_native(const Tensor& elements,
                                 const Tensor& test_elements,
                                 bool /*assume_unique*/, bool invert) {
    isin_dtype_check(elements.dtype());
    isin_dtype_check(test_elements.dtype());
    if (elements.numel() == 0) {
        return ops::full_like(elements, Scalar(invert), DType::Bool);
    }
    const Tensor test_flat = ops::reshape(test_elements, {-1});
    Tensor result = ops::any(
        ops::eq(ops::unsqueeze(elements, -1), test_flat), {-1}, false);
    if (invert) result = ops::logical_not(result);
    return result;
}

Tensor isin_tensor_scalar_native(const Tensor& elements,
                                 const Scalar& test_element,
                                 bool /*assume_unique*/, bool invert) {
    isin_dtype_check(elements.dtype());
    Tensor result = invert ? ops::ne(elements, test_element)
                           : ops::eq(elements, test_element);
    return result;
}

Tensor isin_scalar_tensor_native(const Scalar& element,
                                 const Tensor& test_elements,
                                 bool /*assume_unique*/, bool invert) {
    isin_dtype_check(test_elements.dtype());
    // bool result.
    Tensor result = ops::any(ops::eq(test_elements, element));
    if (invert) result = ops::logical_not(result);
    return result;
}

bool is_nonzero_native(const Tensor& self) {
    const int64_t n = self.numel();
    if (n == 0) {
        TP_THROW(RuntimeError,
                 "Boolean value of Tensor with no values is ambiguous");
    }
    if (n > 1) {
        TP_THROW(RuntimeError,
                 "Boolean value of Tensor with more than one value is ambiguous");
    }
    const Scalar s = self.item();
    if (s.isComplex()) {
        const auto c = s.to<std::complex<double>>();
        return c.real() != 0.0 || c.imag() != 0.0;
    }
    return s.toDouble() != 0.0;
}

TENSORPLAY_LIBRARY_IMPL(Composite, TensorCompareComposite) {
    m.impl("isin.Tensor_Tensor", isin_tensor_tensor_native);
    m.impl("isin.Tensor_Scalar", isin_tensor_scalar_native);
    m.impl("isin.Scalar_Tensor", isin_scalar_tensor_native);
    m.impl("is_nonzero", is_nonzero_native);
}

} // namespace composite
} // namespace tensorplay

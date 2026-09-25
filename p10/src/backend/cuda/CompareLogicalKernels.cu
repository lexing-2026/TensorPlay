// Core operators - CUDA kernels: comparison, logical-connective and floating-point predicate
// operators.
//
// Pointwise kernels over a grid-stride loop; the shared plumbing comes
// from the pointwise header and the per-operator bodies live here.

#include "OpsPointwiseCommon.cuh"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "TypePromotion.h"
#include "CUDARuntime.h"
#include "CUDALoops.cuh"
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <optional>
#include <string>

namespace tensorplay {
namespace cuda {
namespace {

Tensor greater_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, HFn10{}, "greater");
}

Tensor greater_equal_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, HFn11{}, "greater_equal");
}

Tensor less_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, HFn12{}, "less");
}

Tensor less_equal_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, HFn13{}, "less_equal");
}

Tensor not_equal_cuda(const Tensor& a, const Tensor& b) {
    return binary_bool_cuda(a, b, HFn14{}, "not_equal");
}

Tensor signbit_cuda(const Tensor& self) {
    return bool_unary_cuda(self, HFn15{}, "signbit");
}

Tensor logical_not_cuda(const Tensor& self) {
    return logical_unary_cuda(self, HFn16{},
                              "logical_not");
}

Tensor logical_and_cuda(const Tensor& a, const Tensor& b) {
    return logical_binary_cuda(a, b, HFn17{}, "logical_and");
}

Tensor logical_or_cuda(const Tensor& a, const Tensor& b) {
    return logical_binary_cuda(a, b, HFn18{}, "logical_or");
}

Tensor logical_xor_cuda(const Tensor& a, const Tensor& b) {
    return logical_binary_cuda(a, b, HFn19{}, "logical_xor");
}

Tensor isfinite_cuda(const Tensor& self) {
    return bool_unary_cuda(self, HFn20{}, "isfinite");
}

Tensor isinf_cuda(const Tensor& self) {
    return bool_unary_cuda(self, HFn21{}, "isinf");
}

Tensor isnan_cuda(const Tensor& self) {
    return bool_unary_cuda(self, HFn22{}, "isnan");
}

Tensor isneginf_cuda(const Tensor& self) {
    return bool_unary_cuda(self, HFn23{}, "isneginf");
}

Tensor isposinf_cuda(const Tensor& self) {
    return bool_unary_cuda(self, HFn24{}, "isposinf");
}

// ===========================================================================
// Math functions
// ===========================================================================

Tensor isnan_cuda(const Tensor& self);

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, CompareLogicalKernels) {
    m.impl("greater", greater_cuda);
    m.impl("greater_equal", greater_equal_cuda);
    m.impl("less", less_cuda);
    m.impl("less_equal", less_equal_cuda);
    m.impl("not_equal", not_equal_cuda);
    m.impl("signbit", signbit_cuda);
    m.impl("logical_not", logical_not_cuda);
    m.impl("logical_and", logical_and_cuda);
    m.impl("logical_or", logical_or_cuda);
    m.impl("logical_xor", logical_xor_cuda);
    m.impl("isfinite", isfinite_cuda);
    m.impl("isinf", isinf_cuda);
    m.impl("isnan", isnan_cuda);
    m.impl("isneginf", isneginf_cuda);
    m.impl("isposinf", isposinf_cuda);
}

}  // namespace cuda
}  // namespace tensorplay

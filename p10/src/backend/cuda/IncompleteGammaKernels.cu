#include "SpecialKernelUtils.cuh"

#include <SpecialMath.h>

namespace tensorplay::cuda {
namespace {

using special_detail::typed_binary_cuda;
using tensorplay::special_math::calc_igamma;
using tensorplay::special_math::calc_igammac;

struct IgammaFn {
    template <typename T>
    __device__ T operator()(T a, T x) const { return calc_igamma(a, x); }
};

struct IgammacFn {
    template <typename T>
    __device__ T operator()(T a, T x) const { return calc_igammac(a, x); }
};

Tensor gammainc_cuda(const Tensor& a, const Tensor& x) {
    return typed_binary_cuda(a, x, IgammaFn{});
}

Tensor gammaincc_cuda(const Tensor& a, const Tensor& x) {
    return typed_binary_cuda(a, x, IgammacFn{});
}

}  // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, IncompleteGammaKernels) {
    m.impl("gammainc", gammainc_cuda);
    m.impl("gammaincc", gammaincc_cuda);
    m.impl("igamma", gammainc_cuda);
    m.impl("igammac", gammaincc_cuda);
}

}  // namespace tensorplay::cuda

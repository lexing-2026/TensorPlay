#include "cpu/ActivationUnaryKernels.h"
#include "cpu/VecUnary.h"
#include "cpu/vec/vec.h"

// Tier-compiled activation kernels (see TP_CPU_KERNEL_SRCS in
// p10/CMakeLists.txt).  Each copy lands in the CPU_CAPABILITY inline
// namespace; DispatchStub picks the best registered tier at runtime.
// NB: REGISTER_DISPATCH under CPU_CAPABILITY_AVX512 intentionally registers
// REGISTER_AVX512_DISPATCH below, otherwise these ops silently fall back to
// the AVX2 copy.

namespace tensorplay {
namespace cpu {
inline namespace CPU_CAPABILITY {

namespace {
void sigmoid_f32_impl(const float* src, float* dst, int64_t n) {
    vecunary::run_f32(vecunary::VOp::Sigmoid, {}, src, dst, 0, n);
}
void silu_f32_impl(const float* src, float* dst, int64_t n) {
    vecunary::run_f32(vecunary::VOp::Silu, {}, src, dst, 0, n);
}
void sigmoid_f64_impl(const double* src, double* dst, int64_t n) {
    vecunary::run_f64(vecunary::VOp::Sigmoid, {}, src, dst, 0, n);
}
void silu_f64_impl(const double* src, double* dst, int64_t n) {
    vecunary::run_f64(vecunary::VOp::Silu, {}, src, dst, 0, n);
}
} // anonymous namespace
} // inline namespace CPU_CAPABILITY

// One slot per tier TU (the specializations live outside the capability
// namespace, so cross-tier duplicates collide at link time): DEFAULT/AVX2
// copies register their own slot; the AVX512 copy uses ALSO_ instead of
// REGISTER_DISPATCH, which would otherwise null its slot (opt-in design).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(sigmoid_f32_stub, &sigmoid_f32_impl);
REGISTER_DISPATCH(silu_f32_stub, &silu_f32_impl);
REGISTER_DISPATCH(sigmoid_f64_stub, &sigmoid_f64_impl);
REGISTER_DISPATCH(silu_f64_stub, &silu_f64_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(sigmoid_f32_stub, &sigmoid_f32_impl);
ALSO_REGISTER_AVX512_DISPATCH(silu_f32_stub, &silu_f32_impl);
ALSO_REGISTER_AVX512_DISPATCH(sigmoid_f64_stub, &sigmoid_f64_impl);
ALSO_REGISTER_AVX512_DISPATCH(silu_f64_stub, &silu_f64_impl);
#endif

} // namespace cpu
} // namespace tensorplay

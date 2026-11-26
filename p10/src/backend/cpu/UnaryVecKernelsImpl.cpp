// Tier-compiled walk for the unary transcendental map (f32).
//
// Every tier compiles its own copy of this file and registers the slot declared
// in cpu/UnaryVecKernels.h; `vec::Vectorized<float>` resolves to that tier's
// width, and its transcendental members are the same SLEEF-backed bodies the
// GCC build runs through vecunary::run_f32.  The DEFAULT copy (no AVX2) keeps
// the emulated width, so an old host stays correct.
//
// The scalar tail goes through vecunary::scalar_apply, the same reference the
// vector bodies of VecUnary.h are written against, so the last partial vector
// rounds exactly like the rest of the map.

#include "cpu/UnaryVecKernels.h"

#include "DType.h"
#include "Exception.h"
#include "cpu/VecUnary.h"
#include "cpu/vec/vec.h"

#include <cstdint>

namespace tensorplay {
namespace cpu {

namespace {

template <typename T>
bool unary_chunk(const T* s, T* d, int64_t begin, int64_t end, int op,
                 const vecunary::VParams& prm) {
    using V = vec::Vectorized<T>;
    constexpr int64_t W = V::size();
    int64_t i = begin;

    // Vector body: one case per op whose form is a single member call on the
    // vector type, plus the composed activations the hot list leans on.
    switch (static_cast<vecunary::VOp>(op)) {
        case vecunary::VOp::Exp:
            for (; i + W <= end; i += W) V::loadu(s + i).exp().store(d + i);
            break;
        case vecunary::VOp::Exp2:
            for (; i + W <= end; i += W) V::loadu(s + i).exp2().store(d + i);
            break;
        case vecunary::VOp::Expm1:
            for (; i + W <= end; i += W) V::loadu(s + i).expm1().store(d + i);
            break;
        case vecunary::VOp::Log:
            for (; i + W <= end; i += W) V::loadu(s + i).log().store(d + i);
            break;
        case vecunary::VOp::Log2:
            for (; i + W <= end; i += W) V::loadu(s + i).log2().store(d + i);
            break;
        case vecunary::VOp::Log10:
            for (; i + W <= end; i += W) V::loadu(s + i).log10().store(d + i);
            break;
        case vecunary::VOp::Log1p:
            for (; i + W <= end; i += W) V::loadu(s + i).log1p().store(d + i);
            break;
        case vecunary::VOp::Sin:
            for (; i + W <= end; i += W) V::loadu(s + i).sin().store(d + i);
            break;
        case vecunary::VOp::Cos:
            for (; i + W <= end; i += W) V::loadu(s + i).cos().store(d + i);
            break;
        case vecunary::VOp::Tan:
            for (; i + W <= end; i += W) V::loadu(s + i).tan().store(d + i);
            break;
        case vecunary::VOp::Asin:
            for (; i + W <= end; i += W) V::loadu(s + i).asin().store(d + i);
            break;
        case vecunary::VOp::Acos:
            for (; i + W <= end; i += W) V::loadu(s + i).acos().store(d + i);
            break;
        case vecunary::VOp::Atan:
            for (; i + W <= end; i += W) V::loadu(s + i).atan().store(d + i);
            break;
        case vecunary::VOp::Sinh:
            for (; i + W <= end; i += W) V::loadu(s + i).sinh().store(d + i);
            break;
        case vecunary::VOp::Cosh:
            for (; i + W <= end; i += W) V::loadu(s + i).cosh().store(d + i);
            break;
        case vecunary::VOp::Tanh:
            for (; i + W <= end; i += W) V::loadu(s + i).tanh().store(d + i);
            break;
        case vecunary::VOp::Asinh:
            for (; i + W <= end; i += W) V::loadu(s + i).asinh().store(d + i);
            break;
        case vecunary::VOp::Acosh:
            for (; i + W <= end; i += W) V::loadu(s + i).acosh().store(d + i);
            break;
        case vecunary::VOp::Atanh:
            for (; i + W <= end; i += W) V::loadu(s + i).atanh().store(d + i);
            break;
        case vecunary::VOp::Erf:
            for (; i + W <= end; i += W) V::loadu(s + i).erf().store(d + i);
            break;
        case vecunary::VOp::Erfc:
            for (; i + W <= end; i += W) V::loadu(s + i).erfc().store(d + i);
            break;
        case vecunary::VOp::Sigmoid:
            for (; i + W <= end; i += W) {
                const V x = V::loadu(s + i);
                (V(T(1)) / (V(T(1)) + x.neg().exp())).store(d + i);
            }
            break;
        case vecunary::VOp::Silu:
            for (; i + W <= end; i += W) {
                const V x = V::loadu(s + i);
                (x / (V(T(1)) + x.neg().exp())).store(d + i);
            }
            break;
        case vecunary::VOp::GeluNone:
            for (; i + W <= end; i += W) {
                const V x = V::loadu(s + i);
                const T kAlpha = static_cast<T>(0.70710678118654752440);
                (T(0.5) * x * (V(T(1)) + (x * V(kAlpha)).erf())).store(d + i);
            }
            break;
        case vecunary::VOp::GeluTanh:
            for (; i + W <= end; i += W) {
                const V x = V::loadu(s + i);
                // kBeta = 1.41421356237309504880 * 1.12837916709551257390 * 0.5
                const T kBeta = static_cast<T>(0.79788456080286535588);
                const T kKappa = static_cast<T>(0.044715);
                const V inner = V(kBeta) * (x + V(kKappa) * x * x * x);
                (T(0.5) * x * (V(T(1)) + inner.tanh())).store(d + i);
            }
            break;
        case vecunary::VOp::Mish:
            for (; i + W <= end; i += W) {
                const V x = V::loadu(s + i);
                const V sp = (V(T(1)) + x.exp()).log();
                (x * sp.tanh()).store(d + i);
            }
            break;
        default:
            return false;
    }

    for (; i < end; ++i) {
        d[i] = vecunary::scalar_apply(static_cast<vecunary::VOp>(op), prm, s[i]);
    }
    return true;
}

bool unary_vec_impl(const void* src, void* dst, int64_t begin,
                    int64_t end, int op, double p0, double p1, double p2,
                    int dtype) {
    if (dtype != static_cast<int>(DType::Float32)) {
        return false;
    }
    vecunary::VParams prm;
    prm.p0 = p0;
    prm.p1 = p1;
    prm.p2 = p2;
    return unary_chunk(static_cast<const float*>(src),
                       static_cast<float*>(dst), begin, end, op, prm);
}

} // namespace

#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(unary_vec_stub, &unary_vec_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(unary_vec_stub, &unary_vec_impl);
#endif

} // namespace cpu
} // namespace tensorplay


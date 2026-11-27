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
        // Cheap dtype-preserving maps.  On GCC these come from VecUnary.h; on
        // MSVC the header's bodies do not compile and unary_op_kernel would
        // otherwise run the scalar lambda one element at a time (abs over 4K
        // elements: 3.5 us here against 0.3 us for the vector walk).
        case vecunary::VOp::Abs:
            for (; i + W <= end; i += W) V::loadu(s + i).abs().store(d + i);
            break;
        case vecunary::VOp::Neg:
            for (; i + W <= end; i += W) V::loadu(s + i).neg().store(d + i);
            break;
        case vecunary::VOp::Square:
            for (; i + W <= end; i += W) {
                const V x = V::loadu(s + i);
                (x * x).store(d + i);
            }
            break;
        case vecunary::VOp::Relu:
            // x < 0 ? 0 : x, so a NaN lane stays NaN exactly like the scalar
            // functor (clamp_min is that same select).
            for (; i + W <= end; i += W) {
                vec::clamp_min(V::loadu(s + i), V(T(0))).store(d + i);
            }
            break;
        case vecunary::VOp::Reciprocal:
            // The IEEE quotient 1 / x, as the scalar map computes it; the
            // approximate reciprocal member rounds differently.
            for (; i + W <= end; i += W) {
                (V(T(1)) / V::loadu(s + i)).store(d + i);
            }
            break;
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
            // Two vectors per iteration, the way ATen's vectorized_loop does
            // it (native/cpu/Loops.h L230): erf is a reciprocal plus an
            // exponential chained on that result, so one vector at a time
            // leaves the pipeline waiting on the chain.
            for (; i + 2 * W <= end; i += 2 * W) {
                V::loadu(s + i).erf().store(d + i);
                V::loadu(s + i + W).erf().store(d + i + W);
            }
            for (; i + W <= end; i += W) V::loadu(s + i).erf().store(d + i);
            break;
        case vecunary::VOp::Erfc:
            for (; i + 2 * W <= end; i += 2 * W) {
                V::loadu(s + i).erfc().store(d + i);
                V::loadu(s + i + W).erfc().store(d + i + W);
            }
            for (; i + W <= end; i += W) V::loadu(s + i).erfc().store(d + i);
            break;
        case vecunary::VOp::Hardtanh: {
            // clamp(x, lo, hi) keeps a NaN lane a NaN lane: the tier bodies are
            // max(min, x) then min(max, x), and a max whose second operand is
            // NaN answers that NaN.
            const V lo(static_cast<T>(prm.p0));
            const V hi(static_cast<T>(prm.p1));
            for (; i + W <= end; i += W)
                clamp_max(clamp_min(V::loadu(s + i), lo), hi).store(d + i);
            break;
        }
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
        case vecunary::VOp::GeluNone: {
            // Two vectors per iteration, the way ATen's vectorized_loop does it
            // (native/cpu/Loops.h L230): the erf chain is a reciprocal, a Horner
            // tail, an exponential and a product, all dependent, and a wider
            // unroll buys nothing -- at one thread the chain already runs at the
            // 512-bit throughput limit (0.35 ns/element) and at eight threads
            // the box is below its all-core power ceiling for either shape
            // (0.112 ms per 1M against 0.107-0.126 for torch, run to run).
            const T kAlpha = static_cast<T>(0.70710678118654752440);
            for (; i + 2 * W <= end; i += 2 * W) {
                const V x0 = V::loadu(s + i);
                const V x1 = V::loadu(s + i + W);
                (T(0.5) * x0 * (V(T(1)) + (x0 * V(kAlpha)).erf())).store(d + i);
                (T(0.5) * x1 * (V(T(1)) + (x1 * V(kAlpha)).erf())).store(d + i + W);
            }
            for (; i + W <= end; i += W) {
                const V x = V::loadu(s + i);
                (T(0.5) * x * (V(T(1)) + (x * V(kAlpha)).erf())).store(d + i);
            }
            break;
        }
        case vecunary::VOp::GeluTanh:
            for (; i + 2 * W <= end; i += 2 * W) {
                // kBeta = 1.41421356237309504880 * 1.12837916709551257390 * 0.5
                const T kBeta = static_cast<T>(0.79788456080286535588);
                const T kKappa = static_cast<T>(0.044715);
                const V x0 = V::loadu(s + i);
                const V x1 = V::loadu(s + i + W);
                const V inner0 = V(kBeta) * (x0 + V(kKappa) * x0 * x0 * x0);
                const V inner1 = V(kBeta) * (x1 + V(kKappa) * x1 * x1 * x1);
                (T(0.5) * x0 * (V(T(1)) + inner0.tanh())).store(d + i);
                (T(0.5) * x1 * (V(T(1)) + inner1.tanh())).store(d + i + W);
            }
            for (; i + W <= end; i += W) {
                const V x = V::loadu(s + i);
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


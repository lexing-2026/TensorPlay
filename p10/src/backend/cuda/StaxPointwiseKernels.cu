#include "StaxPointwise.h"
#include "CUDARuntime.h"
#include "Macros.h"
#include "Tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef USE_CUDA
#endif

namespace tensorplay {
namespace cuda {
namespace {

#ifdef USE_CUDA

// Program limits mirror the CPU runner: bounded instruction counts keep the
// per-thread temporary pool in registers/local memory instead of global
// scratch.
constexpr int64_t kMaxProgramInstructions = 64;
constexpr int64_t kMaxProgramConstants = 4096;
constexpr int64_t kMaxProgramInputs = 32;

enum class StaxOp : int64_t {
    Add = 1, Sub = 2, Mul = 3, Div = 4, Pow = 5,
    Neg = 6, Pos = 7, Abs = 8, Sin = 9, Cos = 10,
    Exp = 11, Log = 12, Sigmoid = 13, Sqrt = 14, Square = 15,
    Tanh = 16, Relu = 17, ReluGrad = 18, AbsGrad = 19,
    Lt = 20, Le = 21, Gt = 22, Ge = 23, Eq = 24, Ne = 25,
    Where = 26, WhereRest = 27,
    Minimum = 28, Maximum = 29, ClampMin = 30, ClampMax = 31,
    Rsqrt = 32, Exp2 = 33, Erf = 34, Cast = 35,
};

struct Instruction {
    int64_t op;
    int64_t lhs;
    int64_t rhs;
};

// Element-type tags shared with the graph attribute encoding: outputs of one
// program may narrow or widen the float-domain results independently of the
// inputs, so a derivative chain stops paying a separate conversion pass.
constexpr uint8_t kOutF32 = 0;
constexpr uint8_t kOutF64 = 1;
constexpr uint8_t kOutF16 = 2;
constexpr uint8_t kOutBF16 = 3;

uint8_t encode_out_kind(DType dt) {
    switch (dt) {
        case DType::Float32: return kOutF32;
        case DType::Float64: return kOutF64;
        case DType::Float16: return kOutF16;
        case DType::BFloat16: return kOutBF16;
        default:
            throw std::runtime_error(
                "Stax fused pointwise output supports "
                "float16/bfloat16/float32/float64");
    }
}

template <typename compute_t>
__device__ void store_typed(uint8_t kind, void* ptr, int64_t index,
                            compute_t v) {
    switch (kind) {
        case kOutF32:
            static_cast<float*>(ptr)[index] = static_cast<float>(v);
            break;
        case kOutF64:
            static_cast<double*>(ptr)[index] = static_cast<double>(v);
            break;
        case kOutF16:
            static_cast<tensorplay::Half*>(ptr)[index] =
                static_cast<tensorplay::Half>(v);
            break;
        case kOutBF16:
            static_cast<tensorplay::BFloat16*>(ptr)[index] =
                static_cast<tensorplay::BFloat16>(v);
            break;
    }
}

// Stores four consecutive lanes of one output.  Aligned uniform-width
// outputs move in one wide transaction; narrower element types pack into
// their 16-bit patterns first.  Misaligned or unhandled forms fall back to
// the scalar store, so correctness never depends on the wide path firing.
template <typename compute_t>
__device__ void store_group_typed(uint8_t kind, void* ptr, int64_t base,
                                  const compute_t (&lanes)[4]) {
    const bool aligned =
        (reinterpret_cast<uintptr_t>(ptr) & 15) == 0 && (base & 3) == 0;
    if (aligned && kind == kOutF32) {
        const float4 v{static_cast<float>(lanes[0]),
                       static_cast<float>(lanes[1]),
                       static_cast<float>(lanes[2]),
                       static_cast<float>(lanes[3])};
        *reinterpret_cast<float4*>(static_cast<float*>(ptr) + base) = v;
        return;
    }
    if (aligned && kind == kOutF64) {
        double* typed = static_cast<double*>(ptr) + base;
        *reinterpret_cast<double2*>(typed) =
            double2{static_cast<double>(lanes[0]), static_cast<double>(lanes[1])};
        *reinterpret_cast<double2*>(typed + 2) =
            double2{static_cast<double>(lanes[2]), static_cast<double>(lanes[3])};
        return;
    }
    if (aligned && (kind == kOutF16 || kind == kOutBF16)) {
        uint16_t bits[4];
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            bits[lane] = kind == kOutF16
                ? static_cast<tensorplay::Half>(lanes[lane]).x
                : static_cast<tensorplay::BFloat16>(lanes[lane]).x;
        }
        const uint2 w{
            static_cast<unsigned int>(bits[0]) |
                (static_cast<unsigned int>(bits[1]) << 16),
            static_cast<unsigned int>(bits[2]) |
                (static_cast<unsigned int>(bits[3]) << 16)};
        if (kind == kOutF16) {
            *reinterpret_cast<uint2*>(static_cast<tensorplay::Half*>(ptr) + base) = w;
        } else {
            *reinterpret_cast<uint2*>(static_cast<tensorplay::BFloat16*>(ptr) + base) = w;
        }
        return;
    }
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        store_typed(kind, ptr, base + lane, lanes[lane]);
    }
}

// One thread evaluates the full program for one element.  Operands resolve
// through a device-resident pointer table (inputs) and a per-thread
// temporary pool (instruction results); constants ride a flat buffer.
// Each input carries its own storage kind and widens to `compute_t` on
// load, so a single program can mix element widths without a separate
// conversion pass.
template <typename compute_t, bool Flat>
struct ProgramState {    const Instruction* instructions;
    const compute_t* constants;
    const void* const* input_ptrs;
    const uint8_t* input_kinds;
    const int64_t* input_sizes;
    const int64_t* input_strides;
    const int64_t* output_sizes;
    const int64_t* output_strides;
    const uint8_t* input_flat;
    int uniform_kind;
    int64_t input_count;
    int64_t instruction_count;
    int64_t rank;

    // Loads one input element.  When every input shares one storage kind the
    // kind is a single value, so the load compiles to one predictable branch
    // over a directly typed read; only genuinely mixed programs pay the
    // per-input kind lookup and switch.
    __device__ compute_t load_input(int64_t ref, int64_t index) const {
        const void* ptr = input_ptrs[ref];
        switch (uniform_kind) {
            case kOutF32:
                return static_cast<compute_t>(
                    static_cast<const float*>(ptr)[index]);
            case kOutF64:
                return static_cast<compute_t>(
                    static_cast<const double*>(ptr)[index]);
            case kOutF16:
                return static_cast<compute_t>(
                    static_cast<const tensorplay::Half*>(ptr)[index]);
            case kOutBF16:
                return static_cast<compute_t>(
                    static_cast<const tensorplay::BFloat16*>(ptr)[index]);
            default:
                return load_typed(input_kinds[ref], ptr, index);
        }
    }

    __device__ compute_t load_typed(uint8_t kind, const void* ptr,
                                    int64_t index) const {
        switch (kind) {
            case kOutF32:
                return static_cast<compute_t>(
                    static_cast<const float*>(ptr)[index]);
            case kOutF64:
                return static_cast<compute_t>(
                    static_cast<const double*>(ptr)[index]);
            case kOutF16:
                return static_cast<compute_t>(
                    static_cast<const tensorplay::Half*>(ptr)[index]);
            case kOutBF16:
                return static_cast<compute_t>(
                    static_cast<const tensorplay::BFloat16*>(ptr)[index]);
        }
        return compute_t(0);
    }

    __device__ compute_t resolve(int64_t ref, const compute_t* local_temps,
                                 int64_t element) const {
        if (ref >= 0) {
            if (ref < input_count) {
                if constexpr (Flat) {
                    return load_input(ref, element);
                } else {
                    int64_t offset = element;
                    if (!input_flat[ref]) {
                        offset = 0;
                        for (int64_t dim = 0; dim < rank; ++dim) {
                            const int64_t input_extent =
                                input_sizes[ref * rank + dim];
                            if (input_extent != 1) {
                                const int64_t coordinate =
                                    (element / output_strides[dim]) % output_sizes[dim];
                                offset += coordinate *
                                    input_strides[ref * rank + dim];
                            }
                        }
                    }
                    return load_input(ref, offset);
                }
            }
            return local_temps[ref - input_count];
        }
        return constants[-ref - 1];
    }

    __device__ void evaluate(compute_t* local_temps, int64_t* pending_where,
                             int64_t element) const {
        for (int64_t i = 0; i < instruction_count; ++i) {
            const Instruction& inst = instructions[i];
            const int64_t op = inst.op;
            if (op == static_cast<int64_t>(StaxOp::Where)) {
                // First half of the ternary select: stash (cond, then); the
                // value materializes at the paired WhereRest.
                pending_where[0] = inst.lhs;
                pending_where[1] = inst.rhs;
                pending_where[2] = 1;
                local_temps[i] = compute_t(0);
                continue;
            }
            compute_t value;
            if (op == static_cast<int64_t>(StaxOp::WhereRest)) {
                if (pending_where[2] == 1) {
                    const compute_t cond = resolve(pending_where[0], local_temps, element);
                    const compute_t then_v = resolve(pending_where[1], local_temps, element);
                    const compute_t else_v = resolve(inst.rhs, local_temps, element);
                    value = cond != compute_t(0) ? then_v : else_v;
                    pending_where[2] = 0;
                } else {
                    value = compute_t(0);
                }
                local_temps[i] = value;
                continue;
            }
            const compute_t lhs = resolve(inst.lhs, local_temps, element);
            value = lhs;
            switch (op) {
                case static_cast<int64_t>(StaxOp::Add):
                    value = lhs + resolve(inst.rhs, local_temps, element); break;
                case static_cast<int64_t>(StaxOp::Sub):
                    value = lhs - resolve(inst.rhs, local_temps, element); break;
                case static_cast<int64_t>(StaxOp::Mul):
                    value = lhs * resolve(inst.rhs, local_temps, element); break;
                case static_cast<int64_t>(StaxOp::Div):
                    value = lhs / resolve(inst.rhs, local_temps, element); break;
                case static_cast<int64_t>(StaxOp::Pow):
                    value = ::pow(static_cast<double>(lhs),
                                  static_cast<double>(resolve(inst.rhs, local_temps, element)));
                    break;
                case static_cast<int64_t>(StaxOp::Neg): value = -lhs; break;
                case static_cast<int64_t>(StaxOp::Pos): value = lhs; break;
                case static_cast<int64_t>(StaxOp::Abs): value = ::fabs(lhs); break;
                case static_cast<int64_t>(StaxOp::Sin): value = ::sin(lhs); break;
                case static_cast<int64_t>(StaxOp::Cos): value = ::cos(lhs); break;
                case static_cast<int64_t>(StaxOp::Exp): value = ::exp(lhs); break;
                case static_cast<int64_t>(StaxOp::Log): value = ::log(lhs); break;
                case static_cast<int64_t>(StaxOp::Sigmoid):
                    value = compute_t(1) / (compute_t(1) + ::exp(-lhs)); break;
                case static_cast<int64_t>(StaxOp::Sqrt): value = ::sqrt(lhs); break;
                case static_cast<int64_t>(StaxOp::Square): value = lhs * lhs; break;
                case static_cast<int64_t>(StaxOp::Tanh): value = ::tanh(lhs); break;
                case static_cast<int64_t>(StaxOp::Relu):
                    value = lhs > compute_t(0) ? lhs : compute_t(0); break;
                case static_cast<int64_t>(StaxOp::ReluGrad):
                    value = lhs > compute_t(0) ? compute_t(1) : compute_t(0); break;
                case static_cast<int64_t>(StaxOp::AbsGrad):
                    value = (lhs > compute_t(0) ? compute_t(1) : compute_t(0))
                          - (lhs < compute_t(0) ? compute_t(1) : compute_t(0));
                    break;
                case static_cast<int64_t>(StaxOp::Lt):
                    value = lhs < resolve(inst.rhs, local_temps, element)
                        ? compute_t(1) : compute_t(0); break;
                case static_cast<int64_t>(StaxOp::Le):
                    value = lhs <= resolve(inst.rhs, local_temps, element)
                        ? compute_t(1) : compute_t(0); break;
                case static_cast<int64_t>(StaxOp::Gt):
                    value = lhs > resolve(inst.rhs, local_temps, element)
                        ? compute_t(1) : compute_t(0); break;
                case static_cast<int64_t>(StaxOp::Ge):
                    value = lhs >= resolve(inst.rhs, local_temps, element)
                        ? compute_t(1) : compute_t(0); break;
                case static_cast<int64_t>(StaxOp::Eq):
                    value = lhs == resolve(inst.rhs, local_temps, element)
                        ? compute_t(1) : compute_t(0); break;
                case static_cast<int64_t>(StaxOp::Ne):
                    value = lhs != resolve(inst.rhs, local_temps, element)
                        ? compute_t(1) : compute_t(0); break;
                case static_cast<int64_t>(StaxOp::Minimum): {
                    const compute_t r = resolve(inst.rhs, local_temps, element);
                    value = lhs < r ? lhs : r; break;
                }
                case static_cast<int64_t>(StaxOp::Maximum): {
                    const compute_t r = resolve(inst.rhs, local_temps, element);
                    value = lhs > r ? lhs : r; break;
                }
                case static_cast<int64_t>(StaxOp::ClampMin): {
                    const compute_t r = resolve(inst.rhs, local_temps, element);
                    value = lhs < r ? r : lhs; break;
                }
                case static_cast<int64_t>(StaxOp::ClampMax): {
                    const compute_t r = resolve(inst.rhs, local_temps, element);
                    value = lhs > r ? r : lhs; break;
                }
                case static_cast<int64_t>(StaxOp::Rsqrt):
                    value = compute_t(1) / ::sqrt(lhs); break;
                case static_cast<int64_t>(StaxOp::Exp2):
                    value = ::exp2(lhs); break;
                case static_cast<int64_t>(StaxOp::Erf):
                    value = ::erf(lhs); break;
                case static_cast<int64_t>(StaxOp::Cast):
                    // The program runs in one compute dtype; the only cast
                    // it can express is the identity, matching the
                    // float-domain code generator's contract.
                    value = lhs; break;
                default:
                    value = compute_t(0); break;
            }
            local_temps[i] = value;
        }
    }

    // --- Group execution (four consecutive elements per thread) ---------
    // Flat programs address every operand with the same base index, so a
    // group of four lanes is four contiguous values per uniform-width
    // input: one wide transaction instead of four scalar reads.  Mixed
    // kinds and misaligned pointers drop to the scalar load per lane.
    __device__ void load_input_group(int64_t ref, int64_t base,
                                     compute_t (&out)[4]) const {
        const void* ptr = input_ptrs[ref];
        const bool aligned = (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
        if (aligned && uniform_kind == kOutF32) {
            const float4 v =
                *reinterpret_cast<const float4*>(static_cast<const float*>(ptr) + base);
            out[0] = static_cast<compute_t>(v.x);
            out[1] = static_cast<compute_t>(v.y);
            out[2] = static_cast<compute_t>(v.z);
            out[3] = static_cast<compute_t>(v.w);
            return;
        }
        if (aligned && uniform_kind == kOutF64) {
            const double* typed = static_cast<const double*>(ptr) + base;
            const double2 a = *reinterpret_cast<const double2*>(typed);
            const double2 b = *reinterpret_cast<const double2*>(typed + 2);
            out[0] = static_cast<compute_t>(a.x);
            out[1] = static_cast<compute_t>(a.y);
            out[2] = static_cast<compute_t>(b.x);
            out[3] = static_cast<compute_t>(b.y);
            return;
        }
        if (aligned && (uniform_kind == kOutF16 || uniform_kind == kOutBF16)) {
            const uint2 w = uniform_kind == kOutF16
                ? *reinterpret_cast<const uint2*>(static_cast<const tensorplay::Half*>(ptr) + base)
                : *reinterpret_cast<const uint2*>(static_cast<const tensorplay::BFloat16*>(ptr) + base);
            const uint16_t bits[4] = {
                static_cast<uint16_t>(w.x & 0xFFFFu),
                static_cast<uint16_t>(w.x >> 16),
                static_cast<uint16_t>(w.y & 0xFFFFu),
                static_cast<uint16_t>(w.y >> 16),
            };
            #pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                out[lane] = uniform_kind == kOutF16
                    ? static_cast<compute_t>(tensorplay::Half(
                          bits[lane], tensorplay::Half::from_bits()))
                    : static_cast<compute_t>(tensorplay::BFloat16(
                          bits[lane], tensorplay::BFloat16::from_bits()));
            }
            return;
        }
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            out[lane] = load_input(ref, base + lane);
        }
    }

    __device__ void resolve_group(int64_t ref,
                                  const compute_t (*local_temps)[4],
                                  compute_t (&out)[4], int64_t base) const {
        if (ref >= 0) {
            if (ref < input_count) {
                load_input_group(ref, base, out);
                return;
            }
            const compute_t* slot = local_temps[ref - input_count];
            #pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                out[lane] = slot[lane];
            }
            return;
        }
        const compute_t c = constants[-ref - 1];
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            out[lane] = c;
        }
    }

    __device__ void evaluate_group(compute_t (*local_temps)[4],
                                   int64_t* pending_where,
                                   int64_t base) const {
        for (int64_t i = 0; i < instruction_count; ++i) {
            const Instruction& inst = instructions[i];
            const int64_t op = inst.op;
            if (op == static_cast<int64_t>(StaxOp::Where)) {
                pending_where[0] = inst.lhs;
                pending_where[1] = inst.rhs;
                pending_where[2] = 1;
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) local_temps[i][lane] = compute_t(0);
                continue;
            }
            if (op == static_cast<int64_t>(StaxOp::WhereRest)) {
                if (pending_where[2] == 1) {
                    compute_t cond[4], then_v[4], else_v[4];
                    resolve_group(pending_where[0], local_temps, cond, base);
                    resolve_group(pending_where[1], local_temps, then_v, base);
                    resolve_group(inst.rhs, local_temps, else_v, base);
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) {
                        local_temps[i][lane] =
                            cond[lane] != compute_t(0) ? then_v[lane] : else_v[lane];
                    }
                    pending_where[2] = 0;
                } else {
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) local_temps[i][lane] = compute_t(0);
                }
                continue;
            }
            compute_t lhs[4], rhs[4], value[4];
            resolve_group(inst.lhs, local_temps, lhs, base);
            resolve_group(inst.rhs, local_temps, rhs, base);
            switch (op) {
                case static_cast<int64_t>(StaxOp::Add):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = lhs[lane] + rhs[lane];
                    break;
                case static_cast<int64_t>(StaxOp::Sub):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = lhs[lane] - rhs[lane];
                    break;
                case static_cast<int64_t>(StaxOp::Mul):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = lhs[lane] * rhs[lane];
                    break;
                case static_cast<int64_t>(StaxOp::Div):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = lhs[lane] / rhs[lane];
                    break;
                case static_cast<int64_t>(StaxOp::Pow):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = ::pow(static_cast<double>(lhs[lane]),
                                            static_cast<double>(rhs[lane]));
                    break;
                case static_cast<int64_t>(StaxOp::Neg):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = -lhs[lane];
                    break;
                case static_cast<int64_t>(StaxOp::Pos):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = lhs[lane];
                    break;
                case static_cast<int64_t>(StaxOp::Abs):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = ::fabs(lhs[lane]);
                    break;
                case static_cast<int64_t>(StaxOp::Sin):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = ::sin(lhs[lane]);
                    break;
                case static_cast<int64_t>(StaxOp::Cos):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = ::cos(lhs[lane]);
                    break;
                case static_cast<int64_t>(StaxOp::Exp):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = ::exp(lhs[lane]);
                    break;
                case static_cast<int64_t>(StaxOp::Log):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = ::log(lhs[lane]);
                    break;
                case static_cast<int64_t>(StaxOp::Sigmoid):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = compute_t(1) / (compute_t(1) + ::exp(-lhs[lane]));
                    break;
                case static_cast<int64_t>(StaxOp::Sqrt):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = ::sqrt(lhs[lane]);
                    break;
                case static_cast<int64_t>(StaxOp::Square):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = lhs[lane] * lhs[lane];
                    break;
                case static_cast<int64_t>(StaxOp::Tanh):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = ::tanh(lhs[lane]);
                    break;
                case static_cast<int64_t>(StaxOp::Relu):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = lhs[lane] > compute_t(0) ? lhs[lane] : compute_t(0);
                    break;
                case static_cast<int64_t>(StaxOp::ReluGrad):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = lhs[lane] > compute_t(0) ? compute_t(1) : compute_t(0);
                    break;
                case static_cast<int64_t>(StaxOp::AbsGrad):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] =
                            (lhs[lane] > compute_t(0) ? compute_t(1) : compute_t(0)) -
                            (lhs[lane] < compute_t(0) ? compute_t(1) : compute_t(0));
                    break;
                case static_cast<int64_t>(StaxOp::Lt):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = lhs[lane] < rhs[lane] ? compute_t(1) : compute_t(0);
                    break;
                case static_cast<int64_t>(StaxOp::Le):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = lhs[lane] <= rhs[lane] ? compute_t(1) : compute_t(0);
                    break;
                case static_cast<int64_t>(StaxOp::Gt):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = lhs[lane] > rhs[lane] ? compute_t(1) : compute_t(0);
                    break;
                case static_cast<int64_t>(StaxOp::Ge):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = lhs[lane] >= rhs[lane] ? compute_t(1) : compute_t(0);
                    break;
                case static_cast<int64_t>(StaxOp::Eq):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = lhs[lane] == rhs[lane] ? compute_t(1) : compute_t(0);
                    break;
                case static_cast<int64_t>(StaxOp::Ne):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = lhs[lane] != rhs[lane] ? compute_t(1) : compute_t(0);
                    break;
                case static_cast<int64_t>(StaxOp::Minimum):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = lhs[lane] < rhs[lane] ? lhs[lane] : rhs[lane];
                    break;
                case static_cast<int64_t>(StaxOp::Maximum):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = lhs[lane] > rhs[lane] ? lhs[lane] : rhs[lane];
                    break;
                case static_cast<int64_t>(StaxOp::ClampMin):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = lhs[lane] < rhs[lane] ? rhs[lane] : lhs[lane];
                    break;
                case static_cast<int64_t>(StaxOp::ClampMax):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = lhs[lane] > rhs[lane] ? rhs[lane] : lhs[lane];
                    break;
                case static_cast<int64_t>(StaxOp::Rsqrt):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane)
                        value[lane] = compute_t(1) / ::sqrt(lhs[lane]);
                    break;
                case static_cast<int64_t>(StaxOp::Exp2):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = ::exp2(lhs[lane]);
                    break;
                case static_cast<int64_t>(StaxOp::Erf):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = ::erf(lhs[lane]);
                    break;
                case static_cast<int64_t>(StaxOp::Cast):
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = lhs[lane];
                    break;
                default:
                    #pragma unroll
                    for (int lane = 0; lane < 4; ++lane) value[lane] = compute_t(0);
                    break;
            }
            #pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                local_temps[i][lane] = value[lane];
            }
        }
    }
};

template <typename compute_t, bool Flat, int kTemps>
__global__ void stax_fused_pointwise_kernel(
    ProgramState<compute_t, Flat> state,
    void* output,
    uint8_t out_kind,
    int64_t count) {
    compute_t temps[kTemps];
    int64_t pending_where[3] = {0, 0, 0};
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; index < count; index += stride) {
        state.evaluate(temps, pending_where, index);
        store_typed(out_kind, output, index,
                    temps[state.instruction_count - 1]);
    }
}

template <typename compute_t, bool Flat, int kTemps>
__global__ void stax_fused_pointwise_multi_kernel(
    ProgramState<compute_t, Flat> state,
    void* const* temp_outputs,
    const uint8_t* out_kinds,
    const int64_t* temp_refs,
    int64_t temp_output_count,
    int64_t count) {
    compute_t temps[kTemps];
    int64_t pending_where[3] = {0, 0, 0};
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; index < count; index += stride) {
        state.evaluate(temps, pending_where, index);
        for (int64_t o = 0; o < temp_output_count; ++o) {
            store_typed(out_kinds[o], temp_outputs[o], index,
                        temps[temp_refs[o] - state.input_count]);
        }
    }
}

// Vectorized flat program: each thread walks groups of four consecutive
// elements, so aligned uniform-width traffic moves in wide transactions and
// the interpreter's per-instruction switch dispatch amortizes across the
// group.  Group-strided tiles keep every wide access in bounds; the
// trailing partial group re-enters the scalar path.
template <typename compute_t, int kTemps>
__global__ void stax_fused_pointwise_vec_kernel(
    ProgramState<compute_t, true> state,
    void* output,
    uint8_t out_kind,
    int64_t count) {
    compute_t temps[kTemps][4];
    int64_t pending_where[3] = {0, 0, 0};
    const int64_t groups = (count + 3) / 4;
    int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; group < groups; group += stride) {
        const int64_t base = group * 4;
        if (base + 4 <= count) {
            state.evaluate_group(temps, pending_where, base);
            store_group_typed(out_kind, output, base,
                              temps[state.instruction_count - 1]);
        } else {
            compute_t scalar_temps[kTemps];
            for (int64_t element = base; element < count; ++element) {
                state.evaluate(scalar_temps, pending_where, element);
                store_typed(out_kind, output, element,
                            scalar_temps[state.instruction_count - 1]);
            }
        }
    }
}

template <int kTemps, typename compute_t, bool Flat>
void launch_program(
    const ProgramState<compute_t, Flat>& state,
    void* output,
    uint8_t out_kind,
    void* const* temp_outputs,
    const uint8_t* out_kinds,
    const int64_t* temp_refs,
    int64_t temp_output_count,
    int64_t count,
    cudaStream_t stream) {
    const int threads = 256;
    // Wide-path eligibility: flat single-output programs over enough
    // elements to amortize the wider register footprint.  Per-pointer
    // alignment is checked on the device, so a misaligned view silently
    // takes the scalar load instead of failing the launch.
    if constexpr (Flat && kTemps <= 16) {
        if (output != nullptr && count >= 2048) {
            const int64_t groups = (count + 3) / 4;
            const int blocks = static_cast<int>((groups + threads - 1) / threads);
            stax_fused_pointwise_vec_kernel<compute_t, kTemps>
                <<<blocks, threads, 0, stream>>>(state, output, out_kind, count);
            checkCuda(cudaGetLastError(), "stax fused pointwise vector launch");
            return;
        }
    }
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    if (output != nullptr) {
        stax_fused_pointwise_kernel<compute_t, Flat, kTemps>
            <<<blocks, threads, 0, stream>>>(state, output, out_kind, count);
    } else {
        stax_fused_pointwise_multi_kernel<compute_t, Flat, kTemps>
            <<<blocks, threads, 0, stream>>>(
                state, temp_outputs, out_kinds, temp_refs, temp_output_count,
                count);
    }
    checkCuda(cudaGetLastError(), "stax fused pointwise launch");
}

template <typename compute_t, bool Flat>
void dispatch_program_launch(
    const ProgramState<compute_t, Flat>& state,
    void* output,
    uint8_t out_kind,
    void* const* temp_outputs,
    const uint8_t* out_kinds,
    const int64_t* temp_refs,
    int64_t temp_output_count,
    int64_t count,
    cudaStream_t stream) {
    if (output != nullptr) {
        if (state.instruction_count <= 8) {
            launch_program<8, compute_t, Flat>(
                state, output, out_kind, nullptr, nullptr, nullptr, 0, count,
                stream);
        } else if (state.instruction_count <= 16) {
            launch_program<16, compute_t, Flat>(
                state, output, out_kind, nullptr, nullptr, nullptr, 0, count,
                stream);
        } else if (state.instruction_count <= 32) {
            launch_program<32, compute_t, Flat>(
                state, output, out_kind, nullptr, nullptr, nullptr, 0, count,
                stream);
        } else {
            launch_program<64, compute_t, Flat>(
                state, output, out_kind, nullptr, nullptr, nullptr, 0, count,
                stream);
        }
        return;
    }
    if (state.instruction_count <= 8) {
        launch_program<8, compute_t, Flat>(
            state, nullptr, out_kind, temp_outputs, out_kinds, temp_refs,
            temp_output_count, count, stream);
    } else if (state.instruction_count <= 16) {
        launch_program<16, compute_t, Flat>(
            state, nullptr, out_kind, temp_outputs, out_kinds, temp_refs,
            temp_output_count, count, stream);
    } else if (state.instruction_count <= 32) {
        launch_program<32, compute_t, Flat>(
            state, nullptr, out_kind, temp_outputs, out_kinds, temp_refs,
            temp_output_count, count, stream);
    } else {
        launch_program<64, compute_t, Flat>(
            state, nullptr, out_kind, temp_outputs, out_kinds, temp_refs,
            temp_output_count, count, stream);
    }
}


std::vector<int64_t> broadcast_shape(const std::vector<Tensor>& inputs) {
    int64_t rank = 0;
    for (const Tensor& input : inputs) {
        rank = std::max(rank, input.dim());
    }
    std::vector<int64_t> shape(static_cast<size_t>(rank), 1);
    for (const Tensor& input : inputs) {
        const int64_t offset = rank - input.dim();
        for (int64_t dim = 0; dim < input.dim(); ++dim) {
            const int64_t extent = input.size(dim);
            const size_t output_dim = static_cast<size_t>(offset + dim);
            if (extent != 1 && shape[output_dim] != 1 &&
                shape[output_dim] != extent) {
                throw std::runtime_error(
                    "Stax CUDA fused pointwise inputs cannot broadcast");
            }
            if (extent != 1) shape[output_dim] = extent;
        }
    }
    return shape;
}

void check_program_shape(const std::vector<Tensor>& inputs,
                         const std::vector<int64_t>& program,
                         const std::vector<double>& constants,
                         int64_t output_count) {
    if (inputs.empty() || program.empty() || program.size() % 3 != 0 ||
        static_cast<int64_t>(program.size() / 3) > kMaxProgramInstructions ||
        static_cast<int64_t>(constants.size()) > kMaxProgramConstants ||
        static_cast<int64_t>(inputs.size()) > kMaxProgramInputs ||
        output_count < 1) {
        throw std::runtime_error("Stax CUDA fused pointwise program is malformed");
    }
    const Tensor& first = inputs.front();
    if (!first.defined() || !first.device().is_cuda()) {
        throw std::runtime_error(
            "Stax CUDA fused pointwise requires defined CUDA tensors");
    }
    // Inputs may mix element widths within one precision family; float64
    // runs its own family because the arithmetic width follows the widest
    // operand and half storage cannot feed double arithmetic here.
    const bool wide_family = first.dtype() == DType::Float64;
    for (const Tensor& input : inputs) {
        if (!input.defined() || !input.device().is_cuda()) {
            throw std::runtime_error(
                "Stax CUDA fused pointwise requires defined CUDA tensors");
        }
        const bool input_wide = input.dtype() == DType::Float64;
        if (input_wide != wide_family) {
            throw std::runtime_error(
                "Stax CUDA fused pointwise inputs cannot mix float64 with "
                "narrower element types");
        }
    }
    static_cast<void>(broadcast_shape(inputs));
}

Tensor byte_buffer(int64_t bytes, const Tensor& like) {
    return Tensor::empty({bytes > 0 ? bytes : 1}, DType::UInt8, like.device());
}

template <typename compute_t>
std::vector<Tensor> run_program(const std::vector<Tensor>& inputs,
                                const std::vector<int64_t>& program,
                                const std::vector<double>& constants,
                                const std::vector<int64_t>* output_refs,
                                const std::vector<int64_t>* out_dtypes = nullptr) {
    const int64_t input_count = static_cast<int64_t>(inputs.size());
    const int64_t instruction_count = static_cast<int64_t>(program.size() / 3);
    const std::vector<int64_t> output_shape = broadcast_shape(inputs);
    int64_t count = 1;
    for (int64_t extent : output_shape) count *= extent;
    const int64_t rank = static_cast<int64_t>(output_shape.size());
    bool flat_inputs = true;
    for (const Tensor& input : inputs) {
        if (input.shape() != output_shape || !input.is_contiguous()) {
            flat_inputs = false;
            break;
        }
    }

    // Outputs: a ref pointing at an input aliases that input; refs into the
    // temporary pool allocate fresh storage and are written by the kernel.
    std::vector<int64_t> temp_refs;
    std::vector<Tensor> temp_tensors;
    std::vector<int64_t> out_order;
    if (output_refs == nullptr) {
        temp_refs.push_back(input_count + instruction_count - 1);
        out_order.push_back(-1);
    } else {
        for (int64_t ref : *output_refs) {
            if (ref >= 0 && ref < input_count) {
                out_order.push_back(ref); // alias: replay the input tensor
            } else {
                temp_refs.push_back(ref);
                out_order.push_back(-2 - static_cast<int64_t>(temp_tensors.size()));
            }
        }
        if (temp_refs.empty()) {
            // Nothing to compute into fresh storage: pure aliasing.
            std::vector<Tensor> outs;
            for (int64_t code : out_order) {
                outs.push_back(inputs[static_cast<size_t>(code)]);
            }
            return outs;
        }
    }
    const DType in_dtype = inputs.front().dtype();
    // Output-dtype requests are indexed by the public output list; alias
    // slots replay an input tensor and consume no fresh storage, so map each
    // temp slot back to its public output position.
    std::vector<size_t> temp_to_out;
    if (output_refs != nullptr && out_dtypes != nullptr &&
        !out_dtypes->empty()) {
        temp_to_out.reserve(output_refs->size());
        for (size_t slot = 0; slot < output_refs->size(); ++slot) {
            const int64_t ref = (*output_refs)[slot];
            if (ref < 0 || ref >= input_count) {
                temp_to_out.push_back(slot);
            }
        }
    }
    const auto out_dtype_at = [&](size_t temp_slot) -> DType {
        if (temp_to_out.empty()) {
            if (out_dtypes != nullptr && !out_dtypes->empty() &&
                (*out_dtypes)[0] >= 0) {
                return static_cast<DType>((*out_dtypes)[0]);
            }
            return in_dtype;
        }
        const int64_t code =
            (*out_dtypes)[temp_to_out[static_cast<size_t>(temp_slot)]];
        return code >= 0 ? static_cast<DType>(code) : in_dtype;
    };
    for (size_t i = 0; i < temp_refs.size(); ++i) {
        temp_tensors.push_back(Tensor::empty(
            output_shape, out_dtype_at(i), inputs.front().device()));
    }
    std::vector<Tensor> outs;
    outs.reserve(out_order.size());
    size_t next_temp = 0;
    for (int64_t code : out_order) {
        if (code >= 0) {
            outs.push_back(inputs[static_cast<size_t>(code)]);
        } else {
            outs.push_back(temp_tensors[next_temp++]);
        }
    }
    if (count == 0) return outs;

    // Generated straight-line kernels replace the interpreter whenever the
    // program compiles: NVRTC cost is paid once per program content, and the
    // emitted body drops the per-element fetch/dispatch round trip the
    // interpreter pays on every step of the dependency chain.  Broadcast and
    // strided inputs participate too: the address plan folds each input's
    // layout into the generated addressing, so only genuinely unsupported
    // forms fall back to the interpreter.
    if (count >= 1024) {
        std::vector<DType> generated_out_dtypes;
        generated_out_dtypes.reserve(temp_refs.size());
        for (size_t i = 0; i < temp_refs.size(); ++i) {
            generated_out_dtypes.push_back(out_dtype_at(i));
        }
        if (launch_generated_pointwise(inputs, program, constants, temp_refs,
                                       temp_tensors, generated_out_dtypes,
                                       output_shape, count)) {
            return outs;
        }
    }

    // Device staging: pointer tables and program metadata are tiny; tensor
    // data itself is never copied.
    std::vector<const void*> host_input_ptrs;
    std::vector<uint8_t> host_input_kinds;
    host_input_ptrs.reserve(inputs.size());
    host_input_kinds.reserve(inputs.size());
    for (const Tensor& input : inputs) {
        host_input_ptrs.push_back(input.data_ptr());
        host_input_kinds.push_back(encode_out_kind(input.dtype()));
    }
    const int uniform_kind = std::all_of(
        host_input_kinds.begin(), host_input_kinds.end(),
        [&](uint8_t kind) { return kind == host_input_kinds.front(); })
        ? static_cast<int>(host_input_kinds.front())
        : -1;
    std::vector<int64_t> host_input_sizes;
    std::vector<int64_t> host_input_strides;
    std::vector<uint8_t> host_input_flat;
    std::vector<int64_t> host_output_strides;
    if (!flat_inputs) {
        host_input_sizes.assign(static_cast<size_t>(input_count * rank), 1);
        host_input_strides.assign(static_cast<size_t>(input_count * rank), 0);
        host_input_flat.assign(static_cast<size_t>(input_count), 1);
        for (int64_t input_index = 0; input_index < input_count; ++input_index) {
            const Tensor& input = inputs[static_cast<size_t>(input_index)];
            const int64_t offset = rank - input.dim();
            if (input.shape() != output_shape || !input.is_contiguous()) {
                host_input_flat[static_cast<size_t>(input_index)] = 0;
            }
            for (int64_t dim = 0; dim < input.dim(); ++dim) {
                const size_t slot = static_cast<size_t>(
                    input_index * rank + offset + dim);
                host_input_sizes[slot] = input.size(dim);
                host_input_strides[slot] = input.stride(dim);
            }
        }
        host_output_strides.assign(static_cast<size_t>(rank), 1);
        int64_t inner = 1;
        for (int64_t dim = rank - 1; dim >= 0; --dim) {
            host_output_strides[static_cast<size_t>(dim)] = inner;
            inner *= output_shape[static_cast<size_t>(dim)];
        }
    }
    std::vector<void*> host_output_ptrs;
    std::vector<uint8_t> host_output_kinds;
    host_output_ptrs.reserve(temp_tensors.size());
    host_output_kinds.reserve(temp_tensors.size());
    for (size_t i = 0; i < temp_tensors.size(); ++i) {
        host_output_ptrs.push_back(temp_tensors[i].data_ptr());
        host_output_kinds.push_back(encode_out_kind(out_dtype_at(i)));
    }
    std::vector<compute_t> host_constants(constants.size());
    for (size_t i = 0; i < constants.size(); ++i) {
        host_constants[i] = static_cast<compute_t>(constants[i]);
    }
    std::vector<Instruction> host_instructions(instruction_count);
    for (int64_t i = 0; i < instruction_count; ++i) {
        host_instructions[i] = {program[i * 3], program[i * 3 + 1],
                                program[i * 3 + 2]};
    }

    const auto stream = getCurrentCUDAStream().stream();
    const int64_t instruction_bytes =
        static_cast<int64_t>(sizeof(Instruction) * instruction_count);
    const int64_t constants_bytes =
        static_cast<int64_t>(sizeof(compute_t) * constants.size());
    const int64_t input_pointer_bytes =
        static_cast<int64_t>(sizeof(void*) * input_count);
    const int64_t input_kind_bytes =
        static_cast<int64_t>(sizeof(uint8_t) * host_input_kinds.size());
    const auto align_program_offset = [](int64_t value) {
        return (value + 15) & ~int64_t(15);
    };
    const int64_t constants_offset = align_program_offset(instruction_bytes);
    const int64_t input_pointer_offset =
        align_program_offset(constants_offset + constants_bytes);
    const int64_t input_kinds_offset =
        align_program_offset(input_pointer_offset + input_pointer_bytes);
    const int64_t program_bytes = input_kinds_offset + input_kind_bytes;
    Tensor program_buf = byte_buffer(program_bytes, inputs.front());
    std::vector<uint8_t> host_program(static_cast<size_t>(program_bytes), 0);
    if (instruction_bytes > 0) {
        std::memcpy(host_program.data(), host_instructions.data(), instruction_bytes);
    }
    if (constants_bytes > 0) {
        std::memcpy(host_program.data() + constants_offset,
                    host_constants.data(), constants_bytes);
    }
    if (input_pointer_bytes > 0) {
        std::memcpy(host_program.data() + input_pointer_offset,
                    host_input_ptrs.data(), input_pointer_bytes);
    }
    if (input_kind_bytes > 0) {
        std::memcpy(host_program.data() + input_kinds_offset,
                    host_input_kinds.data(), input_kind_bytes);
    }
    checkCuda(cudaMemcpyAsync(program_buf.data_ptr(), host_program.data(),
                               program_bytes, cudaMemcpyHostToDevice, stream),
               "stax fused pointwise program upload");
    const uint8_t* program_ptr =
        reinterpret_cast<const uint8_t*>(program_buf.data_ptr());
    Tensor metadata_buf;
    int64_t input_size_offset = 0;
    int64_t input_stride_offset = 0;
    int64_t output_size_offset = 0;
    int64_t output_stride_offset = 0;
    int64_t input_flat_offset = 0;
    if (!flat_inputs) {
        input_stride_offset = input_size_offset +
            static_cast<int64_t>(sizeof(int64_t) * host_input_sizes.size());
        output_size_offset = input_stride_offset +
            static_cast<int64_t>(sizeof(int64_t) * host_input_strides.size());
        output_stride_offset = output_size_offset +
            static_cast<int64_t>(sizeof(int64_t) * output_shape.size());
        input_flat_offset = output_stride_offset +
            static_cast<int64_t>(sizeof(int64_t) * host_output_strides.size());
        const int64_t metadata_bytes = input_flat_offset +
            static_cast<int64_t>(sizeof(uint8_t) * host_input_flat.size());
        metadata_buf = byte_buffer(metadata_bytes, inputs.front());
        std::vector<uint8_t> host_metadata(
            static_cast<size_t>(metadata_bytes), 0);
        if (!host_input_sizes.empty()) {
            std::memcpy(host_metadata.data() + input_size_offset,
                        host_input_sizes.data(), input_stride_offset - input_size_offset);
        }
        if (!host_input_strides.empty()) {
            std::memcpy(host_metadata.data() + input_stride_offset,
                        host_input_strides.data(), output_size_offset - input_stride_offset);
        }
        if (!output_shape.empty()) {
            std::memcpy(host_metadata.data() + output_size_offset,
                        output_shape.data(), output_stride_offset - output_size_offset);
        }
        if (!host_output_strides.empty()) {
            std::memcpy(host_metadata.data() + output_stride_offset,
                        host_output_strides.data(), input_flat_offset - output_stride_offset);
        }
        if (!host_input_flat.empty()) {
            std::memcpy(host_metadata.data() + input_flat_offset,
                        host_input_flat.data(), metadata_bytes - input_flat_offset);
        }
        checkCuda(cudaMemcpyAsync(metadata_buf.data_ptr(), host_metadata.data(),
                                   metadata_bytes, cudaMemcpyHostToDevice, stream),
                   "stax fused pointwise layout upload");
    }

    void* output = nullptr;
    uint8_t out_kind = kOutF32;
    void* const* output_ptrs = nullptr;
    const uint8_t* output_kind_ptr = nullptr;
    const int64_t* output_ref_ptr = nullptr;
    Tensor outptr_buf;
    Tensor outkind_buf;
    Tensor ref_buf;
    if (output_refs == nullptr) {
        output = temp_tensors[0].data_ptr();
        out_kind = host_output_kinds[0];
    } else {
        outptr_buf = byte_buffer(
            static_cast<int64_t>(sizeof(void*) * host_output_ptrs.size()),
            inputs.front());
        outkind_buf = byte_buffer(
            static_cast<int64_t>(sizeof(uint8_t) * host_output_kinds.size()),
            inputs.front());
        ref_buf = byte_buffer(
            static_cast<int64_t>(sizeof(int64_t) * temp_refs.size()),
            inputs.front());
        checkCuda(cudaMemcpyAsync(outptr_buf.data_ptr(), host_output_ptrs.data(),
                                   sizeof(void*) * host_output_ptrs.size(),
                                   cudaMemcpyHostToDevice, stream),
                   "stax fused pointwise output pointer table upload");
        checkCuda(cudaMemcpyAsync(outkind_buf.data_ptr(), host_output_kinds.data(),
                                   sizeof(uint8_t) * host_output_kinds.size(),
                                   cudaMemcpyHostToDevice, stream),
                   "stax fused pointwise output kinds upload");
        checkCuda(cudaMemcpyAsync(ref_buf.data_ptr(), temp_refs.data(),
                                   sizeof(int64_t) * temp_refs.size(),
                                   cudaMemcpyHostToDevice, stream),
                   "stax fused pointwise output refs upload");
        output_ptrs = reinterpret_cast<void* const*>(outptr_buf.data_ptr());
        output_kind_ptr =
            reinterpret_cast<const uint8_t*>(outkind_buf.data_ptr());
        output_ref_ptr = reinterpret_cast<const int64_t*>(ref_buf.data_ptr());
    }

    if (flat_inputs) {
        ProgramState<compute_t, true> state;
        state.instructions =
            reinterpret_cast<const Instruction*>(program_ptr);
        state.constants = reinterpret_cast<const compute_t*>(program_ptr + constants_offset);
        state.input_ptrs =
            reinterpret_cast<const void* const*>(program_ptr + input_pointer_offset);
        state.input_kinds =
            reinterpret_cast<const uint8_t*>(program_ptr + input_kinds_offset);
        state.input_sizes = nullptr;
        state.input_strides = nullptr;
        state.output_sizes = nullptr;
        state.output_strides = nullptr;
        state.input_flat = nullptr;
        state.uniform_kind = uniform_kind;
        state.input_count = input_count;
        state.instruction_count = instruction_count;
        state.rank = 0;
        dispatch_program_launch(
            state, output, out_kind, output_ptrs, output_kind_ptr,
            output_ref_ptr, static_cast<int64_t>(temp_refs.size()), count,
            stream);
    } else {
        ProgramState<compute_t, false> state;
        state.instructions =
            reinterpret_cast<const Instruction*>(program_ptr);
        state.constants = reinterpret_cast<const compute_t*>(program_ptr + constants_offset);
        state.input_ptrs =
            reinterpret_cast<const void* const*>(program_ptr + input_pointer_offset);
        state.input_kinds =
            reinterpret_cast<const uint8_t*>(program_ptr + input_kinds_offset);
        const uint8_t* metadata_ptr =
            reinterpret_cast<const uint8_t*>(metadata_buf.data_ptr());
        state.input_sizes = reinterpret_cast<const int64_t*>(
            metadata_ptr + input_size_offset);
        state.input_strides = reinterpret_cast<const int64_t*>(
            metadata_ptr + input_stride_offset);
        state.output_sizes = reinterpret_cast<const int64_t*>(
            metadata_ptr + output_size_offset);
        state.output_strides = reinterpret_cast<const int64_t*>(
            metadata_ptr + output_stride_offset);
        state.input_flat = metadata_ptr + input_flat_offset;
        state.uniform_kind = uniform_kind;
        state.input_count = input_count;
        state.instruction_count = instruction_count;
        state.rank = rank;
        dispatch_program_launch(
            state, output, out_kind, output_ptrs, output_kind_ptr,
            output_ref_ptr, static_cast<int64_t>(temp_refs.size()), count,
            stream);
    }
    return outs;
}

#endif // USE_CUDA

} // namespace

#ifdef USE_CUDA

Tensor stax_fused_pointwise_cuda(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    int64_t out_dtype) {
    check_program_shape(inputs, program, constants, 1);
    const DType dt = inputs.front().dtype();
    std::vector<int64_t> kinds;
    if (out_dtype >= 0) {
        kinds.push_back(out_dtype);
    }
    if (dt == DType::Float32) {
        return run_program<float>(inputs, program, constants, nullptr,
                                         kinds.empty() ? nullptr : &kinds)[0];
    }
    if (dt == DType::Float64) {
        return run_program<double>(inputs, program, constants, nullptr,
                                           kinds.empty() ? nullptr : &kinds)[0];
    }
    if (dt == DType::Float16) {
        return run_program<float>(inputs, program, constants, nullptr,
                                                    kinds.empty() ? nullptr : &kinds)[0];
    }
    if (dt == DType::BFloat16) {
        return run_program<float>(inputs, program, constants, nullptr,
                                                        kinds.empty() ? nullptr : &kinds)[0];
    }
    throw std::runtime_error(
        "Stax CUDA fused pointwise supports float16/bfloat16/float32/float64");
}

std::vector<Tensor> stax_fused_pointwise_cuda_multi(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    const std::vector<int64_t>& output_refs,
    const std::vector<int64_t>& out_dtypes) {
    check_program_shape(inputs, program, constants,
                        static_cast<int64_t>(output_refs.size()));
    if (!out_dtypes.empty() && out_dtypes.size() != output_refs.size()) {
        throw std::runtime_error(
            "Stax fused pointwise output dtype count mismatch");
    }
    const DType dt = inputs.front().dtype();
    const std::vector<int64_t>* kinds =
        out_dtypes.empty() ? nullptr : &out_dtypes;
    if (dt == DType::Float32) {
        return run_program<float>(inputs, program, constants, &output_refs, kinds);
    }
    if (dt == DType::Float64) {
        return run_program<double>(inputs, program, constants, &output_refs, kinds);
    }
    if (dt == DType::Float16) {
        return run_program<float>(inputs, program, constants, &output_refs, kinds);
    }
    if (dt == DType::BFloat16) {
        return run_program<float>(inputs, program, constants, &output_refs, kinds);
    }
    throw std::runtime_error(
        "Stax CUDA fused pointwise multi supports float16/bfloat16/float32/float64");
}

#else // !USE_CUDA

Tensor stax_fused_pointwise_cuda(
    const std::vector<Tensor>&,
    const std::vector<int64_t>&,
    const std::vector<double>&,
    int64_t) {
    TP_THROW(NotImplementedError, "stax fused pointwise requires CUDA");
}

std::vector<Tensor> stax_fused_pointwise_cuda_multi(
    const std::vector<Tensor>&,
    const std::vector<int64_t>&,
    const std::vector<double>&,
    const std::vector<int64_t>&,
    const std::vector<int64_t>&) {
    TP_THROW(NotImplementedError, "stax fused pointwise requires CUDA");
}

#endif // USE_CUDA

} // namespace cuda
} // namespace tensorplay

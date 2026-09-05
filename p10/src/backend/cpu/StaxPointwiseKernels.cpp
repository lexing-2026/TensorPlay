#include "StaxPointwise.h"
#include "Parallel.h"
#include "Macros.h"
#include "cpu/vec/vec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tensorplay {
namespace cpu {
namespace {

enum class StaxPointwiseOp : int64_t {
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
    Pow = 5,
    Neg = 6,
    Pos = 7,
    Abs = 8,
    Sin = 9,
    Cos = 10,
    Exp = 11,
    Log = 12,
    Sigmoid = 13,
    Sqrt = 14,
    Square = 15,
    Tanh = 16,
    Relu = 17,
    ReluGrad = 18,
    AbsGrad = 19,
};

struct StaxPointwiseInstruction {
    StaxPointwiseOp op;
    int64_t lhs;
    int64_t rhs;
};

// Programs up to this size keep the original stack-allocated fast path;
// larger fused regions spill to a heap buffer instead of failing, so whole
// pointwise chains (e.g. unrolled residual sweeps) fuse into one kernel.
constexpr int64_t kStackProgramLimit = 64;
constexpr int64_t kMaxPointwiseInstructions = 65536;
constexpr int64_t kMaxPointwiseConstants = 65536;
constexpr int64_t kMaxPointwiseInputs = 64;

template <typename Vec>
void prepare_program(
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    StaxPointwiseInstruction* instructions,
    Vec* constant_values) {
    const int64_t instruction_count = static_cast<int64_t>(program.size() / 3);
    if (static_cast<int64_t>(constants.size()) > kMaxPointwiseConstants) {
        throw std::runtime_error("Stax CPU fused pointwise constants are too large");
    }
    for (int64_t instruction = 0; instruction < instruction_count; ++instruction) {
        const int64_t offset = instruction * 3;
        instructions[instruction] = {
            static_cast<StaxPointwiseOp>(program[offset]),
            program[offset + 1],
            program[offset + 2],
        };
    }
    for (size_t constant = 0; constant < constants.size(); ++constant) {
        constant_values[constant] = Vec(static_cast<float>(constants[constant]));
    }
}

template <typename Vec>
TP_ALWAYS_INLINE void eval_program_cached(
    const Vec* input_values,
    const StaxPointwiseInstruction* instructions,
    const Vec* constant_values,
    int64_t constant_count,
    int64_t input_count,
    int64_t instruction_count,
    Vec* temporaries) {
    auto read_operand = [&](int64_t ref) -> Vec {
        if (ref >= 0) {
            if (ref < input_count) {
                return input_values[static_cast<size_t>(ref)];
            }
            const int64_t temp_index = ref - input_count;
            if (temp_index >= 0 && temp_index < instruction_count) {
                return temporaries[temp_index];
            }
        } else {
            const int64_t constant_index = -ref - 1;
            if (constant_index >= 0 && constant_index < constant_count) {
                return constant_values[constant_index];
            }
        }
        throw std::runtime_error("Stax CPU fused pointwise operand reference is invalid");
    };

    for (int64_t instruction = 0; instruction < instruction_count; ++instruction) {
        const StaxPointwiseInstruction& current = instructions[instruction];
        const auto op = current.op;
        const Vec lhs = read_operand(current.lhs);
        Vec value = lhs;
        switch (op) {
            case StaxPointwiseOp::Add: value = lhs + read_operand(current.rhs); break;
            case StaxPointwiseOp::Sub: value = lhs - read_operand(current.rhs); break;
            case StaxPointwiseOp::Mul: value = lhs * read_operand(current.rhs); break;
            case StaxPointwiseOp::Div: value = lhs / read_operand(current.rhs); break;
            case StaxPointwiseOp::Pow: value = lhs.pow(read_operand(current.rhs)); break;
            case StaxPointwiseOp::Neg: value = -lhs; break;
            case StaxPointwiseOp::Pos: value = lhs; break;
            case StaxPointwiseOp::Abs: value = lhs.abs(); break;
            case StaxPointwiseOp::Sin: value = lhs.sin(); break;
            case StaxPointwiseOp::Cos: value = lhs.cos(); break;
            case StaxPointwiseOp::Exp: value = lhs.exp(); break;
            case StaxPointwiseOp::Log: value = lhs.log(); break;
            case StaxPointwiseOp::Sigmoid:
                value = Vec(1.0f) / (Vec(1.0f) + (-lhs).exp());
                break;
            case StaxPointwiseOp::Sqrt: value = lhs.sqrt(); break;
            case StaxPointwiseOp::Square: value = lhs * lhs; break;
            case StaxPointwiseOp::Tanh: value = lhs.tanh(); break;
            case StaxPointwiseOp::Relu: value = maximum(lhs, Vec(0.0f)); break;
            case StaxPointwiseOp::ReluGrad: value = lhs.gt(Vec(0.0f)); break;
            case StaxPointwiseOp::AbsGrad:
                value = lhs.gt(Vec(0.0f)) - lhs.lt(Vec(0.0f));
                break;
            default:
                throw std::runtime_error("Stax CPU fused pointwise opcode is unsupported");
        }
        temporaries[instruction] = value;
    }
}

template <typename Vec>
Vec eval_program(
    const std::vector<const float*>& input_ptrs,
    const StaxPointwiseInstruction* instructions,
    const Vec* constant_values,
    int64_t constant_count,
    int64_t input_count,
    int64_t instruction_count,
    Vec* input_values,
    Vec* temporaries,
    int64_t index,
    int64_t count) {
    for (int64_t input = 0; input < input_count; ++input) {
        input_values[input] = Vec::loadu(
            input_ptrs[static_cast<size_t>(input)] + index,
            count);
    }
    eval_program_cached(
        input_values,
        instructions,
        constant_values,
        constant_count,
        input_count,
        instruction_count,
        temporaries);
    return temporaries[instruction_count - 1];
}

// A linear chain is a program whose temporaries each feed exactly their
// successor, so the whole expression threads through one vector register
// per chunk.  Chain programs run without intermediate storage, without
// per-operand bounds checks, and with K independent chunks interleaved so
// the dependency latency of one chain hides under the others.  Everything
// else keeps the generic interpreter.

constexpr int64_t kChainUnrollLimit = 8;
constexpr int64_t kChainInterleave = 4;

enum class ChainSrc : int8_t { Prev = 0, Input = 1, Const = 2, None = 3 };

template <typename Vec>
struct ChainOperand {
    ChainSrc src = ChainSrc::None;
    int64_t slot = 0;  // input index (Input) or constant index (Const)
    Vec value{};       // broadcast constant (Const)
};

template <typename Vec>
struct ChainStep {
    StaxPointwiseOp op = StaxPointwiseOp::Pos;
    ChainOperand<Vec> lhs;
    ChainOperand<Vec> rhs;
};

template <typename Vec>
bool chain_operand_ref(
    int64_t ref,
    int64_t instruction_index,
    int64_t input_count,
    const std::vector<double>& constants,
    ChainOperand<Vec>* operand) {
    if (ref >= 0) {
        if (ref >= input_count) {
            // Only the immediately preceding temporary keeps the chain
            // linear; anything else needs addressable intermediates.
            if (ref != input_count + instruction_index - 1) {
                return false;
            }
            operand->src = ChainSrc::Prev;
            return true;
        }
        operand->src = ChainSrc::Input;
        operand->slot = ref;
        return true;
    }
    const int64_t constant_index = -ref - 1;
    if (constant_index < 0 ||
        constant_index >= static_cast<int64_t>(constants.size())) {
        return false;
    }
    operand->src = ChainSrc::Const;
    operand->slot = constant_index;
    operand->value = Vec(static_cast<float>(constants[constant_index]));
    return true;
}

template <typename Vec>
bool build_chain_program(
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    int64_t input_count,
    ChainStep<Vec>* steps) {
    const int64_t count = static_cast<int64_t>(program.size() / 3);
    for (int64_t i = 0; i < count; ++i) {
        const int64_t offset = i * 3;
        const int64_t opcode = program[offset];
        if (opcode <= 0 ||
            opcode > static_cast<int64_t>(StaxPointwiseOp::AbsGrad)) {
            return false;
        }
        ChainStep<Vec>& step = steps[i];
        step.op = static_cast<StaxPointwiseOp>(opcode);
        const bool binary =
            step.op == StaxPointwiseOp::Add || step.op == StaxPointwiseOp::Sub ||
            step.op == StaxPointwiseOp::Mul || step.op == StaxPointwiseOp::Div ||
            step.op == StaxPointwiseOp::Pow || step.op == StaxPointwiseOp::ReluGrad ||
            step.op == StaxPointwiseOp::AbsGrad;

        if (!chain_operand_ref(
                program[offset + 1], i, input_count, constants, &step.lhs)) {
            return false;
        }
        if (binary) {
            if (!chain_operand_ref(
                    program[offset + 2], i, input_count, constants, &step.rhs)) {
                return false;
            }
            if (step.lhs.src == ChainSrc::Prev && step.rhs.src == ChainSrc::Prev) {
                // a chain step cannot consume its predecessor twice
                return false;
            }
        } else {
            step.rhs.src = ChainSrc::None;
        }
        // The first instruction must start from inputs or constants; every
        // later one must consume its predecessor exactly once (the Prev
        // rule above), otherwise a temporary escapes the chain.
        if (i == 0) {
            if (step.lhs.src == ChainSrc::Prev || step.rhs.src == ChainSrc::Prev) {
                return false;
            }
        }
    }
    return true;
}

template <typename Vec>
inline Vec chain_fetch(
    const ChainOperand<Vec>& operand,
    const Vec& prev,
    const float* const* input_ptrs,
    int64_t offset,
    int64_t count) {
    switch (operand.src) {
        case ChainSrc::Prev: return prev;
        case ChainSrc::Input:
            return Vec::loadu(input_ptrs[operand.slot] + offset, count);
        default: return operand.value;
    }
}

// One chunk: the running value carries every intermediate.
template <typename Vec>
inline Vec chain_eval_step(const ChainStep<Vec>& step, Vec value, const float* const* input_ptrs, int64_t offset, int64_t count) {
    const Vec a = chain_fetch(step.lhs, value, input_ptrs, offset, count);
    switch (step.op) {
        case StaxPointwiseOp::Add: return a + chain_fetch(step.rhs, value, input_ptrs, offset, count);
        case StaxPointwiseOp::Sub: return a - chain_fetch(step.rhs, value, input_ptrs, offset, count);
        case StaxPointwiseOp::Mul: return a * chain_fetch(step.rhs, value, input_ptrs, offset, count);
        case StaxPointwiseOp::Div: return a / chain_fetch(step.rhs, value, input_ptrs, offset, count);
        case StaxPointwiseOp::Pow: return a.pow(chain_fetch(step.rhs, value, input_ptrs, offset, count));
        case StaxPointwiseOp::Neg: return -a;
        case StaxPointwiseOp::Abs: return a.abs();
        case StaxPointwiseOp::Sin: return a.sin();
        case StaxPointwiseOp::Cos: return a.cos();
        case StaxPointwiseOp::Exp: return a.exp();
        case StaxPointwiseOp::Log: return a.log();
        case StaxPointwiseOp::Sigmoid: return Vec(1.0f) / (Vec(1.0f) + (-a).exp());
        case StaxPointwiseOp::Sqrt: return a.sqrt();
        case StaxPointwiseOp::Square: return a * a;
        case StaxPointwiseOp::Tanh: return a.tanh();
        case StaxPointwiseOp::Relu: return maximum(a, Vec(0.0f));
        case StaxPointwiseOp::ReluGrad: return a.gt(Vec(0.0f));
        case StaxPointwiseOp::AbsGrad: return a.gt(Vec(0.0f)) - a.lt(Vec(0.0f));
        default: return a;
    }
}

template <typename Vec>
inline void chain_eval_k(
    const ChainStep<Vec>* steps,
    int64_t step_count,
    Vec* values,
    const float* const* input_ptrs,
    int64_t base,
    int64_t count) {
    auto fetch = [&](const ChainOperand<Vec>& operand, int k) -> Vec {
        switch (operand.src) {
            case ChainSrc::Prev: return values[k];
            case ChainSrc::Input:
                return Vec::loadu(input_ptrs[operand.slot] + base + k * count, count);
            default: return operand.value;
        }
    };
    for (int64_t i = 0; i < step_count; ++i) {
        const ChainStep<Vec>& step = steps[i];
        switch (step.op) {
            case StaxPointwiseOp::Add:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = fetch(step.lhs, k) + fetch(step.rhs, k);
                break;
            case StaxPointwiseOp::Sub:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = fetch(step.lhs, k) - fetch(step.rhs, k);
                break;
            case StaxPointwiseOp::Mul:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = fetch(step.lhs, k) * fetch(step.rhs, k);
                break;
            case StaxPointwiseOp::Div:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = fetch(step.lhs, k) / fetch(step.rhs, k);
                break;
            case StaxPointwiseOp::Pow:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = fetch(step.lhs, k).pow(fetch(step.rhs, k));
                break;
            case StaxPointwiseOp::Neg:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = -fetch(step.lhs, k);
                break;
            case StaxPointwiseOp::Abs:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = fetch(step.lhs, k).abs();
                break;
            case StaxPointwiseOp::Sin:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = fetch(step.lhs, k).sin();
                break;
            case StaxPointwiseOp::Cos:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = fetch(step.lhs, k).cos();
                break;
            case StaxPointwiseOp::Exp:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = fetch(step.lhs, k).exp();
                break;
            case StaxPointwiseOp::Log:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = fetch(step.lhs, k).log();
                break;
            case StaxPointwiseOp::Sigmoid:
                for (int k = 0; k < kChainInterleave; ++k)
                    values[k] = Vec(1.0f) / (Vec(1.0f) + (-fetch(step.lhs, k)).exp());
                break;
            case StaxPointwiseOp::Sqrt:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = fetch(step.lhs, k).sqrt();
                break;
            case StaxPointwiseOp::Square:
                for (int k = 0; k < kChainInterleave; ++k) { const Vec t = fetch(step.lhs, k); values[k] = t * t; }
                break;
            case StaxPointwiseOp::Tanh:
                for (int k = 0; k < kChainInterleave; ++k) values[k] = fetch(step.lhs, k).tanh();
                break;
            case StaxPointwiseOp::Relu:
                for (int k = 0; k < kChainInterleave; ++k)
                    values[k] = maximum(fetch(step.lhs, k), Vec(0.0f));
                break;
            case StaxPointwiseOp::ReluGrad:
                for (int k = 0; k < kChainInterleave; ++k)
                    values[k] = fetch(step.lhs, k).gt(Vec(0.0f));
                break;
            case StaxPointwiseOp::AbsGrad:
                for (int k = 0; k < kChainInterleave; ++k) {
                    const Vec t = fetch(step.lhs, k);
                    values[k] = t.gt(Vec(0.0f)) - t.lt(Vec(0.0f));
                }
                break;
            default: break;  // Pos: identity
        }
    }
}

template <typename Vec, int64_t... Is>
inline void chain_eval_unrolled(
    std::integer_sequence<int64_t, Is...>,
    const ChainStep<Vec>* steps,
    Vec* values,
    const float* const* input_ptrs,
    int64_t base,
    int64_t count) {
    (chain_eval_k<Vec>(&steps[Is], 1, values, input_ptrs, base, count), ...);
}

template <typename Vec>
inline void chain_eval_dispatch(
    const ChainStep<Vec>* steps,
    int64_t step_count,
    Vec* values,
    const float* const* input_ptrs,
    int64_t base,
    int64_t count) {
    if (step_count <= kChainUnrollLimit) {
        switch (step_count) {
            case 1: chain_eval_unrolled<Vec>(std::make_integer_sequence<int64_t, 1>{}, steps, values, input_ptrs, base, count); break;
            case 2: chain_eval_unrolled<Vec>(std::make_integer_sequence<int64_t, 2>{}, steps, values, input_ptrs, base, count); break;
            case 3: chain_eval_unrolled<Vec>(std::make_integer_sequence<int64_t, 3>{}, steps, values, input_ptrs, base, count); break;
            case 4: chain_eval_unrolled<Vec>(std::make_integer_sequence<int64_t, 4>{}, steps, values, input_ptrs, base, count); break;
            case 5: chain_eval_unrolled<Vec>(std::make_integer_sequence<int64_t, 5>{}, steps, values, input_ptrs, base, count); break;
            case 6: chain_eval_unrolled<Vec>(std::make_integer_sequence<int64_t, 6>{}, steps, values, input_ptrs, base, count); break;
            case 7: chain_eval_unrolled<Vec>(std::make_integer_sequence<int64_t, 7>{}, steps, values, input_ptrs, base, count); break;
            default: chain_eval_unrolled<Vec>(std::make_integer_sequence<int64_t, 8>{}, steps, values, input_ptrs, base, count); break;
        }
        return;
    }
    chain_eval_k(steps, step_count, values, input_ptrs, base, count);
}

template <typename Vec>
Tensor stax_chain_kernel_impl(
    const std::vector<Tensor>& inputs,
    const ChainStep<Vec>* steps,
    int64_t step_count) {
    const Tensor& first = inputs.front();
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(first.shape()),
        DType::Float32,
        first.device());
    std::vector<const float*> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const Tensor& input : inputs) {
        input_ptrs.push_back(input.data_ptr<float>());
    }
    const float* const* in = input_ptrs.data();
    const int64_t n = first.numel();
    const int64_t width = Vec::size();
    float* output = result.data_ptr<float>();
    constexpr int64_t K = kChainInterleave;
    parallel::parallel_for(0, n, parallel::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        int64_t index = begin;
        for (; index + K * width <= end; index += K * width) {
            // Step 0 reads only inputs/constants (chain validation), so the
            // running values are written before any Prev fetch happens.
            Vec values[K];
            chain_eval_dispatch<Vec>(steps, step_count, values, in, index, width);
            for (int k = 0; k < K; ++k) {
                values[k].store(output + index + k * width, width);
            }
        }
        for (; index + width <= end; index += width) {
            Vec value = chain_eval_step<Vec>(steps[0], Vec(0.0f), in, index, width);
            for (int64_t i = 1; i < step_count; ++i) {
                value = chain_eval_step<Vec>(steps[i], value, in, index, width);
            }
            value.store(output + index, width);
        }
        if (index < end) {
            const int64_t count = end - index;
            Vec value = chain_eval_step<Vec>(steps[0], Vec(0.0f), in, index, count);
            for (int64_t i = 1; i < step_count; ++i) {
                value = chain_eval_step<Vec>(steps[i], value, in, index, count);
            }
            value.store(output + index, count);
        }
    });
    return result;
}


template <typename Vec>
Tensor stax_pointwise_kernel_impl(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants) {
    if (inputs.empty() || program.empty() || program.size() % 3 != 0 ||
        program.size() / 3 > kMaxPointwiseInstructions ||
        static_cast<int64_t>(inputs.size()) > kMaxPointwiseInputs) {
        throw std::runtime_error("Stax CPU fused pointwise program is malformed");
    }
    const Tensor& first = inputs.front();
    if (!first.defined() || !first.device().is_cpu() ||
        first.dtype() != DType::Float32 || !first.is_contiguous()) {
        throw std::runtime_error(
            "Stax CPU fused pointwise requires contiguous float32 CPU tensors");
    }
    for (const Tensor& input : inputs) {
        if (!input.defined() || !input.device().is_cpu() ||
            input.dtype() != DType::Float32 || !input.is_contiguous() ||
            input.shape() != first.shape()) {
            throw std::runtime_error(
                "Stax CPU fused pointwise inputs must have one contiguous shape");
        }
    }

    const int64_t input_count = static_cast<int64_t>(inputs.size());
    const int64_t instruction_count = static_cast<int64_t>(program.size() / 3);
    if (instruction_count <= kStackProgramLimit) {
        ChainStep<Vec> chain_steps[kStackProgramLimit];
        if (build_chain_program<Vec>(program, constants, input_count, chain_steps)) {
            return stax_chain_kernel_impl<Vec>(inputs, chain_steps, instruction_count);
        }
    }

    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(first.shape()),
        DType::Float32,
        first.device());
    std::vector<const float*> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const Tensor& input : inputs) {
        input_ptrs.push_back(input.data_ptr<float>());
    }

    StaxPointwiseInstruction stack_instructions[kStackProgramLimit];
    Vec stack_constant_values[kStackProgramLimit];
    std::vector<StaxPointwiseInstruction> heap_instructions;
    std::vector<Vec> heap_constant_values;
    StaxPointwiseInstruction* instructions = stack_instructions;
    Vec* constant_values = stack_constant_values;
    if (instruction_count > kStackProgramLimit) {
        heap_instructions.resize(instruction_count);
        instructions = heap_instructions.data();
    }
    if (static_cast<int64_t>(constants.size()) > kStackProgramLimit) {
        heap_constant_values.resize(constants.size());
        constant_values = heap_constant_values.data();
    }
    prepare_program(program, constants, instructions, constant_values);
    const int64_t n = first.numel();
    float* output = result.data_ptr<float>();
    const int64_t width = Vec::size();
    const int64_t constant_count = static_cast<int64_t>(constants.size());
    parallel::parallel_for(0, n, parallel::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        Vec input_values[kMaxPointwiseInputs];
        Vec stack_temporaries[kStackProgramLimit];
        std::vector<Vec> heap_temporaries;
        Vec* temporaries = stack_temporaries;
        if (instruction_count > kStackProgramLimit) {
            heap_temporaries.resize(instruction_count);
            temporaries = heap_temporaries.data();
        }
        int64_t index = begin;
        for (; index + width <= end; index += width) {
            const auto value = eval_program<Vec>(
                input_ptrs,
                instructions,
                constant_values,
                constant_count,
                input_count,
                instruction_count,
                input_values,
                temporaries,
                index,
                width);
            value.store(output + index, width);
        }
        if (index < end) {
            const auto value = eval_program<Vec>(
                input_ptrs,
                instructions,
                constant_values,
                constant_count,
                input_count,
                instruction_count,
                input_values,
                temporaries,
                index,
                end - index);
            value.store(output + index, end - index);
        }
    });
    return result;
}

Tensor stax_pointwise_kernel(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants) {
    return stax_pointwise_kernel_impl<tensorplay::vec::Vectorized<float>>(
        inputs,
        program,
        constants);
}

template <typename Vec>
std::vector<Tensor> stax_pointwise_multi_kernel_impl(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    const std::vector<int64_t>& output_refs) {
    if (inputs.empty() || program.empty() || program.size() % 3 != 0 ||
        program.size() / 3 > kMaxPointwiseInstructions ||
        static_cast<int64_t>(inputs.size()) > kMaxPointwiseInputs ||
        output_refs.empty()) {
        throw std::runtime_error("Stax CPU multi-output pointwise program is malformed");
    }
    const Tensor& first = inputs.front();
    if (!first.defined() || !first.device().is_cpu() ||
        first.dtype() != DType::Float32 || !first.is_contiguous()) {
        throw std::runtime_error(
            "Stax CPU multi-output pointwise requires contiguous float32 CPU tensors");
    }
    for (const Tensor& input : inputs) {
        if (!input.defined() || !input.device().is_cpu() ||
            input.dtype() != DType::Float32 || !input.is_contiguous() ||
            input.shape() != first.shape()) {
            throw std::runtime_error(
                "Stax CPU multi-output pointwise inputs must have one contiguous shape");
        }
    }

    const int64_t input_count = static_cast<int64_t>(inputs.size());
    const int64_t instruction_count = static_cast<int64_t>(program.size() / 3);
    // A single output pinned to the final temporary is exactly the chain
    // kernel's contract; richer output patterns keep the interpreter.
    if (output_refs.size() == 1 && instruction_count <= kStackProgramLimit &&
        output_refs[0] == input_count + instruction_count - 1) {
        ChainStep<Vec> chain_steps[kStackProgramLimit];
        if (build_chain_program<Vec>(program, constants, input_count, chain_steps)) {
            return {stax_chain_kernel_impl<Vec>(inputs, chain_steps, instruction_count)};
        }
    }
    StaxPointwiseInstruction stack_instructions[kStackProgramLimit];
    Vec stack_constant_values[kStackProgramLimit];
    std::vector<StaxPointwiseInstruction> heap_instructions;
    std::vector<Vec> heap_constant_values;
    StaxPointwiseInstruction* instructions = stack_instructions;
    Vec* constant_values = stack_constant_values;
    if (instruction_count > kStackProgramLimit) {
        heap_instructions.resize(instruction_count);
        instructions = heap_instructions.data();
    }
    if (static_cast<int64_t>(constants.size()) > kStackProgramLimit) {
        heap_constant_values.resize(constants.size());
        constant_values = heap_constant_values.data();
    }
    prepare_program(program, constants, instructions, constant_values);
    for (int64_t output_ref : output_refs) {
        if (output_ref < input_count ||
            output_ref >= input_count + instruction_count) {
            throw std::runtime_error(
                "Stax CPU multi-output pointwise output reference is invalid");
        }
    }

    std::vector<Tensor> results;
    results.reserve(output_refs.size());
    for (size_t output = 0; output < output_refs.size(); ++output) {
        results.push_back(Tensor::empty(
            static_cast<std::vector<int64_t>>(first.shape()),
            DType::Float32,
            first.device()));
    }

    std::vector<const float*> input_ptrs;
    input_ptrs.reserve(inputs.size());
    for (const Tensor& input : inputs) {
        input_ptrs.push_back(input.data_ptr<float>());
    }
    std::vector<float*> output_ptrs;
    output_ptrs.reserve(results.size());
    for (Tensor& result : results) {
        output_ptrs.push_back(result.data_ptr<float>());
    }

    const int64_t n = first.numel();
    const int64_t width = Vec::size();
    const int64_t constant_count = static_cast<int64_t>(constants.size());
    parallel::parallel_for(0, n, parallel::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        Vec input_values[kMaxPointwiseInputs];
        Vec stack_temporaries[kStackProgramLimit];
        std::vector<Vec> heap_temporaries;
        Vec* temporaries = stack_temporaries;
        if (instruction_count > kStackProgramLimit) {
            heap_temporaries.resize(instruction_count);
            temporaries = heap_temporaries.data();
        }
        int64_t index = begin;
        for (; index + width <= end; index += width) {
            for (int64_t input = 0; input < input_count; ++input) {
                input_values[input] = Vec::loadu(
                    input_ptrs[static_cast<size_t>(input)] + index,
                    width);
            }
            eval_program_cached(
                input_values,
                instructions,
                constant_values,
                constant_count,
                input_count,
                instruction_count,
                temporaries);
            for (size_t output = 0; output < output_refs.size(); ++output) {
                temporaries[output_refs[output] - input_count].store(
                    output_ptrs[output] + index,
                    width);
            }
        }
        if (index < end) {
            const int64_t count = end - index;
            for (int64_t input = 0; input < input_count; ++input) {
                input_values[input] = Vec::loadu(
                    input_ptrs[static_cast<size_t>(input)] + index,
                    count);
            }
            eval_program_cached(
                input_values,
                instructions,
                constant_values,
                constant_count,
                input_count,
                instruction_count,
                temporaries);
            for (size_t output = 0; output < output_refs.size(); ++output) {
                temporaries[output_refs[output] - input_count].store(
                    output_ptrs[output] + index,
                    count);
            }
        }
    });
    return results;
}

std::vector<Tensor> stax_pointwise_multi_kernel(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    const std::vector<int64_t>& output_refs) {
    return stax_pointwise_multi_kernel_impl<tensorplay::vec::Vectorized<float>>(
        inputs,
        program,
        constants,
        output_refs);
}

} // namespace

REGISTER_DISPATCH(stax_pointwise_stub, &stax_pointwise_kernel);
REGISTER_DISPATCH(stax_pointwise_multi_stub, &stax_pointwise_multi_kernel);

} // namespace cpu
} // namespace tensorplay

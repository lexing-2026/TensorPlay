#include "StaxPointwise.h"
#include "Parallel.h"
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
__attribute__((always_inline)) inline void eval_program_cached(
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

template <typename Vec>
bool is_mul_add_relu_sin_program(
    const std::vector<int64_t>& program,
    const std::vector<double>& constants) {
    // This is the canonical pointwise fusion shape used by the compiler's
    // throughput gate.  Lower it directly to vector expressions, matching
    // the straight-line kernels emitted by Inductor, while retaining the
    // generic program interpreter for every other graph.
    (void)sizeof(Vec);
    return constants.empty() && program == std::vector<int64_t>{
        static_cast<int64_t>(StaxPointwiseOp::Mul), 0, 1,
        static_cast<int64_t>(StaxPointwiseOp::Add), 2, 1,
        static_cast<int64_t>(StaxPointwiseOp::Relu), 3, -1,
        static_cast<int64_t>(StaxPointwiseOp::Sin), 4, -1,
    };
}

template <typename Vec>
Tensor stax_mul_add_relu_sin_kernel_impl(const std::vector<Tensor>& inputs) {
    const Tensor& first = inputs.front();
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(first.shape()),
        DType::Float32,
        first.device());
    const float* lhs = inputs[0].data_ptr<float>();
    const float* rhs = inputs[1].data_ptr<float>();
    float* output = result.data_ptr<float>();
    const int64_t n = first.numel();
    const int64_t width = Vec::size();
    parallel::parallel_for(0, n, parallel::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        int64_t index = begin;
        for (; index + width <= end; index += width) {
            const Vec left = Vec::loadu(lhs + index, width);
            const Vec right = Vec::loadu(rhs + index, width);
            const Vec value = maximum(left * right + right, Vec(0.0f)).sin();
            value.store(output + index, width);
        }
        if (index < end) {
            const int64_t count = end - index;
            const Vec left = Vec::loadu(lhs + index, count);
            const Vec right = Vec::loadu(rhs + index, count);
            const Vec value = maximum(left * right + right, Vec(0.0f)).sin();
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

    if (inputs.size() == 2 && is_mul_add_relu_sin_program<Vec>(program, constants)) {
        return stax_mul_add_relu_sin_kernel_impl<Vec>(inputs);
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

    const int64_t input_count = static_cast<int64_t>(inputs.size());
    const int64_t instruction_count = static_cast<int64_t>(program.size() / 3);
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

#include "StaxPointwise.h"
#include "CUDARuntime.h"
#include "Macros.h"
#include "Tensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

// One thread evaluates the full program for one element.  Operands resolve
// through a device-resident pointer table (inputs) and a per-thread
// temporary pool (instruction results); constants ride a flat buffer.
// `io_t` selects the storage format, `compute_t` the arithmetic type; the
// only supported widening is half/bfloat16 storage with float arithmetic.
template <typename io_t, typename compute_t>
struct ProgramState {
    const Instruction* instructions;
    const compute_t* constants;
    const io_t* const* input_ptrs;
    const int64_t* input_sizes;
    const int64_t* input_strides;
    const int64_t* output_sizes;
    const int64_t* output_strides;
    const uint8_t* input_flat;
    int64_t input_count;
    int64_t instruction_count;
    int64_t rank;

    __device__ compute_t load_io(const io_t& v) const {
        return static_cast<compute_t>(v);
    }

    __device__ io_t store_io(compute_t v) const {
        return static_cast<io_t>(v);
    }

    __device__ compute_t resolve(int64_t ref, const compute_t* local_temps,
                                 int64_t element) const {
        if (ref >= 0) {
            if (ref < input_count) {
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
                return load_io(input_ptrs[ref][offset]);
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
};

template <typename io_t, typename compute_t, int kTemps>
__global__ void stax_fused_pointwise_kernel(
    ProgramState<io_t, compute_t> state,
    io_t* output,
    int64_t count) {
    compute_t temps[kTemps];
    int64_t pending_where[3] = {0, 0, 0};
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; index < count; index += stride) {
        state.evaluate(temps, pending_where, index);
        output[index] = state.store_io(
            temps[state.instruction_count - 1]);
    }
}

template <typename io_t, typename compute_t, int kTemps>
__global__ void stax_fused_pointwise_multi_kernel(
    ProgramState<io_t, compute_t> state,
    io_t* const* temp_outputs,
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
            temp_outputs[o][index] =
                state.store_io(temps[temp_refs[o] - state.input_count]);
        }
    }
}

template <int kTemps, typename io_t, typename compute_t>
void launch_program(
    const ProgramState<io_t, compute_t>& state,
    io_t* output,
    io_t* const* temp_outputs,
    const int64_t* temp_refs,
    int64_t temp_output_count,
    int64_t count,
    cudaStream_t stream) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    if (output != nullptr) {
        stax_fused_pointwise_kernel<io_t, compute_t, kTemps>
            <<<blocks, threads, 0, stream>>>(state, output, count);
    } else {
        stax_fused_pointwise_multi_kernel<io_t, compute_t, kTemps>
            <<<blocks, threads, 0, stream>>>(
                state, temp_outputs, temp_refs, temp_output_count, count);
    }
    checkCuda(cudaGetLastError(), "stax fused pointwise launch");
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
    for (const Tensor& input : inputs) {
        if (!input.defined() || !input.device().is_cuda()) {
            throw std::runtime_error(
                "Stax CUDA fused pointwise requires defined CUDA tensors");
        }
        if (input.dtype() != first.dtype()) {
            throw std::runtime_error(
                "Stax CUDA fused pointwise inputs must share one dtype");
        }
    }
    static_cast<void>(broadcast_shape(inputs));
}

Tensor byte_buffer(int64_t bytes, const Tensor& like) {
    return Tensor::empty({bytes > 0 ? bytes : 1}, DType::UInt8, like.device());
}

template <typename io_t, typename compute_t>
std::vector<Tensor> run_program(const std::vector<Tensor>& inputs,
                                const std::vector<int64_t>& program,
                                const std::vector<double>& constants,
                                const std::vector<int64_t>* output_refs) {
    const int64_t input_count = static_cast<int64_t>(inputs.size());
    const int64_t instruction_count = static_cast<int64_t>(program.size() / 3);
    const std::vector<int64_t> output_shape = broadcast_shape(inputs);
    int64_t count = 1;
    for (int64_t extent : output_shape) count *= extent;
    const int64_t rank = static_cast<int64_t>(output_shape.size());

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
    for (size_t i = 0; i < temp_refs.size(); ++i) {
        temp_tensors.push_back(Tensor::empty(
            output_shape, inputs.front().dtype(),
            inputs.front().device()));
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

    // Device staging: pointer tables and program metadata are tiny; tensor
    // data itself is never copied.
    std::vector<const io_t*> host_input_ptrs;
    host_input_ptrs.reserve(inputs.size());
    for (const Tensor& input : inputs) {
        host_input_ptrs.push_back(input.data_ptr<io_t>());
    }
    std::vector<int64_t> host_input_sizes(
        static_cast<size_t>(input_count * rank), 1);
    std::vector<int64_t> host_input_strides(
        static_cast<size_t>(input_count * rank), 0);
    std::vector<uint8_t> host_input_flat(
        static_cast<size_t>(input_count), 1);
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
    std::vector<int64_t> host_output_strides(
        static_cast<size_t>(rank), 1);
    int64_t inner = 1;
    for (int64_t dim = rank - 1; dim >= 0; --dim) {
        host_output_strides[static_cast<size_t>(dim)] = inner;
        inner *= output_shape[static_cast<size_t>(dim)];
    }
    std::vector<io_t*> host_output_ptrs;
    host_output_ptrs.reserve(temp_tensors.size());
    for (const Tensor& t : temp_tensors) {
        host_output_ptrs.push_back(t.data_ptr<io_t>());
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
    Tensor instr_buf = byte_buffer(
        static_cast<int64_t>(sizeof(Instruction) * instruction_count),
        inputs.front());
    Tensor const_buf = byte_buffer(
        static_cast<int64_t>(sizeof(compute_t) * constants.size()),
        inputs.front());
    Tensor inptr_buf = byte_buffer(
        static_cast<int64_t>(sizeof(const io_t*) * input_count),
        inputs.front());
    Tensor size_buf = byte_buffer(
        static_cast<int64_t>(sizeof(int64_t) * host_input_sizes.size()),
        inputs.front());
    Tensor stride_buf = byte_buffer(
        static_cast<int64_t>(sizeof(int64_t) * host_input_strides.size()),
        inputs.front());
    Tensor output_size_buf = byte_buffer(
        static_cast<int64_t>(sizeof(int64_t) * output_shape.size()),
        inputs.front());
    Tensor output_stride_buf = byte_buffer(
        static_cast<int64_t>(sizeof(int64_t) * host_output_strides.size()),
        inputs.front());
    Tensor input_flat_buf = byte_buffer(
        static_cast<int64_t>(sizeof(uint8_t) * host_input_flat.size()),
        inputs.front());
    checkCuda(cudaMemcpyAsync(instr_buf.data_ptr(), host_instructions.data(),
                               sizeof(Instruction) * instruction_count,
                               cudaMemcpyHostToDevice, stream),
               "stax fused pointwise program upload");
    if (!constants.empty()) {
        checkCuda(cudaMemcpyAsync(const_buf.data_ptr(), host_constants.data(),
                                   sizeof(compute_t) * constants.size(),
                                   cudaMemcpyHostToDevice, stream),
                   "stax fused pointwise constants upload");
    }
    checkCuda(cudaMemcpyAsync(inptr_buf.data_ptr(), host_input_ptrs.data(),
                               sizeof(const io_t*) * input_count,
                               cudaMemcpyHostToDevice, stream),
               "stax fused pointwise input pointer table upload");
    if (!host_input_sizes.empty()) {
        checkCuda(cudaMemcpyAsync(
                      size_buf.data_ptr(), host_input_sizes.data(),
                      sizeof(int64_t) * host_input_sizes.size(),
                      cudaMemcpyHostToDevice, stream),
                  "stax fused pointwise input size upload");
        checkCuda(cudaMemcpyAsync(
                      stride_buf.data_ptr(), host_input_strides.data(),
                      sizeof(int64_t) * host_input_strides.size(),
                      cudaMemcpyHostToDevice, stream),
                  "stax fused pointwise input stride upload");
    }
    if (!output_shape.empty()) {
        checkCuda(cudaMemcpyAsync(
                      output_size_buf.data_ptr(), output_shape.data(),
                      sizeof(int64_t) * output_shape.size(),
                      cudaMemcpyHostToDevice, stream),
                  "stax fused pointwise output size upload");
        checkCuda(cudaMemcpyAsync(
                      output_stride_buf.data_ptr(), host_output_strides.data(),
                      sizeof(int64_t) * host_output_strides.size(),
                      cudaMemcpyHostToDevice, stream),
                  "stax fused pointwise output stride upload");
    }
    checkCuda(cudaMemcpyAsync(
                  input_flat_buf.data_ptr(), host_input_flat.data(),
                  sizeof(uint8_t) * host_input_flat.size(),
                  cudaMemcpyHostToDevice, stream),
              "stax fused pointwise input layout upload");

    ProgramState<io_t, compute_t> state;
    state.instructions = reinterpret_cast<const Instruction*>(instr_buf.data_ptr());
    state.constants = reinterpret_cast<const compute_t*>(const_buf.data_ptr());
    state.input_ptrs = reinterpret_cast<const io_t* const*>(inptr_buf.data_ptr());
    state.input_sizes = reinterpret_cast<const int64_t*>(size_buf.data_ptr());
    state.input_strides = reinterpret_cast<const int64_t*>(stride_buf.data_ptr());
    state.output_sizes = reinterpret_cast<const int64_t*>(output_size_buf.data_ptr());
    state.output_strides = reinterpret_cast<const int64_t*>(output_stride_buf.data_ptr());
    state.input_flat = reinterpret_cast<const uint8_t*>(input_flat_buf.data_ptr());
    state.input_count = input_count;
    state.instruction_count = instruction_count;
    state.rank = rank;

    if (output_refs == nullptr) {
        io_t* out_ptr = temp_tensors[0].data_ptr<io_t>();
        if (instruction_count <= 8) {
            launch_program<8, io_t, compute_t>(
                state, out_ptr, nullptr, nullptr, 0, count, stream);
        } else if (instruction_count <= 16) {
            launch_program<16, io_t, compute_t>(
                state, out_ptr, nullptr, nullptr, 0, count, stream);
        } else if (instruction_count <= 32) {
            launch_program<32, io_t, compute_t>(
                state, out_ptr, nullptr, nullptr, 0, count, stream);
        } else {
            launch_program<64, io_t, compute_t>(
                state, out_ptr, nullptr, nullptr, 0, count, stream);
        }
        return outs;
    }
    Tensor outptr_buf = byte_buffer(
        static_cast<int64_t>(sizeof(io_t*) * host_output_ptrs.size()),
        inputs.front());
    Tensor ref_buf = byte_buffer(
        static_cast<int64_t>(sizeof(int64_t) * temp_refs.size()),
        inputs.front());
    checkCuda(cudaMemcpyAsync(outptr_buf.data_ptr(), host_output_ptrs.data(),
                               sizeof(io_t*) * host_output_ptrs.size(),
                               cudaMemcpyHostToDevice, stream),
               "stax fused pointwise output pointer table upload");
    checkCuda(cudaMemcpyAsync(ref_buf.data_ptr(), temp_refs.data(),
                               sizeof(int64_t) * temp_refs.size(),
                               cudaMemcpyHostToDevice, stream),
               "stax fused pointwise output refs upload");
    io_t* const* output_ptrs =
        reinterpret_cast<io_t* const*>(outptr_buf.data_ptr());
    const int64_t* output_ref_ptr =
        reinterpret_cast<const int64_t*>(ref_buf.data_ptr());
    if (instruction_count <= 8) {
        launch_program<8, io_t, compute_t>(
            state, nullptr, output_ptrs, output_ref_ptr,
            static_cast<int64_t>(temp_refs.size()), count, stream);
    } else if (instruction_count <= 16) {
        launch_program<16, io_t, compute_t>(
            state, nullptr, output_ptrs, output_ref_ptr,
            static_cast<int64_t>(temp_refs.size()), count, stream);
    } else if (instruction_count <= 32) {
        launch_program<32, io_t, compute_t>(
            state, nullptr, output_ptrs, output_ref_ptr,
            static_cast<int64_t>(temp_refs.size()), count, stream);
    } else {
        launch_program<64, io_t, compute_t>(
            state, nullptr, output_ptrs, output_ref_ptr,
            static_cast<int64_t>(temp_refs.size()), count, stream);
    }
    return outs;
}

#endif // USE_CUDA

} // namespace

#ifdef USE_CUDA

Tensor stax_fused_pointwise_cuda(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants) {
    check_program_shape(inputs, program, constants, 1);
    const DType dt = inputs.front().dtype();
    if (dt == DType::Float32) {
        return run_program<float, float>(inputs, program, constants, nullptr)[0];
    }
    if (dt == DType::Float64) {
        return run_program<double, double>(inputs, program, constants, nullptr)[0];
    }
    if (dt == DType::Float16) {
        return run_program<tensorplay::Half, float>(inputs, program, constants, nullptr)[0];
    }
    if (dt == DType::BFloat16) {
        return run_program<tensorplay::BFloat16, float>(inputs, program, constants, nullptr)[0];
    }
    throw std::runtime_error(
        "Stax CUDA fused pointwise supports float16/bfloat16/float32/float64");
}

std::vector<Tensor> stax_fused_pointwise_cuda_multi(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    const std::vector<int64_t>& output_refs) {
    check_program_shape(inputs, program, constants,
                        static_cast<int64_t>(output_refs.size()));
    const DType dt = inputs.front().dtype();
    if (dt == DType::Float32) {
        return run_program<float, float>(inputs, program, constants, &output_refs);
    }
    if (dt == DType::Float64) {
        return run_program<double, double>(inputs, program, constants, &output_refs);
    }
    if (dt == DType::Float16) {
        return run_program<tensorplay::Half, float>(inputs, program, constants, &output_refs);
    }
    if (dt == DType::BFloat16) {
        return run_program<tensorplay::BFloat16, float>(inputs, program, constants, &output_refs);
    }
    throw std::runtime_error(
        "Stax CUDA fused pointwise multi supports float16/bfloat16/float32/float64");
}

#else // !USE_CUDA

Tensor stax_fused_pointwise_cuda(
    const std::vector<Tensor>&,
    const std::vector<int64_t>&,
    const std::vector<double>&) {
    TP_THROW(NotImplementedError, "stax fused pointwise requires CUDA");
}

std::vector<Tensor> stax_fused_pointwise_cuda_multi(
    const std::vector<Tensor>&,
    const std::vector<int64_t>&,
    const std::vector<double>&,
    const std::vector<int64_t>&) {
    TP_THROW(NotImplementedError, "stax fused pointwise requires CUDA");
}

#endif // USE_CUDA

} // namespace cuda
} // namespace tensorplay

#include "StaxPointwise.h"

#include "Macros.h"
#include "backend/cuda/JitUtils.h"

#include <cuda.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef USE_CUDA

namespace tensorplay {
namespace cuda {
namespace {

// Instruction opcodes of the shared pointwise program encoding.  The values
// are part of the program format the graph builder emits; they mirror the
// interpreter's decode table.
enum class GenOp : int64_t {
    Add = 1, Sub = 2, Mul = 3, Div = 4, Pow = 5,
    Neg = 6, Pos = 7, Abs = 8, Sin = 9, Cos = 10,
    Exp = 11, Log = 12, Sigmoid = 13, Sqrt = 14, Square = 15,
    Tanh = 16, Relu = 17, ReluGrad = 18, AbsGrad = 19,
    Lt = 20, Le = 21, Gt = 22, Ge = 23, Eq = 24, Ne = 25,
    Where = 26, WhereRest = 27,
    Minimum = 28, Maximum = 29, ClampMin = 30, ClampMax = 31,
    Rsqrt = 32, Exp2 = 33, Erf = 34, Cast = 35,
};

constexpr int64_t kGenMaxInstructions = 64;
constexpr int64_t kGenMaxInputs = 32;
constexpr int64_t kGenMinCount = 1024;

// The program evaluates in one arithmetic width: double only when an
// operand is float64, float otherwise (half storage widens on load).
bool wide_family(const std::vector<Tensor>& inputs) {
    for (const Tensor& input : inputs) {
        if (input.dtype() == DType::Float64) return true;
    }
    return false;
}

std::string pointer_type(DType dt) {
    switch (dt) {
        case DType::Float32: return "float";
        case DType::Float64: return "double";
        case DType::Float16: return "__half";
        case DType::BFloat16: return "__nv_bfloat16";
        default:
            throw std::runtime_error(
                "generated pointwise supports "
                "float16/bfloat16/float32/float64 only");
    }
}

std::string load_expr(DType dt, const std::string& ptr, const std::string& idx) {
    switch (dt) {
        case DType::Float32:
        case DType::Float64:
            return "(T)(" + ptr + "[" + idx + "])";
        case DType::Float16:
            return "(T)__half2float(" + ptr + "[" + idx + "])";
        case DType::BFloat16:
            return "(T)__bfloat162float(" + ptr + "[" + idx + "])";
        default:
            throw std::runtime_error(
                "generated pointwise supports "
                "float16/bfloat16/float32/float64 only");
    }
}

std::string store_expr(DType dt, const std::string& place, const std::string& value) {
    switch (dt) {
        case DType::Float32:
        case DType::Float64:
            return place + " = (" + value + ");";
        case DType::Float16:
            return place + " = __float2half((float)(" + value + "));";
        case DType::BFloat16:
            return place + " = __float2bfloat16((float)(" + value + "));";
        default:
            throw std::runtime_error(
                "generated pointwise supports "
                "float16/bfloat16/float32/float64 only");
    }
}

// A float literal that survives text-to-C++ round trip: enough digits for
// the double value, with an explicit fractional part so the token stays
// floating point.
std::string literal(double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "generated pointwise constants must be finite");
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    std::string text(buffer);
    if (text.find('.') == std::string::npos &&
        text.find('e') == std::string::npos &&
        text.find('E') == std::string::npos) {
        text += ".0";
    }
    return text;
}

// One element's straight-line body: every input loads once, every
// instruction expands into a scalar expression over named temporaries, and
// every requested output stores its value.  `idx` names the element index
// variable and `suffix` disambiguates the temporaries of sibling elements
// handled by the same thread.
std::string emit_element_body(
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    const std::vector<DType>& input_dtypes,
    const std::vector<int64_t>& out_temp_refs,
    const std::vector<DType>& out_dtypes,
    const std::string& idx,
    const std::string& suffix) {
    const int64_t input_count = static_cast<int64_t>(input_dtypes.size());
    const int64_t instruction_count =
        static_cast<int64_t>(program.size() / 3);

    auto resolve = [&](int64_t ref, int64_t current) -> std::string {
        if (ref < 0) {
            const int64_t index = -ref - 1;
            if (index >= static_cast<int64_t>(constants.size())) {
                throw std::runtime_error(
                    "generated pointwise constant reference is invalid");
            }
            return "T(" + literal(constants[index]) + ")";
        }
        if (ref < input_count) {
            return "x" + std::to_string(ref) + suffix;
        }
        const int64_t temp = ref - input_count;
        if (temp < 0 || temp >= current) {
            throw std::runtime_error(
                "generated pointwise value reference is invalid");
        }
        return "t" + std::to_string(temp) + suffix;
    };

    std::ostringstream body;
    for (int64_t i = 0; i < input_count; ++i) {
        body << "      const T x" << i << suffix << " = "
             << load_expr(input_dtypes[i], "in" + std::to_string(i), idx)
             << ";\n";
    }

    std::string where_cond;
    std::string where_then;
    bool pending_where = false;
    for (int64_t instruction = 0; instruction < instruction_count;
         ++instruction) {
        const int64_t op = program[instruction * 3];
        const int64_t lhs_ref = program[instruction * 3 + 1];
        const int64_t rhs_ref = program[instruction * 3 + 2];
        const std::string temp =
            "t" + std::to_string(instruction) + suffix;
        std::string expr;
        const std::string lhs = resolve(lhs_ref, instruction);
        switch (op) {
            case static_cast<int64_t>(GenOp::Where): {
                if (pending_where) {
                    throw std::runtime_error(
                        "generated pointwise where instruction is unpaired");
                }
                where_cond = lhs;
                where_then = resolve(rhs_ref, instruction);
                pending_where = true;
                expr = "T(0)";
                break;
            }
            case static_cast<int64_t>(GenOp::WhereRest): {
                if (!pending_where) {
                    throw std::runtime_error(
                        "generated pointwise where_rest has no condition");
                }
                expr = "((" + where_cond + ") != T(0) ? (" + where_then +
                       ") : (" + resolve(rhs_ref, instruction) + "))";
                pending_where = false;
                break;
            }
            case static_cast<int64_t>(GenOp::Add):
                expr = "(" + lhs + " + " + resolve(rhs_ref, instruction) + ")";
                break;
            case static_cast<int64_t>(GenOp::Sub):
                expr = "(" + lhs + " - " + resolve(rhs_ref, instruction) + ")";
                break;
            case static_cast<int64_t>(GenOp::Mul):
                expr = "(" + lhs + " * " + resolve(rhs_ref, instruction) + ")";
                break;
            case static_cast<int64_t>(GenOp::Div):
                expr = "(" + lhs + " / " + resolve(rhs_ref, instruction) + ")";
                break;
            case static_cast<int64_t>(GenOp::Pow):
                expr = "(T)::pow((double)(" + lhs + "), (double)(" +
                       resolve(rhs_ref, instruction) + "))";
                break;
            case static_cast<int64_t>(GenOp::Neg):
                expr = "(-(" + lhs + "))";
                break;
            case static_cast<int64_t>(GenOp::Pos):
                expr = lhs;
                break;
            case static_cast<int64_t>(GenOp::Abs):
                expr = "::fabs(" + lhs + ")";
                break;
            case static_cast<int64_t>(GenOp::Sin):
                expr = "::sin(" + lhs + ")";
                break;
            case static_cast<int64_t>(GenOp::Cos):
                expr = "::cos(" + lhs + ")";
                break;
            case static_cast<int64_t>(GenOp::Exp):
                expr = "::exp(" + lhs + ")";
                break;
            case static_cast<int64_t>(GenOp::Log):
                expr = "::log(" + lhs + ")";
                break;
            case static_cast<int64_t>(GenOp::Sigmoid):
                expr = "(T(1) / (T(1) + ::exp(-(" + lhs + "))))";
                break;
            case static_cast<int64_t>(GenOp::Sqrt):
                expr = "::sqrt(" + lhs + ")";
                break;
            case static_cast<int64_t>(GenOp::Square):
                expr = "((" + lhs + ") * (" + lhs + "))";
                break;
            case static_cast<int64_t>(GenOp::Tanh):
                expr = "::tanh(" + lhs + ")";
                break;
            case static_cast<int64_t>(GenOp::Relu):
                expr = "((" + lhs + ") > T(0) ? (" + lhs + ") : T(0))";
                break;
            case static_cast<int64_t>(GenOp::ReluGrad):
                expr = "((" + lhs + ") > T(0) ? T(1) : T(0))";
                break;
            case static_cast<int64_t>(GenOp::AbsGrad):
                expr = "(((" + lhs + ") > T(0) ? T(1) : T(0)) - ((" + lhs +
                       ") < T(0) ? T(1) : T(0)))";
                break;
            case static_cast<int64_t>(GenOp::Lt):
                expr = "((" + lhs + ") < (" + resolve(rhs_ref, instruction) +
                       ") ? T(1) : T(0))";
                break;
            case static_cast<int64_t>(GenOp::Le):
                expr = "((" + lhs + ") <= (" + resolve(rhs_ref, instruction) +
                       ") ? T(1) : T(0))";
                break;
            case static_cast<int64_t>(GenOp::Gt):
                expr = "((" + lhs + ") > (" + resolve(rhs_ref, instruction) +
                       ") ? T(1) : T(0))";
                break;
            case static_cast<int64_t>(GenOp::Ge):
                expr = "((" + lhs + ") >= (" + resolve(rhs_ref, instruction) +
                       ") ? T(1) : T(0))";
                break;
            case static_cast<int64_t>(GenOp::Eq):
                expr = "((" + lhs + ") == (" + resolve(rhs_ref, instruction) +
                       ") ? T(1) : T(0))";
                break;
            case static_cast<int64_t>(GenOp::Ne):
                expr = "((" + lhs + ") != (" + resolve(rhs_ref, instruction) +
                       ") ? T(1) : T(0))";
                break;
            case static_cast<int64_t>(GenOp::Minimum):
                expr = "((" + lhs + ") < (" + resolve(rhs_ref, instruction) +
                       ") ? (" + lhs + ") : (" +
                       resolve(rhs_ref, instruction) + "))";
                break;
            case static_cast<int64_t>(GenOp::Maximum):
                expr = "((" + lhs + ") > (" + resolve(rhs_ref, instruction) +
                       ") ? (" + lhs + ") : (" +
                       resolve(rhs_ref, instruction) + "))";
                break;
            case static_cast<int64_t>(GenOp::ClampMin):
                expr = "((" + lhs + ") < (" + resolve(rhs_ref, instruction) +
                       ") ? (" + resolve(rhs_ref, instruction) + ") : (" +
                       lhs + "))";
                break;
            case static_cast<int64_t>(GenOp::ClampMax):
                expr = "((" + lhs + ") > (" + resolve(rhs_ref, instruction) +
                       ") ? (" + resolve(rhs_ref, instruction) + ") : (" +
                       lhs + "))";
                break;
            case static_cast<int64_t>(GenOp::Rsqrt):
                expr = "(T(1) / ::sqrt(" + lhs + "))";
                break;
            case static_cast<int64_t>(GenOp::Exp2):
                expr = "::exp2(" + lhs + ")";
                break;
            case static_cast<int64_t>(GenOp::Erf):
                expr = "::erf(" + lhs + ")";
                break;
            case static_cast<int64_t>(GenOp::Cast):
                // One compute width per program: the only cast the encoding
                // can express is the identity.
                expr = lhs;
                break;
            default:
                throw std::runtime_error(
                    "generated pointwise opcode is unsupported");
        }
        body << "      T " << temp << " = " << expr << ";\n";
    }
    if (pending_where) {
        throw std::runtime_error(
            "generated pointwise where instruction is unpaired");
    }
    for (size_t port = 0; port < out_temp_refs.size(); ++port) {
        const int64_t temp = out_temp_refs[port] - input_count;
        if (temp < 0 || temp >= instruction_count) {
            throw std::runtime_error(
                "generated pointwise output reference is invalid");
        }
        body << "      "
             << store_expr(out_dtypes[port],
                           "out" + std::to_string(port) + "[" + idx + "]",
                           "t" + std::to_string(temp) + suffix)
             << "\n";
    }
    return body.str();
}

std::string emit_kernel_source(
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    bool wide,
    const std::vector<DType>& input_dtypes,
    const std::vector<int64_t>& out_temp_refs,
    const std::vector<DType>& out_dtypes,
    int elements_per_thread,
    const std::string& name) {
    std::ostringstream source;
    source << "#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n\n";
    source << "typedef " << (wide ? "double" : "float") << " T;\n\n";
    source << "extern \"C\" __global__ void " << name << "_kernel(\n";
    for (size_t port = 0; port < out_dtypes.size(); ++port) {
        source << "    " << pointer_type(out_dtypes[port]) << "* out" << port
               << ",\n";
    }
    for (size_t i = 0; i < input_dtypes.size(); ++i) {
        source << "    const " << pointer_type(input_dtypes[i]) << "* in" << i
               << ",\n";
    }
    source << "    long long n) {\n";
    source << "  const long long base = ((long long)blockIdx.x * blockDim.x) * "
           << elements_per_thread << " + threadIdx.x;\n";
    for (int element = 0; element < elements_per_thread; ++element) {
        const std::string suffix =
            elements_per_thread > 1 ? "_" + std::to_string(element) : "";
        const std::string idx =
            elements_per_thread > 1 ? "i" + std::to_string(element) : "i0";
        source << "  {\n";
        source << "    const long long " << idx << " = base + " << element
               << "LL * (long long)blockDim.x;\n";
        source << "    if (" << idx << " < n) {\n";
        source << emit_element_body(program, constants, input_dtypes,
                                    out_temp_refs, out_dtypes, idx, suffix);
        source << "    }\n";
        source << "  }\n";
    }
    source << "}\n";
    return source.str();
}

bool generated_form_supported(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    const std::vector<int64_t>& temp_refs) {
    const int64_t input_count = static_cast<int64_t>(inputs.size());
    const int64_t instruction_count =
        static_cast<int64_t>(program.size() / 3);
    if (instruction_count < 1 || instruction_count > kGenMaxInstructions ||
        input_count < 1 || input_count > kGenMaxInputs || temp_refs.empty()) {
        return false;
    }
    for (const double value : constants) {
        if (!std::isfinite(value)) return false;
    }
    for (const int64_t ref : temp_refs) {
        if (ref < input_count || ref >= input_count + instruction_count) {
            // Output refs into fresh storage must name program temporaries;
            // aliased outputs replay inputs and stay on the interpreter.
            return false;
        }
    }
    return true;
}

}  // namespace

bool launch_generated_pointwise(
    const std::vector<Tensor>& inputs,
    const std::vector<int64_t>& program,
    const std::vector<double>& constants,
    const std::vector<int64_t>& temp_refs,
    const std::vector<Tensor>& temp_tensors,
    const std::vector<DType>& out_dtypes,
    int64_t count) {
    static std::mutex cache_lock;
    static std::unordered_map<std::string, jit::NvrtcFunction> cache;
    static std::unordered_set<std::string> unsupported;

    // Escape hatch for diagnosis: a plain off switch keeps every launch on
    // the interpreter.
    static const bool enabled = [] {
        const char* value = std::getenv("TP_STAX_PROGRAM_CODEGEN");
        return value == nullptr || std::string(value) != "0";
    }();
    if (!enabled) return false;

    if (count < kGenMinCount || temp_refs.size() != temp_tensors.size() ||
        temp_refs.size() != out_dtypes.size() ||
        !generated_form_supported(inputs, program, constants, temp_refs)) {
        return false;
    }

    std::ostringstream key;
    key << program.size() << ':' << inputs.size() << ':' << out_dtypes.size();
    for (int64_t op : program) key << op << ',';
    for (double value : constants) {
        try {
            key << literal(value) << ',';
        } catch (const std::exception&) {
            return false;
        }
    }
    for (int64_t ref : temp_refs) key << ref << ',';
    for (const Tensor& input : inputs) {
        key << static_cast<int>(input.dtype()) << ',';
    }
    for (DType dt : out_dtypes) key << static_cast<int>(dt) << ',';
    const std::string cache_key = key.str();

    {
        const std::lock_guard<std::mutex> lock(cache_lock);
        if (unsupported.count(cache_key)) return false;
    }

    try {
        std::vector<DType> input_dtypes;
        input_dtypes.reserve(inputs.size());
        for (const Tensor& input : inputs) {
            input_dtypes.push_back(input.dtype());
        }
        // Sibling elements per thread: independent straight-line chains the
        // compiler can interleave, unlike the interpreter whose dispatch
        // separates every instruction.  The chain length bounds the
        // register budget, so deep programs take fewer siblings.
        const int64_t instruction_count =
            static_cast<int64_t>(program.size() / 3);
        const int elements_per_thread =
            instruction_count <= 16 ? 4 : (instruction_count <= 32 ? 2 : 1);

        const size_t hash = std::hash<std::string>()(cache_key);
        char name_buffer[32];
        std::snprintf(name_buffer, sizeof(name_buffer), "stax_gen_pw_%zx",
                      hash);
        const std::string name(name_buffer);

        const std::string source = emit_kernel_source(
            program, constants, wide_family(inputs), input_dtypes, temp_refs,
            out_dtypes, elements_per_thread, name);

        jit::NvrtcFunction fn;
        {
            const std::lock_guard<std::mutex> lock(cache_lock);
            auto it = cache.find(cache_key);
            if (it != cache.end()) {
                fn = it->second;
            } else {
                fn = jit::jit_pwise_function(source, name);
                cache.emplace(cache_key, fn);
            }
        }

        // cuLaunchKernel reads each argument's VALUE through an extra
        // indirection, so the array holds addresses of stable storage that
        // carries the pointer values themselves.
        std::vector<void*> out_values;
        std::vector<void*> in_values;
        out_values.reserve(temp_tensors.size());
        in_values.reserve(inputs.size());
        for (const Tensor& out : temp_tensors) {
            out_values.push_back(out.data_ptr());
        }
        for (const Tensor& input : inputs) {
            in_values.push_back(input.data_ptr());
        }
        long long n = static_cast<long long>(count);
        std::vector<const void*> args;
        args.reserve(out_values.size() + in_values.size() + 1);
        for (size_t i = 0; i < out_values.size(); ++i) {
            args.push_back(&out_values[i]);
        }
        for (size_t i = 0; i < in_values.size(); ++i) {
            args.push_back(&in_values[i]);
        }
        args.push_back(&n);

        const long long block = 128;
        const long long span = block * elements_per_thread;
        const unsigned grid = static_cast<unsigned>((n + span - 1) / span);
        jit::launch_jitted_pwise_function(
            fn, args.data(), {grid, 1u, 1u}, {128u, 1u, 1u}, 0);
        return true;
    } catch (const std::exception&) {
        // Unsupported program forms and toolchain failures keep the
        // interpreter route; remember the miss so later calls skip the
        // emission attempt.
        const std::lock_guard<std::mutex> lock(cache_lock);
        unsupported.insert(cache_key);
        return false;
    }
}

}  // namespace cuda
}  // namespace tensorplay

#else  // !USE_CUDA

namespace tensorplay {
namespace cuda {

bool launch_generated_pointwise(
    const std::vector<Tensor>&,
    const std::vector<int64_t>&,
    const std::vector<double>&,
    const std::vector<int64_t>&,
    const std::vector<Tensor>&,
    const std::vector<DType>&,
    int64_t) {
    return false;
}

}  // namespace cuda
}  // namespace tensorplay

#endif  // USE_CUDA

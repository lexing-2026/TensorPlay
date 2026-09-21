#include "backend/cuda/JitUtils.h"

#include <limits>
#include <sstream>
#include <unordered_map>

#include "CUDARuntime.h"
#include "Device.h"
#include "Exception.h"

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kJitNumThreads = 128;
constexpr int kJitThreadWorkSize = 4;
constexpr int kJitBlockWork = kJitNumThreads * kJitThreadWorkSize;

size_t jit_itemsize(ScalarType dtype) {
    switch (dtype) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Int16:
        case ScalarType::UInt16:
            return 2;
        case ScalarType::Float64:
        case ScalarType::Int64:
        case ScalarType::UInt64:
            return 8;
        case ScalarType::Bool:
        case ScalarType::Int8:
        case ScalarType::UInt8:
            return 1;
        default:
            return 4;
    }
}

}  // namespace

std::vector<Tensor> compile_and_launch_jiterator(
    const std::string& code_string,
    const std::string& kernel_name,
    int num_outputs,
    const std::vector<Tensor>& tensors,
    const std::vector<jit::JitExtraArg>& extra_args,
    bool return_by_ref) {
    TP_CHECK(!tensors.empty(), "jiterator needs at least one tensor input");
    const int nInputs = static_cast<int>(tensors.size());
    const int64_t N = tensors[0].numel();
    const ScalarType dtype = tensors[0].dtype();
    const Device device = tensors[0].device();
    CUDAGuard guard(device);

    std::vector<Tensor> outputs;
    outputs.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
        outputs.emplace_back(std::vector<int64_t>{N}, dtype, device);
    }

    if (N == 0) {
        return outputs;
    }
    TP_CHECK(N <= std::numeric_limits<int32_t>::max(),
             "jiterator does not support tensors with more than 2^31 elements");

    std::vector<void*> ptrs;
    ptrs.reserve(nInputs + num_outputs);
    for (Tensor& out : outputs) {
        ptrs.push_back(out.data_ptr());
    }
    for (const Tensor& t : tensors) {
        ptrs.push_back(t.data_ptr());
    }
    const int vec_size = jit::can_vectorize_up_to(jit_itemsize(dtype), ptrs);

    std::vector<std::string> extra_types;
    extra_types.reserve(extra_args.size());
    for (const auto& arg : extra_args) {
        extra_types.push_back(arg.type_name);
    }

    // Cache key: every input to generate_code plus the enclosing scope.
    // The functor source is part of the key: two kernels may share a name
    // and dtype but compute different expressions.
    std::ostringstream key;
    key << nInputs << 'x' << num_outputs << '|' << kernel_name << '|'
        << std::hash<std::string>{}(code_string) << '|'
        << static_cast<int>(dtype) << '|' << vec_size << '|'
        << return_by_ref << '|' << static_cast<int>(device.index());
    for (const auto& type : extra_types) {
        key << '|' << type;
    }
    const std::string cache_key = key.str();

    static std::mutex cache_lock;
    static std::unordered_map<std::string, jit::NvrtcFunction> cache;
    jit::NvrtcFunction* fn = &cache[cache_key];
    if (!fn->function) {
        const std::lock_guard<std::mutex> lock(cache_lock);
        if (!fn->function) {
            const std::string code = jit::generate_code(
                nInputs, num_outputs, code_string, kernel_name,
                jit::type_name(dtype), jit::compute_type_name(dtype),
                jit::type_name(dtype), extra_types, kJitThreadWorkSize,
                vec_size, return_by_ref);
            *fn = jit::jit_pwise_function(code, kernel_name);
        }
    }

    const unsigned grid =
        static_cast<unsigned>((N + kJitBlockWork - 1) / kJitBlockWork);

    // The generated kernel takes N by value as int; pass an int-typed slot.
    const int n_launch = static_cast<int>(N);
    std::vector<const void*> arg_ptrs;
    arg_ptrs.reserve(2 + extra_args.size());
    arg_ptrs.push_back(&n_launch);
    arg_ptrs.push_back(ptrs.data());
    for (const auto& arg : extra_args) {
        arg_ptrs.push_back(
            std::visit([](const auto& v) { return static_cast<const void*>(&v); },
                       arg.value));
    }

    jit::launch_jitted_pwise_function(*fn, arg_ptrs.data(), {grid, 1u, 1u},
                                      {kJitNumThreads, 1u, 1u});
    return outputs;
}

}  // namespace cuda
}  // namespace tensorplay

#pragma once

#include <cuda.h>
#include <string>
#include <variant>
#include <vector>
#include <vector_types.h>

#include "DType.h"
#include "Macros.h"
#include "Tensor.h"

namespace tensorplay {
namespace cuda {
namespace jit {

// A compiled JIT kernel. `module` owns the cubin/PTX image; the CUfunction
// handle is only valid while the module is alive, so the pair is kept as one
// cache entry.
struct P10_API NvrtcFunction {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
};

// Extra runtime argument carried verbatim into the generated kernel. The
// `type_name` must be a CUDA type name (double, long long, bool, ...); the
// stored value keeps the exact bits of that type.
struct P10_API JitExtraArg {
    std::string type_name;
    std::variant<bool, long long, double> value;
};

// Generate the source of an elementwise kernel that evaluates `func` for
// every element of `nInputs` inputs and writes `nOutputs` outputs. All
// tensors are contiguous and share `f_inputs_type`; the functor is invoked
// with `compute_type` values; results are stored as `result_type`.
// `vec_size` is 1, 2 or 4 (the kernel falls back to a scalar loop when the
// tail is too short to vectorize). `thread_work_size` is the number of
// elements each thread handles per block.
P10_API std::string generate_code(
    int nInputs,
    int nOutputs,
    const std::string& func,
    const std::string& name,
    const std::string& f_inputs_type,
    const std::string& compute_type,
    const std::string& result_type,
    const std::vector<std::string>& extra_args_types,
    int thread_work_size,
    int vec_size,
    bool return_by_ref);

// Compile `code` with NVRTC and look up `kernel_name` in the resulting
// module. Throws on compilation failure, including the driver log.
P10_API NvrtcFunction jit_pwise_function(
    const std::string& code, const std::string& kernel_name);

// Launch a compiled kernel on the current CUDA stream.
P10_API void launch_jitted_pwise_function(
    NvrtcFunction function,
    const void* args[],
    const dim3 nBlocks,
    const dim3 kBlockSize,
    const int smem = 0);

// The device type name used inside generated kernels for `dtype`.
P10_API std::string type_name(ScalarType dtype);

// The compute (working) type for `dtype`: half-precision operands are
// evaluated in float, everything else in its own type.
P10_API std::string compute_type_name(ScalarType dtype);

// Largest vector width (1, 2 or 4) that fits every pointer in `ptrs`
// given `itemsize`-sized elements.
P10_API int can_vectorize_up_to(size_t itemsize,
                                const std::vector<void*>& ptrs);

}  // namespace jit

// Compile `code_string` with the jiterator pipeline and launch it over the
// elementwise tensors. Inputs must already be contiguous, broadcast to a
// common shape, and share one dtype (the Python wrapper normalizes them).
// Returns `num_outputs` fresh tensors of that dtype.
std::vector<Tensor> compile_and_launch_jiterator(
    const std::string& code_string,
    const std::string& kernel_name,
    int num_outputs,
    const std::vector<Tensor>& tensors,
    const std::vector<jit::JitExtraArg>& extra_args,
    bool return_by_ref);

}  // namespace cuda
}  // namespace tensorplay

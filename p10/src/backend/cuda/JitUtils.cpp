#include "backend/cuda/JitUtils.h"

#include <cuda_runtime.h>
#include <nvrtc.h>

#include "backend/cuda/DriverApi.h"

#include <cstdlib>
#include <sstream>
#include <unordered_map>

#include "CUDARuntime.h"
#include "Exception.h"

namespace tensorplay {
namespace cuda {
namespace jit {

namespace {

void check_nvrtc(nvrtcResult result, const char* what) {
    if (result != NVRTC_SUCCESS) {
        TP_THROW(RuntimeError, std::string("NVRTC ") + what + " failed: " +
                                   nvrtcGetErrorString(result));
    }
}

// Compute capability of the current device, used as the NVRTC arch target.
std::string current_arch() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        TP_THROW(RuntimeError, "no CUDA device for jiterator compilation");
    }
    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) !=
            cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) !=
            cudaSuccess) {
        TP_THROW(RuntimeError,
                 "could not determine the compute capability of CUDA device " +
                     std::to_string(device));
    }
    return "sm_" + std::to_string(major) + std::to_string(minor);
}

std::string scalar_type_name(ScalarType dtype) {
    switch (dtype) {
        case ScalarType::Float32:
            return "float";
        case ScalarType::Float64:
            return "double";
        case ScalarType::Float16:
            return "__half";
        case ScalarType::BFloat16:
            return "__nv_bfloat16";
        case ScalarType::Int8:
            return "signed char";
        case ScalarType::Int16:
            return "short";
        case ScalarType::Int32:
            return "int";
        case ScalarType::Int64:
            return "long long";
        case ScalarType::UInt8:
            return "unsigned char";
        case ScalarType::UInt16:
            return "unsigned short";
        case ScalarType::UInt32:
            return "unsigned int";
        case ScalarType::UInt64:
            return "unsigned long long";
        case ScalarType::Bool:
            return "bool";
        default:
            TP_THROW(NotImplementedError,
                     "jiterator does not support this scalar type");
    }
}

std::string scalar_compute_name(ScalarType dtype) {
    switch (dtype) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
            return "float";
        default:
            return scalar_type_name(dtype);
    }
}

// Emit the per-lane load/call/store statements for one element.
void emit_lane(std::ostringstream& out, const std::string& offset_expr,
               const std::string& name, int nInputs, int nOutputs,
               const std::string& f_inputs_type,
               const std::string& compute_type,
               const std::string& result_type,
               const std::string& extra_call_args, bool return_by_ref) {
    std::ostringstream lane_args;
    for (int i = 0; i < nInputs; ++i) {
        out << "    const " << compute_type << " arg" << i
            << " = static_cast<" << compute_type << ">(jit_load<"
            << f_inputs_type << ">(data.data[" << std::to_string(i + nOutputs)
            << "], " << offset_expr << "));\n";
        lane_args << "arg" << i << ", ";
    }
    std::string lane = lane_args.str();
    if (!lane.empty()) {
        lane.pop_back();
        lane.pop_back();
    }
    if (return_by_ref) {
        std::ostringstream outs;
        for (int i = 0; i < nOutputs; ++i) {
            outs << ", out" << i;
        }
        out << "    " << name << "<" << compute_type << ">(" << lane
            << extra_call_args << outs.str() << ");\n";
        for (int i = 0; i < nOutputs; ++i) {
            out << "    jit_store<" << result_type << ">(data.data[" << i
                << "], " << offset_expr << ", static_cast<" << result_type
                << ">(out" << i << "));\n";
        }
    } else {
        out << "    jit_store<" << result_type << ">(data.data[0], "
            << offset_expr << ", static_cast<" << result_type << ">(" << name
            << "<" << compute_type << ">(" << lane << extra_call_args
            << ")));\n";
    }
}

}  // namespace

std::string type_name(ScalarType dtype) { return scalar_type_name(dtype); }

std::string compute_type_name(ScalarType dtype) {
    return scalar_compute_name(dtype);
}

int can_vectorize_up_to(size_t itemsize, const std::vector<void*>& ptrs) {
    int result = 1;
    for (void* ptr : ptrs) {
        auto ip = reinterpret_cast<uintptr_t>(ptr);
        int candidate = 1;
        if (itemsize <= 4 && ip % (16 * itemsize) == 0) {
            candidate = 4;
        } else if (ip % (8 * itemsize) == 0) {
            candidate = 4;
        } else if (ip % (4 * itemsize) == 0) {
            candidate = 2;
        }
        result = std::min(result, candidate);
    }
    return result;
}

std::string generate_code(
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
    bool return_by_ref) {
    const int nTensors = nInputs + nOutputs;

    std::ostringstream extra_params;
    std::ostringstream extra_call_args;
    for (size_t i = 0; i < extra_args_types.size(); ++i) {
        const std::string arg_name = "extra_arg_" + std::to_string(i);
        extra_params << ", " << extra_args_types[i] << " " << arg_name;
        extra_call_args << ", " << arg_name;
    }

    std::ostringstream kernel;
    kernel << R"(
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T, int N>
struct JitData {
    T data[N];
};

template <typename T>
__device__ __forceinline__ T jit_load(const char* base, int idx) {
    return reinterpret_cast<const T*>(base)[idx];
}

template <typename T>
__device__ __forceinline__ void jit_store(char* base, int idx, T value) {
    reinterpret_cast<T*>(base)[idx] = value;
}

)";
    kernel << func << "\n\n";

    kernel << "extern \"C\" __global__ void " << name << "_kernel(\n"
           << "    int N,\n"
           << "    JitData<char*, " << nTensors << "> data"
           << extra_params.str() << ") {\n"
           << "  constexpr int TWS = " << thread_work_size << ";\n"
           << "  constexpr int NT = 128;\n"
           << "  constexpr int BLOCK_WORK = TWS * NT;\n"
           << "  constexpr int VEC = " << std::max(vec_size, 1) << ";\n"
           << "  const int idx = blockIdx.x;\n"
           << "  const int remain = N - BLOCK_WORK * idx;\n"
           << "  const int base = BLOCK_WORK * idx;\n";

    // Scalar path: also covers the final partial block and VEC == 1.
    kernel << "  if (remain < BLOCK_WORK || VEC == 1) {\n"
           << "    int tid = threadIdx.x;\n"
           << "#pragma unroll\n"
           << "    for (int k = 0; k < TWS; ++k) {\n"
           << "      if (tid >= remain) break;\n"
           << "      const int linear = base + tid;\n";
    if (return_by_ref) {
        for (int i = 0; i < nOutputs; ++i) {
            kernel << "      " << compute_type << " out" << i << ";\n";
        }
    }
    emit_lane(kernel, "linear", name, nInputs, nOutputs, f_inputs_type,
              compute_type, result_type, extra_call_args.str(), return_by_ref);
    kernel << "      tid += NT;\n"
           << "    }\n"
           << "    return;\n"
           << "  }\n";

    // Vectorized path: BLOCK_WORK is a multiple of VEC, so every lane is
    // fully in bounds here.
    kernel << "  constexpr int ROUNDS = TWS / VEC;\n"
           << "#pragma unroll\n"
           << "  for (int r = 0; r < ROUNDS; ++r) {\n"
           << "    const int linear = base + (threadIdx.x + r * NT) * VEC;\n";
    if (return_by_ref) {
        for (int i = 0; i < nOutputs; ++i) {
            kernel << "    " << compute_type << " out" << i << ";\n";
        }
    }
    kernel << "#pragma unroll\n"
           << "    for (int v = 0; v < VEC; ++v) {\n";
    emit_lane(kernel, "linear + v", name, nInputs, nOutputs, f_inputs_type,
              compute_type, result_type, extra_call_args.str(), return_by_ref);
    kernel << "    }\n"
           << "  }\n"
           << "}\n";
    return kernel.str();
}

namespace {

using CuModuleLoadDataExFn = CUresult (*)(CUmodule*, const void*, unsigned int,
                                          const CUjit_option*, void**);
using CuModuleGetFunctionFn = CUresult (*)(CUfunction*, CUmodule, const char*);
using CuLaunchKernelFn =
    CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                 unsigned int, unsigned int, unsigned int, unsigned int,
                 CUstream, void**, void**);

// Driver entry points are resolved lazily so the wheel imports on machines
// without the driver; the JIT path needs them, so a missing driver surfaces
// here with a clear error instead of at module load.
CuModuleLoadDataExFn cu_module_load_data_ex() {
    static const CuModuleLoadDataExFn fn =
        driver::resolve_symbol<CuModuleLoadDataExFn>("cuModuleLoadDataEx");
    if (fn == nullptr) {
        TP_THROW(RuntimeError, "the CUDA driver is not available");
    }
    return fn;
}

CuModuleGetFunctionFn cu_module_get_function() {
    static const CuModuleGetFunctionFn fn =
        driver::resolve_symbol<CuModuleGetFunctionFn>("cuModuleGetFunction");
    if (fn == nullptr) {
        TP_THROW(RuntimeError, "the CUDA driver is not available");
    }
    return fn;
}

CuLaunchKernelFn cu_launch_kernel() {
    static const CuLaunchKernelFn fn =
        driver::resolve_symbol<CuLaunchKernelFn>("cuLaunchKernel");
    if (fn == nullptr) {
        TP_THROW(RuntimeError, "the CUDA driver is not available");
    }
    return fn;
}

}  // namespace

NvrtcFunction jit_pwise_function(const std::string& code,
                                 const std::string& kernel_name) {
    nvrtcProgram program;
    check_nvrtc(nvrtcCreateProgram(&program, code.c_str(), "jiterator.cu", 0,
                                   nullptr, nullptr),
                "create program");

    const std::string arch = current_arch();
    const std::string arch_option = "--gpu-architecture=" + arch;
    // Modern NVRTC no longer ships the toolkit headers as built-ins.
    const char* cuda_home = std::getenv("CUDA_HOME");
    const std::string include_dir =
        (cuda_home ? std::string(cuda_home) : "/usr/local/cuda") + "/include";
    const std::string include_option = "--include-path=" + include_dir;
    const char* options[] = {"--std=c++17", arch_option.c_str(),
                             include_option.c_str(), "-default-device"};
    nvrtcResult compile_result =
        nvrtcCompileProgram(program, 4, options);
    if (compile_result != NVRTC_SUCCESS) {
        size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        std::string log(log_size, '\0');
        if (log_size > 0) {
            nvrtcGetProgramLog(program, log.data());
        }
        nvrtcDestroyProgram(&program);
        TP_THROW(RuntimeError,
                 "jiterator kernel compilation failed:\n" + log);
    }

    size_t ptx_size = 0;
    check_nvrtc(nvrtcGetPTXSize(program, &ptx_size), "get PTX size");
    std::string ptx(ptx_size, '\0');
    check_nvrtc(nvrtcGetPTX(program, ptx.data()), "get PTX");
    nvrtcDestroyProgram(&program);

    NvrtcFunction fn;
    if (cu_module_load_data_ex()(&fn.module, ptx.c_str(), 0, nullptr, nullptr) !=
        CUDA_SUCCESS) {
        TP_THROW(RuntimeError, "failed to load the jiterator kernel module");
    }
    const std::string symbol = kernel_name + "_kernel";
    if (cu_module_get_function()(&fn.function, fn.module, symbol.c_str()) !=
        CUDA_SUCCESS) {
        TP_THROW(RuntimeError,
                 std::string("kernel ") + symbol +
                     " not found in the jiterator module");
    }
    return fn;
}

void launch_jitted_pwise_function(NvrtcFunction function, const void* args[],
                                  const dim3 nBlocks, const dim3 kBlockSize,
                                  const int smem) {
    uint32_t grid_x = nBlocks.x;
    uint32_t grid_y = nBlocks.y;
    uint32_t grid_z = nBlocks.z;
    CUresult result = cu_launch_kernel()(
        function.function, grid_x, grid_y, grid_z, kBlockSize.x, kBlockSize.y,
        kBlockSize.z, smem, getCurrentCUDAStream().stream(),
        const_cast<void**>(args), nullptr);
    if (result != CUDA_SUCCESS) {
        TP_THROW(RuntimeError, "failed to launch the jiterator kernel");
    }
}

}  // namespace jit
}  // namespace cuda
}  // namespace tensorplay

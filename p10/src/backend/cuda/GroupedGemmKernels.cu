// Ragged grouped GEMM on tensor cores: one persistent kernel walks all
// expert groups.
//
// The per-group cuBLAS entry pays a heuristic lookup plus a launch per
// expert; the persistent scheduler amortizes both by covering every group
// inside a single grid.  Per-group descriptors -- problem shape, operand
// pointers, leading dimensions -- are assembled on the device from the
// offsets tensor, so the launch path carries no host staging at all.
// Half inputs run the native mma with an f32 accumulator; Float32 runs the
// reduced-exponent f32 tensor-op path, gated by the same Context flag that
// selects the f32 tensor-op compute mode for the cuBLAS entry.

#include "CudaGemm.h"
#include "CUDARuntime.h"
#include "Context.h"
#include "Exception.h"

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

#define CUDA_CHECK(condition)                                                \
    do {                                                                     \
        const cudaError_t _tp_cuda_err = (condition);                        \
        if (_tp_cuda_err != cudaSuccess) {                                   \
            throw ::tensorplay::RuntimeError(                                \
                {__FILE__, __func__, __LINE__},                              \
                ::tensorplay::detail::format_msg(                            \
                    std::string("CUDA Error: ") +                            \
                    cudaGetErrorString(_tp_cuda_err)));                      \
        }                                                                    \
    } while (0)

using GemmCoord = cutlass::gemm::GemmCoord;

// Carve points of the flat descriptor buffer written by the meta kernel.
// The problem-size array is packed at sizeof(GemmCoord)=12 bytes per entry
// (the scheduler indexes it as a C array); everything after it is 8-byte
// aligned.  Pointers and leading dimensions are per-group arrays, as the
// grouped Arguments expect.
struct MetaLayout {
    int64_t problem_sizes;
    int64_t ptr_a;
    int64_t ptr_b;
    int64_t ptr_c;
    int64_t ptr_d;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldd;
    int64_t total;
};

MetaLayout meta_layout(int64_t groups) {
    MetaLayout m{};
    int64_t off = 0;
    m.problem_sizes = off;
    off += groups * 12;
    off = (off + 7) & ~int64_t(7);
    m.ptr_a = off; off += groups * 8;
    m.ptr_b = off; off += groups * 8;
    m.ptr_c = off; off += groups * 8;
    m.ptr_d = off; off += groups * 8;
    m.lda = off; off += groups * 8;
    m.ldb = off; off += groups * 8;
    m.ldc = off; off += groups * 8;
    m.ldd = off; off += groups * 8;
    m.total = off;
    return m;
}

// Group g covers rows [offs[g-1], offs[g]) of A; its B expert is slice g of
// the operand stack.  Offsets were validated non-decreasing in [0, M] by the
// caller, so the descriptors need no clamping.
template <typename ElementT, typename OffsetT>
__global__ void build_grouped_meta_kernel(
    char* meta, MetaLayout layout, const OffsetT* offs, int64_t groups,
    const ElementT* a_base, const ElementT* b_base, ElementT* d_base,
    int64_t k_dim, int64_t n_dim) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const int64_t start = g ? int64_t(offs[g - 1]) : 0;
    const int64_t end = int64_t(offs[g]);
    reinterpret_cast<GemmCoord*>(meta + layout.problem_sizes)[g] =
        GemmCoord(int(end - start), int(n_dim), int(k_dim));
    reinterpret_cast<const ElementT**>(meta + layout.ptr_a)[g] =
        a_base + start * k_dim;
    reinterpret_cast<const ElementT**>(meta + layout.ptr_b)[g] =
        b_base + g * k_dim * n_dim;
    reinterpret_cast<ElementT**>(meta + layout.ptr_c)[g] =
        d_base + start * n_dim;
    reinterpret_cast<ElementT**>(meta + layout.ptr_d)[g] =
        d_base + start * n_dim;
    reinterpret_cast<int64_t*>(meta + layout.lda)[g] = k_dim;
    reinterpret_cast<int64_t*>(meta + layout.ldb)[g] = n_dim;
    reinterpret_cast<int64_t*>(meta + layout.ldc)[g] = n_dim;
    reinterpret_cast<int64_t*>(meta + layout.ldd)[g] = n_dim;
}

// Tensor-op config for half inputs: mma.m16n8k16 with an f32 accumulator.
// The 128x128 tile feeds the tensor cores on prefill-sized groups; the
// 64x64 tile keeps small decode groups from reserving whole 128-row waves.
template <typename ElementT, int TbM, int TbN, int TbK, int WarpM, int WarpN,
          int WarpK>
struct HalfGroupedGemm {
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementT, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone,
        8, ElementT, cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone, 8, ElementT,
        cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<TbM, TbN, TbK>,
        cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<ElementT, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmGrouped<Kernel>;
};

// Reduced-exponent f32 tensor-op config: mma.m16n8k8 over tf32-converted
// operands (OpMultiplyAddFastF32), f32 accumulator.  The f32 epilogue
// staging tile keeps the tile at 128x64 so the shared-memory footprint
// stays within the consumer-ampere budget.
struct TF32GroupedGemm {
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 4,
        float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 4,
        float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 64, 16>,
        cutlass::gemm::GemmShape<64, 32, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
        cutlass::arch::OpMultiplyAddFastF32>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmGrouped<Kernel>;
};

// Occupancy of a config is device-constant for the process; cache it per
// instantiation.  A failure here only shrinks the grid, never correctness.
template <typename Gemm>
int grouped_gemm_max_active() {
    static const int active = []() {
        const int blocks = Gemm::maximum_active_blocks();
        return blocks > 0 ? blocks : 1;
    }();
    return active;
}

// Tiles assigned to the persistent grid: every tile of every group, capped
// by the co-resident capacity -- extra CTAs would only rescan the problem
// list.
int64_t total_tile_count(const int64_t* ends_host, int64_t groups, int64_t n,
                         int tb_m, int tb_n) {
    int64_t tiles = 0;
    int64_t start = 0;
    for (int64_t g = 0; g < groups; ++g) {
        const int64_t rows = ends_host[g] - start;
        start = ends_host[g];
        tiles += ((rows + tb_m - 1) / tb_m) * ((n + tb_n - 1) / tb_n);
    }
    return tiles;
}

template <typename Gemm, typename ElementT>
bool launch_grouped_gemm(const Tensor& meta_buf, const MetaLayout& layout,
                         int64_t groups, int threadblock_count,
                         cudaStream_t stream) {
    char* meta = static_cast<char*>(meta_buf.data_ptr());
    typename Gemm::Arguments args(
        reinterpret_cast<GemmCoord*>(meta + layout.problem_sizes),
        static_cast<int>(groups), threadblock_count, {1.0f, 0.0f},
        reinterpret_cast<ElementT**>(meta + layout.ptr_a),
        reinterpret_cast<ElementT**>(meta + layout.ptr_b),
        reinterpret_cast<ElementT**>(meta + layout.ptr_c),
        reinterpret_cast<ElementT**>(meta + layout.ptr_d),
        reinterpret_cast<int64_t*>(meta + layout.lda),
        reinterpret_cast<int64_t*>(meta + layout.ldb),
        reinterpret_cast<int64_t*>(meta + layout.ldc),
        reinterpret_cast<int64_t*>(meta + layout.ldd));
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
    Tensor ws_buf;
    void* ws = nullptr;
    const size_t ws_bytes = gemm.get_workspace_size(args);
    if (ws_bytes) {
        ws_buf = Tensor::empty({static_cast<int64_t>(ws_bytes)}, DType::UInt8,
                               Device(DeviceType::CUDA));
        ws = ws_buf.data_ptr();
    }
    if (gemm.initialize(args, ws, stream) != cutlass::Status::kSuccess)
        return false;
    return gemm.run(stream) == cutlass::Status::kSuccess;
}

// Build the descriptors on the device and hand the whole ragged batch to one
// persistent kernel.  `tb_m`/`tb_n` are the tile of `Gemm` and only feed the
// host-side tile arithmetic.
template <typename Gemm, typename ElementT>
bool run_grouped_config(const Tensor& offs_dev, const void* a_ptr,
                        const void* b_ptr, void* d_ptr,
                        const int64_t* ends_host, int64_t groups,
                        int64_t k_dim, int64_t n_dim, int tb_m, int tb_n,
                        int sm_count, cudaStream_t stream) {
    const int64_t tiles =
        total_tile_count(ends_host, groups, n_dim, tb_m, tb_n);
    if (tiles == 0) return true;  // caller's zero-filled output is the result
    MetaLayout layout = meta_layout(groups);
    Tensor meta =
        Tensor::empty({layout.total}, DType::UInt8, Device(DeviceType::CUDA));
    char* meta_base = static_cast<char*>(meta.data_ptr());
    const int blocks = static_cast<int>((groups + 255) / 256);
    if (offs_dev.dtype() == DType::Int32) {
        build_grouped_meta_kernel<ElementT, int32_t><<<blocks, 256, 0, stream>>>(
            meta_base, layout, static_cast<const int32_t*>(offs_dev.data_ptr()),
            groups, static_cast<const ElementT*>(a_ptr),
            static_cast<const ElementT*>(b_ptr),
            static_cast<ElementT*>(d_ptr), k_dim, n_dim);
    } else {
        build_grouped_meta_kernel<ElementT, int64_t><<<blocks, 256, 0, stream>>>(
            meta_base, layout, static_cast<const int64_t*>(offs_dev.data_ptr()),
            groups, static_cast<const ElementT*>(a_ptr),
            static_cast<const ElementT*>(b_ptr),
            static_cast<ElementT*>(d_ptr), k_dim, n_dim);
    }
    CUDA_CHECK(cudaGetLastError());
    const int threadblock_count = static_cast<int>(std::min(
        tiles, int64_t(sm_count) * grouped_gemm_max_active<Gemm>()));
    return launch_grouped_gemm<Gemm, ElementT>(meta, layout, groups,
                                               threadblock_count, stream);
}

template <typename ElementT>
bool run_half_grouped_gemm(const Tensor& offs_dev, const void* a_ptr,
                           const void* b_ptr, void* d_ptr,
                           const int64_t* ends_host, int64_t groups,
                           int64_t k_dim, int64_t n_dim,
                           int64_t max_group_rows, int sm_count,
                           int smem_optin, cudaStream_t stream) {
    using Big = HalfGroupedGemm<ElementT, 128, 128, 32, 64, 64, 32>;
    using Small = HalfGroupedGemm<ElementT, 64, 64, 32, 32, 32, 32>;
    if (max_group_rows >= 64 &&
        int(sizeof(typename Big::Kernel::SharedStorage)) <= smem_optin) {
        return run_grouped_config<typename Big::Gemm, ElementT>(
            offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, k_dim, n_dim,
            128, 128, sm_count, stream);
    }
    if (int(sizeof(typename Small::Kernel::SharedStorage)) > smem_optin)
        return false;
    return run_grouped_config<typename Small::Gemm, ElementT>(
        offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, k_dim, n_dim, 64,
        64, sm_count, stream);
}

}  // namespace

bool try_grouped_gemm_tensor_op(const Tensor& self, const Tensor& mat2,
                                const Tensor& offs, Tensor& out,
                                const int64_t* ends_host,
                                int64_t max_group_rows) {
    const int64_t groups = mat2.size(0);
    const int64_t k_dim = self.size(1);
    const int64_t n_dim = mat2.size(2);

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int cc_major = 0, sm_count = 0, smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &cc_major, cudaDevAttrComputeCapabilityMajor, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &sm_count, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(
        &smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    if (cc_major < 8) return false;

    const DType dt = self.dtype();
    const bool f32_tensor_op =
        dt == DType::Float32 && tensorplay::globalContext().allowTF32CuBLAS();
    if (dt != DType::Float16 && dt != DType::BFloat16 && !f32_tensor_op)
        return false;
    // Row starts must carry the mainloop's 16-byte access granularity.
    const int64_t align = f32_tensor_op ? 4 : 8;
    if ((k_dim % align) != 0 || (n_dim % align) != 0) return false;
    if (self.size(0) > INT32_MAX || k_dim > INT32_MAX || n_dim > INT32_MAX)
        return false;

    Tensor a = self.is_contiguous() ? self : self.contiguous();
    Tensor b = mat2.is_contiguous() ? mat2 : mat2.contiguous();
    if (!offs.is_contiguous() || !out.is_contiguous()) return false;
    const Device out_dev = a.device();
    Tensor offs_dev;
    if (offs.device().type() == DeviceType::CUDA) {
        if (offs.device().index() != out_dev.index()) return false;
        offs_dev = offs;
    } else {
        offs_dev = offs.to(out_dev);
    }

    cudaStream_t stream = getCurrentCUDAStream().stream();
    const void* a_ptr = a.data_ptr();
    const void* b_ptr = b.data_ptr();
    void* d_ptr = out.data_ptr();

    if (dt == DType::Float16) {
        return run_half_grouped_gemm<cutlass::half_t>(
            offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, k_dim, n_dim,
            max_group_rows, sm_count, smem_optin, stream);
    }
    if (dt == DType::BFloat16) {
        return run_half_grouped_gemm<cutlass::bfloat16_t>(
            offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, k_dim, n_dim,
            max_group_rows, sm_count, smem_optin, stream);
    }
    return run_grouped_config<typename TF32GroupedGemm::Gemm, float>(
        offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, k_dim, n_dim, 128,
        64, sm_count, stream);
}

}  // namespace cuda
}  // namespace tensorplay

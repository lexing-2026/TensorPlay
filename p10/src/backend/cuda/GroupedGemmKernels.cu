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

// The persistent tensor-op route below consumes the standalone CUTLASS
// headers through the include directory the build adds only when a vendored
// tree is present.  Builds without that tree (and the HIP lane) keep the
// plain per-group cuBLAS fallback in grouped_mm; the guard hides the whole
// route and leaves a stub that declines every dispatch.
#if !defined(USE_ROCM) && __has_include("cutlass/cutlass.h") && \
    __has_include("cutlass/gemm/device/gemm_grouped.h")
#define TP_GROUPED_GEMM_TENSOR_OP 1
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#endif

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#if defined(TP_GROUPED_GEMM_TENSOR_OP)

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
// the operand stack.  Every boundary is clamped into [0, m_total] and an
// inverted span collapses to zero rows, so offsets that were not validated
// on the host can neither launch out-of-bounds tiles nor carry negative
// extents; the launch path therefore needs no device-to-host round trip.
// `ldb_b` is B's leading dimension: N for row-major experts, K for
// column-major ones (the [E, out, in] weight stacks read through their
// transpose).
template <typename ElementT, typename OffsetT>
__global__ void build_grouped_meta_kernel(
    char* meta, MetaLayout layout, const OffsetT* offs, int64_t groups,
    int64_t m_total, const ElementT* a_base, const ElementT* b_base,
    ElementT* d_base, int64_t k_dim, int64_t n_dim, int64_t ldb_b) {
    const int64_t g = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    int64_t start = g ? int64_t(offs[g - 1]) : 0;
    int64_t end = int64_t(offs[g]);
    start = start < 0 ? 0 : (start > m_total ? m_total : start);
    end = end < 0 ? 0 : (end > m_total ? m_total : end);
    const int64_t rows = end > start ? end - start : 0;
    reinterpret_cast<GemmCoord*>(meta + layout.problem_sizes)[g] =
        GemmCoord(int(rows), int(n_dim), int(k_dim));
    reinterpret_cast<const ElementT**>(meta + layout.ptr_a)[g] =
        a_base + start * k_dim;
    reinterpret_cast<const ElementT**>(meta + layout.ptr_b)[g] =
        b_base + g * k_dim * n_dim;
    reinterpret_cast<ElementT**>(meta + layout.ptr_c)[g] =
        d_base + start * n_dim;
    reinterpret_cast<ElementT**>(meta + layout.ptr_d)[g] =
        d_base + start * n_dim;
    reinterpret_cast<int64_t*>(meta + layout.lda)[g] = k_dim;
    reinterpret_cast<int64_t*>(meta + layout.ldb)[g] = ldb_b;
    reinterpret_cast<int64_t*>(meta + layout.ldc)[g] = n_dim;
    reinterpret_cast<int64_t*>(meta + layout.ldd)[g] = n_dim;
}

// Rows beyond the last offset belong to no group; the grouped kernel leaves
// them untouched, so the synchronization-free path zeroes that tail on the
// device instead of deciding on the host.
template <typename ElementT, typename OffsetT>
__global__ void grouped_gemm_tail_zero_kernel(ElementT* d,
                                              const OffsetT* offs,
                                              int64_t groups, int64_t m_total,
                                              int64_t n) {
    __shared__ int64_t s_end;
    if (threadIdx.x == 0) {
        int64_t end = groups ? int64_t(offs[groups - 1]) : 0;
        end = end < 0 ? 0 : (end > m_total ? m_total : end);
        s_end = end;
    }
    __syncthreads();
    const int64_t total = (m_total - s_end) * n;
    const int64_t stride = int64_t(gridDim.x) * blockDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        d[s_end * n + i] = ElementT(0);
    }
}

template <typename ElementT>
bool launch_grouped_tail_zero(void* d_ptr, const Tensor& offs_dev,
                              int64_t groups, int64_t m_total, int64_t n_dim,
                              cudaStream_t stream) {
    const int64_t total = m_total * n_dim;
    if (total <= 0) return true;
    const int blocks =
        static_cast<int>(std::min<int64_t>((total + 255) / 256, 4096));
    if (offs_dev.dtype() == DType::Int32) {
        grouped_gemm_tail_zero_kernel<ElementT, int32_t><<<blocks, 256, 0, stream>>>(
            static_cast<ElementT*>(d_ptr),
            static_cast<const int32_t*>(offs_dev.data_ptr()), groups, m_total,
            n_dim);
    } else {
        grouped_gemm_tail_zero_kernel<ElementT, int64_t><<<blocks, 256, 0, stream>>>(
            static_cast<ElementT*>(d_ptr),
            static_cast<const int64_t*>(offs_dev.data_ptr()), groups, m_total,
            n_dim);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// Tensor-op config for half inputs: mma.m16n8k16 with an f32 accumulator.
// The 128x128 tile feeds the tensor cores on prefill-sized groups; the
// 64x64 tile keeps small decode groups from reserving whole 128-row waves.
// LayoutB selects how the per-expert [K, N] operand reads: RowMajor for
// dense stacks, ColumnMajor for stacks of [N, K] row-major weights viewed
// through their transpose (the linear-layer convention).
template <typename ElementT, typename LayoutB, int TbM, int TbN, int TbK,
          int WarpM, int WarpN, int WarpK>
struct HalfGroupedGemm {
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementT, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone,
        8, ElementT, LayoutB,
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
template <typename LayoutB>
struct TF32GroupedGemm {
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 4,
        float, LayoutB, cutlass::ComplexTransform::kNone, 4,
        float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 64, 16>,
        cutlass::gemm::GemmShape<64, 32, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
        cutlass::arch::OpMultiplyAddFastF32>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmGrouped<Kernel>;
};

// Scalar fp32 grouped GEMM: plain FMA arithmetic with fp32 accumulation, so
// numerics match the per-group cublas path bit-for-bit in spirit while the
// device-side descriptor machinery stays identical to the tensor-op routes.
// This is the entry point that keeps full-precision fp32 callers on the
// synchronization-free fast path.
template <typename LayoutB, int TbM, int TbN, int TbK,
          int WarpM, int WarpN, int WarpK>
struct SimtGroupedGemm {
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        float, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,
        float, LayoutB, cutlass::ComplexTransform::kNone, 1,
        float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<TbM, TbN, TbK>,
        cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
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
// list.  Without host-validated offsets an upper bound from the row total
// stands in for the per-group arithmetic; the device scheduler sizes the
// work from the clamped descriptors, so an overestimate only means some
// CTAs exit without tiles.
int64_t total_tile_count(const int64_t* ends_host, int64_t groups,
                         int64_t m_total, int64_t n, int64_t tb_m,
                         int64_t tb_n) {
    if (ends_host) {
        int64_t tiles = 0;
        int64_t start = 0;
        for (int64_t g = 0; g < groups; ++g) {
            const int64_t rows = ends_host[g] - start;
            start = ends_host[g];
            tiles += ((rows + tb_m - 1) / tb_m) * ((n + tb_n - 1) / tb_n);
        }
        return tiles;
    }
    return groups * ((m_total + tb_m - 1) / tb_m) * ((n + tb_n - 1) / tb_n);
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
                        int64_t m_total, int64_t k_dim, int64_t n_dim,
                        bool b_col, int tb_m, int tb_n, int sm_count,
                        cudaStream_t stream) {
    const int64_t tiles =
        total_tile_count(ends_host, groups, m_total, n_dim, tb_m, tb_n);
    if (tiles == 0) return true;  // zero-filled rows [0, M) are the result
    MetaLayout layout = meta_layout(groups);
    Tensor meta =
        Tensor::empty({layout.total}, DType::UInt8, Device(DeviceType::CUDA));
    char* meta_base = static_cast<char*>(meta.data_ptr());
    const int blocks = static_cast<int>((groups + 255) / 256);
    const int64_t ldb_b = b_col ? k_dim : n_dim;
    if (offs_dev.dtype() == DType::Int32) {
        build_grouped_meta_kernel<ElementT, int32_t><<<blocks, 256, 0, stream>>>(
            meta_base, layout, static_cast<const int32_t*>(offs_dev.data_ptr()),
            groups, m_total, static_cast<const ElementT*>(a_ptr),
            static_cast<const ElementT*>(b_ptr),
            static_cast<ElementT*>(d_ptr), k_dim, n_dim, ldb_b);
    } else {
        build_grouped_meta_kernel<ElementT, int64_t><<<blocks, 256, 0, stream>>>(
            meta_base, layout, static_cast<const int64_t*>(offs_dev.data_ptr()),
            groups, m_total, static_cast<const ElementT*>(a_ptr),
            static_cast<const ElementT*>(b_ptr),
            static_cast<ElementT*>(d_ptr), k_dim, n_dim, ldb_b);
    }
    CUDA_CHECK(cudaGetLastError());
    const int threadblock_count = static_cast<int>(std::min(
        tiles, int64_t(sm_count) * grouped_gemm_max_active<Gemm>()));
    return launch_grouped_gemm<Gemm, ElementT>(meta, layout, groups,
                                               threadblock_count, stream);
}

template <typename ElementT, typename LayoutB>
bool run_half_layout(const Tensor& offs_dev, const void* a_ptr,
                     const void* b_ptr, void* d_ptr,
                     const int64_t* ends_host, int64_t groups,
                     int64_t m_total, int64_t k_dim, int64_t n_dim,
                     int64_t max_rows_bound, int sm_count, int smem_optin,
                     cudaStream_t stream) {
    using Big = HalfGroupedGemm<ElementT, LayoutB, 128, 128, 32, 64, 64, 32>;
    using Small = HalfGroupedGemm<ElementT, LayoutB, 64, 64, 32, 32, 32, 32>;
    if (max_rows_bound >= 64 &&
        int(sizeof(typename Big::Kernel::SharedStorage)) <= smem_optin) {
        return run_grouped_config<typename Big::Gemm, ElementT>(
            offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
            n_dim, /*b_col=*/!std::is_same<LayoutB, cutlass::layout::RowMajor>::value,
            128, 128, sm_count, stream);
    }
    if (int(sizeof(typename Small::Kernel::SharedStorage)) > smem_optin)
        return false;
    return run_grouped_config<typename Small::Gemm, ElementT>(
        offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
        n_dim, /*b_col=*/!std::is_same<LayoutB, cutlass::layout::RowMajor>::value,
        64, 64, sm_count, stream);
}

template <typename ElementT>
bool run_half_grouped_gemm(const Tensor& offs_dev, const void* a_ptr,
                           const void* b_ptr, void* d_ptr,
                           const int64_t* ends_host, int64_t groups,
                           int64_t m_total, int64_t k_dim, int64_t n_dim,
                           bool b_col, int64_t max_rows_bound, int sm_count,
                           int smem_optin, cudaStream_t stream) {
    if (b_col) {
        return run_half_layout<ElementT, cutlass::layout::ColumnMajor>(
            offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
            n_dim, max_rows_bound, sm_count, smem_optin, stream);
    }
    return run_half_layout<ElementT, cutlass::layout::RowMajor>(
        offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
        n_dim, max_rows_bound, sm_count, smem_optin, stream);
}

// Scalar-precision tile selection.  Tile choice drives two opposite costs:
// wide accumulators pay one register each and strand residency, while small
// tiles multiply the number of B-matrix passes.  The row count per group is
// unknown on the host (no synchronization here), so selection runs on an
// even-split estimate -- malformed offsets only shift work between
// equivalent tiles, never correctness.
template <typename LayoutB>
bool run_simt_layout(const Tensor& offs_dev, const void* a_ptr,
                     const void* b_ptr, void* d_ptr,
                     const int64_t* ends_host, int64_t groups,
                     int64_t m_total, int64_t k_dim, int64_t n_dim,
                     int sm_count, cudaStream_t stream) {
    using Big = SimtGroupedGemm<LayoutB, 128, 128, 8, 64, 64, 8>;
    using Small = SimtGroupedGemm<LayoutB, 64, 128, 8, 32, 32, 8>;
    using Skinny = SimtGroupedGemm<LayoutB, 32, 128, 8, 32, 32, 8>;
    const bool b_col =
        !std::is_same<LayoutB, cutlass::layout::RowMajor>::value;
    const int64_t rows_est = (m_total + groups - 1) / groups;
    if (rows_est <= 32) {
        // Each group covers at most one skinny M tile: the grid walks the
        // full B matrices exactly once, which is what tiny-M launches are
        // bound by.
        return run_grouped_config<typename Skinny::Gemm, float>(
            offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
            n_dim, b_col, 32, 128, sm_count, stream);
    }
    const int64_t big_tiles =
        groups * ((rows_est + 127) / 128) * ((n_dim + 127) / 128);
    if (big_tiles > 2 * sm_count) {
        return run_grouped_config<typename Big::Gemm, float>(
            offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
            n_dim, b_col, 128, 128, sm_count, stream);
    }
    return run_grouped_config<typename Small::Gemm, float>(
        offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
        n_dim, b_col, 64, 128, sm_count, stream);
}

template <typename LayoutB>
bool run_simt_grouped_gemm(const Tensor& offs_dev, const void* a_ptr,
                           const void* b_ptr, void* d_ptr,
                           const int64_t* ends_host, int64_t groups,
                           int64_t m_total, int64_t k_dim, int64_t n_dim,
                           bool b_col, int sm_count, cudaStream_t stream) {
    if (b_col) {
        return run_simt_layout<cutlass::layout::ColumnMajor>(
            offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
            n_dim, sm_count, stream);
    }
    return run_simt_layout<cutlass::layout::RowMajor>(
        offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
        n_dim, sm_count, stream);
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
    const bool is_f32 = dt == DType::Float32;
    const bool f32_tensor_op =
        is_f32 && tensorplay::globalContext().allowTF32CuBLAS();
    if (dt != DType::Float16 && dt != DType::BFloat16 && !is_f32)
        return false;
    // Mainloop access granularity: the vectorized tensor-op routes need
    // 8- or 4-element row alignment; the scalar fp32 route reads
    // element-wise.
    const int64_t align = f32_tensor_op ? 4 : (is_f32 ? 1 : 8);
    if ((k_dim % align) != 0 || (n_dim % align) != 0) return false;
    if (self.size(0) > INT32_MAX || k_dim > INT32_MAX || n_dim > INT32_MAX)
        return false;

    // B reads each expert as a [K, N] matrix.  A dense [E, K, N] stack is
    // row-major; a stack of [N, K] row-major weights viewed through its
    // transpose is column-major with zero-copy consumption.  Any other
    // stride pattern pays one materializing copy.
    const int64_t batch_span = k_dim * n_dim;
    bool b_col = false;
    Tensor b = mat2;
    const bool dense_stack = mat2.size(0) == 1 || mat2.stride(0) == batch_span;
    if (dense_stack && mat2.stride(2) == 1 && mat2.stride(1) == n_dim) {
        b_col = false;
    } else if (dense_stack && mat2.stride(1) == 1 && mat2.stride(2) == k_dim) {
        b_col = true;
    } else {
        b = mat2.contiguous();
        b_col = false;
    }
    Tensor a = self.is_contiguous() ? self : self.contiguous();
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
    const int64_t m_total = self.size(0);

    if (ends_host == nullptr) {
        // Unvalidated offsets: zero the never-covered tail on the device so
        // no host read of the offsets is needed anywhere on this path.
        if (dt == DType::Float16) {
            launch_grouped_tail_zero<cutlass::half_t>(
                d_ptr, offs_dev, groups, m_total, n_dim, stream);
        } else if (dt == DType::BFloat16) {
            launch_grouped_tail_zero<cutlass::bfloat16_t>(
                d_ptr, offs_dev, groups, m_total, n_dim, stream);
        } else {
            launch_grouped_tail_zero<float>(
                d_ptr, offs_dev, groups, m_total, n_dim, stream);
        }
    }

    if (dt == DType::Float16) {
        return run_half_grouped_gemm<cutlass::half_t>(
            offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
            n_dim, b_col, max_group_rows, sm_count, smem_optin, stream);
    }
    if (dt == DType::BFloat16) {
        return run_half_grouped_gemm<cutlass::bfloat16_t>(
            offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
            n_dim, b_col, max_group_rows, sm_count, smem_optin, stream);
    }
    if (is_f32 && !f32_tensor_op) {
        return run_simt_grouped_gemm<cutlass::layout::RowMajor>(
            offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
            n_dim, b_col, sm_count, stream);
    }
    if (b_col) {
        return run_grouped_config<
            TF32GroupedGemm<cutlass::layout::ColumnMajor>::Gemm, float>(
            offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
            n_dim, /*b_col=*/true, 128, 64, sm_count, stream);
    }
    return run_grouped_config<TF32GroupedGemm<cutlass::layout::RowMajor>::Gemm,
                              float>(
        offs_dev, a_ptr, b_ptr, d_ptr, ends_host, groups, m_total, k_dim,
        n_dim, /*b_col=*/false, 128, 64, sm_count, stream);
}

#else

namespace tensorplay {
namespace cuda {

// No vendored CUTLASS tree: grouped_mm stays on its per-group cuBLAS
// fallback, so every dispatch is declined here.
bool try_grouped_gemm_tensor_op(const Tensor&, const Tensor&,
                                const Tensor&, Tensor&, const int64_t*,
                                int64_t) {
    return false;
}

}  // namespace cuda
}  // namespace tensorplay

#endif  // TP_GROUPED_GEMM_TENSOR_OP

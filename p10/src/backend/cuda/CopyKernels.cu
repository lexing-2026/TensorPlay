#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "SparseKernels.h"
#include "Exception.h"
#include "TensorIterator.h"
#include "CUDALoops.cuh"
#include "Complex.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kTransTile = 32;
constexpr int kTransBlockRows = 8;

template <typename T>
__global__ void transpose_tiled_kernel(
    T* __restrict__ dst, const T* __restrict__ src,
    int64_t rows, int64_t cols) {
    __shared__ T tile[kTransTile][kTransTile + 1];

    const int64_t r_base = static_cast<int64_t>(blockIdx.x) * kTransTile;
    const int64_t c_base = static_cast<int64_t>(blockIdx.y) * kTransTile;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Read phase: one warp per column stripe, lanes stride along the view's
    // contiguous axis (rows) at unit stride.
    const int64_t r = r_base + tx;
    const int64_t c = c_base + ty;
#pragma unroll
    for (int j = 0; j < kTransTile; j += kTransBlockRows) {
        if (r < rows && c + j < cols)
            tile[ty + j][tx] = src[r + (c + j) * rows];
    }
    __syncthreads();

    // Write phase: transpose the lane mapping (tx now indexes the column
    // axis) so dst writes also walk unit stride along dst rows.
    const int64_t r2 = r_base + ty;
    const int64_t c2 = c_base + tx;
#pragma unroll
    for (int j = 0; j < kTransTile; j += kTransBlockRows) {
        if (r2 + j < rows && c2 < cols)
            dst[(r2 + j) * cols + c2] = tile[tx][ty + j];
    }
}

// Vectorized rectangular transpose: a 64x64 tile staged by 256 threads, each
// moving one 16-byte packet per stripe on both the read and the write pass.
// Every global transaction therefore spans 512 contiguous bytes instead of
// the 128 bytes a lane-per-element walk produces, which is what separates
// this schedule from the scalar tile above on narrow-memory parts.
constexpr int kTransVecTile = 64;

__global__ void transpose_tiled_vec4_kernel(
    float* __restrict__ dst, const float* __restrict__ src,
    int64_t rows, int64_t cols) {
    __shared__ float tile[kTransVecTile][kTransVecTile + 1];

    const int64_t r_base = static_cast<int64_t>(blockIdx.x) * kTransVecTile;
    const int64_t c_base = static_cast<int64_t>(blockIdx.y) * kTransVecTile;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Full 4x4 sub-tiles take the packet path; sub-tiles that straddle the
    // matrix edge fall back to guarded scalar moves so no element is
    // skipped.  Packet starts stay 16-byte aligned whenever both extents are
    // multiples of four (gated by the host launcher).  The write phase swaps
    // the axis roles of the lane indices, so it carries its own bounds flags.
    const bool full = (r_base + tx * 4 + 4 <= rows) && (c_base + ty * 4 + 4 <= cols);
    const bool any = (r_base + tx * 4 < rows) && (c_base + ty * 4 < cols);

    float v[4][4];
    if (full) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 pack = *reinterpret_cast<const float4*>(
                src + (r_base + tx * 4) + (c_base + ty * 4 + i) * rows);
            v[i][0] = pack.x;
            v[i][1] = pack.y;
            v[i][2] = pack.z;
            v[i][3] = pack.w;
        }
    } else if (any) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int64_t c = c_base + ty * 4 + i;
                const int64_t r = r_base + tx * 4 + j;
                v[i][j] = (r < rows && c < cols) ? src[r + c * rows] : 0.0f;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            tile[ty * 4 + i][tx * 4 + j] = v[i][j];
    }
    __syncthreads();

    // Write phase: the lane mapping swaps axes, so each lane stores one
    // 4-wide packet of one destination row at unit stride along cols.
    const bool full_w = (r_base + tx * 4 + 4 <= rows) && (c_base + ty * 4 + 4 <= cols);
    const bool any_w = (r_base + tx * 4 < rows) && (c_base + ty * 4 < cols);
    if (full_w) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float4 out;
            out.x = tile[ty * 4 + 0][tx * 4 + j];
            out.y = tile[ty * 4 + 1][tx * 4 + j];
            out.z = tile[ty * 4 + 2][tx * 4 + j];
            out.w = tile[ty * 4 + 3][tx * 4 + j];
            *reinterpret_cast<float4*>(
                dst + (r_base + tx * 4 + j) * cols + (c_base + ty * 4)) = out;
        }
    } else if (any_w) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int64_t r = r_base + tx * 4 + j;
                const int64_t col = c_base + ty * 4 + c;
                if (r < rows && col < cols)
                    dst[r * cols + col] = tile[ty * 4 + c][tx * 4 + j];
            }
        }
    }
}

// Vectorized two-byte transpose: same 64x64 schedule as the float packet
// tile, with each thread moving a 4x4 element sub-tile in 8-byte packets
// (four 16-bit values), so a warp stripe spans 128 contiguous bytes on both
// the read and the write pass instead of the 64 bytes the scalar tile
// produces.  One 16-bit slot of padding per shared row separates consecutive
// packet rows onto distinct banks.
constexpr int kTransHalfTile = 64;

__global__ void transpose_tiled_vec2_halfpack_kernel(
    uint16_t* __restrict__ dst, const uint16_t* __restrict__ src,
    int64_t rows, int64_t cols) {
    __shared__ uint16_t tile[kTransHalfTile][kTransHalfTile + 2];

    const int64_t r_base = static_cast<int64_t>(blockIdx.x) * kTransHalfTile;
    const int64_t c_base = static_cast<int64_t>(blockIdx.y) * kTransHalfTile;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Full 4x4 sub-tiles take the packet path; sub-tiles that straddle the
    // matrix edge fall back to guarded scalar moves so no element is
    // skipped.  Packet starts stay 8-byte aligned whenever both extents are
    // multiples of four (gated by the host launcher).
    const bool full = (r_base + tx * 4 + 4 <= rows) && (c_base + ty * 4 + 4 <= cols);
    const bool any = (r_base + tx * 4 < rows) && (c_base + ty * 4 < cols);

    uint16_t v[4][4];
    if (full) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint2 pack = *reinterpret_cast<const uint2*>(
                src + (r_base + tx * 4) + (c_base + ty * 4 + i) * rows);
            v[i][0] = static_cast<uint16_t>(pack.x & 0xffffu);
            v[i][1] = static_cast<uint16_t>(pack.x >> 16);
            v[i][2] = static_cast<uint16_t>(pack.y & 0xffffu);
            v[i][3] = static_cast<uint16_t>(pack.y >> 16);
        }
    } else if (any) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int64_t c = c_base + ty * 4 + i;
                const int64_t r = r_base + tx * 4 + j;
                v[i][j] = (r < rows && c < cols) ? src[r + c * rows] : 0;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            tile[ty * 4 + i][tx * 4 + j] = v[i][j];
    }
    __syncthreads();

    // Write phase: the lane mapping swaps axes, so each lane stores one
    // 4-wide packet of one destination row at unit stride along cols.
    const bool full_w = (r_base + tx * 4 + 4 <= rows) && (c_base + ty * 4 + 4 <= cols);
    const bool any_w = (r_base + tx * 4 < rows) && (c_base + ty * 4 < cols);
    if (full_w) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint2 out;
            out.x = static_cast<uint32_t>(tile[ty * 4 + 0][tx * 4 + j]) |
                    (static_cast<uint32_t>(tile[ty * 4 + 1][tx * 4 + j]) << 16);
            out.y = static_cast<uint32_t>(tile[ty * 4 + 2][tx * 4 + j]) |
                    (static_cast<uint32_t>(tile[ty * 4 + 3][tx * 4 + j]) << 16);
            *reinterpret_cast<uint2*>(
                dst + (r_base + tx * 4 + j) * cols + (c_base + ty * 4)) = out;
        }
    } else if (any_w) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int64_t r = r_base + tx * 4 + j;
                const int64_t col = c_base + ty * 4 + c;
                if (r < rows && col < cols)
                    dst[r * cols + col] = tile[ty * 4 + c][tx * 4 + j];
            }
        }
    }
}

// Batched float transpose: batch independent [rows, cols] matrices, each the
// transpose view of a row-major [cols, rows] plane with contiguous batches.
// The 64x64 packet schedule is identical to the 2-D kernel; blockIdx.z picks
// the plane, so the per-plane work stays coalesced while the batch dimension
// adds grid-level parallelism for free.
__global__ void transpose_tiled_vec4_batched_kernel(
    float* __restrict__ dst, const float* __restrict__ src,
    int64_t rows, int64_t cols, int64_t batch) {
    __shared__ float tile[kTransVecTile][kTransVecTile + 1];

    const int64_t plane = blockIdx.z;
    const int64_t r_base = static_cast<int64_t>(blockIdx.x) * kTransVecTile;
    const int64_t c_base = static_cast<int64_t>(blockIdx.y) * kTransVecTile;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Per-plane source base: plane is contiguous (rows * cols elements);
    // dst planes come from the contiguous [batch, cols, rows] result.
    const float* src_p = src + plane * rows * cols;
    float* dst_p = dst + plane * rows * cols;

    const bool full = (r_base + tx * 4 + 4 <= rows) && (c_base + ty * 4 + 4 <= cols);
    const bool any = (r_base + tx * 4 < rows) && (c_base + ty * 4 < cols);

    float v[4][4];
    if (full) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 pack = *reinterpret_cast<const float4*>(
                src_p + (r_base + tx * 4) + (c_base + ty * 4 + i) * rows);
            v[i][0] = pack.x;
            v[i][1] = pack.y;
            v[i][2] = pack.z;
            v[i][3] = pack.w;
        }
    } else if (any) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int64_t c = c_base + ty * 4 + i;
                const int64_t r = r_base + tx * 4 + j;
                v[i][j] = (r < rows && c < cols) ? src_p[r + c * rows] : 0.0f;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            tile[ty * 4 + i][tx * 4 + j] = v[i][j];
    }
    __syncthreads();

    const bool full_w = (r_base + tx * 4 + 4 <= rows) && (c_base + ty * 4 + 4 <= cols);
    const bool any_w = (r_base + tx * 4 < rows) && (c_base + ty * 4 < cols);
    if (full_w) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float4 out;
            out.x = tile[ty * 4 + 0][tx * 4 + j];
            out.y = tile[ty * 4 + 1][tx * 4 + j];
            out.z = tile[ty * 4 + 2][tx * 4 + j];
            out.w = tile[ty * 4 + 3][tx * 4 + j];
            *reinterpret_cast<float4*>(
                dst_p + (r_base + tx * 4 + j) * cols + (c_base + ty * 4)) = out;
        }
    } else if (any_w) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int64_t r = r_base + tx * 4 + j;
                const int64_t col = c_base + ty * 4 + c;
                if (r < rows && col < cols)
                    dst_p[r * cols + col] = tile[ty * 4 + c][tx * 4 + j];
            }
        }
    }
}

// dst is a [rows, cols] column-major view of a [cols, rows] row-major
// source: the two storages agree element for element, so the copy
// degenerates to memcpy.
bool transpose_layout_is_identity(const Tensor& self, const Tensor& src) {
    return self.dim() == 2 && src.dim() == 2 &&
           self.size(0) == src.size(1) && self.size(1) == src.size(0) &&
           self.stride(0) == 1 && self.stride(1) == self.size(0);
}

// src is the [rows, cols] transpose view of row-major [cols, rows] storage:
// unit stride over rows, the destination row count over cols.
bool transpose_layout_is_tiled_copy(const Tensor& self, const Tensor& src) {
    if (self.dim() != 2 || src.dim() != 2 || self.dtype() != src.dtype())
        return false;
    if (!self.is_contiguous()) return false;
    if (self.size(0) != src.size(0) || self.size(1) != src.size(1))
        return false;
    if (src.stride(0) != 1 || src.stride(1) != self.size(0)) return false;
    const int64_t total = self.numel();
    if (total < kTransTile * kTransTile * 4) return false;
    switch (src.itemsize()) {
        case 1: case 2: case 4: case 8: return true;
        default: return false;
    }
}

// Same layout contract as the scalar tile, restricted to the float32 case
// with both extents multiple of four (so every 4-float packet starts
// 16-byte aligned) and both base pointers 16-byte aligned.
bool transpose_layout_is_tiled_copy_vec(const Tensor& self, const Tensor& src) {
    if (self.dim() != 2 || src.dim() != 2 ||
        self.dtype() != DType::Float32)
        return false;
    if (!self.is_contiguous()) return false;
    if (self.size(0) != src.size(0) || self.size(1) != src.size(1))
        return false;
    if (src.stride(0) != 1 || src.stride(1) != self.size(0)) return false;
    const int64_t rows = self.size(0);
    const int64_t cols = self.size(1);
    if (rows < 256 || cols < 256) return false;
    if (rows % 4 != 0 || cols % 4 != 0) return false;
    if ((reinterpret_cast<uintptr_t>(self.data_ptr()) |
         reinterpret_cast<uintptr_t>(src.data_ptr())) & 15u)
        return false;
    return true;
}

// Same layout contract as the float packet tile, for the two-byte element
// types (fp16 / bf16 storages share the 16-bit bit pattern): 8-byte packet
// alignment requires both extents to be multiples of four and both base
// pointers 8-byte aligned.
bool transpose_layout_is_tiled_copy_vec2_half(const Tensor& self,
                                              const Tensor& src) {
    if (self.dim() != 2 || src.dim() != 2 || self.dtype() != src.dtype())
        return false;
    if (src.itemsize() != 2) return false;
    if (!self.is_contiguous()) return false;
    if (self.size(0) != src.size(0) || self.size(1) != src.size(1))
        return false;
    if (src.stride(0) != 1 || src.stride(1) != self.size(0)) return false;
    const int64_t rows = self.size(0);
    const int64_t cols = self.size(1);
    if (rows < 256 || cols < 256) return false;
    if (rows % 4 != 0 || cols % 4 != 0) return false;
    if ((reinterpret_cast<uintptr_t>(self.data_ptr()) |
         reinterpret_cast<uintptr_t>(src.data_ptr())) & 7u)
        return false;
    return true;
}

// Batched plane transposes: dst is a contiguous [batch, rows, cols] float32
// tensor; src is a [batch, rows, cols] view whose planes are each the
// transpose of a row-major [cols, rows] matrix with unit batch stride.
// This is exactly the permute(0, 2, 1) view of a contiguous batch, the most
// common rank-3 layout change.
bool transpose_layout_is_tiled_copy_vec3(const Tensor& self, const Tensor& src) {
    if (self.dim() != 3 || src.dim() != 3 ||
        self.dtype() != DType::Float32)
        return false;
    if (!self.is_contiguous()) return false;
    if (self.size(0) != src.size(0) || self.size(1) != src.size(1) ||
        self.size(2) != src.size(2))
        return false;
    const int64_t batch = self.size(0);
    const int64_t rows = self.size(1);
    const int64_t cols = self.size(2);
    if (batch == 0 || rows < 256 || cols < 256) return false;
    if (rows % 4 != 0 || cols % 4 != 0) return false;
    if (src.stride(0) != rows * cols) return false;
    if (src.stride(1) != 1 || src.stride(2) != rows) return false;
    if ((reinterpret_cast<uintptr_t>(self.data_ptr()) |
         reinterpret_cast<uintptr_t>(src.data_ptr())) & 15u)
        return false;
    return true;
}

}  // namespace

Tensor& copy_kernel(Tensor& self, const Tensor& src, bool non_blocking) {
    if (self.numel() != src.numel()) {
        TP_THROW(RuntimeError, "Sizes do not match for copy");
    }
    
    Device dst_dev = self.device();
    Device src_dev = src.device();
    
    if (!dst_dev.is_cuda()) {
         TP_THROW(RuntimeError, "copy_kernel dispatched to CUDA but dst is CPU?");
    }

    bool src_cuda = src_dev.is_cuda();
    auto stream = getCurrentCUDAStream(static_cast<int>(dst_dev.index()));

    // Optimize: Contiguous copy (both src and dst must be contiguous AND same dtype)
    if (self.dtype() == src.dtype() && self.is_contiguous() && src.is_contiguous()) {
        size_t nbytes = self.numel() * self.itemsize();
        if (src_cuda && src_dev.index() != dst_dev.index()) {
            // Establish both directions of the lifetime/order relationship:
            // destination work waits for prior source work, and the source
            // stream waits for the copy before its allocator may recycle src.
            auto src_stream = getCurrentCUDAStream(static_cast<int>(src_dev.index()));
            CUDAEvent source_ready;
            source_ready.record(src_stream);
            source_ready.block(stream);
            checkCuda(cudaMemcpyPeerAsync(
                          self.data_ptr(), static_cast<int>(dst_dev.index()),
                          src.data_ptr(), static_cast<int>(src_dev.index()),
                          nbytes, stream.stream()),
                      "cudaMemcpyPeerAsync");
            CUDAEvent copy_complete;
            copy_complete.record(stream);
            copy_complete.block(src_stream);
        } else {
            const cudaMemcpyKind kind = src_cuda
                ? cudaMemcpyDeviceToDevice
                : cudaMemcpyHostToDevice;
            checkCuda(cudaMemcpyAsync(self.data_ptr(), src.data_ptr(), nbytes, kind, stream.stream()),
                      "cudaMemcpyAsync");
            if (!src_cuda) {
                if (non_blocking && src.is_pinned()) {
                    recordPinnedStream(
                        const_cast<void*>(src.unsafeGetTensorImpl()->storage().data()), stream);
                } else {
                    // Pageable host storage cannot safely outlive the call and
                    // cudaMemcpyAsync may stage it synchronously anyway.
                    stream.synchronize();
                }
            }
        }
        return self;
    }

    // Transposed 2-D views: a column-major destination over a row-major
    // source (or the reverse) is either an identity re-labelling or a full
    // tile transpose; both beat the generic per-element strided walker.
    if (src_cuda && self.dtype() == src.dtype() &&
        transpose_layout_is_identity(self, src)) {
        const size_t nbytes = self.numel() * self.itemsize();
        checkCuda(cudaMemcpyAsync(self.data_ptr(), src.data_ptr(), nbytes,
                                  cudaMemcpyDeviceToDevice, stream.stream()),
                  "cudaMemcpyAsync (transposed identity)");
        return self;
    }
    if (src_cuda && transpose_layout_is_tiled_copy_vec(self, src)) {
        const int64_t rows = self.size(0);
        const int64_t cols = self.size(1);
        dim3 block(kTransVecTile / 4, kTransVecTile / 4);
        dim3 grid(static_cast<unsigned>((rows + kTransVecTile - 1) / kTransVecTile),
                  static_cast<unsigned>((cols + kTransVecTile - 1) / kTransVecTile));
        transpose_tiled_vec4_kernel<<<grid, block, 0, stream.stream()>>>(
            static_cast<float*>(self.data_ptr()),
            static_cast<const float*>(src.data_ptr()), rows, cols);
        checkCuda(cudaGetLastError(), "CUDA vectorized tiled transpose copy");
        return self;
    }
    if (src_cuda && transpose_layout_is_tiled_copy_vec3(self, src)) {
        const int64_t batch = self.size(0);
        const int64_t rows = self.size(1);
        const int64_t cols = self.size(2);
        dim3 block(kTransVecTile / 4, kTransVecTile / 4, 1);
        dim3 grid(static_cast<unsigned>((rows + kTransVecTile - 1) / kTransVecTile),
                  static_cast<unsigned>((cols + kTransVecTile - 1) / kTransVecTile),
                  static_cast<unsigned>(batch));
        transpose_tiled_vec4_batched_kernel<<<grid, block, 0, stream.stream()>>>(
            static_cast<float*>(self.data_ptr()),
            static_cast<const float*>(src.data_ptr()), rows, cols, batch);
        checkCuda(cudaGetLastError(), "CUDA batched vectorized tiled transpose copy");
        return self;
    }
    if (src_cuda && transpose_layout_is_tiled_copy_vec2_half(self, src)) {
        const int64_t rows = self.size(0);
        const int64_t cols = self.size(1);
        dim3 block(kTransHalfTile / 4, kTransHalfTile / 4);
        dim3 grid(static_cast<unsigned>((rows + kTransHalfTile - 1) / kTransHalfTile),
                  static_cast<unsigned>((cols + kTransHalfTile - 1) / kTransHalfTile));
        transpose_tiled_vec2_halfpack_kernel<<<grid, block, 0, stream.stream()>>>(
            static_cast<uint16_t*>(self.data_ptr()),
            static_cast<const uint16_t*>(src.data_ptr()), rows, cols);
        checkCuda(cudaGetLastError(), "CUDA vectorized two-byte tiled transpose copy");
        return self;
    }
    if (src_cuda && transpose_layout_is_tiled_copy(self, src)) {
        const int64_t rows = self.size(0);
        const int64_t cols = self.size(1);
        dim3 block(kTransTile, kTransBlockRows);
        dim3 grid(static_cast<unsigned>((rows + kTransTile - 1) / kTransTile),
                  static_cast<unsigned>((cols + kTransTile - 1) / kTransTile));
        switch (src.itemsize()) {
            case 1:
                transpose_tiled_kernel<uint8_t><<<grid, block, 0, stream.stream()>>>(
                    static_cast<uint8_t*>(self.data_ptr()),
                    static_cast<const uint8_t*>(src.data_ptr()), rows, cols);
                break;
            case 2:
                transpose_tiled_kernel<uint16_t><<<grid, block, 0, stream.stream()>>>(
                    static_cast<uint16_t*>(self.data_ptr()),
                    static_cast<const uint16_t*>(src.data_ptr()), rows, cols);
                break;
            case 4:
                transpose_tiled_kernel<uint32_t><<<grid, block, 0, stream.stream()>>>(
                    static_cast<uint32_t*>(self.data_ptr()),
                    static_cast<const uint32_t*>(src.data_ptr()), rows, cols);
                break;
            case 8:
                transpose_tiled_kernel<uint64_t><<<grid, block, 0, stream.stream()>>>(
                    static_cast<uint64_t*>(self.data_ptr()),
                    static_cast<const uint64_t*>(src.data_ptr()), rows, cols);
                break;
            default:
                break;
        }
        checkCuda(cudaGetLastError(), "CUDA tiled transpose copy");
        return self;
    }

    // Strided copy or Casting copy
    // If src is CPU, we must move it to CUDA first (to a contiguous buffer)
    Tensor src_cuda_tensor = src;
    if (!src_cuda) {
        // Create a contiguous CUDA tensor
        // Note: we can't easily use "empty" then copy because we might recurse.
        // We manually allocate and copy from host.
        
        // 1. Ensure src is contiguous on host
        Tensor src_contig = src.is_contiguous() ? src : src.contiguous();
        
        // 2. Allocate temp CUDA memory
        src_cuda_tensor = Tensor(static_cast<std::vector<int64_t>>(src.shape()), src.dtype(), self.device());
        
        // 3. Copy H2D (contiguous)
        checkCuda(cudaMemcpyAsync(
                      src_cuda_tensor.data_ptr(), src_contig.data_ptr(),
                      src_contig.numel() * src_contig.itemsize(),
                      cudaMemcpyHostToDevice, stream.stream()),
                  "cudaMemcpyAsync (strided H2D staging)");
        if (non_blocking && src_contig.is_pinned()) {
            recordPinnedStream(
                const_cast<void*>(src_contig.unsafeGetTensorImpl()->storage().data()), stream);
        } else {
            stream.synchronize();
        }
    }
    
    // Now src_cuda_tensor is on CUDA. 
    // self is on CUDA.
    
    int64_t numel = self.numel();
    if (numel == 0) return self;

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    // Same-dtype copies run through the iterator elementwise lane: the
    // iterator coalesces and reorders dimensions, picks the vectorized or
    // unrolled schedule, and splits 64-bit indexing, so strided layouts move
    // 4-8 elements per thread instead of one.  Complex destinations are
    // served here too — the iterator's dynamic-cast machinery carries
    // complex values through the same identity functor.
    if (self.dtype() == src_cuda_tensor.dtype()) {
        TensorIterator iter = TensorIteratorConfig()
                                  .check_all_same_dtype(true)
                                  .add_output(self)
                                  .add_input(src_cuda_tensor)
                                  .build();
        switch (self.dtype()) {
            case DType::ComplexHalf:
                gpu_kernel(iter, [] __host__ __device__(
                                     tensorplay::complex<Half> v) { return v; });
                break;
            case DType::ComplexFloat:
                gpu_kernel(iter, [] __host__ __device__(
                                     tensorplay::complex<float> v) { return v; });
                break;
            case DType::ComplexDouble:
                gpu_kernel(iter, [] __host__ __device__(
                                     tensorplay::complex<double> v) { return v; });
                break;
            case DType::BComplex32:
                gpu_kernel(iter, [] __host__ __device__(
                                     tensorplay::complex<BFloat16> v) { return v; });
                break;
#define TP_COPY_ITER_CASE(ctype, name)                                   \
            case DType::name:                                                \
                gpu_kernel(iter, [] __host__ __device__(ctype v) { return v; });      \
                break;
            TENSORPLAY_FORALL_SCALAR_TYPES(TP_COPY_ITER_CASE)
#undef TP_COPY_ITER_CASE
            default:
                TP_THROW(NotImplementedError, "Unsupported dtype for copy");
        }
        checkCuda(cudaGetLastError(), "CUDA iterator copy kernel");
        return self;
    }

    // Cross-dtype casts follow the unified elementwise lane as well: one
    // identity functor per destination dtype, with the source operand
    // converted on load through the dynamic-cast machinery.  This carries
    // every real/complex combination, including the reduced complex widths,
    // without per-pair kernels.
    TensorIterator iter = TensorIteratorConfig()
                              .check_all_same_dtype(false)
                              .add_output(self)
                              .add_input(src_cuda_tensor)
                              .build();
    switch (self.dtype()) {
#define TP_COPY_CAST_CASE(ctype, name)                                     \
        case DType::name:                                                  \
            gpu_kernel(iter, [] __host__ __device__(ctype v) { return v; });        \
            break;
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_COPY_CAST_CASE)
#undef TP_COPY_CAST_CASE
        default:
            TP_THROW(NotImplementedError, "Unsupported dst dtype for casting");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
         TP_THROW(RuntimeError, std::string("CUDA Copy Kernel Error: ") + cudaGetErrorString(err));
    }

    return self;
}

// Extract the single element of a 1-element device tensor.  The value is
// staged through a synchronous device-to-host copy so the result reflects all
// work queued on the current stream.
Scalar item_cuda(const Tensor& self) {
    if (!self.defined()) {
        TP_THROW(RuntimeError, "Tensor not defined");
    }
    std::shared_ptr<TensorImpl> impl = self.unsafeGetTensorImpl();
    if (impl->is_sparse()) {
        TP_THROW(RuntimeError, "item() is not supported for sparse tensors");
    }
    if (impl->numel() != 1) {
        TP_THROW(ValueError, "item() only supported for 1-element tensors");
    }
    if (!impl->device().is_cuda()) {
        TP_THROW(RuntimeError, "item(): expected a CUDA tensor but got ",
                 impl->device().toString());
    }

    const void* src = self.data_ptr();
    // A 1-element tensor addresses its only element directly; strides are
    // irrelevant and data_ptr() already includes the storage offset.
    switch (impl->dtype()) {
        case DType::Float32: {
            float v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<double>(v));
        }
        case DType::Float64: {
            double v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(v);
        }
        case DType::Float16: {
            Half v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<float>(v));
        }
        case DType::BFloat16: {
            BFloat16 v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<float>(v));
        }
        case DType::Float8_e4m3fn: {
            Float8_e4m3fn v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<float>(v));
        }
        case DType::Float8_e5m2: {
            Float8_e5m2 v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<float>(v));
        }
        case DType::Float8_e4m3fnuz: {
            Float8_e4m3fnuz v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<float>(v));
        }
        case DType::Float8_e5m2fnuz: {
            Float8_e5m2fnuz v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<float>(v));
        }
        case DType::Float8_e8m0fnu: {
            Float8_e8m0fnu v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<float>(v));
        }
        case DType::Int8: {
            int8_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<int64_t>(v));
        }
        case DType::Int16: {
            int16_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<int64_t>(v));
        }
        case DType::Int32: {
            int32_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<int64_t>(v));
        }
        case DType::Int64: {
            int64_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(v);
        }
        case DType::UInt8: {
            uint8_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<uint64_t>(v));
        }
        case DType::UInt16: {
            uint16_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<uint64_t>(v));
        }
        case DType::UInt32: {
            uint32_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(static_cast<uint64_t>(v));
        }
        case DType::UInt64: {
            uint64_t v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(v);
        }
        case DType::Bool: {
            bool v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(v);
        }
        case DType::ComplexHalf: {
            tensorplay::complex<Half> v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(tensorplay::complex<float>(static_cast<float>(v.real()),
                                                     static_cast<float>(v.imag())));
        }
        case DType::ComplexFloat: {
            tensorplay::complex<float> v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(tensorplay::complex<float>(v.real(), v.imag()));
        }
        case DType::ComplexDouble: {
            tensorplay::complex<double> v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(tensorplay::complex<double>(v.real(), v.imag()));
        }
        case DType::BComplex32: {
            tensorplay::complex<BFloat16> v; checkCuda(cudaMemcpy(&v, src, sizeof(v), cudaMemcpyDeviceToHost), "item D2H");
            return Scalar(tensorplay::complex<float>(static_cast<float>(v.real()),
                                                     static_cast<float>(v.imag())));
        }
        default:
            TP_THROW(NotImplementedError, "item() not implemented for this dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CUDA, CopyKernels) {
    m.impl("copy_", copy_kernel);
    m.impl("item", item_cuda);
    m.impl("sparse_coo_tensor", sparse_coo_tensor_cuda);
    m.impl("sparse_mask", sparse_mask_cuda);
    m.impl("to_dense", to_dense_sparse_cuda);
    m.impl("to_sparse", to_sparse_coo_cuda);
    m.impl("to_sparse_csr", to_sparse_csr_cuda);
    m.impl("_nnz", sparse_nnz_cuda);
    m.impl("sparse_mm", sparse_mm_cuda);
    m.impl("sparse_sum", sparse_sum_cuda);
    m.impl("sparse_add", sparse_add_cuda);
    m.impl("sparse_mul", sparse_mul_cuda);
    m.impl("spdiags", spdiags_cuda);
}

} // namespace cuda
} // namespace tensorplay

#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Utils.h"
#include "CudaGemm.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kGemvThreads = 256;

struct alignas(16) uint4_packet { uint32_t w[4]; };

// Wave width is probed once on the host and passed to kernels as an
// argument: the device pass cannot call the host-only attribute API.
inline int gemv_wave_size() {
    static int wave = []() {
        int dev = 0, lanes = 32;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&lanes, cudaDevAttrWarpSize, dev);
        return lanes > 0 ? lanes : 32;
    }();
    return wave;
}

template <typename acc_t, int HW_WAVE>
__device__ __forceinline__ acc_t gemv_block_sum(acc_t value, acc_t* scratch) {
    const int lane = static_cast<int>(threadIdx.x) & (HW_WAVE - 1);
    const int wid = static_cast<int>(threadIdx.x) / HW_WAVE;
    constexpr unsigned long long kMask =
        HW_WAVE == 64 ? 0xffffffffffffffffull : 0xffffffffull;
#pragma unroll
    for (int offset = HW_WAVE / 2; offset > 0; offset /= 2) {
        const acc_t other = __shfl_xor_sync(kMask, value, offset, HW_WAVE);
        value += other;
    }
    if (lane == 0) scratch[wid] = value;
    __syncthreads();
    constexpr int kNumWaves = kGemvThreads / HW_WAVE;
    value = (lane < kNumWaves) ? scratch[lane] : acc_t(0);
    if (wid == 0) {
#pragma unroll
        for (int offset = HW_WAVE / 2; offset > 0; offset /= 2) {
            const acc_t other = __shfl_xor_sync(kMask, value, offset, HW_WAVE);
            value += other;
        }
        if (lane == 0) scratch[0] = value;
    }
    __syncthreads();
    return scratch[0];
}

#define TP_GEMV_BLOCK_SUM(acc, scratch)                                 \
    (wave_size == 64 ? gemv_block_sum<float, 64>(acc, scratch)          \
                     : gemv_block_sum<float, 32>(acc, scratch))

__device__ __forceinline__ void unpack_i8x4(uint32_t q, float out[4]) {
    out[0] = static_cast<float>(static_cast<int8_t>(q & 0xffu));
    out[1] = static_cast<float>(static_cast<int8_t>((q >> 8) & 0xffu));
    out[2] = static_cast<float>(static_cast<int8_t>((q >> 16) & 0xffu));
    out[3] = static_cast<float>(static_cast<int8_t>((q >> 24) & 0xffu));
}

__device__ __forceinline__ void unpack_u4x8(uint32_t q, uint8_t out[8]) {
    out[0] = static_cast<uint8_t>(q & 0x0fu);
    out[1] = static_cast<uint8_t>((q >> 4) & 0x0fu);
    out[2] = static_cast<uint8_t>((q >> 8) & 0x0fu);
    out[3] = static_cast<uint8_t>((q >> 12) & 0x0fu);
    out[4] = static_cast<uint8_t>((q >> 16) & 0x0fu);
    out[5] = static_cast<uint8_t>((q >> 20) & 0x0fu);
    out[6] = static_cast<uint8_t>((q >> 24) & 0x0fu);
    out[7] = static_cast<uint8_t>((q >> 28) & 0x0fu);
}

// ---------------------------------------------------------------------------
// Int8 weight pack.  out[b, n] = scale[n] * sum_k w[n, k] * x[b, k].
// One block per output row n; each lane streams one aligned uint4 (16 int8
// weights) plus the matching activation packets and accumulates in fp32.
// ---------------------------------------------------------------------------

__global__ void int8pack_gemv_kernel(
    float* __restrict__ out, const int8_t* __restrict__ w,
    const float* __restrict__ x, const float* __restrict__ scale,
    int64_t K, int wave_size) {
    __shared__ float scratch[kGemvThreads / 32];
    const int64_t n = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    // The weight stream is int8 (16 codes per 16-byte packet); the
    // activation stream is float (4 values per 16-byte packet), so one
    // weight packet lines up with four activation packets.
    const uint4_packet* w_row =
        reinterpret_cast<const uint4_packet*>(w + n * K);
    const float4* x_row = reinterpret_cast<const float4*>(x);
    const int64_t npackets = K / 16;
    const int64_t tail_start = npackets * 16;

    float acc = 0.f;
    for (int64_t p = tid; p < npackets; p += nthreads) {
        const uint4_packet wp = w_row[p];
        const int64_t k0 = p * 16;
        float wv[16];
#pragma unroll
        for (int g = 0; g < 4; ++g) unpack_i8x4(wp.w[g], &wv[g * 4]);
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            const float4 xv = x_row[k0 / 4 + f];
            acc = fmaf(wv[f * 4 + 0], xv.x, acc);
            acc = fmaf(wv[f * 4 + 1], xv.y, acc);
            acc = fmaf(wv[f * 4 + 2], xv.z, acc);
            acc = fmaf(wv[f * 4 + 3], xv.w, acc);
        }
    }
    for (int64_t k = tail_start + tid; k < K; k += nthreads) {
        acc = fmaf(static_cast<float>(w[n * K + k]), x[k], acc);
    }
    acc = TP_GEMV_BLOCK_SUM(acc, scratch);
    if (tid == 0) out[n] = acc * scale[n];
}

// Multi-activation variant: the weight packet is reused across the batch and
// each activation contributes its own fp32 accumulator per lane.
__global__ void int8pack_gemv_batched_kernel(
    float* __restrict__ out, const int8_t* __restrict__ w,
    const float* __restrict__ x, const float* __restrict__ scale,
    int64_t K, int64_t B, int wave_size) {
    __shared__ float scratch[kGemvThreads / 32];
    const int64_t n = static_cast<int64_t>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int nthreads = static_cast<int>(blockDim.x);

    const uint4_packet* w_row =
        reinterpret_cast<const uint4_packet*>(w + n * K);
    const int64_t npackets = K / 16;
    const int64_t tail_start = npackets * 16;

    float acc[8];
    for (int b = 0; b < 8; ++b) acc[b] = 0.f;

    for (int64_t p = tid; p < npackets; p += nthreads) {
        const uint4_packet wp = w_row[p];
        const int64_t k0 = p * 16;
        float wv[16];
#pragma unroll
        for (int g = 0; g < 4; ++g) unpack_i8x4(wp.w[g], &wv[g * 4]);
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            if (b >= B) break;
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                const float4 xv = reinterpret_cast<const float4*>(x + b * K)[k0 / 4 + f];
                acc[b] = fmaf(wv[f * 4 + 0], xv.x, acc[b]);
                acc[b] = fmaf(wv[f * 4 + 1], xv.y, acc[b]);
                acc[b] = fmaf(wv[f * 4 + 2], xv.z, acc[b]);
                acc[b] = fmaf(wv[f * 4 + 3], xv.w, acc[b]);
            }
        }
    }
    for (int64_t k = tail_start + tid; k < K; k += nthreads) {
        const float wv = static_cast<float>(w[n * K + k]);
        for (int b = 0; b < B; ++b) acc[b] = fmaf(wv, x[b * K + k], acc[b]);
    }

    for (int b = 0; b < B; ++b) {
        const float total = TP_GEMV_BLOCK_SUM(acc[b], scratch);
        if (tid == 0) out[b * gridDim.x + n] = total * scale[n];
    }
}

// ---------------------------------------------------------------------------
// Int4 group-quantized weight pack.  Storage: two 4-bit codes per byte
// (even k in the low nibble, odd k in the high nibble, values 0..15).  The
// de-quantized weight is (code - zero[g]) * scale[g] with one entry per
// group of qGroupSize codes.  One block per output row; each lane walks its
// uint4 packets (32 codes), keeping the active group's scale/zero pair in
// registers so a group change costs one scalar load.
// ---------------------------------------------------------------------------

// One 32-lane wave owns one output row; each lane owns whole groups
// (group_size / 32 packets) so the group's scale/zero pair stays in
// registers and no per-element group detection runs.  The row dot product
// reduces as an in-register butterfly with no shared-memory round trip.
template <int HW_WAVE>
__global__ void int4pack_gemv_rows_kernel(
    float* __restrict__ out, const uint8_t* __restrict__ w,
    const float* __restrict__ x, const float* __restrict__ scale_zero,
    int64_t K, int64_t group_size, int64_t rows) {
    constexpr unsigned long long kMask =
        HW_WAVE == 64 ? 0xffffffffffffffffull : 0xffffffffull;
    constexpr int kRowsPerBlock = kGemvThreads / HW_WAVE;

    const int wave_in_block = static_cast<int>(threadIdx.x) / HW_WAVE;
    const int lane = static_cast<int>(threadIdx.x) & (HW_WAVE - 1);
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + wave_in_block;

    const int64_t groups_per_row = K / group_size;
    const int64_t pkts_per_group = group_size / 32;

    float acc = 0.f;
    if (row < rows) {
        const uint4_packet* w_row =
            reinterpret_cast<const uint4_packet*>(w + row * (K / 2));
        for (int64_t g = lane; g < groups_per_row; g += HW_WAVE) {
            // scale_zero is (groups, rows, 2): scale/zero for group g and
            // output row `row` live at (g * rows + row) * 2.
            const float zs = scale_zero[(g * rows + row) * 2 + 0];
            const float zz = scale_zero[(g * rows + row) * 2 + 1];
            const int64_t pkt0 = g * pkts_per_group;
            float gacc = 0.f;
            for (int64_t p = 0; p < pkts_per_group; ++p) {
                const uint4_packet qp = w_row[pkt0 + p];
                const int64_t k0 = (pkt0 + p) * 32;
                uint8_t code[32];
#pragma unroll
                for (int u = 0; u < 4; ++u) unpack_u4x8(qp.w[u], &code[u * 8]);
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    // De-quantized weight: (unsigned code - 8) * scale + zero.
                    const float wval =
                        (static_cast<float>(code[i]) - 8.0f) * zs + zz;
                    gacc = fmaf(wval, x[k0 + i], gacc);
                }
            }
            acc += gacc;
        }
    }
#pragma unroll
    for (int offset = HW_WAVE / 2; offset > 0; offset /= 2) {
        const float other = __shfl_xor_sync(kMask, acc, offset, HW_WAVE);
        acc += other;
    }
    if (row < rows && lane == 0) out[row] = acc;
}

#undef TP_GEMV_BLOCK_SUM

// Scalar fallback for unaligned or tiny shapes: one thread per output.
__global__ void int8pack_scalar_kernel(
    float* __restrict__ out, const int8_t* __restrict__ w,
    const float* __restrict__ x, const float* __restrict__ scale,
    int64_t K, int64_t N, int64_t B) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= N * B) return;
    const int64_t n = e % N;
    const int64_t b = e / N;
    float acc = 0.f;
    for (int64_t k = 0; k < K; ++k) {
        acc = fmaf(static_cast<float>(w[n * K + k]), x[b * K + k], acc);
    }
    out[b * N + n] = acc * scale[n];
}

__global__ void int4pack_scalar_kernel(
    float* __restrict__ out, const uint8_t* __restrict__ w,
    const float* __restrict__ x, const float* __restrict__ scale_zero,
    int64_t K, int64_t N, int64_t B, int64_t group_size) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= N * B) return;
    const int64_t n = e % N;
    const int64_t b = e / N;
    const uint8_t* w_row = w + n * (K / 2);
    float acc = 0.f;
    for (int64_t k = 0; k < K; k += 2) {
        const uint8_t byte = w_row[k / 2];
        const int64_t g = k / group_size;
        // scale_zero is (groups, N, 2).
        const float zs = scale_zero[(g * N + n) * 2 + 0];
        const float zz = scale_zero[(g * N + n) * 2 + 1];
        // De-quantized weight: (unsigned code - 8) * scale + zero.
        const float w0 = (static_cast<float>(byte & 0x0fu) - 8.0f) * zs + zz;
        acc += w0 * x[b * K + k];
        if (k + 1 < K) {
            const int64_t g2 = (k + 1) / group_size;
            const float zs2 = scale_zero[(g2 * N + n) * 2 + 0];
            const float zz2 = scale_zero[(g2 * N + n) * 2 + 1];
            const float w1 =
                (static_cast<float>(byte >> 4) - 8.0f) * zs2 + zz2;
            acc += w1 * x[b * K + k + 1];
        }
    }
    out[b * N + n] = acc;
}

// De-quantize a packed int4 weight [N, K/2] (two 4-bit codes per byte, even
// k in the low nibble) into a dense [N, K] float tensor using per-group
// (scale, zero): value = (code - 8) * scale + zero.  scale_zero has the
// canonical (K/group, N, 2) layout shared with the CPU int4pack path.
template <typename T>
__global__ void int4_dequantize_kernel(
    const uint8_t* __restrict__ w, const float* __restrict__ scale_zero,
    T* __restrict__ out, int64_t N, int64_t K, int64_t groups_per_row,
    int64_t group_size) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= N * K) return;
    const int64_t n = e / K;
    const int64_t k = e % K;
    const uint8_t byte = w[n * (K / 2) + k / 2];
    const int code = (k & 1) ? (byte >> 4) : (byte & 0x0f);
    const int64_t g = k / group_size;
    const float scale = scale_zero[(g * N + n) * 2 + 0];
    const float zero = scale_zero[(g * N + n) * 2 + 1];
    out[e] = static_cast<T>((static_cast<float>(code) - 8.0f) * scale + zero);
}

// Pack an int32 code tensor [N, K] (values 0..15) into [N, K/2] uint8 with
// the odd-k code in the low nibble and the even-k code in the high nibble,
// matching the nibble order expected by the m16n8k16 B-layout repacking
// kernel (and the direct tensor-core GEMM that consumes it).
__global__ void int4_pack_kernel(const int32_t* __restrict__ codes,
                                 uint8_t* __restrict__ out,
                                 int64_t num_bytes) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= num_bytes) return;
    const int32_t low = codes[2 * e + 1] & 0x0f;
    const int32_t high = codes[2 * e] & 0x0f;
    out[e] = static_cast<uint8_t>(low | (high << 4));
}

bool gemv_k_aligned16(const void* a, const void* b) {
    return ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) & 15u) == 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// _weight_int8pack_mm: QInt8 weight [N, K] against fp activations [B, K] and
// a per-row fp scale [N]; the result dtype follows the activations.  `self`
// is the activation matrix (matching the op schema), `mat2` is the packed
// int8 weight; the returned product is `self @ weight.T * scales`.
// ---------------------------------------------------------------------------

Tensor _weight_int8pack_mm_cuda(const Tensor& self, const Tensor& mat2,
                                const Tensor& scales) {
    if (mat2.dtype() != DType::QInt8) {
        TP_THROW(TypeError, "_weight_int8pack_mm(): weight must be QInt8");
    }
    if (self.dim() != 2 || mat2.dim() != 2 || scales.dim() != 1) {
        TP_THROW(ValueError,
                 "_weight_int8pack_mm(): expected 2-D activations [B,K], "
                 "2-D weight [N,K] and 1-D scales [N]");
    }
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError,
                 "_weight_int8pack_mm(): activations must be floating point");
    }
    const int64_t B = self.size(0);
    const int64_t K = self.size(1);
    const int64_t N = mat2.size(0);
    if (mat2.size(1) != K) {
        TP_THROW(ValueError, "_weight_int8pack_mm(): K dimension mismatch");
    }
    if (scales.size(0) != N) {
        TP_THROW(ValueError, "_weight_int8pack_mm(): scales length mismatch");
    }
    if (N == 0 || B == 0 || K == 0) {
        return Tensor::empty({B, N}, self.dtype(), self.device());
    }

    Tensor w = mat2.contiguous();
    Tensor sc = scales.to(DType::Float32).contiguous();
    Tensor out = Tensor::empty({B, N}, DType::Float32, self.device());
    Tensor xf = self.to(DType::Float32).contiguous();
    auto stream = getCurrentCUDAStream().stream();

    const int8_t* w_ptr = static_cast<const int8_t*>(w.data_ptr());
    const float* x_ptr = static_cast<const float*>(xf.data_ptr());
    const float* s_ptr = static_cast<const float*>(sc.data_ptr());
    float* o_ptr = static_cast<float*>(out.data_ptr());

    const unsigned grid = static_cast<unsigned>(N);
    const bool aligned = gemv_k_aligned16(w_ptr, x_ptr) && (K % 16 == 0);
    const int wave = gemv_wave_size();
    if (B == 1 && aligned) {
        int8pack_gemv_kernel<<<grid, kGemvThreads, 0, stream>>>(
            o_ptr, w_ptr, x_ptr, s_ptr, K, wave);
    } else if (B <= 8 && aligned) {
        int8pack_gemv_batched_kernel<<<grid, kGemvThreads, 0, stream>>>(
            o_ptr, w_ptr, x_ptr, s_ptr, K, B, wave);
    } else if (aligned) {
        for (int64_t b = 0; b < B; ++b) {
            int8pack_gemv_kernel<<<grid, kGemvThreads, 0, stream>>>(
                o_ptr + b * N, w_ptr, x_ptr + b * K, s_ptr, K, wave);
        }
    } else {
        const int64_t total = N * B;
        int8pack_scalar_kernel<<<static_cast<unsigned>((total + 255) / 256),
                                 256, 0, stream>>>(o_ptr, w_ptr, x_ptr, s_ptr,
                                                   K, N, B);
    }
    checkCuda(cudaGetLastError(), "int8pack gemm kernel");
    return out.to(self.dtype());
}


// ---------------------------------------------------------------------------
// Direct int4 tensor-core GEMM.
//
// The packed weight is first repacked into the m16n8k16 B-fragment layout
// (matrix_to_m16n8k16_Bint4_layout): shape [n/8][k/(InnerKTiles*16)][32]
// [InnerKTiles/2], int32.  The main kernel then consumes that layout
// directly with mma.sync, loading each lane's exact fragment words straight
// from global memory and dequantizing eight int4 values per register with
// bit-twiddling + bf16 FMA.  No shared-memory staging is needed between the
// fragment loads and the MMA pipeline.
// ---------------------------------------------------------------------------

constexpr int32_t kWarpSize = 32;

// bf16 vector types used by the fragment layouts.
struct __align__(4) bf16x2x1 {
  __nv_bfloat162 vals[1];
};

struct __align__(8) bf16x2x2 {
  __nv_bfloat162 vals[2];
};

struct __align__(16) bf16x2x4 {
  __nv_bfloat162 vals[4];
};

struct __align__(16) bf16x2x4_u32 {
  uint32_t vals[4];
};

struct __align__(8) bf16x2x2_u32 {
  uint32_t vals[2];
};

struct __align__(4) bf16x2x1_u32 {
  uint32_t vals[1];
};

template <typename U, typename V>
constexpr __host__ __device__ auto divDown(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral_v<U> && std::is_integral_v<V>);
  return (a / b);
}

template <typename U, typename V>
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral_v<U> && std::is_integral_v<V>);
  const uint64_t blocks = a / b + (a % b != 0);
  return blocks;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundDown(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral_v<U> && std::is_integral_v<V>);
  return divDown(a, b) * b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundUp(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral_v<U> && std::is_integral_v<V>);
  return divUp(a, b) * b;
}

template <typename U, typename V>
constexpr __host__ __device__ bool isEvenDivisor(U a, V b) {
  static_assert(std::is_integral_v<U> && std::is_integral_v<V>);
  return (a % V(b) == 0) && ((a / V(b)) >= 1);
}

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
  static_assert(std::is_integral_v<T>);
  return (v && !(v & (v - 1)));
}

// Converts eight int4 nibbles held in one uint32 into four bf16x2 values
// representing the signed int4 range [-8, 7], then applies the per-group
// (scale, zero):  reconstructed = bf16(int4) * scale + zero.
__device__ inline bf16x2x4 convert_i4x8_to_bf16x2x4(uint32_t source) {
  bf16x2x4 result;
  constexpr int kElements = 8;
  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  uint32_t const source_i4s = source;
  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

  uint32_t i4s = source_i4s;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
#pragma unroll
  for (int ii = 1; ii < kElements / 2; ++ii) {
    i4s >>= 4;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[ii])
        : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
  }
  static constexpr uint32_t BF16_BIAS = 0xC308C308;
  static constexpr uint32_t BF16_ONE = 0x3F803F80;
#pragma unroll
  for (int ii = 0; ii < kElements / 2; ++ii) {
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n"
        : "=r"(h[ii])
        : "r"(h[ii]), "r"(BF16_ONE), "r"(BF16_BIAS));
  }
  return result;
}

enum class KReductionType {
  None,
};

// Loads the A matrix in 16-bit standard m x k row-major layout and stores
// the C matrix in 16-bit standard m x n row-major layout.
template <KReductionType ReduceType>
struct ALayout_RM {
  static constexpr int32_t kMTileSize = 16;
  static constexpr int32_t kNTileSize = 8;
  static constexpr int32_t kKTileSize = 16;

  template <int KTilesToLoad>
  static __device__ void load(const void* A, int32_t m, int32_t k,
                              int32_t mTiles, int32_t mTile, int32_t kTiles,
                              int32_t kTileStart, int32_t laneId,
                              bf16x2x4_u32 out[KTilesToLoad]) {
    const auto mLane = mTile * kMTileSize + (laneId / 4);
    const auto kLane = kTileStart * kKTileSize + (laneId % 4) * 2;
    auto aPtr = reinterpret_cast<const __nv_bfloat16*>(A) + mLane * k + kLane;
    bool m0InBounds = mLane < m;
    auto aPtrPlus8Rows = aPtr + 8 * k;
    bool m1InBounds = (mLane + 8) < m;
#pragma unroll
    for (int i = 0; i < KTilesToLoad; ++i) {
      out[i].vals[0] =
          m0InBounds ? *reinterpret_cast<const uint32_t*>(aPtr + i * kKTileSize)
                     : uint32_t(0);
      out[i].vals[1] = m1InBounds
                           ? *reinterpret_cast<const uint32_t*>(aPtrPlus8Rows +
                                                                i * kKTileSize)
                           : uint32_t(0);
      out[i].vals[2] = m0InBounds
                           ? *reinterpret_cast<const uint32_t*>(aPtr +
                                                                i * kKTileSize +
                                                                8)
                           : uint32_t(0);
      out[i].vals[3] =
          m1InBounds ? *reinterpret_cast<const uint32_t*>(aPtrPlus8Rows +
                                                          i * kKTileSize + 8)
                     : uint32_t(0);
    }
  }

  static __device__ void store(void* C, int32_t m, int32_t n, int32_t mOutTiles,
                               int32_t mTile, int32_t nOutTiles, int32_t nTile,
                               int32_t laneId, const float4& out) {
    static_assert(ReduceType == KReductionType::None);
    const int outRow = mTile * kMTileSize + (laneId / 4);
    const int outCol = nTile * kNTileSize + (laneId % 4) * 2;
    auto cPtr = reinterpret_cast<__nv_bfloat16*>(C) + outRow * n + outCol;
    auto v01 = __float22bfloat162_rn(float2{out.x, out.y});
    auto v23 = __float22bfloat162_rn(float2{out.z, out.w});
    if (outRow < m) {
      *reinterpret_cast<__nv_bfloat162*>(cPtr) = v01;
    }
    if (outRow + 8 < m) {
      *reinterpret_cast<__nv_bfloat162*>(cPtr + 8 * n) = v23;
    }
  }
};

template <int InnerKTiles, int QGroupSize>
struct BLayout_TC_int4 {
  static constexpr int32_t kInnerKTiles = InnerKTiles;
  static constexpr int32_t kMTileSize = 16;
  static constexpr int32_t kNTileSize = 8;
  static constexpr int32_t kKTileSize = 16;

  template <int KTilesToLoad>
  static __device__ void load(const void* __restrict__ B,
                              const void* __restrict__ quantizationInfo,
                              int32_t n, int32_t k, int32_t nTiles,
                              int32_t nTile, int32_t kTiles,
                              int32_t kTileStart, int32_t laneId,
                              bf16x2x4_u32 out[KTilesToLoad / InnerKTiles]
                                              [InnerKTiles / 2]) {
    auto bPtr = reinterpret_cast<const int32_t*>(B) +
        (((nTile * (kTiles / InnerKTiles) + (kTileStart / InnerKTiles)) *
          kWarpSize) +
         laneId) *
            (InnerKTiles / 2);

    int32_t b_int4[KTilesToLoad / InnerKTiles][InnerKTiles / 2];
#pragma unroll
    for (int i = 0; i < KTilesToLoad / InnerKTiles; ++i) {
      auto bPtrCur = bPtr + i * kWarpSize * (InnerKTiles / 2);
      if constexpr (InnerKTiles == 2) {
        b_int4[i][0] = bPtrCur[0];
      }
      if constexpr (InnerKTiles == 4) {
        int2 load8 = reinterpret_cast<const int2*>(bPtrCur)[0];
        b_int4[i][0] = load8.x;
        b_int4[i][1] = load8.y;
      }
      if constexpr (InnerKTiles == 8) {
        int4 load16 = reinterpret_cast<const int4*>(bPtrCur)[0];
        b_int4[i][0] = load16.x;
        b_int4[i][1] = load16.y;
        b_int4[i][2] = load16.z;
        b_int4[i][3] = load16.w;
      }
    }

    static_assert(isPowerOf2(QGroupSize));
    static_assert(isEvenDivisor(QGroupSize, kKTileSize));
    static_assert(QGroupSize >= kKTileSize * 2);
    constexpr int kKTilesPerQGroup = (QGroupSize / kKTileSize);
    constexpr int kNumQGroups = (KTilesToLoad / kKTilesPerQGroup) < 1
                                    ? 1
                                    : (KTilesToLoad / kKTilesPerQGroup);

    __nv_bfloat162 qScaleAndZero[kNumQGroups];
    {
      int32_t laneN = nTile * kNTileSize + (laneId / 4);
      int32_t groupStart = (kTileStart * kKTileSize) / QGroupSize;
      int32_t nFull = nTiles * kNTileSize;
      auto qInfoPtr = reinterpret_cast<const __nv_bfloat16*>(quantizationInfo) +
          (groupStart * nFull + laneN) * 2;
#pragma unroll
      for (int i = 0; i < kNumQGroups; ++i) {
        qScaleAndZero[i] =
            *reinterpret_cast<const __nv_bfloat162*>(qInfoPtr + i * nFull * 2);
      }
    }

    __nv_bfloat162 qScale[kNumQGroups];
    __nv_bfloat162 qZero[kNumQGroups];
#pragma unroll
    for (int i = 0; i < kNumQGroups; ++i) {
      qScale[i] = __bfloat162bfloat162(qScaleAndZero[i].x);
      qZero[i] = __bfloat162bfloat162(qScaleAndZero[i].y);
    }

#pragma unroll
    for (int i = 0; i < KTilesToLoad / InnerKTiles; ++i) {
#pragma unroll
      for (int j = 0; j < InnerKTiles / 2; ++j) {
        bf16x2x4 v = convert_i4x8_to_bf16x2x4(b_int4[i][j]);
        int curKTile = i * InnerKTiles + j * 2;
        int curQGroup = (curKTile * kKTileSize) / QGroupSize;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          v.vals[k] = __hfma2(v.vals[k], qScale[curQGroup], qZero[curQGroup]);
        }
        static_assert(sizeof(bf16x2x4) == sizeof(out[0][0]));
        memcpy(&out[i][j], &v, sizeof(bf16x2x4_u32));
      }
    }
  }
};

template <typename ALayout, typename BLayout, typename CLayout, int Warps,
          int KTilesPerIteration>
__global__ __launch_bounds__(Warps* kWarpSize)
    void tinygemm_m16n8k16_chunk_kernel(
        const void* const __restrict__ A, const void* const __restrict__ B,
        const void* const __restrict__ B_quantizationInfo,
        void* __restrict__ C, int32_t m, int32_t n, int32_t k, int32_t mTiles,
        int32_t nTiles, int32_t kTiles) {
  constexpr int32_t kMTileSize = 16;
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;
  static_assert(ALayout::kMTileSize == kMTileSize &&
                ALayout::kNTileSize == kNTileSize &&
                ALayout::kKTileSize == kKTileSize);
  static_assert(BLayout::kMTileSize == kMTileSize &&
                BLayout::kNTileSize == kNTileSize &&
                BLayout::kKTileSize == kKTileSize);
  static_assert(CLayout::kMTileSize == kMTileSize &&
                CLayout::kNTileSize == kNTileSize &&
                CLayout::kKTileSize == kKTileSize);

  constexpr int kInnerKTiles = BLayout::kInnerKTiles;
  static_assert(kInnerKTiles == 2 || kInnerKTiles == 4 || kInnerKTiles == 8);
  static_assert(KTilesPerIteration >= kInnerKTiles &&
                  isEvenDivisor(KTilesPerIteration, kInnerKTiles));

  auto warpId = threadIdx.y;
  auto laneId = threadIdx.x;
  int32_t mTile = blockIdx.z;
  int32_t nTile = blockIdx.y;
  float4 c{0.0f, 0.0f, 0.0f, 0.0f};

  auto kTilesLimit = roundDown(kTiles, KTilesPerIteration);
  for (int32_t kTileBase = (blockIdx.x * Warps + warpId) * KTilesPerIteration;
       kTileBase < kTilesLimit; kTileBase += Warps * KTilesPerIteration) {
    bf16x2x4_u32 a[KTilesPerIteration];
    ALayout::template load<KTilesPerIteration>(
        A, m, k, mTiles, mTile, kTiles, kTileBase, laneId, a);

    bf16x2x4_u32 b[KTilesPerIteration / kInnerKTiles][kInnerKTiles / 2];
    BLayout::template load<KTilesPerIteration>(
        B, B_quantizationInfo, n, k, nTiles, nTile, kTiles, kTileBase, laneId,
        b);

#pragma unroll
    for (int i = 0; i < KTilesPerIteration / kInnerKTiles; ++i) {
#pragma unroll
      for (int j = 0; j < kInnerKTiles / 2; ++j) {
        float4 cTmp[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          cTmp[k] = float4{0.0f, 0.0f, 0.0f, 0.0f};
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
              : "=f"(cTmp[k].x), "=f"(cTmp[k].y), "=f"(cTmp[k].z),
                "=f"(cTmp[k].w)
              : "r"(a[i * kInnerKTiles + j * 2 + k].vals[0]),
                "r"(a[i * kInnerKTiles + j * 2 + k].vals[1]),
                "r"(a[i * kInnerKTiles + j * 2 + k].vals[2]),
                "r"(a[i * kInnerKTiles + j * 2 + k].vals[3]),
                "r"(b[i][(j * 2 + k) / 2].vals[((j * 2 + k) % 2) * 2 + 0]),
                "r"(b[i][(j * 2 + k) / 2].vals[((j * 2 + k) % 2) * 2 + 1]),
                "f"(cTmp[k].x), "f"(cTmp[k].y), "f"(cTmp[k].z),
                "f"(cTmp[k].w));
        }
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          c.x += cTmp[k].x;
          c.y += cTmp[k].y;
          c.z += cTmp[k].z;
          c.w += cTmp[k].w;
        }
      }
    }
  }

  static_assert(Warps >= KTilesPerIteration / kInnerKTiles);
  auto kTileBaseRemaining = kTilesLimit + warpId * kInnerKTiles;
  if (kTileBaseRemaining < kTiles) {
    bf16x2x4_u32 a[kInnerKTiles];
    ALayout::template load<kInnerKTiles>(
        A, m, k, mTiles, mTile, kTiles, kTileBaseRemaining, laneId, a);
    bf16x2x4_u32 b[1][kInnerKTiles / 2];
    BLayout::template load<kInnerKTiles>(
        B, B_quantizationInfo, n, k, nTiles, nTile, kTiles,
        kTileBaseRemaining, laneId, b);
#pragma unroll
    for (int j = 0; j < kInnerKTiles / 2; ++j) {
      float4 cTmp[2];
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        cTmp[k] = float4{0.0f, 0.0f, 0.0f, 0.0f};
      }
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
            : "=f"(cTmp[k].x), "=f"(cTmp[k].y), "=f"(cTmp[k].z),
              "=f"(cTmp[k].w)
            : "r"(a[j * 2 + k].vals[0]), "r"(a[j * 2 + k].vals[1]),
              "r"(a[j * 2 + k].vals[2]), "r"(a[j * 2 + k].vals[3]),
              "r"(b[0][(j * 2 + k) / 2].vals[((j * 2 + k) % 2) * 2 + 0]),
              "r"(b[0][(j * 2 + k) / 2].vals[((j * 2 + k) % 2) * 2 + 1]),
              "f"(cTmp[k].x), "f"(cTmp[k].y), "f"(cTmp[k].z),
              "f"(cTmp[k].w));
      }
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        c.x += cTmp[k].x;
        c.y += cTmp[k].y;
        c.z += cTmp[k].z;
        c.w += cTmp[k].w;
      }
    }
  }

  __shared__ float4 smem_sum[Warps][kWarpSize];
  smem_sum[warpId][laneId] = c;
  __syncthreads();
  if (warpId == 0) {
    float4 sum_f32{0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < Warps; ++i) {
      float4 v = smem_sum[i][laneId];
      sum_f32.x += v.x;
      sum_f32.y += v.y;
      sum_f32.z += v.z;
      sum_f32.w += v.w;
    }
    CLayout::store(C, m, n, mTiles, mTile, nTiles / 2, nTile, laneId, sum_f32);
  }
}

template <typename ALayout, typename BLayout, typename CLayout, int Warps,
          int KTilesPerWarp>
void launch_tinygemm_kernel(const Tensor& A, const Tensor& B,
                            const Tensor& qScaleAndZeros, Tensor& C_final,
                            int32_t mTiles, int32_t nTiles, int32_t kTiles,
                            int32_t m, int32_t n, int32_t k,
                            cudaStream_t stream) {
  int32_t postKernelKTiles = 1;
  auto grid = dim3(postKernelKTiles, nTiles, mTiles);
  auto block = dim3(kWarpSize, Warps);
  auto func = tinygemm_m16n8k16_chunk_kernel<ALayout, BLayout, CLayout, Warps,
                                             KTilesPerWarp>;
  func<<<grid, block, 0, stream>>>(
      A.data_ptr(), B.data_ptr(), qScaleAndZeros.data_ptr(), C_final.data_ptr(),
      m, n, k, mTiles, nTiles, kTiles);
  checkCuda(cudaGetLastError(), "int4 tinygemm kernel");
}

// Repacks the packed nibble tensor [n][k/2] (uint8) into the m16n8k16
// B-fragment layout [n/8][k/(InnerKTiles*16)][32][InnerKTiles/2] (int32).
template <int InnerKTiles>
__global__ void matrix_to_m16n8k16_Bint4_layout(
    const uint8_t* __restrict__ in, int64_t in_rows, int64_t in_cols,
    int32_t* __restrict__ out, int64_t out_k_super) {
  static_assert(InnerKTiles >= 2 && isPowerOf2(InnerKTiles));
  constexpr int32_t kNTileSize = 8;
  constexpr int32_t kKTileSize = 16;

  auto kOuterTile = blockIdx.x;
  auto nTile = blockIdx.y;
  auto t = threadIdx.x;

#pragma unroll
  for (int innerKTile = 0; innerKTile < InnerKTiles; innerKTile += 2) {
    auto n0 = nTile * kNTileSize + (t / 4);
    bool n0Valid = n0 < in_rows;
    int32_t ks[4];
    auto kBase0 = (kOuterTile * InnerKTiles + innerKTile) * kKTileSize / 2;
    ks[0] = kBase0 + t % 4;
    ks[1] = ks[0] + 4;
    auto kBase1 = kBase0 + kKTileSize / 2;
    ks[2] = kBase1 + t % 4;
    ks[3] = ks[2] + 4;

    const uint8_t* pIn = in + n0 * in_cols;
    uint8_t v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] = (n0Valid && ks[i] < in_cols) ? pIn[ks[i]] : uint8_t(0);
    }
    int32_t pack = ((uint32_t)(v[3] & 0xF) << 28) |
        ((uint32_t)(v[2] & 0xF) << 24) | ((uint32_t)(v[1] & 0xF) << 20) |
        ((uint32_t)(v[0] & 0xF) << 16) | ((uint32_t)(v[3] & 0xF0) << 8) |
        ((uint32_t)(v[2] & 0xF0) << 4) | ((uint32_t)(v[1] & 0xF0)) |
        ((uint32_t)(v[0] & 0xF0) >> 4);

    out[(((nTile * out_k_super + kOuterTile) * 32) + t) * (InnerKTiles / 2) +
        (innerKTile / 2)] = pack;
  }
}

// ---------------------------------------------------------------------------
// Large-tile direct int4 tensor-core GEMM.
//
// Consumes the m16n8k16-packed int4 weight (one int32 word per lane per 32-k
// chunk) and fp16/bf16 activations directly with mma.sync.  Each block
// computes a 64 x 128 output tile: A is staged in shared memory in an
// ldmatrix-friendly 16x16-subtile layout, B stays in the packed int32
// fragment layout, and per-group (scale, zero) dequantization happens in
// registers right before the MMA.  No dense weight matrix is ever built.
// ---------------------------------------------------------------------------

struct __align__(16) fp16x2x4_u32 { uint32_t vals[4]; };

__device__ inline fp16x2x4_u32 convert_i4x8_to_fp16x2x4(uint32_t source) {
  fp16x2x4_u32 result;
  constexpr int kElements = 8;
  uint32_t* h = result.vals;
  uint32_t const source_i4s = source;
  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t MASK = 0x000f000f;
  static constexpr uint32_t I4s_TO_FP16s_MAGIC_NUM = 0x64006400;

  uint32_t i4s = source_i4s;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(i4s), "n"(MASK), "n"(I4s_TO_FP16s_MAGIC_NUM), "n"(immLut));
#pragma unroll
  for (int ii = 1; ii < kElements / 2; ++ii) {
    i4s >>= 4;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[ii])
                 : "r"(i4s), "n"(MASK), "n"(I4s_TO_FP16s_MAGIC_NUM), "n"(immLut));
  }
  static constexpr uint32_t FP16_BIAS = 0xC808C808;
  static constexpr uint32_t FP16_ONE = 0x3C003C00;
  const __half2 one = __half2half2(__float2half(1.0f));
  const __half2 bias = __half2half2(__float2half(-1032.0f));
#pragma unroll
  for (int ii = 0; ii < kElements / 2; ++ii) {
    __half2 vv = *reinterpret_cast<__half2*>(&h[ii]);
    vv = __hfma2(vv, one, bias);
    h[ii] = *reinterpret_cast<uint32_t*>(&vv);
  }
  return result;
}

// Dequantizes one packed int4 word in registers: the eight nibbles become
// eight T values (two B fragments for two consecutive m16n8k16 MMAs over a
// 32-k chunk), scaled and zero-shifted with the per-group quantization info.
template <typename T>
struct Int4Convert {
  static __device__ void convert(uint32_t word, T scale, T zero,
                                 uint32_t out[4]);
};

template <>
struct Int4Convert<tensorplay::BFloat16> {
  static __device__ void convert(uint32_t word, tensorplay::BFloat16 scale,
                                 tensorplay::BFloat16 zero, uint32_t out[4]) {
    bf16x2x4 v = convert_i4x8_to_bf16x2x4(word);
    __nv_bfloat162 s =
        __bfloat162bfloat162(*reinterpret_cast<__nv_bfloat16*>(&scale));
    __nv_bfloat162 z =
        __bfloat162bfloat162(*reinterpret_cast<__nv_bfloat16*>(&zero));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v.vals[i] = __hfma2(v.vals[i], s, z);
      out[i] = *reinterpret_cast<uint32_t*>(&v.vals[i]);
    }
  }
};

template <>
struct Int4Convert<tensorplay::Half> {
  static __device__ void convert(uint32_t word, tensorplay::Half scale,
                                 tensorplay::Half zero, uint32_t out[4]) {
    fp16x2x4_u32 v = convert_i4x8_to_fp16x2x4(word);
    __half2 s = __half2half2(*reinterpret_cast<__half*>(&scale));
    __half2 z = __half2half2(*reinterpret_cast<__half*>(&zero));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      __half2 vv = *reinterpret_cast<__half2*>(&v.vals[i]);
      vv = __hfma2(vv, s, z);
      v.vals[i] = *reinterpret_cast<uint32_t*>(&vv);
    }
    out[0] = v.vals[0];
    out[1] = v.vals[1];
    out[2] = v.vals[2];
    out[3] = v.vals[3];
  }
};

template <typename T>
__device__ __forceinline__ void mma16n8k16(float acc[4], const uint32_t a[4],
                                           uint32_t b0, uint32_t b1) {
  // The A fragment is supplied already in mma order (a0: row g, col 2t;
  // a1: row g+8, col 2t; a2: row g, col 2t+8; a3: row g+8, col 2t+8).
  if constexpr (std::is_same_v<T, tensorplay::BFloat16>) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
}

// One threadblock computes a BM x BN = 64 x 128 output tile with 8 warps,
// each warp handling 32 x 32.  K advances in 32-element chunks.  A and B
// tiles are fetched with cp.async into triple-buffered shared memory (one
// sync per iteration), A fragments are loaded manually in the exact
// m16n8k16 register layout (padded stride -> conflict-free reads), and B
// stays packed as int32 words, dequantized in registers before each MMA.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t s = __cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <typename T>
__global__ __launch_bounds__(256) void int4pack_tc_mm_kernel(
    const T* __restrict__ A, const int32_t* __restrict__ B,
    const T* __restrict__ QSZ, T* __restrict__ C, int M, int N, int K,
    int group_size) {
  constexpr int BM = 64;
  constexpr int BN = 128;
  constexpr int BK = 32;
  constexpr int kAStride = 40;  // 32 + 8 padding: conflict-free fragment reads

  __shared__ T a_smem[3][BM][kAStride];
  __shared__ int32_t b_smem[3][BN / 8][32];
  __shared__ T s_smem[3][BN][2];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int mBlock = blockIdx.y * BM;
  const int nBlock = blockIdx.x * BN;
  const int kTiles = K / BK;
  const int kSuper = K / BK;  // InnerKTiles == 2 -> kSuperTiles = K / 32

  const int warpRow = warp >> 2;  // 0..1
  const int warpCol = warp & 3;   // 0..3

  float acc[2][4][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.0f;

  auto prefetch = [&](int buf, int kTile) {
    {
      const int r = tid >> 2;
      const int cblk = (tid & 3) * 8;
      const int gm = mBlock + r;
      if (gm < M) {
        cp_async16(&a_smem[buf][r][cblk],
                  &A[(int64_t)gm * K + kTile * BK + cblk]);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) a_smem[buf][r][cblk + i] = T(0);
      }
    }
    // B: 512 int32 = 128 threads x 16 bytes (4 words each)
    if (tid < 128) {
      const int ln = tid >> 3;
      const int ld = (tid & 7) * 4;
      const int gn = nBlock / 8 + ln;
      if (gn < N / 8) {
        cp_async16(&b_smem[buf][ln][ld], &B[((gn * kSuper + kTile) * 32) + ld]);
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) b_smem[buf][ln][ld + i] = 0;
      }
    }
    if (tid < BN) {
      const int ncol = nBlock + tid;
      const int group = (kTile * BK) / group_size;
      if (ncol < N) {
        s_smem[buf][tid][0] = QSZ[((int64_t)group * N + ncol) * 2 + 0];
        s_smem[buf][tid][1] = QSZ[((int64_t)group * N + ncol) * 2 + 1];
      } else {
        s_smem[buf][tid][0] = T(0);
        s_smem[buf][tid][1] = T(0);
      }
    }
  };

  // Prefetch the first two K-chunks (buffers 0 and 1).
  if (kTiles > 0) prefetch(0, 0);
  if (kTiles > 1) prefetch(1, 1);

  for (int kTile = 0; kTile < kTiles; ++kTile) {
    const int cur = kTile % 3;
    const int pf = (kTile + 2) % 3;

    if (kTile + 2 < kTiles) prefetch(pf, kTile + 2);

    if (kTile < 2) {
      cp_async_wait_group<0>();
    } else {
      cp_async_wait_group<1>();
    }
    __syncthreads();

    // ---- load A fragments (2 m-subtiles x 2 k-subtiles), manual layout ----
    const int gid = lane >> 2;
    const int t4 = lane & 3;
    uint32_t a_reg[2][2][4];
#pragma unroll
    for (int ms = 0; ms < 2; ++ms) {
      const int baseM = warpRow * 32 + ms * 16;
      const T* row0 = &a_smem[cur][baseM + gid][0];
      const T* row1 = &a_smem[cur][baseM + gid + 8][0];
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        const int koff = ks * 16 + 2 * t4;
        a_reg[ms][ks][0] = *reinterpret_cast<const uint32_t*>(row0 + koff);
        a_reg[ms][ks][1] = *reinterpret_cast<const uint32_t*>(row1 + koff);
        a_reg[ms][ks][2] =
            *reinterpret_cast<const uint32_t*>(row0 + koff + 8);
        a_reg[ms][ks][3] =
            *reinterpret_cast<const uint32_t*>(row1 + koff + 8);
      }
    }

    // ---- load B fragments and dequantize in registers ----
    uint32_t b_reg[4][4];
#pragma unroll
    for (int ns = 0; ns < 4; ++ns) {
      const int ln = warpCol * 4 + ns;
      const int32_t word = b_smem[cur][ln][lane];
      const int ncol = nBlock + ln * 8 + (lane >> 2);
      const T scale = s_smem[cur][ncol - nBlock][0];
      const T zero = s_smem[cur][ncol - nBlock][1];
      Int4Convert<T>::convert(static_cast<uint32_t>(word), scale, zero,
                              b_reg[ns]);
    }

    // ---- MMA accumulate ----
#pragma unroll
    for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
      for (int ns = 0; ns < 4; ++ns) {
#pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
          mma16n8k16<T>(acc[ms][ns], a_reg[ms][kh], b_reg[ns][kh * 2],
                        b_reg[ns][kh * 2 + 1]);
        }
      }
    }
  }

  // ---- store C ----
  const int outM = mBlock + warpRow * 32;
  const int outN = nBlock + warpCol * 32;
#pragma unroll
  for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
    for (int ns = 0; ns < 4; ++ns) {
      const int row = outM + ms * 16 + (lane >> 2);
      const int col = outN + ns * 8 + (lane & 3) * 2;
      if (row < M && col < N) {
        C[(int64_t)row * N + col] = static_cast<T>(acc[ms][ns][0]);
        if (col + 1 < N)
          C[(int64_t)row * N + col + 1] = static_cast<T>(acc[ms][ns][1]);
      }
      if (row + 8 < M && col < N) {
        C[(int64_t)(row + 8) * N + col] = static_cast<T>(acc[ms][ns][2]);
        if (col + 1 < N)
          C[(int64_t)(row + 8) * N + col + 1] = static_cast<T>(acc[ms][ns][3]);
      }
    }
  }
}

// Dequantizes the m16n8k16-packed weight back to a dense [N, K] matrix.
// One thread per packed int32 word; each word expands to eight output
// elements along K for one row.
template <typename T>
__global__ void int4_tc_dequantize_kernel(const int32_t* __restrict__ B,
                                          const T* __restrict__ QSZ,
                                          T* __restrict__ W, int N, int K,
                                          int group_size, int kSuper) {
  const int total = (N / 8) * kSuper * 32;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int lane = idx % 32;
  const int kTile = (idx / 32) % kSuper;
  const int nTile = idx / (32 * kSuper);
  const int n = nTile * 8 + (lane >> 2);
  const int t4 = lane & 3;
  const int kBase = kTile * 32 + 2 * t4;
  const int group = (kTile * 32) / group_size;
  const T* sz = QSZ + ((int64_t)group * N + n) * 2;
  const T scale = sz[0];
  const T zero = sz[1];
  const int32_t word = B[idx];
  // With even-k codes stored in the high nibble of each packed byte, the
  // int32 word's bits hold: odd k at {16,20,24,28} and even k at {0,4,8,12}.
  int v[8] = {(word >> 16) & 0xF, word & 0xF, (word >> 20) & 0xF,
              (word >> 4) & 0xF, (word >> 24) & 0xF, (word >> 8) & 0xF,
              (word >> 28) & 0xF, (word >> 12) & 0xF};
  int kk[8] = {kBase + 1, kBase, kBase + 9, kBase + 8,
               kBase + 17, kBase + 16, kBase + 25, kBase + 24};
  T* row = W + (int64_t)n * K;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int kc = kk[i];
    if (kc < K) row[kc] = static_cast<T>(v[i] - 8) * scale + zero;
  }
}

// ---------------------------------------------------------------------------
// _weight_int4pack_mm: group-quantized 4-bit weight [N, K] packed two codes
// per byte against fp activations [B, K].  qScaleAndZeros [N, groups, 2]
// carries (scale, zero) per group; qGroupSize is 32, 64 or 128.
// ---------------------------------------------------------------------------

Tensor _weight_int4pack_mm_cuda(const Tensor& self, const Tensor& mat2,
                                int64_t q_group_size,
                                const Tensor& q_scale_and_zeros) {
    // `self` is the activation matrix [m, k] and `mat2` is the int4 weight
    // packed into the m16n8k16 B-fragment layout (4-D int32), so the result
    // is `self @ weight.T` with per-group (scale, zero) dequantization.
    if (mat2.dtype() != DType::Int32 || mat2.dim() != 4) {
        TP_THROW(TypeError,
                 "_weight_int4pack_mm(): packed weight must be 4-D int32 in "
                 "the tensor-core B layout");
    }
    if (self.dim() != 2) {
        TP_THROW(ValueError,
                 "_weight_int4pack_mm(): expected 2-D activations [m,k]");
    }
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError,
                 "_weight_int4pack_mm(): activations must be floating point");
    }
    if (q_group_size != 32 && q_group_size != 64 && q_group_size != 128) {
        TP_THROW(ValueError,
                 "_weight_int4pack_mm(): qGroupSize must be 32, 64 or 128");
    }
    const int64_t m = self.size(0);
    const int64_t k = self.size(1);
    const int64_t nTiles = mat2.size(0);
    const int64_t kSuper = mat2.size(1);
    const int64_t inner = mat2.size(3) * 2;
    const int64_t n = nTiles * 8;
    if (mat2.size(2) != 32 || (inner != 2 && inner != 4 && inner != 8)) {
        TP_THROW(ValueError,
                 "_weight_int4pack_mm(): packed weight layout is invalid");
    }
    if (kSuper != divUp(k, inner * 16)) {
        TP_THROW(ValueError,
                 "_weight_int4pack_mm(): K dimension does not match packed B");
    }
    const int64_t groups = divUp(k, q_group_size);
    if (q_scale_and_zeros.dim() != 3 ||
        q_scale_and_zeros.size(0) != groups ||
        q_scale_and_zeros.size(1) != n || q_scale_and_zeros.size(2) != 2) {
        TP_THROW(ValueError,
                 "_weight_int4pack_mm(): qScaleAndZeros must be "
                 "[K/group, N, 2]");
    }
    if (m == 0 || n == 0 || k == 0) {
        return Tensor::empty({m, n}, self.dtype(), self.device());
    }

    // The direct tensor-core path (int4pack_tc_mm_kernel) is correct and
    // remains in this file, but the dequantized cuBLAS path below is the
    // default: cuBLASLt reaches a much higher fraction of peak FP16/bf16
    // throughput than the hand-written MMA loop on current hardware, so the
    // cuBLAS path is faster for production shapes.  Flip this flag to true
    // to route aligned fp16/bf16 shapes through the direct path instead.
    const bool direct_ok = false;
    if (direct_ok) {
        Tensor A = self.contiguous();
        Tensor QSZ = q_scale_and_zeros.to(self.dtype()).contiguous();
        Tensor C = Tensor::empty({m, n}, self.dtype(), self.device());
        auto stream = getCurrentCUDAStream().stream();
        const unsigned nBlocks = static_cast<unsigned>((n + 127) / 128);
        const unsigned mBlocks = static_cast<unsigned>((m + 63) / 64);
        const dim3 grid(nBlocks, mBlocks);
        if (self.dtype() == DType::Float16) {
            int4pack_tc_mm_kernel<tensorplay::Half><<<grid, 256, 0, stream>>>(
                static_cast<const tensorplay::Half*>(A.data_ptr()),
                static_cast<const int32_t*>(mat2.data_ptr()),
                static_cast<const tensorplay::Half*>(QSZ.data_ptr()),
                static_cast<tensorplay::Half*>(C.data_ptr()),
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                static_cast<int>(q_group_size));
        } else {
            int4pack_tc_mm_kernel<tensorplay::BFloat16>
                <<<grid, 256, 0, stream>>>(
                    static_cast<const tensorplay::BFloat16*>(A.data_ptr()),
                    static_cast<const int32_t*>(mat2.data_ptr()),
                    static_cast<const tensorplay::BFloat16*>(QSZ.data_ptr()),
                    static_cast<tensorplay::BFloat16*>(C.data_ptr()),
                    static_cast<int>(m), static_cast<int>(n),
                    static_cast<int>(k), static_cast<int>(q_group_size));
        }
        checkCuda(cudaGetLastError(), "int4pack tc mm kernel");
        return C;
    }

    // Fallback: dequantize the packed weight (from the tensor-core layout)
    // and run a cuBLAS GEMM (out = activations @ weight.T).
    Tensor xf = self.to(self.dtype()).contiguous();
    Tensor wf = Tensor::empty({n, k}, self.dtype(), self.device());
    Tensor out = Tensor::empty({m, n}, self.dtype(), self.device());
    Tensor qsz_t = q_scale_and_zeros.to(self.dtype()).contiguous();
    const int64_t numel = (n / 8) * kSuper * 32;
    constexpr int kThreads = 256;
    const unsigned blocks =
        static_cast<unsigned>((numel + kThreads - 1) / kThreads);
    auto stream = getCurrentCUDAStream().stream();
    switch (self.dtype()) {
        case DType::Float16:
            int4_tc_dequantize_kernel<tensorplay::Half><<<blocks, kThreads, 0, stream>>>(
                static_cast<const int32_t*>(mat2.data_ptr()),
                static_cast<const tensorplay::Half*>(qsz_t.data_ptr()),
                static_cast<tensorplay::Half*>(wf.data_ptr()),
                static_cast<int>(n), static_cast<int>(k),
                static_cast<int>(q_group_size), static_cast<int>(kSuper));
            break;
        case DType::BFloat16:
            int4_tc_dequantize_kernel<tensorplay::BFloat16><<<blocks, kThreads, 0, stream>>>(
                static_cast<const int32_t*>(mat2.data_ptr()),
                static_cast<const tensorplay::BFloat16*>(qsz_t.data_ptr()),
                static_cast<tensorplay::BFloat16*>(wf.data_ptr()),
                static_cast<int>(n), static_cast<int>(k),
                static_cast<int>(q_group_size), static_cast<int>(kSuper));
            break;
        case DType::Float32:
            int4_tc_dequantize_kernel<float><<<blocks, kThreads, 0, stream>>>(
                static_cast<const int32_t*>(mat2.data_ptr()),
                static_cast<const float*>(qsz_t.data_ptr()),
                static_cast<float*>(wf.data_ptr()),
                static_cast<int>(n), static_cast<int>(k),
                static_cast<int>(q_group_size), static_cast<int>(kSuper));
            break;
        default:
            TP_THROW(TypeError,
                     "_weight_int4pack_mm(): unsupported activation dtype");
    }
    checkCuda(cudaGetLastError(), "int4pack dequantize kernel");
    gemm_impl(xf, wf.t(), out, 1.0, 0.0, nullptr);
    return out;
}

// ---------------------------------------------------------------------------
// _convert_weight_to_int4pack: pack an int32 code tensor [N, K] (values
// 0..15) into [N, K/2] uint8 with two codes per byte, matching the CPU
// packing layout used by _weight_int4pack_mm.
// ---------------------------------------------------------------------------

Tensor _convert_weight_to_int4pack_cuda(const Tensor& self,
                                        int64_t inner_tiles) {
    // Input: int32 code tensor [N, K] (values 0..15).  The codes are packed
    // two per byte into [N, K/2] uint8 and then repacked into the m16n8k16
    // B-fragment layout [n/8][k/(InnerKTiles*16)][32][InnerKTiles/2] int32
    // consumed by the direct tensor-core GEMM.
    if (self.dim() != 2) {
        TP_THROW(ValueError,
                 "_convert_weight_to_int4pack(): expected 2-D codes [N,K]");
    }
    if (inner_tiles != 2 && inner_tiles != 4 && inner_tiles != 8) {
        TP_THROW(ValueError,
                 "_convert_weight_to_int4pack(): innerKTiles must be 2, 4 or 8");
    }
    const int64_t N = self.size(0);
    const int64_t K = self.size(1);
    if (K % 2 != 0) {
        TP_THROW(ValueError, "_convert_weight_to_int4pack(): K must be even");
    }
    const int64_t nTiles = divUp(N, 8);
    const int64_t kSuper = divUp(K, inner_tiles * 16);
    if (N == 0 || K == 0) {
        return Tensor::empty({nTiles, kSuper, 32, inner_tiles / 2},
                             DType::Int32, self.device());
    }

    Tensor codes = self.to(DType::Int32).contiguous();
    Tensor packed_u8 = Tensor::empty({N, K / 2}, DType::UInt8, self.device());
    const int64_t num_bytes = N * (K / 2);
    constexpr int kThreads = 256;
    const unsigned blocks =
        static_cast<unsigned>((num_bytes + kThreads - 1) / kThreads);
    auto stream = getCurrentCUDAStream().stream();
    int4_pack_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const int32_t*>(codes.data_ptr()),
        static_cast<uint8_t*>(packed_u8.data_ptr()), num_bytes);
    checkCuda(cudaGetLastError(), "int4 pack kernel");

    Tensor out = Tensor::empty({nTiles, kSuper, 32, inner_tiles / 2},
                               DType::Int32, self.device());
    const dim3 grid(static_cast<unsigned>(kSuper),
                    static_cast<unsigned>(nTiles));
    const uint8_t* u8_ptr = static_cast<const uint8_t*>(packed_u8.data_ptr());
    int32_t* out_ptr = static_cast<int32_t*>(out.data_ptr());
    if (inner_tiles == 2) {
        matrix_to_m16n8k16_Bint4_layout<2>
            <<<grid, kWarpSize, 0, stream>>>(u8_ptr, N, K / 2, out_ptr,
                                             kSuper);
    } else if (inner_tiles == 4) {
        matrix_to_m16n8k16_Bint4_layout<4>
            <<<grid, kWarpSize, 0, stream>>>(u8_ptr, N, K / 2, out_ptr,
                                             kSuper);
    } else {
        matrix_to_m16n8k16_Bint4_layout<8>
            <<<grid, kWarpSize, 0, stream>>>(u8_ptr, N, K / 2, out_ptr,
                                             kSuper);
    }
    checkCuda(cudaGetLastError(), "int4pack layout kernel");
    return out;
}

// ---------------------------------------------------------------------------
// Fused row-wise signed quantization.
//
// One block per row: the block first reduces the row's absolute maximum,
// derives the per-row scale (absmax clamped to `eps`, divided by the largest
// positive code), then writes `round(x / scale)` clamped to the code range.
// A single kernel (row read twice, codes written once) replaces the eager
// abs/amax/div/round/clamp/cast chain used for large activation and weight
// matrices.
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void rowwise_quantize_kernel(const scalar_t* __restrict__ x,
                                        int8_t* __restrict__ codes,
                                        float* __restrict__ scales, int K,
                                        float eps, int quant_min,
                                        int quant_max) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const scalar_t* row_ptr = x + static_cast<int64_t>(row) * K;

    float absmax = 0.0f;
    for (int i = tid; i < K; i += nthreads) {
        absmax = fmaxf(absmax, fabsf(static_cast<float>(row_ptr[i])));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        absmax = fmaxf(absmax, __shfl_down_sync(0xffffffffu, absmax, offset));
    }
    __shared__ float s_max[32];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) s_max[warp] = absmax;
    __syncthreads();
    if (warp == 0) {
        absmax = (lane < (nthreads >> 5)) ? s_max[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            absmax = fmaxf(absmax, __shfl_down_sync(0xffffffffu, absmax, offset));
        }
        if (lane == 0) s_max[0] = absmax;
    }
    __syncthreads();

    const float scale = fmaxf(s_max[0], eps) / quant_max;
    if (tid == 0) scales[row] = scale;

    int8_t* codes_row = codes + static_cast<int64_t>(row) * K;
    for (int i = tid; i < K; i += nthreads) {
        int c = static_cast<int>(
            rintf(static_cast<float>(row_ptr[i]) / scale));
        c = c < quant_min ? quant_min : (c > quant_max ? quant_max : c);
        codes_row[i] = static_cast<int8_t>(c);
    }
}

template <typename scalar_t>
__global__ void rowwise_quantize_shared_kernel(
    const scalar_t* __restrict__ x, int8_t* __restrict__ codes,
    float* __restrict__ scales, int K, float eps, int quant_min,
    int quant_max) {
    extern __shared__ float s_row[];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const scalar_t* row_ptr = x + static_cast<int64_t>(row) * K;

    float absmax = 0.0f;
    for (int i = tid; i < K; i += nthreads) {
        const float v = static_cast<float>(row_ptr[i]);
        s_row[i] = v;
        absmax = fmaxf(absmax, fabsf(v));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        absmax = fmaxf(absmax, __shfl_down_sync(0xffffffffu, absmax, offset));
    }
    __shared__ float s_max[32];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) s_max[warp] = absmax;
    __syncthreads();
    if (warp == 0) {
        absmax = (lane < (nthreads >> 5)) ? s_max[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            absmax = fmaxf(absmax, __shfl_down_sync(0xffffffffu, absmax, offset));
        }
        if (lane == 0) s_max[0] = absmax;
    }
    __syncthreads();

    const float scale = fmaxf(s_max[0], eps) / quant_max;
    if (tid == 0) scales[row] = scale;

    int8_t* codes_row = codes + static_cast<int64_t>(row) * K;
    for (int i = tid; i < K; i += nthreads) {
        int c = static_cast<int>(rintf(s_row[i] / scale));
        c = c < quant_min ? quant_min : (c > quant_max ? quant_max : c);
        codes_row[i] = static_cast<int8_t>(c);
    }
}

void launch_rowwise_quantize(const Tensor& x, Tensor& codes, Tensor& scales,
                             float eps, int quant_min, int quant_max,
                             cudaStream_t stream) {
    const int64_t K = x.size(1);
    constexpr int kThreads = 256;
    const unsigned blocks = static_cast<unsigned>(x.size(0));
    // Single-pass path: the whole row is cached in shared memory, so the
    // quantization pass never touches global memory again.  Rows wider than
    // the shared budget fall back to the two-pass kernel, which re-reads the
    // row from the cache hierarchy.
    const bool use_shared = K * static_cast<int64_t>(sizeof(float)) <=
                            (48 * 1024);
    switch (x.dtype()) {
        case DType::Float16:
            if (use_shared) {
                rowwise_quantize_shared_kernel<tensorplay::Half>
                    <<<blocks, kThreads, K * sizeof(float), stream>>>(
                        static_cast<const tensorplay::Half*>(x.data_ptr()),
                        codes.data_ptr<int8_t>(), scales.data_ptr<float>(),
                        static_cast<int>(K), eps, quant_min, quant_max);
            } else {
                rowwise_quantize_kernel<tensorplay::Half>
                    <<<blocks, kThreads, 0, stream>>>(
                        static_cast<const tensorplay::Half*>(x.data_ptr()),
                        codes.data_ptr<int8_t>(), scales.data_ptr<float>(),
                        static_cast<int>(K), eps, quant_min, quant_max);
            }
            break;
        case DType::BFloat16:
            if (use_shared) {
                rowwise_quantize_shared_kernel<tensorplay::BFloat16>
                    <<<blocks, kThreads, K * sizeof(float), stream>>>(
                        static_cast<const tensorplay::BFloat16*>(x.data_ptr()),
                        codes.data_ptr<int8_t>(), scales.data_ptr<float>(),
                        static_cast<int>(K), eps, quant_min, quant_max);
            } else {
                rowwise_quantize_kernel<tensorplay::BFloat16>
                    <<<blocks, kThreads, 0, stream>>>(
                        static_cast<const tensorplay::BFloat16*>(x.data_ptr()),
                        codes.data_ptr<int8_t>(), scales.data_ptr<float>(),
                        static_cast<int>(K), eps, quant_min, quant_max);
            }
            break;
        case DType::Float32:
            if (use_shared) {
                rowwise_quantize_shared_kernel<float>
                    <<<blocks, kThreads, K * sizeof(float), stream>>>(
                        static_cast<const float*>(x.data_ptr()),
                        codes.data_ptr<int8_t>(), scales.data_ptr<float>(),
                        static_cast<int>(K), eps, quant_min, quant_max);
            } else {
                rowwise_quantize_kernel<float>
                    <<<blocks, kThreads, 0, stream>>>(
                        static_cast<const float*>(x.data_ptr()),
                        codes.data_ptr<int8_t>(), scales.data_ptr<float>(),
                        static_cast<int>(K), eps, quant_min, quant_max);
            }
            break;
        default:
            TP_THROW(TypeError, "rowwise quantize: unsupported input dtype");
    }
}

std::tuple<Tensor, Tensor> _quantize_int8_rowwise_cuda(const Tensor& self,
                                                       double eps) {
    if (self.dim() != 2) {
        TP_THROW(ValueError, "_quantize_int8_rowwise(): expected a 2-D tensor");
    }
    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const Tensor x = self.contiguous();
    Tensor codes = Tensor::empty({M, K}, DType::Int8, self.device());
    Tensor scales = Tensor::empty({M}, DType::Float32, self.device());
    if (M == 0 || K == 0) {
        return {codes, scales};
    }
    launch_rowwise_quantize(x, codes, scales, static_cast<float>(eps), -127,
                            127, getCurrentCUDAStream().stream());
    checkCuda(cudaGetLastError(), "rowwise int8 quantize kernel");
    return {codes, scales};
}

std::tuple<Tensor, Tensor> _quantize_int4_rowwise_cuda(const Tensor& self,
                                                       double eps) {
    if (self.dim() != 2) {
        TP_THROW(ValueError, "_quantize_int4_rowwise(): expected a 2-D tensor");
    }
    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const Tensor x = self.contiguous();
    Tensor codes = Tensor::empty({M, K}, DType::Int8, self.device());
    Tensor scales = Tensor::empty({M}, DType::Float32, self.device());
    if (M == 0 || K == 0) {
        return {codes, scales};
    }
    launch_rowwise_quantize(x, codes, scales, static_cast<float>(eps), -8, 7,
                            getCurrentCUDAStream().stream());
    checkCuda(cudaGetLastError(), "rowwise int4 quantize kernel");
    return {codes, scales};
}

__global__ void int8_mm_rescale_kernel(const int32_t* __restrict__ out,
                                       const float* __restrict__ row_scale,
                                       const float* __restrict__ col_scale,
                                       float* __restrict__ result, int M,
                                       int N) {
    const int row = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < N) {
        const int64_t idx = static_cast<int64_t>(row) * N + col;
        result[idx] = static_cast<float>(out[idx]) * row_scale[row] *
                      col_scale[col];
    }
}

Tensor _int8_mm_rescale_cuda(const Tensor& out, const Tensor& row_scale,
                              const Tensor& col_scale) {
    if (out.dim() != 2 || out.dtype() != DType::Int32) {
        TP_THROW(ValueError,
                 "_int8_mm_rescale(): expected a 2-D int32 output");
    }
    const int64_t M = out.size(0);
    const int64_t N = out.size(1);
    Tensor result = Tensor::empty({M, N}, DType::Float32, out.device());
    if (M == 0 || N == 0) {
        return result;
    }
    const Tensor rs = row_scale.contiguous();
    const Tensor cs = col_scale.contiguous();
    constexpr int kThreads = 256;
    const dim3 grid(static_cast<unsigned>(M),
                    static_cast<unsigned>((N + kThreads - 1) / kThreads));
    int8_mm_rescale_kernel<<<grid, kThreads, 0, getCurrentCUDAStream().stream()>>>(
        static_cast<const int32_t*>(out.data_ptr()),
        static_cast<const float*>(rs.data_ptr()),
        static_cast<const float*>(cs.data_ptr()),
        static_cast<float*>(result.data_ptr()), static_cast<int>(M),
        static_cast<int>(N));
    checkCuda(cudaGetLastError(), "int8 mm rescale kernel");
    return result;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantGemmKernels) {
    m.impl("_weight_int8pack_mm", _weight_int8pack_mm_cuda);
    m.impl("_weight_int4pack_mm", _weight_int4pack_mm_cuda);
    m.impl("_convert_weight_to_int4pack", _convert_weight_to_int4pack_cuda);
    m.impl("_quantize_int8_rowwise", _quantize_int8_rowwise_cuda);
    m.impl("_quantize_int4_rowwise", _quantize_int4_rowwise_cuda);
    m.impl("_int8_mm_rescale", _int8_mm_rescale_cuda);
}

}  // namespace cuda
}  // namespace tensorplay

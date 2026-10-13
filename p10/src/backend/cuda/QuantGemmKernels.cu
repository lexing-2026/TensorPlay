#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Utils.h"
#include "CudaGemm.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
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
// the even-k code in the low nibble and the odd-k code in the high nibble.
__global__ void int4_pack_kernel(const int32_t* __restrict__ codes,
                                 uint8_t* __restrict__ out,
                                 int64_t num_bytes) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= num_bytes) return;
    const int32_t low = codes[2 * e] & 0x0f;
    const int32_t high = codes[2 * e + 1] & 0x0f;
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
// _weight_int4pack_mm: group-quantized 4-bit weight [N, K] packed two codes
// per byte against fp activations [B, K].  qScaleAndZeros [N, groups, 2]
// carries (scale, zero) per group; qGroupSize is 32, 64 or 128.
// ---------------------------------------------------------------------------

Tensor _weight_int4pack_mm_cuda(const Tensor& self, const Tensor& mat2,
                                int64_t q_group_size,
                                const Tensor& q_scale_and_zeros) {
    // `self` is the activation matrix [B, K] and `mat2` is the packed int4
    // weight [N, K/2] (matching the op schema), so the result is
    // `self @ weight.T` with per-group (scale, zero) dequantization.
    if (mat2.dtype() != DType::UInt8 && mat2.dtype() != DType::QInt8) {
        TP_THROW(TypeError,
                 "_weight_int4pack_mm(): packed weight must be Byte/QInt8");
    }
    if (self.dim() != 2 || mat2.dim() != 2) {
        TP_THROW(ValueError,
                 "_weight_int4pack_mm(): expected 2-D activations [B,K] and "
                 "2-D packed weight [N,K/2]");
    }
    if (!isFloatingType(self.dtype())) {
        TP_THROW(TypeError,
                 "_weight_int4pack_mm(): activations must be floating point");
    }
    if (q_group_size != 32 && q_group_size != 64 && q_group_size != 128) {
        TP_THROW(ValueError,
                 "_weight_int4pack_mm(): qGroupSize must be 32, 64 or 128");
    }
    const int64_t B = self.size(0);
    const int64_t K = self.size(1);
    const int64_t N = mat2.size(0);
    if (mat2.size(1) * 2 != K) {
        TP_THROW(ValueError, "_weight_int4pack_mm(): K dimension mismatch");
    }
    const int64_t groups = (K + q_group_size - 1) / q_group_size;
    if (q_scale_and_zeros.dim() != 3 ||
        q_scale_and_zeros.size(0) != groups ||
        q_scale_and_zeros.size(1) != N || q_scale_and_zeros.size(2) != 2) {
        TP_THROW(ValueError,
                 "_weight_int4pack_mm(): qScaleAndZeros must be [K/group, N, 2]");
    }
    if (N == 0 || B == 0 || K == 0) {
        return Tensor::empty({B, N}, self.dtype(), self.device());
    }

    Tensor w = mat2.contiguous();
    Tensor sz = q_scale_and_zeros.to(DType::Float32).contiguous();
    auto stream = getCurrentCUDAStream().stream();

    const uint8_t* w_ptr = static_cast<const uint8_t*>(w.data_ptr());
    const float* s_ptr = static_cast<const float*>(sz.data_ptr());

    // Custom GEMV/scalar kernels cover the memory-bound skinny shapes and
    // shapes the tensor-core GEMM path cannot express (unaligned K).  The
    // tensor-core path dequantizes the weight once per call and runs a
    // fp16/bf16/fp32 GEMM.
    const bool aligned = gemv_k_aligned16(w_ptr, self.data_ptr()) &&
                         (K % 32 == 0) && (q_group_size % 32 == 0);
    const bool use_gemv = B == 1 && aligned;
    const bool use_scalar =
        !use_gemv && !(B >= 4 && K % q_group_size == 0 && K >= 64);
    if (use_gemv || use_scalar) {
        Tensor out = Tensor::empty({B, N}, DType::Float32, self.device());
        Tensor xf = self.to(DType::Float32).contiguous();
        const float* x_ptr = static_cast<const float*>(xf.data_ptr());
        float* o_ptr = static_cast<float*>(out.data_ptr());
        const unsigned grid = static_cast<unsigned>(N);
        const int wave = gemv_wave_size();
        if (use_gemv) {
            const int rpb = kGemvThreads / wave;
            const unsigned blocks =
                static_cast<unsigned>((N + rpb - 1) / rpb);
            if (wave == 64) {
                int4pack_gemv_rows_kernel<64><<<blocks, kGemvThreads, 0, stream>>>(
                    o_ptr, w_ptr, x_ptr, s_ptr, K, q_group_size, N);
            } else {
                int4pack_gemv_rows_kernel<32><<<blocks, kGemvThreads, 0, stream>>>(
                    o_ptr, w_ptr, x_ptr, s_ptr, K, q_group_size, N);
            }
        } else {
            const int64_t total = N * B;
            int4pack_scalar_kernel<<<static_cast<unsigned>((total + 255) / 256),
                                     256, 0, stream>>>(o_ptr, w_ptr, x_ptr, s_ptr,
                                                       K, N, B, q_group_size);
        }
        checkCuda(cudaGetLastError(), "int4pack gemm kernel");
        return out.to(self.dtype());
    }

    // Tensor-core path: dequantize the packed weight to the activation dtype
    // and run a cuBLAS GEMM (out = activations @ weight.T).
    Tensor xf = self.to(self.dtype()).contiguous();
    Tensor wf = Tensor::empty({N, K}, self.dtype(), self.device());
    Tensor out = Tensor::empty({B, N}, self.dtype(), self.device());
    const int64_t numel = N * K;
    constexpr int kThreads = 256;
    const unsigned blocks =
        static_cast<unsigned>((numel + kThreads - 1) / kThreads);
    const int64_t groups_per_row = K / q_group_size;
    switch (self.dtype()) {
        case DType::Float16:
            int4_dequantize_kernel<tensorplay::Half><<<blocks, kThreads, 0, stream>>>(
                w_ptr, s_ptr, static_cast<tensorplay::Half*>(wf.data_ptr()), N,
                K, groups_per_row, q_group_size);
            break;
        case DType::BFloat16:
            int4_dequantize_kernel<tensorplay::BFloat16><<<blocks, kThreads, 0, stream>>>(
                w_ptr, s_ptr, static_cast<tensorplay::BFloat16*>(wf.data_ptr()),
                N, K, groups_per_row, q_group_size);
            break;
        case DType::Float32:
            int4_dequantize_kernel<float><<<blocks, kThreads, 0, stream>>>(
                w_ptr, s_ptr, static_cast<float*>(wf.data_ptr()), N, K,
                groups_per_row, q_group_size);
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
    (void)inner_tiles;
    if (self.dim() != 2) {
        TP_THROW(ValueError,
                 "_convert_weight_to_int4pack(): expected 2-D codes [N,K]");
    }
    const int64_t N = self.size(0);
    const int64_t K = self.size(1);
    if (K % 2 != 0) {
        TP_THROW(ValueError, "_convert_weight_to_int4pack(): K must be even");
    }
    if (N == 0 || K == 0) {
        return Tensor::empty({N, K / 2}, DType::UInt8, self.device());
    }
    Tensor codes = self.to(DType::Int32).contiguous();
    Tensor out = Tensor::empty({N, K / 2}, DType::UInt8, self.device());
    const int64_t num_bytes = N * (K / 2);
    constexpr int kThreads = 256;
    const unsigned blocks =
        static_cast<unsigned>((num_bytes + kThreads - 1) / kThreads);
    int4_pack_kernel<<<blocks, kThreads, 0, getCurrentCUDAStream().stream()>>>(
        static_cast<const int32_t*>(codes.data_ptr()),
        static_cast<uint8_t*>(out.data_ptr()), num_bytes);
    checkCuda(cudaGetLastError(), "int4 pack kernel");
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, QuantGemmKernels) {
    m.impl("_weight_int8pack_mm", _weight_int8pack_mm_cuda);
    m.impl("_weight_int4pack_mm", _weight_int4pack_mm_cuda);
    m.impl("_convert_weight_to_int4pack", _convert_weight_to_int4pack_cuda);
}

}  // namespace cuda
}  // namespace tensorplay

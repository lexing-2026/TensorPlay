#pragma once

// Sixteen-by-sixteen float transpose, the primitive behind every NCHW <->
// channel-blocked conversion.  A 16-channel block stores one pixel's channels
// contiguously, so turning that into per-channel planes (or back) is a plain
// matrix transpose once the work is cut into 16x16 tiles: one 64-byte line per
// row in, one per row out, with no read amplification.
//
// The tile is done as four 8x8 AVX2 transposes, which keeps the kernel in the
// baseline ISA of every translation unit that needs it.

#include <immintrin.h>
#include <cstdint>

namespace tensorplay {
namespace cpu {
namespace blk16 {

// dst[k][j] = src[j][k] for j, k in [0, 16), with rows spaced by the given
// element strides.  Use (16, spatial) for blocked -> NCHW and (spatial, 16)
// for NCHW -> blocked.
//
// NT=true writes the tile with non-temporal stores, which is the right choice
// whenever the destination is a fresh buffer the consumer will read from DRAM
// anyway: it drops the read-for-ownership traffic (a 25 MB plane costs 77 MB
// of bandwidth with plain stores and 51 MB without).  The caller must then
// guarantee 32-byte alignment on every destination row.
template <bool NT>
inline void transpose_tile(const float* src, int64_t src_row_stride,
                           float* dst, int64_t dst_row_stride) {
    auto store = [](float* p, __m256 v) {
        if constexpr (NT) _mm256_stream_ps(p, v);
        else _mm256_storeu_ps(p, v);
    };
    for (int h = 0; h < 2; ++h) {
        for (int p = 0; p < 2; ++p) {
            __m256 r0 = _mm256_loadu_ps(src + (p * 8 + 0) * src_row_stride + h * 8);
            __m256 r1 = _mm256_loadu_ps(src + (p * 8 + 1) * src_row_stride + h * 8);
            __m256 r2 = _mm256_loadu_ps(src + (p * 8 + 2) * src_row_stride + h * 8);
            __m256 r3 = _mm256_loadu_ps(src + (p * 8 + 3) * src_row_stride + h * 8);
            __m256 r4 = _mm256_loadu_ps(src + (p * 8 + 4) * src_row_stride + h * 8);
            __m256 r5 = _mm256_loadu_ps(src + (p * 8 + 5) * src_row_stride + h * 8);
            __m256 r6 = _mm256_loadu_ps(src + (p * 8 + 6) * src_row_stride + h * 8);
            __m256 r7 = _mm256_loadu_ps(src + (p * 8 + 7) * src_row_stride + h * 8);

            __m256 t0 = _mm256_unpacklo_ps(r0, r1);
            __m256 t1 = _mm256_unpackhi_ps(r0, r1);
            __m256 t2 = _mm256_unpacklo_ps(r2, r3);
            __m256 t3 = _mm256_unpackhi_ps(r2, r3);
            __m256 t4 = _mm256_unpacklo_ps(r4, r5);
            __m256 t5 = _mm256_unpackhi_ps(r4, r5);
            __m256 t6 = _mm256_unpacklo_ps(r6, r7);
            __m256 t7 = _mm256_unpackhi_ps(r6, r7);

            __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
            __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
            __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
            __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
            __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
            __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
            __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
            __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);

            float* out = dst + (h * 8) * dst_row_stride + p * 8;
            store(out + 0 * dst_row_stride, _mm256_permute2f128_ps(s0, s4, 0x20));
            store(out + 1 * dst_row_stride, _mm256_permute2f128_ps(s1, s5, 0x20));
            store(out + 2 * dst_row_stride, _mm256_permute2f128_ps(s2, s6, 0x20));
            store(out + 3 * dst_row_stride, _mm256_permute2f128_ps(s3, s7, 0x20));
            store(out + 4 * dst_row_stride, _mm256_permute2f128_ps(s0, s4, 0x31));
            store(out + 5 * dst_row_stride, _mm256_permute2f128_ps(s1, s5, 0x31));
            store(out + 6 * dst_row_stride, _mm256_permute2f128_ps(s2, s6, 0x31));
            store(out + 7 * dst_row_stride, _mm256_permute2f128_ps(s3, s7, 0x31));
        }
    }
}

// Whether a plane of `spatial` pixels can take the non-temporal path: every
// destination row then starts at a 32-byte boundary.
inline bool nt_eligible(const void* dst, int64_t spatial) {
    return (spatial % 8 == 0) && ((reinterpret_cast<uintptr_t>(dst) % 32) == 0);
}

// Converts `blocks` channel blocks of `spatial` pixels, stored as
// [block][pixel][16], into [channel][pixel] planes.  The tail of a plane whose
// pixel count is not a multiple of 16 is copied element by element.
inline void blocked_to_planes(const float* src, float* dst,
                              int64_t blocks, int64_t spatial, bool nt) {
    const bool fast = nt && nt_eligible(dst, spatial);
    for (int64_t b = 0; b < blocks; ++b) {
        const float* in = src + b * spatial * 16;
        float* out = dst + b * 16 * spatial;
        int64_t j = 0;
        if (fast) {
            for (; j + 16 <= spatial; j += 16)
                transpose_tile<true>(in + j * 16, 16, out + j, spatial);
            _mm_sfence();
        } else {
            for (; j + 16 <= spatial; j += 16)
                transpose_tile<false>(in + j * 16, 16, out + j, spatial);
        }
        for (; j < spatial; ++j) {
            for (int c = 0; c < 16; ++c) out[c * spatial + j] = in[j * 16 + c];
        }
    }
}

// The inverse: [channel][pixel] planes into [block][pixel][16] blocks.
inline void planes_to_blocks(const float* src, float* dst,
                             int64_t blocks, int64_t spatial, bool nt) {
    const bool fast = nt && nt_eligible(dst, spatial);
    for (int64_t b = 0; b < blocks; ++b) {
        const float* in = src + b * 16 * spatial;
        float* out = dst + b * spatial * 16;
        int64_t j = 0;
        if (fast) {
            for (; j + 16 <= spatial; j += 16)
                transpose_tile<true>(in + j, spatial, out + j * 16, 16);
            _mm_sfence();
        } else {
            for (; j + 16 <= spatial; j += 16)
                transpose_tile<false>(in + j, spatial, out + j * 16, 16);
        }
        for (; j < spatial; ++j) {
            for (int c = 0; c < 16; ++c) out[j * 16 + c] = in[c * spatial + j];
        }
    }
}

// [channel][pixel] planes into a pixel-major buffer whose pixels are
// `pixel_stride` floats wide and whose channel 16b..16b+15 belong to block b.
// This is the ordinary NCHW -> NHWC conversion (stride 16 only when the image
// happens to have exactly 16 channels).
inline void planes_to_pixels(const float* src, float* dst,
                             int64_t blocks, int64_t spatial,
                             int64_t pixel_stride, bool nt) {
    const bool fast = nt && (pixel_stride % 8 == 0) &&
                      ((reinterpret_cast<uintptr_t>(dst) % 32) == 0);
    for (int64_t b = 0; b < blocks; ++b) {
        const float* in = src + b * 16 * spatial;
        float* out = dst + b * 16;
        int64_t j = 0;
        if (fast) {
            for (; j + 16 <= spatial; j += 16)
                transpose_tile<true>(in + j, spatial, out + j * pixel_stride, pixel_stride);
            _mm_sfence();
        } else {
            for (; j + 16 <= spatial; j += 16)
                transpose_tile<false>(in + j, spatial, out + j * pixel_stride, pixel_stride);
        }
        for (; j < spatial; ++j) {
            for (int c = 0; c < 16; ++c)
                out[j * pixel_stride + c] = in[c * spatial + j];
        }
    }
}

// The whole 64-channel group of one image: a 16-pixel tile is assembled in a
// staging block and then written to the pixel-major destination as one dense
// 4 KB run.  Writing all four channel blocks of a pixel together is what makes
// the destination stores sequential, and a sequential destination is what
// makes non-temporal stores pay off: the same conversion then reads 25 MB and
// writes 25 MB instead of also reading the destination for ownership.
inline void planes_to_pixels64(const float* src, float* dst,
                               int64_t spatial, int64_t pixel_stride, bool nt) {
    const bool fast = nt && (pixel_stride % 8 == 0) &&
                      ((reinterpret_cast<uintptr_t>(dst) % 32) == 0);
    alignas(64) float tile[16 * 64];
    int64_t j = 0;
    for (; j + 16 <= spatial; j += 16) {
        for (int g = 0; g < 4; ++g)
            transpose_tile<false>(src + (g * 16) * spatial + j, spatial,
                                  tile + g * 16, 64);
        float* out = dst + j * pixel_stride;
        if (fast) {
            for (int p = 0; p < 16; ++p) {
                const float* t = tile + p * 64;
                for (int v = 0; v < 8; ++v)
                    _mm256_stream_ps(out + p * pixel_stride + v * 8,
                                     _mm256_loadu_ps(t + v * 8));
            }
            _mm_sfence();
        } else {
            for (int p = 0; p < 16; ++p) {
                const float* t = tile + p * 64;
                for (int v = 0; v < 8; ++v)
                    _mm256_storeu_ps(out + p * pixel_stride + v * 8,
                                     _mm256_loadu_ps(t + v * 8));
            }
        }
    }
    for (; j < spatial; ++j) {
        for (int c = 0; c < 64; ++c)
            dst[j * pixel_stride + c] = src[c * spatial + j];
    }
}

} // namespace blk16
} // namespace cpu
} // namespace tensorplay

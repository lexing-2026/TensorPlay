#pragma once

// WMMA primitive subset for the HIP toolchain, built on the wave32
// f32_16x16x16_f16 WMMA instruction of RDNA3 targets.  The public surface
// (fragment / load_matrix_sync / store_matrix_sync / fill_fragment /
// mma_sync) matches the CUDA-toolchain primitive API restricted to the
// shapes and layouts the attention kernels use:
//   fragment<matrix_a, 16, 16, 16, __half, row_major>
//   fragment<matrix_b, 16, 16, 16, __half, row_major | col_major>
//   fragment<accumulator, 16, 16, 16, float>
//
// Per-wave instruction semantics (one wave = 32 lanes computes one 16x16x16
// product D = A*B + C):
//   A operand: lane 2*m     slot k  -> A(m, k)   for m in [0, 8)   (upper half)
//              lane 2*m + 17 slot k  -> A(m, k)   for m in [8, 16)  (lower half)
//              remaining lanes are not read
//   B operand: lane n       slot k  -> B(k, n)   for n in [0, 16)
//              lane 16 + n  slot k  -> B(k, n)   (duplicate needed for the
//                                             lower result half)
//   D/C operand: lane n + 16*(m >= 8), slot m & 7 -> (m, n)
//
// The accumulator exposes its elements through x[] / num_elements in that
// instruction order; kernel code that walks fragment elements directly must
// use this mapping (a_lane/i helpers below) instead of any other layout.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <type_traits>

namespace nvcuda {
namespace wmma {

struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};

// Memory-order tags for accumulator load/store (enumerator values, matching
// the primitive API's usage as call-site arguments).
enum layout_t { mem_row_major, mem_col_major };

namespace wmma_rocm_detail {

using f16x16 = _Float16 __attribute__((ext_vector_type(16)));
using f32x8 = float __attribute__((ext_vector_type(8)));

__device__ inline f32x8 wmma_f32_f16(f16x16 a, f16x16 b, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// A-operand matrix row served by a lane; lanes the instruction never reads
// report -1.
__device__ inline int a_row_of_lane(int lane) {
    if (lane < 16 && (lane & 1) == 0) return lane / 2;
    if (lane >= 17 && (lane & 1) == 1) return (lane - 17) / 2 + 8;
    return -1;
}

// Coordinate of accumulator element i in lane l inside the 16x16 tile.
__device__ inline void acc_coord(int lane, int i, int* row, int* col) {
    *row = (i & 7) + 8 * (lane >= 16);
    *col = lane & 15;
}

}  // namespace wmma_rocm_detail

template <class Use, int M, int N, int K, class T, class Layout = void>
class fragment;

template <int M, int N, int K, class Layout>
class fragment<matrix_a, M, N, K, __half, Layout> {
 public:
    __device__ fragment() {
#pragma unroll
        for (int i = 0; i < 16; ++i) x[i] = (_Float16)0.0f;
    }
    wmma_rocm_detail::f16x16 x;
    static constexpr int num_elements = 16;
};

template <int M, int N, int K, class Layout>
class fragment<matrix_b, M, N, K, __half, Layout> {
 public:
    __device__ fragment() {
#pragma unroll
        for (int i = 0; i < 16; ++i) x[i] = (_Float16)0.0f;
    }
    wmma_rocm_detail::f16x16 x;
    static constexpr int num_elements = 16;
};

template <int M, int N, int K>
class fragment<accumulator, M, N, K, float, void> {
 public:
    __device__ fragment() {
#pragma unroll
        for (int i = 0; i < 8; ++i) x[i] = 0.f;
    }
    wmma_rocm_detail::f32x8 x;
    static constexpr int num_elements = 8;
};

// A-operand load; the fragment's Layout parameter selects the memory order.
template <int M, int N, int K, class Layout>
__device__ inline void load_matrix_sync(
    fragment<matrix_a, M, N, K, __half, Layout>& f,
    const __half* ptr, unsigned ldm) {
    const _Float16* p = reinterpret_cast<const _Float16*>(ptr);
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int m = wmma_rocm_detail::a_row_of_lane(lane);
    if constexpr (std::is_same<Layout, row_major>::value) {
#pragma unroll
        for (int k = 0; k < 16; ++k)
            f.x[k] = (m >= 0) ? p[m * ldm + k] : (_Float16)0.0f;
    } else {
#pragma unroll
        for (int k = 0; k < 16; ++k)
            f.x[k] = (m >= 0) ? p[m + k * ldm] : (_Float16)0.0f;
    }
}

// B-operand load; both lane halves must carry the same K x N tile because
// the instruction pairs the upper result rows with lanes 0..15 and the
// lower rows with lanes 16..31.
template <int M, int N, int K, class Layout>
__device__ inline void load_matrix_sync(
    fragment<matrix_b, M, N, K, __half, Layout>& f,
    const __half* ptr, unsigned ldm) {
    const _Float16* p = reinterpret_cast<const _Float16*>(ptr);
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int n = lane & 15;
    if constexpr (std::is_same<Layout, col_major>::value) {
#pragma unroll
        for (int k = 0; k < 16; ++k) f.x[k] = p[k + n * ldm];
    } else {
#pragma unroll
        for (int k = 0; k < 16; ++k) f.x[k] = p[k * ldm + n];
    }
}

// Accumulator load/store; the memory-order enumerator selects the layout.
template <int M, int N, int K>
__device__ inline void load_matrix_sync(
    fragment<accumulator, M, N, K, float, void>& f,
    const float* ptr, unsigned ldm, layout_t layout) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r, c;
        wmma_rocm_detail::acc_coord(lane, i, &r, &c);
        f.x[i] = (layout == mem_row_major) ? ptr[r * ldm + c]
                                            : ptr[r + c * ldm];
    }
}

template <int M, int N, int K>
__device__ inline void store_matrix_sync(
    float* ptr, const fragment<accumulator, M, N, K, float, void>& f,
    unsigned ldm, layout_t layout) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r, c;
        wmma_rocm_detail::acc_coord(lane, i, &r, &c);
        if (layout == mem_row_major) {
            ptr[r * ldm + c] = f.x[i];
        } else {
            ptr[r + c * ldm] = f.x[i];
        }
    }
}

template <int M, int N, int K>
__device__ inline void store_matrix_sync(
    __half* ptr, const fragment<accumulator, M, N, K, float, void>& f,
    unsigned ldm, layout_t layout) {
    _Float16* p = reinterpret_cast<_Float16*>(ptr);
    const int lane = static_cast<int>(threadIdx.x & 31u);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r, c;
        wmma_rocm_detail::acc_coord(lane, i, &r, &c);
        if (layout == mem_row_major) {
            p[r * ldm + c] = (_Float16)f.x[i];
        } else {
            p[r + c * ldm] = (_Float16)f.x[i];
        }
    }
}

template <int M, int N, int K>
__device__ inline void fill_fragment(
    fragment<accumulator, M, N, K, float, void>& f, float value) {
#pragma unroll
    for (int i = 0; i < 8; ++i) f.x[i] = value;
}

template <int M, int N, int K, class Layout>
__device__ inline void fill_fragment(
    fragment<matrix_a, M, N, K, __half, Layout>& f, __half value) {
    const _Float16 v = reinterpret_cast<const _Float16&>(value);
#pragma unroll
    for (int i = 0; i < 16; ++i) f.x[i] = v;
}

template <int M, int N, int K, class Layout>
__device__ inline void fill_fragment(
    fragment<matrix_b, M, N, K, __half, Layout>& f, __half value) {
    const _Float16 v = reinterpret_cast<const _Float16&>(value);
#pragma unroll
    for (int i = 0; i < 16; ++i) f.x[i] = v;
}

// d = a * b + c, one 16x16x16 product per wave.
template <int M, int N, int K, class LayoutA, class LayoutB>
__device__ inline void mma_sync(
    fragment<accumulator, M, N, K, float, void>& d,
    const fragment<matrix_a, M, N, K, __half, LayoutA>& a,
    const fragment<matrix_b, M, N, K, __half, LayoutB>& b,
    const fragment<accumulator, M, N, K, float, void>& c) {
    const wmma_rocm_detail::f32x8 r = wmma_rocm_detail::wmma_f32_f16(
        a.x, b.x, c.x);
#pragma unroll
    for (int i = 0; i < 8; ++i) d.x[i] = r[i];
}

}  // namespace wmma
}  // namespace nvcuda

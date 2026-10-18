#include <metal_array>
#include <metal_simdgroup>
#include <metal_stdlib>

#include "common.h"

using namespace metal;
using p10::metal::simdgroup_size;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ cumsum/cumprod ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Each thread owns one (outer, inner) slice and scans `dim_size` elements
// sequentially.  Consecutive elements along the scan dim are `inner` elements
// apart in the contiguous buffer.

template <typename T, typename AccT>
kernel void cumsum_kernel(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant int& dim_size [[buffer(2)]],
    constant int& inner [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int base = pos.y * dim_size * inner + pos.x;
  AccT acc = 0;
  for (int j = 0; j < dim_size; ++j) {
    acc += static_cast<AccT>(input[base + j * inner]);
    output[base + j * inner] = static_cast<T>(acc);
  }
}

template <typename T, typename AccT>
kernel void cumprod_kernel(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant int& dim_size [[buffer(2)]],
    constant int& inner [[buffer(3)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int base = pos.y * dim_size * inner + pos.x;
  AccT acc = 1;
  for (int j = 0; j < dim_size; ++j) {
    acc *= static_cast<AccT>(input[base + j * inner]);
    output[base + j * inner] = static_cast<T>(acc);
  }
}

// A scan down a row of consecutive elements, the shape an innermost-axis scan
// takes. One threadgroup takes one row: its threads own consecutive runs of
// kScanReads elements, scan their run, then fold the run totals across the
// threadgroup, so the row's serial chain is as long as its share of it. Rows
// longer than one pass carry the solved prefix into the next pass.
//
// The running value is `acc` -- a sum or a product, chosen at compile time --
// and `identity` is its neutral element.

P10_METAL_CONSTEXPR int kScanReads = 4;
using ScanSizes = metal::array<uint32_t, 2>;

template <typename AccT, bool Product>
inline AccT scan_combine(AccT a, AccT b) {
  return Product ? a * b : a + b;
}

template <typename AccT, bool Product>
inline AccT scan_identity() {
  return Product ? AccT(1) : AccT(0);
}

// Shuffles move 32-bit values; a 64-bit accumulator travels as the two halves
// of its bit pattern.
inline long simd_shuffle_and_fill_up(long data, long filling, ushort delta) {
  return as_type<long>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

// Inclusive scan of one value per lane, folded across the SIMD group. The
// shifting step fills the lanes below the window with the identity, so the
// result in lane j covers lanes 0..j only.
template <typename AccT, bool Product>
inline AccT scan_inclusive(AccT x) {
  for (int delta = 1; delta <= 16; delta *= 2) {
    const AccT other = simd_shuffle_and_fill_up(
        x, scan_identity<AccT, Product>(), static_cast<ushort>(delta));
    x = scan_combine<AccT, Product>(x, other);
  }
  return x;
}

template <typename AccT, bool Product>
inline AccT scan_exclusive(AccT x) {
  return simd_shuffle_and_fill_up(scan_inclusive<AccT, Product>(x),
                                  scan_identity<AccT, Product>(),
                                  static_cast<ushort>(1));
}

template <typename T, typename AccT, bool Product>
kernel void scan_rows_inner(
    device const T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant ScanSizes& sizes [[buffer(2)]],
    threadgroup AccT* partials [[threadgroup(0)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]]) {
  const uint len = sizes[1];
  device const T* src = in + (uint64_t)tgid * (uint64_t)len;
  device T* dst = out + (uint64_t)tgid * (uint64_t)len;
  const uint groups = tptg / simdgroup_size;
  const AccT identity = scan_identity<AccT, Product>();
  // The fold reads one entry per lane, so the lanes beyond the last group need
  // a value of their own; the identity leaves them out of every prefix.
  if (tid < simdgroup_size) {
    partials[tid] = identity;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  AccT carried = identity;

  for (uint base = 0; base < len; base += tptg * kScanReads) {
    const uint offset = base + tid * kScanReads;
    AccT values[kScanReads];
    for (int i = 0; i < kScanReads; ++i) {
      const uint at = offset + static_cast<uint>(i);
      values[i] = at < len ? static_cast<AccT>(src[at]) : identity;
    }
    // The thread's own run is scanned in registers; the run total joins the
    // threadgroup-wide scan.
    for (int i = 1; i < kScanReads; ++i) {
      values[i] = scan_combine<AccT, Product>(values[i - 1], values[i]);
    }
    const AccT prior = scan_exclusive<AccT, Product>(values[kScanReads - 1]);
    if (lane == simdgroup_size - 1) {
      partials[group] =
          scan_combine<AccT, Product>(prior, values[kScanReads - 1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0) {
      partials[lane] = scan_exclusive<AccT, Product>(partials[lane]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < kScanReads; ++i) {
      const uint at = offset + static_cast<uint>(i);
      if (at < len) {
        const AccT within_group =
            scan_combine<AccT, Product>(values[i], prior);
        const AccT within_pass =
            scan_combine<AccT, Product>(within_group, partials[group]);
        dst[at] = static_cast<T>(
            scan_combine<AccT, Product>(within_pass, carried));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The last thread holds the largest prefix of this pass; it becomes the
    // carry of the next one.
    if (group == groups - 1 && lane == simdgroup_size - 1) {
      const AccT within_pass = scan_combine<AccT, Product>(
          scan_combine<AccT, Product>(values[kScanReads - 1], prior),
          partials[group]);
      partials[0] = scan_combine<AccT, Product>(within_pass, carried);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    carried = partials[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ sort ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Bitonic sort.  One threadgroup sorts one slice of `sort_size` elements; the
// threadgroup size is the next power of two.  The slice's elements are
// `inner` apart in the contiguous buffer (slice index = threadgroup id).

template <typename T>
kernel void sort_kernel(
    device const T* input [[buffer(0)]],
    device T* values [[buffer(1)]],
    device long* indices [[buffer(2)]],
    constant int& sort_size [[buffer(3)]],
    constant int& inner [[buffer(4)]],
    constant int& descending [[buffer(5)]],
    constant T& sentinel [[buffer(6)]],
    threadgroup T* local_vals [[threadgroup(0)]],
    threadgroup long* local_idx [[threadgroup(1)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  const int base = (tgid / inner) * sort_size * inner + (tgid % inner);
  const int P = static_cast<int>(tptg);
  const bool desc = descending != 0;

  if (tid < sort_size) {
    local_vals[tid] = input[base + tid * inner];
    local_idx[tid] = static_cast<long>(tid);
  }
  for (int i = tid; i < P; i += tptg) {
    if (i >= sort_size) {
      local_vals[i] = sentinel;
      local_idx[i] = -1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 2; k <= P; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int ixj = tid ^ j;
      if (ixj > tid) {
        const bool up = (((tid & k) == 0) != desc);
        const T a = local_vals[tid];
        const T b = local_vals[ixj];
        const bool swap = up ? (a > b) : (a < b);
        if (swap) {
          const long tmp_i = local_idx[tid];
          local_vals[tid] = b;
          local_idx[tid] = local_idx[ixj];
          local_vals[ixj] = a;
          local_idx[ixj] = tmp_i;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  if (tid < sort_size) {
    values[base + tid * inner] = local_vals[tid];
    indices[base + tid * inner] = local_idx[tid];
  }
}

#define REGISTER_CUM_OPS(DTYPE, ACC_DTYPE, PREFIX)                        \
  template [[host_name("cumsum_" #PREFIX)]] kernel void                 \
  cumsum_kernel<DTYPE, ACC_DTYPE>(                                        \
      device const DTYPE*, device DTYPE*, constant int&, constant int&, \
      uint2);                                                            \
  template [[host_name("cumprod_" #PREFIX)]] kernel void                 \
  cumprod_kernel<DTYPE, ACC_DTYPE>(                                       \
      device const DTYPE*, device DTYPE*, constant int&, constant int&,  \
      uint2);

REGISTER_CUM_OPS(float, float, float)
REGISTER_CUM_OPS(half, float, half)
REGISTER_CUM_OPS(bfloat, float, bfloat)
REGISTER_CUM_OPS(int, int, int)
REGISTER_CUM_OPS(long, long, long)
REGISTER_CUM_OPS(short, int, short)
REGISTER_CUM_OPS(char, int, char)
REGISTER_CUM_OPS(uchar, uint, uchar)
REGISTER_CUM_OPS(bool, int, bool)

#define REGISTER_SCAN_ROWS(DTYPE, ACC_DTYPE, PREFIX)                          \
  template [[host_name("scan_rows_sum_" #PREFIX)]] kernel void                \
  scan_rows_inner<DTYPE, ACC_DTYPE, false>(                                   \
      device const DTYPE*, device DTYPE*, constant ScanSizes&,                \
      threadgroup ACC_DTYPE*, uint, uint, uint, uint, uint);                  \
  template [[host_name("scan_rows_prod_" #PREFIX)]] kernel void               \
  scan_rows_inner<DTYPE, ACC_DTYPE, true>(                                    \
      device const DTYPE*, device DTYPE*, constant ScanSizes&,                \
      threadgroup ACC_DTYPE*, uint, uint, uint, uint, uint);

REGISTER_SCAN_ROWS(float, float, float)
REGISTER_SCAN_ROWS(half, float, half)
REGISTER_SCAN_ROWS(bfloat, float, bfloat)
REGISTER_SCAN_ROWS(int, int, int)
REGISTER_SCAN_ROWS(long, long, long)
REGISTER_SCAN_ROWS(short, int, short)
REGISTER_SCAN_ROWS(char, int, char)
REGISTER_SCAN_ROWS(uchar, uint, uchar)
REGISTER_SCAN_ROWS(bool, int, bool)

#define REGISTER_SORT(DTYPE)                                               \
  template [[host_name("sort_" #DTYPE)]] kernel void sort_kernel<DTYPE>(   \
      device const DTYPE*, device DTYPE*, device long*, constant int&, \
      constant int&, constant int&, constant DTYPE&, threadgroup DTYPE*, \
      threadgroup long*, uint, uint, uint);

REGISTER_SORT(float)
REGISTER_SORT(half)
REGISTER_SORT(bfloat)
REGISTER_SORT(int)
REGISTER_SORT(long)
REGISTER_SORT(short)
REGISTER_SORT(char)
REGISTER_SORT(uchar)
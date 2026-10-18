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
// One threadgroup sorts one slice. Each thread keeps four consecutive elements
// in registers; a SIMD-group bitonic network orders 128 elements without a
// barrier, then pairwise merges of ordered runs double the span inside
// threadgroup memory until the slice is whole. The slice's elements are
// `inner` apart in the contiguous buffer (slice index = threadgroup id).

template <typename T>
inline bool sort_less(T a, T b, bool desc) {
  // NaN takes the extreme position so the order stays a total order: last
  // ascending, first descending. Without it equal-pair decisions disagree
  // across lanes and the same position can be emitted twice.
  if IF_CONSTEXPR (metal::is_floating_point_v<T>) {
    const bool a_nan = a != a;
    const bool b_nan = b != b;
    if (a_nan || b_nan) {
      return desc ? (a_nan && !b_nan) : (b_nan && !a_nan);
    }
  }
  return desc ? (a > b) : (a < b);
}

// Total order over (value, original position). Positions are unique, so equal
// values stay deterministic, and padding (the largest position) follows every
// real element that shares its value.
template <typename T>
inline bool sort_before(T va, uint ia, T vb, uint ib, bool desc) {
  if (sort_less(va, vb, desc)) {
    return true;
  }
  if (sort_less(vb, va, desc)) {
    return false;
  }
  return ia < ib;
}

template <typename T, metal::enable_if_t<sizeof(T) == 1, bool> = true>
inline T sort_shuffle(T v, ushort delta) {
  return as_type<T>(uchar(simd_shuffle_xor(uint(as_type<uchar>(v)), delta)));
}

template <typename T, metal::enable_if_t<sizeof(T) == 2, bool> = true>
inline T sort_shuffle(T v, ushort delta) {
  return as_type<T>(ushort(simd_shuffle_xor(uint(as_type<ushort>(v)), delta)));
}

template <typename T, metal::enable_if_t<sizeof(T) == 4, bool> = true>
inline T sort_shuffle(T v, ushort delta) {
  return as_type<T>(simd_shuffle_xor(as_type<uint>(v), delta));
}

template <typename T, metal::enable_if_t<sizeof(T) == 8, bool> = true>
inline T sort_shuffle(T v, ushort delta) {
  const ulong u = as_type<ulong>(v);
  const uint lo = simd_shuffle_xor(uint(u), delta);
  const uint hi = simd_shuffle_xor(uint(u >> 32), delta);
  return as_type<T>(ulong(lo) | (ulong(hi) << 32));
}

inline uint sort_shuffle_index(uint v, ushort delta) {
  return simd_shuffle_xor(v, delta);
}

// Exchange partner elements held by the same thread.
template <typename T, short TN, int K, int OFFSET>
inline void sort_substage_reg(thread T (&v)[TN], thread uint (&idx)[TN],
                              uint lane, bool desc) {
  const uint pbase = lane * TN;
#pragma unroll
  for (short i = 0; i < TN; ++i) {
    const short pi = i ^ OFFSET;
    if (pi > i) {
      const bool ascending = (((pbase + i) & K) == 0);
      const bool before = sort_before(v[i], idx[i], v[pi], idx[pi], desc);
      const bool swap = ascending ? !before : before;
      const T tv = v[i];
      const uint ti = idx[i];
      v[i] = swap ? v[pi] : v[i];
      idx[i] = swap ? idx[pi] : idx[i];
      v[pi] = swap ? tv : v[pi];
      idx[pi] = swap ? ti : idx[pi];
    }
  }
}

// Exchange partner elements held by another lane of the SIMD group.
template <typename T, short TN, int K, int OFFSET>
inline void sort_substage_simd(thread T (&v)[TN], thread uint (&idx)[TN],
                               uint lane, bool desc) {
  constexpr ushort kLaneOffset = OFFSET / TN;
  const bool low = (lane & uint(kLaneOffset)) == 0;
  const bool ascending = (((lane * TN) & K) == 0);
#pragma unroll
  for (short i = 0; i < TN; ++i) {
    const T vp = sort_shuffle(v[i], kLaneOffset);
    const uint ip = sort_shuffle_index(idx[i], kLaneOffset);
    const bool first = sort_before(v[i], idx[i], vp, ip, desc);
    const bool take = first != (ascending == low);
    v[i] = take ? vp : v[i];
    idx[i] = take ? ip : idx[i];
  }
}

// Sort 32 lanes of TN elements: a bitonic network over 128 positions whose
// cross-lane stages are register exchanges inside the SIMD group.
template <typename T, short TN>
inline void sort_simd_group(thread T (&v)[TN], thread uint (&idx)[TN],
                            uint lane, bool desc) {
  sort_substage_reg<T, TN, 2, 1>(v, idx, lane, desc);

  sort_substage_reg<T, TN, 4, 2>(v, idx, lane, desc);
  sort_substage_reg<T, TN, 4, 1>(v, idx, lane, desc);

  sort_substage_simd<T, TN, 8, 4>(v, idx, lane, desc);
  sort_substage_reg<T, TN, 8, 2>(v, idx, lane, desc);
  sort_substage_reg<T, TN, 8, 1>(v, idx, lane, desc);

  sort_substage_simd<T, TN, 16, 8>(v, idx, lane, desc);
  sort_substage_simd<T, TN, 16, 4>(v, idx, lane, desc);
  sort_substage_reg<T, TN, 16, 2>(v, idx, lane, desc);
  sort_substage_reg<T, TN, 16, 1>(v, idx, lane, desc);

  sort_substage_simd<T, TN, 32, 16>(v, idx, lane, desc);
  sort_substage_simd<T, TN, 32, 8>(v, idx, lane, desc);
  sort_substage_simd<T, TN, 32, 4>(v, idx, lane, desc);
  sort_substage_reg<T, TN, 32, 2>(v, idx, lane, desc);
  sort_substage_reg<T, TN, 32, 1>(v, idx, lane, desc);

  sort_substage_simd<T, TN, 64, 32>(v, idx, lane, desc);
  sort_substage_simd<T, TN, 64, 16>(v, idx, lane, desc);
  sort_substage_simd<T, TN, 64, 8>(v, idx, lane, desc);
  sort_substage_simd<T, TN, 64, 4>(v, idx, lane, desc);
  sort_substage_reg<T, TN, 64, 2>(v, idx, lane, desc);
  sort_substage_reg<T, TN, 64, 1>(v, idx, lane, desc);

  sort_substage_simd<T, TN, 128, 64>(v, idx, lane, desc);
  sort_substage_simd<T, TN, 128, 32>(v, idx, lane, desc);
  sort_substage_simd<T, TN, 128, 16>(v, idx, lane, desc);
  sort_substage_simd<T, TN, 128, 8>(v, idx, lane, desc);
  sort_substage_simd<T, TN, 128, 4>(v, idx, lane, desc);
  sort_substage_reg<T, TN, 128, 2>(v, idx, lane, desc);
  sort_substage_reg<T, TN, 128, 1>(v, idx, lane, desc);
}

// Number of A elements among the first `diag` outputs of merging two ordered
// runs of `asz` and `bsz` elements.
template <typename T>
inline int sort_partition(threadgroup const T* A, threadgroup const uint* ai,
                          threadgroup const T* B, threadgroup const uint* bi,
                          int asz, int bsz, int diag, bool desc) {
  int lo = max(0, diag - bsz);
  int hi = min(diag, asz);
  while (lo < hi) {
    const int m = lo + (hi - lo) / 2;
    if (sort_before(B[diag - 1 - m], bi[diag - 1 - m], A[m], ai[m], desc)) {
      hi = m;
    } else {
      lo = m + 1;
    }
  }
  return lo;
}

// Merges TN consecutive outputs starting at the run-local offsets (a0, b0).
template <typename T, short TN>
inline void sort_merge_step(threadgroup const T* A, threadgroup const uint* ai,
                            threadgroup const T* B, threadgroup const uint* bi,
                            int asz, int bsz, int a0, int b0,
                            thread T (&v)[TN], thread uint (&idx)[TN],
                            bool desc) {
  int a = a0;
  int b = b0;
#pragma unroll
  for (short i = 0; i < TN; ++i) {
    const bool take_b =
        b < bsz && (a >= asz || sort_before(B[b], bi[b], A[a], ai[a], desc));
    v[i] = take_b ? B[min(b, bsz - 1)] : A[min(a, asz - 1)];
    idx[i] = take_b ? bi[min(b, bsz - 1)] : ai[min(a, asz - 1)];
    a += take_b ? 0 : 1;
    b += take_b ? 1 : 0;
  }
}

// Merges the two halves of every `MT * TN`-element group in threadgroup
// memory; each thread emits its TN outputs into registers.
template <typename T, short TN, int MT>
inline void sort_merge_pass(threadgroup T* tv, threadgroup uint* ti, uint lid,
                            thread T (&v)[TN], thread uint (&idx)[TN],
                            bool desc) {
  constexpr int kRun = MT * TN;
  const uint grp = lid / uint(MT);
  const uint lane = lid % uint(MT);
  const int st = kRun * int(grp);
  constexpr int kHalf = kRun / 2;
  const int diag = TN * int(lane);
  const int p = sort_partition(tv + st, ti + st, tv + st + kHalf,
                               ti + st + kHalf, kHalf, kHalf, diag, desc);
  sort_merge_step<T, TN>(tv + st, ti + st, tv + st + kHalf, ti + st + kHalf,
                         kHalf, kHalf, p, diag - p, v, idx, desc);
}

template <typename T, short TPTG, short TN>
kernel void sort_merge_kernel(
    device const T* input [[buffer(0)]],
    device T* values [[buffer(1)]],
    device long* indices [[buffer(2)]],
    constant int& sort_size [[buffer(3)]],
    constant int& inner [[buffer(4)]],
    constant int& descending [[buffer(5)]],
    constant T& sentinel [[buffer(6)]],
    threadgroup T* local_vals [[threadgroup(0)]],
    threadgroup uint* local_idx [[threadgroup(1)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  const int base = (tgid / inner) * sort_size * inner + (tgid % inner);
  const bool desc = descending != 0;
  const uint pos0 = tid * TN;

  thread T v[TN];
  thread uint idx[TN];
#pragma unroll
  for (short i = 0; i < TN; ++i) {
    const int p = int(pos0) + i;
    v[i] = p < sort_size ? input[base + min(p, sort_size - 1) * inner]
                         : sentinel;
    idx[i] = p < sort_size ? uint(p) : 0xFFFFFFFFu;
  }

  sort_simd_group<T, TN>(v, idx, tid & 31u, desc);

  if (TPTG >= 64) {
#pragma unroll
    for (short i = 0; i < TN; ++i) {
      local_vals[pos0 + i] = v[i];
      local_idx[pos0 + i] = idx[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sort_merge_pass<T, TN, 64>(local_vals, local_idx, tid, v, idx, desc);
  }
  if (TPTG >= 128) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (short i = 0; i < TN; ++i) {
      local_vals[pos0 + i] = v[i];
      local_idx[pos0 + i] = idx[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sort_merge_pass<T, TN, 128>(local_vals, local_idx, tid, v, idx, desc);
  }
  if (TPTG >= 256) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (short i = 0; i < TN; ++i) {
      local_vals[pos0 + i] = v[i];
      local_idx[pos0 + i] = idx[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sort_merge_pass<T, TN, 256>(local_vals, local_idx, tid, v, idx, desc);
  }

#pragma unroll
  for (short i = 0; i < TN; ++i) {
    const int p = int(pos0) + i;
    if (p < sort_size) {
      values[base + p * inner] = v[i];
      indices[base + p * inner] = static_cast<long>(idx[i]);
    }
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

#define REGISTER_SORT_TPTG(DTYPE, TPTG)                                     \
  template [[host_name("sort_merge_" #DTYPE "_tptg" #TPTG)]] kernel void    \
  sort_merge_kernel<DTYPE, TPTG, 4>(                                        \
      device const DTYPE*, device DTYPE*, device long*, constant int&,      \
      constant int&, constant int&, constant DTYPE&, threadgroup DTYPE*,    \
      threadgroup uint*, uint, uint);

#define REGISTER_SORT(DTYPE)    \
  REGISTER_SORT_TPTG(DTYPE, 32) \
  REGISTER_SORT_TPTG(DTYPE, 64) \
  REGISTER_SORT_TPTG(DTYPE, 128) \
  REGISTER_SORT_TPTG(DTYPE, 256)

REGISTER_SORT(float)
REGISTER_SORT(half)
REGISTER_SORT(bfloat)
REGISTER_SORT(int)
REGISTER_SORT(long)
REGISTER_SORT(short)
REGISTER_SORT(char)
REGISTER_SORT(uchar)
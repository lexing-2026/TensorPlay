#include <metal_stdlib>

using namespace metal;

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
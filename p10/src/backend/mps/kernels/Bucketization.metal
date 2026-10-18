#include <metal_stdlib>
using namespace metal;

// Boundary lookups for searchsorted and bucketize.

template <typename input_t>
int64_t lower_bound(
    constant input_t* data_ss,
    int64_t start,
    int64_t end,
    const input_t val,
    constant int64_t* data_sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add
  // the non-updated start as an offset i.e. the second row of a 3x3 tensors
  // starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_ss[orig_start + data_sort[mid]];
    if (!(mid_val >= val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename input_t>
int64_t lower_bound(
    constant input_t* data_ss,
    int64_t start,
    int64_t end,
    const input_t val) {
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_ss[mid];
    if (!(mid_val >= val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename input_t>
int64_t upper_bound(
    constant input_t* data_ss,
    int64_t start,
    int64_t end,
    const input_t val,
    constant int64_t* data_sort) {
  // sorter gives relative ordering for ND tensors, so we need to save and add
  // the non-updated start as an offset i.e. the second row of a 3x3 tensors
  // starts at element 3 but sorter's second row only contains 0, 1, or 2
  const int64_t orig_start = start;
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_ss[orig_start + data_sort[mid]];
    if (!(mid_val > val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename input_t>
int64_t upper_bound(
    constant input_t* data_ss,
    int64_t start,
    int64_t end,
    const input_t val) {
  while (start < end) {
    const int64_t mid = start + ((end - start) >> 1);
    const input_t mid_val = data_ss[mid];
    if (!(mid_val > val)) {
      start = mid + 1;
    } else {
      end = mid;
    }
  }
  return start;
}

template <typename input_t, typename output_t>
kernel void searchsorted_sorter(
    constant input_t* data_in [[buffer(0)]],
    constant input_t* data_bd [[buffer(1)]],
    device output_t* data_out [[buffer(2)]],
    constant int64_t& idim_in [[buffer(3)]],
    constant int64_t& idim_bd [[buffer(4)]],
    constant int64_t& numel_in [[buffer(5)]],
    constant int64_t& right [[buffer(6)]],
    constant int64_t& is_1d_boundaries [[buffer(7)]],
    constant int64_t* data_sort [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpg [[threadgroups_per_grid]]) {
  for (int64_t tid = tgid.x * tptg.x + tid2.x; tid < numel_in;
       tid += tptg.x * tgpg.x) {
    // If boundaries tensor is 1d, we always search the entire boundary tensor
    int64_t start_bd = is_1d_boundaries ? 0 : tid / idim_in * idim_bd;
    int64_t end_bd = start_bd + idim_bd;

    int64_t pos = !right
        ? lower_bound<input_t>(
              data_bd, start_bd, end_bd, data_in[tid], data_sort) -
            start_bd
        : upper_bound<input_t>(
              data_bd, start_bd, end_bd, data_in[tid], data_sort) -
            start_bd;

    // type conversion might happen here
    data_out[tid] = pos;
  }
}

template <typename input_t, typename output_t>
kernel void searchsorted(
    constant input_t* data_in [[buffer(0)]],
    constant input_t* data_bd [[buffer(1)]],
    device output_t* data_out [[buffer(2)]],
    constant int64_t& idim_in [[buffer(3)]],
    constant int64_t& idim_bd [[buffer(4)]],
    constant int64_t& numel_in [[buffer(5)]],
    constant int64_t& right [[buffer(6)]],
    constant int64_t& is_1d_boundaries [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 tptg [[threads_per_threadgroup]],
    uint2 tgpg [[threadgroups_per_grid]]) {
  for (int64_t tid = tgid.x * tptg.x + tid2.x; tid < numel_in;
       tid += tptg.x * tgpg.x) {
    // If boundaries tensor is 1d, we always search the entire boundary tensor
    int64_t start_bd = is_1d_boundaries ? 0 : tid / idim_in * idim_bd;
    int64_t end_bd = start_bd + idim_bd;

    int64_t pos = !right
        ? lower_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid]) -
            start_bd
        : upper_bound<input_t>(data_bd, start_bd, end_bd, data_in[tid]) -
            start_bd;

    // type conversion might happen here
    data_out[tid] = pos;
  }
}

// A single boundary run of power-of-two length lets every search take the same
// number of steps, each probe fixing one bit of the position. Four queries
// share a thread and one step loop, so their probes are independent loads in
// flight together.
template <typename input_t, typename output_t, int kRight>
kernel void searchsorted_pow2(
    const device input_t* data_in [[buffer(0)]],
    const device input_t* data_bd [[buffer(1)]],
    device output_t* data_out [[buffer(2)]],
    constant uint& idim_bd [[buffer(3)]],
    constant uint& numel_in [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  const uint last = numel_in - 1;
  const uint base = tid * 4;
  const bool a0 = base <= last;
  const bool a1 = base + 1 <= last;
  const bool a2 = base + 2 <= last;
  const bool a3 = base + 3 <= last;
  const input_t v0 = data_in[min(base, last)];
  const input_t v1 = data_in[min(base + 1, last)];
  const input_t v2 = data_in[min(base + 2, last)];
  const input_t v3 = data_in[min(base + 3, last)];

  uint p0 = 0;
  uint p1 = 0;
  uint p2 = 0;
  uint p3 = 0;
  for (uint step = idim_bd >> 1; step != 0; step >>= 1) {
    const input_t b0 = data_bd[p0 + step - 1];
    const input_t b1 = data_bd[p1 + step - 1];
    const input_t b2 = data_bd[p2 + step - 1];
    const input_t b3 = data_bd[p3 + step - 1];
    const bool g0 = kRight != 0 ? !(b0 > v0) : !(b0 >= v0);
    const bool g1 = kRight != 0 ? !(b1 > v1) : !(b1 >= v1);
    const bool g2 = kRight != 0 ? !(b2 > v2) : !(b2 >= v2);
    const bool g3 = kRight != 0 ? !(b3 > v3) : !(b3 >= v3);
    p0 += g0 ? step : 0;
    p1 += g1 ? step : 0;
    p2 += g2 ? step : 0;
    p3 += g3 ? step : 0;
  }
  const input_t e0 = data_bd[p0];
  const input_t e1 = data_bd[p1];
  const input_t e2 = data_bd[p2];
  const input_t e3 = data_bd[p3];
  const uint r0 = p0 + ((kRight != 0 ? !(e0 > v0) : !(e0 >= v0)) ? 1 : 0);
  const uint r1 = p1 + ((kRight != 0 ? !(e1 > v1) : !(e1 >= v1)) ? 1 : 0);
  const uint r2 = p2 + ((kRight != 0 ? !(e2 > v2) : !(e2 >= v2)) ? 1 : 0);
  const uint r3 = p3 + ((kRight != 0 ? !(e3 > v3) : !(e3 >= v3)) ? 1 : 0);
  if (a0) {
    data_out[base] = static_cast<output_t>(r0);
  }
  if (a1) {
    data_out[base + 1] = static_cast<output_t>(r1);
  }
  if (a2) {
    data_out[base + 2] = static_cast<output_t>(r2);
  }
  if (a3) {
    data_out[base + 3] = static_cast<output_t>(r3);
  }
}

#define REGISTER_SEARCHSORTED_OP(INPUT_T, OUTPUT_T)                          \
  template [[host_name("searchsorted_" #INPUT_T "_" #OUTPUT_T                \
                       "_sorter")]] kernel void                              \
  searchsorted_sorter<INPUT_T, OUTPUT_T>(                                    \
      constant INPUT_T * data_in [[buffer(0)]],                              \
      constant INPUT_T * data_bd [[buffer(1)]],                              \
      device OUTPUT_T * data_out [[buffer(2)]],                              \
      constant int64_t& idim_in [[buffer(3)]],                               \
      constant int64_t& idim_bd [[buffer(4)]],                               \
      constant int64_t& numel_in [[buffer(5)]],                              \
      constant int64_t& right [[buffer(6)]],                                 \
      constant int64_t& is_1d_boundaries [[buffer(7)]],                      \
      constant int64_t* data_sort [[buffer(8)]],                             \
      uint2 tgid [[threadgroup_position_in_grid]],                           \
      uint2 tid2 [[thread_position_in_threadgroup]],                         \
      uint2 tptg [[threads_per_threadgroup]],                                \
      uint2 tgpg [[threadgroups_per_grid]]);                                 \
  template [[host_name("searchsorted_" #INPUT_T "_" #OUTPUT_T)]] kernel void \
  searchsorted<INPUT_T, OUTPUT_T>(                                           \
      constant INPUT_T * data_in [[buffer(0)]],                              \
      constant INPUT_T * data_bd [[buffer(1)]],                              \
      device OUTPUT_T * data_out [[buffer(2)]],                              \
      constant int64_t& idim_in [[buffer(3)]],                               \
      constant int64_t& idim_bd [[buffer(4)]],                               \
      constant int64_t& numel_in [[buffer(5)]],                              \
      constant int64_t& right [[buffer(6)]],                                 \
      constant int64_t& is_1d_boundaries [[buffer(7)]],                      \
      uint2 tgid [[threadgroup_position_in_grid]],                           \
      uint2 tid2 [[thread_position_in_threadgroup]],                         \
      uint2 tptg [[threads_per_threadgroup]],                                \
      uint2 tgpg [[threadgroups_per_grid]]);

REGISTER_SEARCHSORTED_OP(float, int);
REGISTER_SEARCHSORTED_OP(float, long);
REGISTER_SEARCHSORTED_OP(half, int);
REGISTER_SEARCHSORTED_OP(half, long);
REGISTER_SEARCHSORTED_OP(bfloat, int);
REGISTER_SEARCHSORTED_OP(bfloat, long);
REGISTER_SEARCHSORTED_OP(char, int);
REGISTER_SEARCHSORTED_OP(char, long);
REGISTER_SEARCHSORTED_OP(uchar, int);
REGISTER_SEARCHSORTED_OP(uchar, long);
REGISTER_SEARCHSORTED_OP(short, int);
REGISTER_SEARCHSORTED_OP(short, long);
REGISTER_SEARCHSORTED_OP(int, int);
REGISTER_SEARCHSORTED_OP(int, long);
REGISTER_SEARCHSORTED_OP(long, int);
REGISTER_SEARCHSORTED_OP(long, long);

#define REGISTER_SEARCHSORTED_POW2(INPUT_T, OUTPUT_T)                        \
  template [[host_name("searchsorted_" #INPUT_T "_" #OUTPUT_T                \
                       "_pow2_left")]] kernel void                            \
  searchsorted_pow2<INPUT_T, OUTPUT_T, 0>(                                   \
      const device INPUT_T * data_in [[buffer(0)]],                          \
      const device INPUT_T * data_bd [[buffer(1)]],                          \
      device OUTPUT_T * data_out [[buffer(2)]],                              \
      constant uint & idim_bd [[buffer(3)]],                                 \
      constant uint & numel_in [[buffer(4)]],                                \
      uint tid [[thread_position_in_grid]]);                                 \
  template [[host_name("searchsorted_" #INPUT_T "_" #OUTPUT_T                \
                       "_pow2_right")]] kernel void                           \
  searchsorted_pow2<INPUT_T, OUTPUT_T, 1>(                                   \
      const device INPUT_T * data_in [[buffer(0)]],                          \
      const device INPUT_T * data_bd [[buffer(1)]],                          \
      device OUTPUT_T * data_out [[buffer(2)]],                              \
      constant uint & idim_bd [[buffer(3)]],                                 \
      constant uint & numel_in [[buffer(4)]],                                \
      uint tid [[thread_position_in_grid]]);

REGISTER_SEARCHSORTED_POW2(float, int);
REGISTER_SEARCHSORTED_POW2(float, long);
REGISTER_SEARCHSORTED_POW2(half, int);
REGISTER_SEARCHSORTED_POW2(half, long);
REGISTER_SEARCHSORTED_POW2(bfloat, int);
REGISTER_SEARCHSORTED_POW2(bfloat, long);
REGISTER_SEARCHSORTED_POW2(char, int);
REGISTER_SEARCHSORTED_POW2(char, long);
REGISTER_SEARCHSORTED_POW2(uchar, int);
REGISTER_SEARCHSORTED_POW2(uchar, long);
REGISTER_SEARCHSORTED_POW2(short, int);
REGISTER_SEARCHSORTED_POW2(short, long);
REGISTER_SEARCHSORTED_POW2(int, int);
REGISTER_SEARCHSORTED_POW2(int, long);
REGISTER_SEARCHSORTED_POW2(long, int);
REGISTER_SEARCHSORTED_POW2(long, long);

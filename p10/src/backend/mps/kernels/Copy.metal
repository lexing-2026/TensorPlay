// Contiguous-input block copy into a regularly strided destination.
//
// The input is a contiguous run of bytes. Consecutive `slice_bytes` runs land
// at `off_bytes + k * out_stride_bytes` in the destination; `base` is the
// offset of the chunk this launch covers, and `nbytes` its length. The copy
// unit is 16 bytes and the tail (or a run that would cross a slice boundary)
// falls back to per-byte stores.

#include <metal_stdlib>
#include "Copy.h"
#include "utils.h"

using namespace metal;
using namespace p10::metal;

kernel void strided_block_copy(
    constant uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant StridedBlockCopyParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  const uint pos = tid * 16u;
  if (pos >= params.nbytes) {
    return;
  }
  const ulong g = params.base + pos;
  const uint run = min(16u, params.nbytes - pos);
  // The whole input is one slice: the destination is a single run.
  if (params.flat != 0) {
    copy_bytes_aligned(output + params.off_bytes + g, input + g, run);
    return;
  }
  const ulong o = g / params.slice_bytes;
  const ulong j = g - o * params.slice_bytes;
  device uchar* dst =
      output + o * params.out_stride_bytes + params.off_bytes + j;
  constant uchar* src = input + g;
  if (params.slice_bytes - j >= run) {
    copy_bytes_aligned(dst, src, run);
    return;
  }
  for (uint i = 0; i < run; ++i) {
    const ulong gg = g + i;
    const ulong oo = gg / params.slice_bytes;
    output[oo * params.out_stride_bytes + params.off_bytes +
           (gg - oo * params.slice_bytes)] = input[gg];
  }
}

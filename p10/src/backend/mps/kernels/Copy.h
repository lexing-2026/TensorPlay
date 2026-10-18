#pragma once
// The Metal Standard Library provides the fixed-width integer types; the host
// translation unit gets them from the standard header.
#ifndef __METAL__
#include <cstdint>
#endif

#include "common.h"

// Byte geometry for one launch of strided_block_copy: consecutive
// `slice_bytes` runs of the contiguous input are placed at
// `off_bytes + k * out_stride_bytes` in the destination. `base` and `nbytes`
// select the chunk the launch covers so a copy larger than the 32-bit length
// field is split across launches.
struct StridedBlockCopyParams {
  uint64_t base;
  uint64_t slice_bytes;
  uint64_t out_stride_bytes;
  uint64_t off_bytes;
  uint32_t nbytes;
  uint32_t flat;
};

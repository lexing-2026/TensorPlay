#pragma once
// Tier-compiled channels-last kernels for grid_sampler_2d (bilinear, nearest
// and bicubic interpolation).
//
// Input layout: channels-last (N, C, H, W); grid: standard contiguous
// (N, oH, oW, 2) with dtype matching the input; output: channels-last
// (N, C, oH, oW). Channel is the vectorized dimension: the tap pixels of one
// output pixel are loaded for a whole block of channels with one vector load
// each, scaled by the tap weight and accumulated over the block.
//
// The per-pixel coordinate/weight section is the scalar frame's own
// arithmetic (same helpers, same order and rounding), so the two frames are
// bit-identical by construction; no vector coordinate math is involved.
//
// The stub switches interpolation/padding at run time inside each tier
// binary; align_corners is a run-time bool (one branch per output pixel).

#include "DispatchStub.h"

namespace tensorplay {
namespace cpu {

using grid_sample2d_cl_fn = void (*)(const void* in, const void* grid, void* out,
                                     int64_t N, int64_t C, int64_t H, int64_t W,
                                     int64_t oH, int64_t oW,
                                     int interpolation_mode, int padding_mode,
                                     int align_corners, int dtype);

DECLARE_DISPATCH(grid_sample2d_cl_fn, grid_sample2d_cl_stub)

// Backward variant. grad_output and input are channels-last; grad_input is
// written channels-last (pre-zeroed by the caller); grad_grid is standard
// contiguous (N, oH, oW, 2). need_grad_input / need_grad_grid gate which
// outputs are produced.
using grid_sample2d_backward_cl_fn = void (*)(const void* grad_output, const void* in,
                                              const void* grid, void* grad_input,
                                              void* grad_grid,
                                              int64_t N, int64_t C, int64_t H, int64_t W,
                                              int64_t oH, int64_t oW,
                                              int interpolation_mode, int padding_mode,
                                              int align_corners, int dtype,
                                              int need_grad_input, int need_grad_grid);

DECLARE_DISPATCH(grid_sample2d_backward_cl_fn, grid_sample2d_backward_cl_stub)

}  // namespace cpu
}  // namespace tensorplay

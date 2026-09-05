#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Int8 variant of the single-axis slice: signed-byte textures in and out,
// one byte per channel lane.
layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the output, for bounds checks
  ivec4 out_sizes;
  int axis; // sliced axis, counted from the innermost (0: W, 1: H, 2: C, 3: N)
  int start;
  int step;
  // 1 when the sliced axis is dropped (a select) instead of kept
  int removed;
  int in_c_depth; // ceil(in C / 4)
  int out_c_depth; // ceil(out C / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Strided slice along one axis over Int8 storage, addressed in
 * innermost-first axis order (0: width, 1: height, 2: channel, 3: batch).
 * The invocation maps its output coordinates back to input coordinates: the
 * sliced axis moves by start + step * position, a dropped axis pins to
 * `start` and shifts all outer axes inward by one slot.  Elements are
 * copied per channel lane so slicing the channel axis reroutes lanes and z
 * positions correctly.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  const int n_out = pos.z / uBlock.out_c_depth;
  const int c4 = pos.z % uBlock.out_c_depth;

  ivec4 out_texel;
  for (int i = 0; i < 4; ++i) {
    // Output coordinates in innermost-first order.
    int o[4];
    o[0] = pos.x;
    o[1] = pos.y;
    o[2] = c4 * 4 + i;
    o[3] = n_out;

    int s[4];
    for (int q = 0; q < 4; ++q) {
      s[q] = o[q];
    }

    if (uBlock.removed != 0) {
      for (int q = 0; q < 4; ++q) {
        if (q < uBlock.axis) {
          // Inner axes keep their coordinates.
          s[q] = o[q];
        } else if (q == uBlock.axis) {
          s[q] = uBlock.start;
        } else {
          // Outer axes shift inward by one slot.
          s[q] = o[q - 1];
        }
      }
    } else {
      s[uBlock.axis] = uBlock.start + uBlock.step * o[uBlock.axis];
    }

    out_texel[i] = imageLoad(
        uInput,
        ivec3(s[0], s[1], s[3] * uBlock.in_c_depth + s[2] / 4))[s[2] % 4];
  }

  imageStore(uOutput, pos, out_texel);
}

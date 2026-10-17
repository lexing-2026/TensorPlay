#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer concatenation: the output and every input are plain
// row-major buffer payloads (the shapes that exceed the 3-D image axis
// limit).  Each pass scatters one input's elements into the output by
// shifting only the concatenated axis back through the running offset;
// lanes whose shifted coordinate falls outside this input belong to a
// neighboring input and are left untouched.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  float data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  float data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // Logical sizes of the output, prepadded to {N, C, H, W}
  ivec4 out_sizes;
  // Logical sizes of this input, prepadded to {N, C, H, W}
  ivec4 in_sizes;
  // Dense (contiguous) strides of this input in {N, C, H, W} order
  ivec4 in_strides;
  // Concatenated axis counted from the innermost (0: W, 1: H, 2: C, 3: N)
  int axis;
  // Element offset of this input along the concatenated axis
  int offset;
  int out_c_depth; // ceil(out C / 4), drives the grid
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Concatenation along any axis for linear buffer payloads.  Each invocation
 * maps its output coordinates {n, c, h, w} to input coordinates by shifting
 * only the concatenated axis back through the running offset; every other
 * coordinate carries over.  A lane whose shifted coordinate falls outside
 * this input belongs to a neighboring input and is skipped.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.w || pos.y >= uBlock.out_sizes.z ||
      pos.z >= uBlock.out_sizes.x * uBlock.out_c_depth) {
    return;
  }

  const int n_out = pos.z / uBlock.out_c_depth;
  const int c4 = pos.z % uBlock.out_c_depth;

  for (int i = 0; i < 4; ++i) {
    const int c = c4 * 4 + i;
    if (c >= uBlock.out_sizes.y) {
      break;
    }

    // Input coordinates: only the concatenated axis moves.
    int s0 = n_out;
    int s1 = c;
    int s2 = pos.y;
    int s3 = pos.x;
    if (uBlock.axis == 0) {
      s3 -= uBlock.offset;
    } else if (uBlock.axis == 1) {
      s2 -= uBlock.offset;
    } else if (uBlock.axis == 2) {
      s1 -= uBlock.offset;
    } else {
      s0 -= uBlock.offset;
    }

    if (s0 < 0 || s0 >= uBlock.in_sizes.x || s1 < 0 ||
        s1 >= uBlock.in_sizes.y || s2 < 0 || s2 >= uBlock.in_sizes.z ||
        s3 < 0 || s3 >= uBlock.in_sizes.w) {
      continue; // belongs to another input
    }

    const int in_linear = s0 * uBlock.in_strides.x +
        s1 * uBlock.in_strides.y + s2 * uBlock.in_strides.z +
        s3 * uBlock.in_strides.w;

    // Buffer payloads are plain row-major memory: the output linear element
    // is the logical position of (n_out, c, h, w).
    const int out_linear = ((n_out * uBlock.out_sizes.y + c) *
                               uBlock.out_sizes.z +
                           pos.y) *
                              uBlock.out_sizes.w +
                          pos.x;
    uOutput.data[out_linear] = uInput.data[in_linear];
  }
}
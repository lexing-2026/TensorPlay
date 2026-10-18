#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Buffer-to-buffer flip: input and output are plain row-major buffer
// payloads (the shapes that exceed the 3-D image axis limit).  Each output
// invocation maps its coordinates {n, c, h, w} back through the flagged
// axes (x: width, y: height, z: channel, w: batch), computes the flipped
// input linear position over the dense layout, and copies one element per
// channel lane.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  vec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  vec4 data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // Logical sizes of the output, prepadded to {N, C, H, W}
  ivec4 out_sizes;
  // Logical sizes of the input, prepadded to {N, C, H, W}
  ivec4 in_sizes;
  // Dense (contiguous) strides of the input in {N, C, H, W} order
  ivec4 in_strides;
  // One flag per input axis (x: W, y: H, z: C, w: N)
  ivec4 flip_axes;
  int out_c_depth; // ceil(out C / 4), drives the grid
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

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

    int s0 = n_out;
    int s1 = c;
    int s2 = pos.y;
    int s3 = pos.x;
    if (uBlock.flip_axes.w != 0) {
      s0 = uBlock.in_sizes.x - 1 - n_out;
    }
    if (uBlock.flip_axes.z != 0) {
      s1 = uBlock.in_sizes.y - 1 - c;
    }
    if (uBlock.flip_axes.y != 0) {
      s2 = uBlock.in_sizes.z - 1 - pos.y;
    }
    if (uBlock.flip_axes.x != 0) {
      s3 = uBlock.in_sizes.w - 1 - pos.x;
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
    uOutput.data[out_linear >> 2][out_linear & 3] =
        uInput.data[in_linear >> 2][in_linear & 3];
  }
}
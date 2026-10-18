#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Half variant of the linear flip: input and output are row-major payloads
// that pack two 16-bit lanes per word (the shapes whose extents exceed the
// 3-D image axis limit).  One invocation owns one destination word and
// reverses both of its lanes, so a flipped neighbour never shares a store
// with another invocation.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
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
  int out_c_depth; // unused here: the grid runs over destination words
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Reverses one output element through the flagged axes and picks the 16-bit
 * field at the mirrored input position.  Coordinates are recovered from the
 * logical row-major index because the destination is addressed by words, not
 * by channel groups.
 */
float flip_element(const uint element) {
  uint rest = element;
  const int w = int(rest % uint(uBlock.out_sizes.w));
  rest /= uint(uBlock.out_sizes.w);
  const int h = int(rest % uint(uBlock.out_sizes.z));
  rest /= uint(uBlock.out_sizes.z);
  const int c = int(rest % uint(uBlock.out_sizes.y));
  const int n = int(rest / uint(uBlock.out_sizes.y));

  int s0 = n;
  int s1 = c;
  int s2 = h;
  int s3 = w;
  if (uBlock.flip_axes.w != 0) {
    s0 = uBlock.in_sizes.x - 1 - s0;
  }
  if (uBlock.flip_axes.z != 0) {
    s1 = uBlock.in_sizes.y - 1 - s1;
  }
  if (uBlock.flip_axes.y != 0) {
    s2 = uBlock.in_sizes.z - 1 - s2;
  }
  if (uBlock.flip_axes.x != 0) {
    s3 = uBlock.in_sizes.w - 1 - s3;
  }

  const uint in_linear = uint(
      s0 * uBlock.in_strides.x + s1 * uBlock.in_strides.y +
      s2 * uBlock.in_strides.z + s3 * uBlock.in_strides.w);
  return unpackHalf2x16(uInput.data[in_linear >> 1u])[in_linear & 1u];
}

void main() {
  const uint word = gl_GlobalInvocationID.x;

  const uint out_numel = uint(uBlock.out_sizes.x) *
      uint(uBlock.out_sizes.y) * uint(uBlock.out_sizes.z) *
      uint(uBlock.out_sizes.w);
  if ((word << 1u) >= out_numel) {
    return;
  }

  vec2 values = vec2(0.0);
  for (uint lane = 0u; lane < 2u; ++lane) {
    const uint element = (word << 1u) + lane;
    if (element >= out_numel) {
      break;
    }
    values[int(lane)] = flip_element(element);
  }

  uOutput.data[word] = packHalf2x16(values);
}

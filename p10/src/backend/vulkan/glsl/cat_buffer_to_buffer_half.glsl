#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Half variant of the linear concatenation pass: the payloads are dense
// row-major 16-bit lanes, two sharing one word.  One invocation owns one
// destination word, so the whole word is read, its two lanes are filled from
// whichever input they belong to, and it is written back.  A concatenated axis
// whose length is odd leaves a word split between two passes; the surviving
// bits are what the other pass stored, since passes run in order through the
// barrier system.
layout(set = 0, binding = 0) buffer PRECISION restrict Output {
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
  // Logical sizes of this input, prepadded to {N, C, H, W}
  ivec4 in_sizes;
  // Dense (contiguous) strides of this input in {N, C, H, W} order
  ivec4 in_strides;
  // Concatenated axis counted from the innermost (0: W, 1: H, 2: C, 3: N)
  int axis;
  // Element offset of this input along the concatenated axis
  int offset;
  int out_c_depth; // unused here: the grid runs over destination words
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint out_numel = uint(uBlock.out_sizes.x) *
      uint(uBlock.out_sizes.y) * uint(uBlock.out_sizes.z) *
      uint(uBlock.out_sizes.w);
  if ((word << 1u) >= out_numel) {
    return;
  }

  uint packed = uOutput.data[word];
  for (uint lane = 0u; lane < 2u; ++lane) {
    const uint element = (word << 1u) + lane;
    if (element >= out_numel) {
      break;
    }

    // Walk the logical row-major position back into output coordinates.
    uint rest = element;
    int s3 = int(rest % uint(uBlock.out_sizes.w));
    rest /= uint(uBlock.out_sizes.w);
    int s2 = int(rest % uint(uBlock.out_sizes.z));
    rest /= uint(uBlock.out_sizes.z);
    int s1 = int(rest % uint(uBlock.out_sizes.y));
    rest /= uint(uBlock.out_sizes.y);
    int s0 = int(rest);

    // Undo this pass's shift on the concatenated axis; every other coordinate
    // carries over.
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
      // The lane belongs to another pass, so its stored bits stay.
      continue;
    }

    const uint in_linear = uint(s0) * uint(uBlock.in_strides.x) +
        uint(s1) * uint(uBlock.in_strides.y) +
        uint(s2) * uint(uBlock.in_strides.z) +
        uint(s3) * uint(uBlock.in_strides.w);

    const uint bits =
        (uInput.data[in_linear >> 1u] >> ((in_linear & 1u) * 16u)) & 0xffffu;
    packed = (packed & ~(0xffffu << (16u * lane))) | (bits << (16u * lane));
  }
  uOutput.data[word] = packed;
}

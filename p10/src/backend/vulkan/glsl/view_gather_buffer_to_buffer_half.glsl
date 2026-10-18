#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Half variant of the linear-to-linear strided-view materialization gather:
// both the source view and the dense output are row-major payloads of two
// 16-bit lanes per word (the shapes that exceed the 3-D image axis limit).
// One invocation owns one destination word, so the pair it covers is gathered
// and written in a single store, and a view whose strides split a word is
// still safe.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uint data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // Logical sizes of the dense input, prepadded to {d0, d1, d2, d3} = {N, C, H, W}
  ivec4 in_sizes;
  // Dense strides of the input in the same prepadded order
  ivec4 in_strides;
  // Logical sizes of the output, prepadded the same way
  ivec4 out_sizes;
  // Strides of the source view over the logical element order, in the same
  // prepadded order; a prepadded axis contributes zero
  ivec4 out_strides;
  int in_c_depth;  // unused here: a linear payload has no texel columns
  int out_c_depth; // unused here: the grid runs over destination words
  int offset;      // element offset of the view into the input
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Maps one output element's logical coordinates through the view strides to a
// linear position in the input's element order and picks its 16-bit field.
// Zero strides re-read one position, which materializes broadcast expansions
// without a separate kernel.
float gather_element(const ivec4 coord) {
  const uint linear_offset = uint(
      uBlock.offset + coord.x * uBlock.out_strides.x +
      coord.y * uBlock.out_strides.y + coord.z * uBlock.out_strides.z +
      coord.w * uBlock.out_strides.w);
  return unpackHalf2x16(uInput.data[linear_offset >> 1u])[linear_offset & 1u];
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
    // Walk the logical row-major position back into {N, C, H, W}.
    uint rest = element;
    ivec4 coord;
    coord.w = int(rest % uint(uBlock.out_sizes.w));
    rest /= uint(uBlock.out_sizes.w);
    coord.z = int(rest % uint(uBlock.out_sizes.z));
    rest /= uint(uBlock.out_sizes.z);
    coord.y = int(rest % uint(uBlock.out_sizes.y));
    rest /= uint(uBlock.out_sizes.y);
    coord.x = int(rest);
    values[int(lane)] = gather_element(coord);
  }

  uOutput.data[word] = packHalf2x16(values);
}

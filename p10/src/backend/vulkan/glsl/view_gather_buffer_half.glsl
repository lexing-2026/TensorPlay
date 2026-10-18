#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Half buffer-output variant of the strided-view materialization gather: a half
// texture feeds a dense row-major payload of two 16-bit lanes per word.  Two
// neighbouring output elements share a word whatever the view strides look
// like, so one invocation owns one word and writes both lanes in a single
// store; the texel's four lanes are widened exactly by the image format and
// rounded back on the way out.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uint data[];
}
uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
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
  int in_c_depth;  // ceil(in C / 4)
  int out_c_depth; // unused here: the grid runs over destination words
  int offset;      // element offset of the view into the input
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Maps one output element's logical coordinates through the view strides to a
// linear position in the input, decomposes that position over the input's dense
// layout and fetches the lane.  Zero strides re-read one position, which
// materializes broadcast expansions without a separate kernel.
float gather_element(const ivec4 coord) {
  const int linear_offset = uBlock.offset + coord.x * uBlock.out_strides.x +
      coord.y * uBlock.out_strides.y + coord.z * uBlock.out_strides.z +
      coord.w * uBlock.out_strides.w;

  const int n_in = linear_offset / uBlock.in_strides.x;
  int r = linear_offset - n_in * uBlock.in_strides.x;
  const int c_in = r / uBlock.in_strides.y;
  r -= c_in * uBlock.in_strides.y;
  const int h_in = r / uBlock.in_strides.z;
  const int w_in = r - h_in * uBlock.in_strides.z;

  return texelFetch(
      uInput,
      ivec3(w_in, h_in, n_in * uBlock.in_c_depth + c_in / 4),
      0)[c_in % 4];
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

#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * 64-bit-code buffer-to-buffer variant of the strided-view materialization
 * gather: both the source view and the dense output are linear payloads.  A
 * wide element owns a pair of words, so the payload is read and written as
 * whole elements; no lane of an element is ever moved on its own, which is what
 * keeps a code that names a number past the reach of a word intact.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  uvec2 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  uvec2 data[];
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
  int in_c_depth;  // ceil(in C / 4)
  int out_c_depth; // ceil(out C / 4)
  int offset;      // element offset of the view into the input
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Each invocation maps its output coordinates {n, c, h, w} through the view
 * strides to a linear position in the input's logical order and copies the
 * whole code it finds there.  Zero strides re-read one position, which
 * materializes broadcast expansions without a separate kernel.
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

    const int linear_offset = uBlock.offset + n_out * uBlock.out_strides.x +
        c * uBlock.out_strides.y + pos.y * uBlock.out_strides.z +
        pos.x * uBlock.out_strides.w;

    // A linear payload is plain row-major memory in the input's logical order,
    // so the position the view strides name is the element to read.
    const uvec2 code = uInput.data[linear_offset];

    // The output keeps the same rule: its linear element is the logical
    // position of (n_out, c, h, w).
    const int out_linear = ((n_out * uBlock.out_sizes.y + c) *
                               uBlock.out_sizes.z +
                           pos.y) *
                              uBlock.out_sizes.w +
                          pos.x;
    uOutput.data[out_linear] = code;
  }
}

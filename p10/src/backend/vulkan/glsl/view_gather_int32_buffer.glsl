#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Int32 buffer-output variant of the strided-view materialization gather:
// signed-word textures in, plain row-major buffer out.
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  int data[];
}
uOutput;
layout(set = 0, binding = 1, rgba32i) uniform PRECISION restrict readonly iimage3D uInput;
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
 * Generic strided gather for Int32 buffer-stored outputs.  Each invocation
 * maps its output coordinates {n, c, h, w} through the view strides to a
 * linear position in the input's logical order, decomposes that position over
 * the input's dense layout, and fetches one element per channel lane.  Zero
 * strides re-read one position, which materializes broadcast expansions
 * without a separate kernel.
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

    // Decompose the linear position over the input's dense layout.
    const int n_in = linear_offset / uBlock.in_strides.x;
    int r = linear_offset - n_in * uBlock.in_strides.x;
    const int c_in = r / uBlock.in_strides.y;
    r -= c_in * uBlock.in_strides.y;
    const int h_in = r / uBlock.in_strides.z;
    const int w_in = r - h_in * uBlock.in_strides.z;

    const int value = imageLoad(
        uInput,
        ivec3(w_in, h_in, n_in * uBlock.in_c_depth + c_in / 4))[c_in % 4];

    // Buffer payloads are plain row-major memory: the output linear element
    // is the logical position of (n_out, c, h, w).
    const int out_linear = ((n_out * uBlock.out_sizes.y + c) *
                               uBlock.out_sizes.z +
                           pos.y) *
                              uBlock.out_sizes.w +
                          pos.x;
    uOutput.data[out_linear] = value;
  }
}
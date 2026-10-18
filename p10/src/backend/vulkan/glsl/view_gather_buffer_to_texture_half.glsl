#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// Half texture-output variant of the strided-view materialization gather: a
// row-major payload of two 16-bit lanes per word feeds an image whose texels
// carry four half lanes each.  This is the path for slicing a small range out
// of a linear-stored payload, for example the leading rows of a long kv cache.
// A texel is stored whole, so word sharing between lanes never reaches the
// destination; the fields are widened exactly before the store rounds them
// back into the format.
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
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
  int out_c_depth; // ceil(out C / 4)
  int offset;      // element offset of the view into the input
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Strided gather from a linear half payload into an image.  Each invocation
 * maps its output coordinates {n, c, h, w} through the view strides to a
 * linear position in the input's element order, takes the 16-bit field stored
 * there, and writes the four lanes of its texel in one store.  Zero strides
 * re-read one position, which materializes broadcast expansions without a
 * separate kernel.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.w || pos.y >= uBlock.out_sizes.z ||
      pos.z >= uBlock.out_sizes.x * uBlock.out_c_depth) {
    return;
  }

  const int n_out = pos.z / uBlock.out_c_depth;
  const int c4 = pos.z % uBlock.out_c_depth;

  vec4 out_texel = vec4(0.0);
  for (int i = 0; i < 4; ++i) {
    const int c = c4 * 4 + i;
    if (c >= uBlock.out_sizes.y) {
      break;
    }

    const uint linear_offset = uint(
        uBlock.offset + n_out * uBlock.out_strides.x +
        c * uBlock.out_strides.y + pos.y * uBlock.out_strides.z +
        pos.x * uBlock.out_strides.w);

    out_texel[i] =
        unpackHalf2x16(uInput.data[linear_offset >> 1u])[linear_offset & 1u];
  }

  imageStore(uOutput, pos, out_texel);
}

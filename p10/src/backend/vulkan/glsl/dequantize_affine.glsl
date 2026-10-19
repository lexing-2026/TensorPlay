#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

/*
 * Undoing a block-wise affine quantization over texel storage: an element
 * recovers (code - zero point of its block) * scale of its block, where the
 * block grid is the logical shape tiled by one extent per axis.
 *
 * A texel of a signed-byte code payload carries four channel lanes, so the
 * texel coordinates name the four innermost logical axes: x is the width, y the
 * height, z a group of four channels within a batch, and the lane is the
 * channel inside the group.  Lanes that name no channel belong to the padding
 * of the texel, so they leave the run before any parameter is read.
 */
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uFloatOutput;
layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uCodes;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Scales {
  float data[];
}
uScales;
layout(set = 0, binding = 3) buffer PRECISION restrict readonly Zeros {
  float data[];
}
uZeros;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  uvec4 sizes;         // logical extents, outer axis first, padded with ones
  uvec4 size_strides;  // element stride of each logical axis
  uvec4 blocks;        // elements one parameter covers, per axis
  uvec4 block_strides; // stride through the block grid, per axis
  uvec4 grid;          // (texel x, texel y, texel z, channel groups)
  uvec4 flags;         // (element count, payload is unsigned, scale count, zero point count)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// The block a logical element belongs to: its coordinate along each axis,
// shrunk to the block that spans it, read through the block grid's strides.
uint block_of(const uvec4 coord) {
  return (coord.x / uBlock.blocks.x) * uBlock.block_strides.x +
      (coord.y / uBlock.blocks.y) * uBlock.block_strides.y +
      (coord.z / uBlock.blocks.z) * uBlock.block_strides.z +
      (coord.w / uBlock.blocks.w) * uBlock.block_strides.w;
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, ivec3(uBlock.grid.xyz)))) {
    return;
  }

  const ivec4 codes = imageLoad(uCodes, pos);
  // The image's depth runs over batches of channel groups, so the batch and the
  // group within it come apart through the group depth of the logical shape.
  const uint batch = uint(pos.z) / uBlock.grid.w;
  const uint channel_group = uint(pos.z) - batch * uBlock.grid.w;

  vec4 result = vec4(0.0);
  for (int lane = 0; lane < 4; ++lane) {
    const uvec4 coord = uvec4(
        batch,
        channel_group * 4u + uint(lane),
        uint(pos.y),
        uint(pos.x));
    if (any(greaterThanEqual(coord, uBlock.sizes))) {
      continue;
    }

    const uint block = block_of(coord);
    // A single parameter covers the whole payload, whatever the block grid
    // looks like, so its slot is the first one.
    const uint scale_slot = uBlock.flags.z == 1u ? 0u : block;
    const uint zero_slot = uBlock.flags.w == 1u ? 0u : block;
    const float zero_point =
        uBlock.flags.w == 0u ? 0.0 : uZeros.data[zero_slot];
    int stored = codes[lane];
    if (uBlock.flags.y != 0u) {
      // An unsigned payload read through a signed image hands back the low
      // eight bits; those name the code as it was stored.
      stored = int(uint(stored) & 0xFFu);
    }
    result[lane] =
        (float(stored) - zero_point) * uScales.data[scale_slot];
  }

  imageStore(uFloatOutput, pos, result);
}

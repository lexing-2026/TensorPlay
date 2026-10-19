#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Undoing a block-wise affine quantization over linear storage: an element
 * recovers (code - zero point of its block) * scale of its block.
 *
 * Four codes share one word of the payload and four results fill one word of
 * the output, so one invocation names the same four positions on both sides.
 * The four codes of a word are consecutive in the logical order, and the block
 * grid is coarser than that order only along the outer axes, so each lane
 * resolves its own block: a group boundary can fall inside a word.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  vec4 data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Codes {
  uint data[];
}
uCodes;
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
  uvec4 grid;          // unused here; the linear run is one dimensional
  uvec4 flags;         // (element count, payload is unsigned, scale count, zero point count)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Bit 7 of a lane is the sign of a two's-complement int8 code, so a byte of 128
// or more names a negative value and the remaining seven bits name its magnitude.
float signed_byte(const uint byte_value) {
  return float(int(byte_value & 0x7Fu) - int(byte_value & 0x80u));
}

void main() {
  const uint word = gl_GlobalInvocationID.x;
  const uint first = word << 2u;
  if (first >= uBlock.flags.x) {
    return;
  }

  const uint packed = uCodes.data[word];
  vec4 result = vec4(0.0);
  for (int lane = 0; lane < 4; ++lane) {
    const uint index = first + uint(lane);
    if (index >= uBlock.flags.x) {
      break;
    }

    const uvec4 coord = uvec4(
        (index / uBlock.size_strides.x) % uBlock.sizes.x,
        (index / uBlock.size_strides.y) % uBlock.sizes.y,
        (index / uBlock.size_strides.z) % uBlock.sizes.z,
        (index / uBlock.size_strides.w) % uBlock.sizes.w);
    const uint block = (coord.x / uBlock.blocks.x) * uBlock.block_strides.x +
        (coord.y / uBlock.blocks.y) * uBlock.block_strides.y +
        (coord.z / uBlock.blocks.z) * uBlock.block_strides.z +
        (coord.w / uBlock.blocks.w) * uBlock.block_strides.w;
    // A single parameter covers the whole payload, whatever the block grid
    // looks like, so its slot is the first one.
    const uint scale_slot = uBlock.flags.z == 1u ? 0u : block;
    const uint zero_slot = uBlock.flags.w == 1u ? 0u : block;
    const float zero_point =
        uBlock.flags.w == 0u ? 0.0 : uZeros.data[zero_slot];
    const uint byte_value = (packed >> (uint(lane) * 8u)) & 0xFFu;
    const float code = uBlock.flags.y == 0u ? signed_byte(byte_value)
                                           : float(byte_value);
    result[lane] = (code - zero_point) * uScales.data[scale_slot];
  }

  uOutput.data[word] = result;
}

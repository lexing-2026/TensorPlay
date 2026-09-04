#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the input
  ivec4 in_sizes;
  // reduction axis: 0 = W, 1 = H, 2 = C, 3 = N
  int axis;
  int c_depth; // ceil(C / 4)
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Inclusive prefix scan along one axis.  Each invocation accumulates the
 * input line up to and including its own position; the small tensors
 * targeted by this backend keep the linear walk affordable.  Width, height,
 * and batch scans accumulate texel lanes independently.  A channel scan
 * walks the texel depth of one batch: each lane adds its own value from the
 * earlier texels, then the lower lanes of its own texel, so lane L of the
 * result carries the total of channels up to and including c4*4 + L.
 * Lanes past the channel count are written as zero so the padded texel
 * region stays defined.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.in_sizes.x || pos.y >= uBlock.in_sizes.y ||
      pos.z >= uBlock.in_sizes.w * uBlock.c_depth) {
    return;
  }

  vec4 acc = vec4(0.0f);

  if (uBlock.axis == 0) {
    for (int x = 0; x <= pos.x; ++x) {
      acc += texelFetch(uInput, ivec3(x, pos.y, pos.z), 0);
    }
    imageStore(uOutput, pos, acc);
  } else if (uBlock.axis == 1) {
    for (int y = 0; y <= pos.y; ++y) {
      acc += texelFetch(uInput, ivec3(pos.x, y, pos.z), 0);
    }
    imageStore(uOutput, pos, acc);
  } else if (uBlock.axis == 2) {
    // Channel scan: pos.z addresses (n * c_depth + c4).
    const int n = pos.z / uBlock.c_depth;
    const int c4 = pos.z - n * uBlock.c_depth;
    const int channels = uBlock.in_sizes.z;
    for (int z = n * uBlock.c_depth; z < pos.z; ++z) {
      acc += texelFetch(uInput, ivec3(pos.x, pos.y, z), 0);
    }
    const vec4 self_texel =
        texelFetch(uInput, ivec3(pos.x, pos.y, pos.z), 0);
    acc.x += self_texel.x;
    acc.y += acc.x + self_texel.y;
    acc.z += acc.y + self_texel.z;
    acc.w += acc.z + self_texel.w;
    const vec4 lane_mask =
        vec4(lessThan(ivec4(c4 * 4, c4 * 4 + 1, c4 * 4 + 2, c4 * 4 + 3),
                      ivec4(channels)));
    imageStore(uOutput, pos, acc * lane_mask);
  } else {
    // Batch scan: one batch step moves c_depth texels along z; the channel
    // slot (pos.z mod c_depth) stays fixed.
    const int c4 = pos.z % uBlock.c_depth;
    for (int n = 0; n <= pos.z / uBlock.c_depth; ++n) {
      acc += texelFetch(
          uInput, ivec3(pos.x, pos.y, n * uBlock.c_depth + c4), 0);
    }
    imageStore(uOutput, pos, acc);
  }
}

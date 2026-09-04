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
  // (W, H, C, N) sizes of the output
  ivec4 out_sizes;
  int axis; // glu split axis: 0 = W, 1 = H, 2 = C, 3 = N
  int in_c_depth;
  int out_c_depth;
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Logistic gate, applied per component.
vec4 gate(const vec4 x) {
  return 1.0f / (1.0f + exp(-x));
}

/*
 * Gated linear unit: the input is split along one axis into equal halves,
 * the output is first_half * sigmoid(second_half).  Each output position
 * gathers the two source elements directly, so no intermediate tensor is
 * materialized.  A channel-axis split crosses texel lanes: the halves are
 * addressed through the texel depth, and the sigmoid is applied per source
 * lane before the product.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  const int half_len = (uBlock.axis == 2)
      ? uBlock.in_sizes.z / 2
      : ((uBlock.axis == 0)
             ? uBlock.in_sizes.x / 2
             : ((uBlock.axis == 1) ? uBlock.in_sizes.y / 2
                                   : uBlock.in_sizes.w / 2));

  if (uBlock.axis == 0) {
    const vec4 a = texelFetch(uInput, ivec3(pos.x, pos.y, pos.z), 0);
    const vec4 b =
        texelFetch(uInput, ivec3(pos.x + half_len, pos.y, pos.z), 0);
    imageStore(uOutput, pos, a * gate(b));
  } else if (uBlock.axis == 1) {
    const vec4 a = texelFetch(uInput, ivec3(pos.x, pos.y, pos.z), 0);
    const vec4 b =
        texelFetch(uInput, ivec3(pos.x, pos.y + half_len, pos.z), 0);
    imageStore(uOutput, pos, a * gate(b));
  } else if (uBlock.axis == 2) {
    // Source channels: c and c + half_len.  Each maps to a (texel, lane)
    // pair inside the same batch block.
    const int channels = uBlock.in_sizes.z;
    const int n = pos.z / uBlock.out_c_depth;
    const int c4 = pos.z - n * uBlock.out_c_depth;
    vec4 r;
    for (int lane = 0; lane < 4; ++lane) {
      const int c = c4 * 4 + lane;
      if (c >= channels / 2) {
        continue;
      }
      const int t0 = n * uBlock.in_c_depth + c / 4;
      const int c2 = c + half_len;
      const int t1 = n * uBlock.in_c_depth + c2 / 4;
      const vec4 a = texelFetch(uInput, ivec3(pos.x, pos.y, t0), 0);
      const vec4 b = texelFetch(uInput, ivec3(pos.x, pos.y, t1), 0);
      // Pick the requested channel out of both source texels.
      const float sa = (c % 4 == 0)
          ? a.x
          : ((c % 4 == 1) ? a.y : ((c % 4 == 2) ? a.z : a.w));
      const float sb = (c2 % 4 == 0)
          ? b.x
          : ((c2 % 4 == 1) ? b.y : ((c2 % 4 == 2) ? b.z : b.w));
      r[lane] = sa * (1.0f / (1.0f + exp(-sb)));
    }
    imageStore(uOutput, pos, r);
  } else {
    // Batch split: one batch step moves in_c_depth texels along z.
    const int c4 = pos.z % uBlock.in_c_depth;
    const vec4 a = texelFetch(uInput, ivec3(pos.x, pos.y, pos.z), 0);
    const vec4 b = texelFetch(
        uInput, ivec3(pos.x, pos.y, (pos.z / uBlock.in_c_depth + half_len) *
                uBlock.in_c_depth + c4), 0);
    imageStore(uOutput, pos, a * gate(b));
  }
}

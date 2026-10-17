#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define LOG_SOFTMAX ${LOG_SOFTMAX}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the tensor being normalized
  ivec4 sizes;
  int c_depth; // ceil(C / 4)
  // 0: normalize along W, 1: along H, 3: along N
  int axis;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Normalizes along the width, height, or batch axis.  Each invocation owns one
 * texel of the output and derives statistics from the full extent of the
 * chosen axis at that texel's position on the remaining axes.  The walk
 * therefore starts at the axis origin (base) and advances by the axis step,
 * while the channel lane stays fixed; statistics are tracked per lane so
 * every channel is normalized independently.  Padding lanes carry garbage
 * through the same formula and are never read back.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // Start of the walk and the step between two of its elements expressed in
  // texel coordinates.  The start zeroes the component of the axis being
  // reduced so every invocation of that axis observes the same elements.
  ivec3 base = pos;
  ivec3 step;
  int length;
  if (uBlock.axis == 0) {
    base.x = 0;
    step = ivec3(1, 0, 0);
    length = uBlock.sizes.x;
  } else if (uBlock.axis == 1) {
    base.y = 0;
    step = ivec3(0, 1, 0);
    length = uBlock.sizes.y;
  } else {
    // Batch axis: one batch step moves c_depth texels along z, and the walk
    // starts at the channel that pos points into.
    base.z = pos.z % uBlock.c_depth;
    step = ivec3(0, 0, uBlock.c_depth);
    length = uBlock.sizes.w;
  }

  vec4 vmax = vec4(-3.0e38f);
  for (int i = 0; i < length; ++i) {
    vmax = max(vmax, texelFetch(uInput, base + step * i, 0));
  }

  vec4 vsum = vec4(0.0f);
  for (int i = 0; i < length; ++i) {
    vsum += exp(texelFetch(uInput, base + step * i, 0) - vmax);
  }

  // clang-format off
  $if LOG_SOFTMAX:
    const vec4 v = texelFetch(uInput, pos, 0);
    imageStore(uOutput, pos, v - vmax - log(vsum));
  $else:
    const vec4 v = texelFetch(uInput, pos, 0);
    imageStore(uOutput, pos, exp(v - vmax) / vsum);
  // clang-format on
}

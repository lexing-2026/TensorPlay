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
  // (W, H, C, N)
  ivec4 sizes;
  int c_depth; // ceil(C / 4)
  int channels;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Normalizes along the channel axis.  The running statistics cross texel
 * lanes, so channels beyond the tensor's channel count are masked out; the
 * corresponding output lanes keep whatever the formula produces and are
 * never read back.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int n = pos.z / uBlock.c_depth;
  const int c4 = pos.z % uBlock.c_depth;

  const vec4 lane_mask = vec4(lessThan(
      ivec4(c4 * 4) + ivec4(0, 1, 2, 3),
      ivec4(uBlock.channels)));
  const vec4 neg_inf = vec4(-3.0e38f);

  vec4 vmax = neg_inf;
  for (int i = 0; i < uBlock.c_depth; ++i) {
    const vec4 v = texelFetch(
        uInput, ivec3(pos.x, pos.y, n * uBlock.c_depth + i), 0);
    vmax = max(vmax, mix(neg_inf, v, lane_mask));
  }
  const float max_val = max(max(vmax.x, vmax.y), max(vmax.z, vmax.w));

  float sum = 0.0f;
  for (int i = 0; i < uBlock.c_depth; ++i) {
    const vec4 v = texelFetch(
        uInput, ivec3(pos.x, pos.y, n * uBlock.c_depth + i), 0);
    sum += dot(lane_mask, exp(v - max_val));
  }

  // clang-format off
  $if LOG_SOFTMAX:
    const vec4 v = texelFetch(uInput, pos, 0);
    imageStore(uOutput, pos, v - max_val - log(sum));
  $else:
    const vec4 v = texelFetch(uInput, pos, 0);
    imageStore(uOutput, pos, exp(v - max_val) / sum);
  // clang-format on
}

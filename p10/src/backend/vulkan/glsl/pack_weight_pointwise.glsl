#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 weight_sizes; // (O, C, O4, C4) of the source and its group depths
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Regroups a pointwise weight payload {O, C, 1, 1} for the tiled pointwise
 * kernel: the destination texel (ic4, o4, lane) carries, per component, the
 * weight w[oc = 4*o4 + comp][ic = 4*ic4 + lane].  The source texture stores
 * element (oc, ic) at texel (0, 0, oc * C4 + ic4) lane ic % 4.  Output
 * channels past the edge are written as zeros.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.weight_sizes.w || pos.y >= uBlock.weight_sizes.z ||
      pos.z >= 4) {
    return;
  }

  vec4 out_t = vec4(0.0f);
  for (int comp = 0; comp < 4; ++comp) {
    const int oc = pos.y * 4 + comp;
    if (oc < uBlock.weight_sizes.x) {
      const int src_z = oc * uBlock.weight_sizes.w + pos.x;
      out_t[comp] = texelFetch(uInput, ivec3(0, 0, src_z), 0)[pos.z];
    }
  }

  imageStore(uOutput, pos, out_t);
}

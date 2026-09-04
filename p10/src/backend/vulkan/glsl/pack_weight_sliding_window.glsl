#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 weight_sizes; // (O, C, KH, KW) logical sizes of the source
  int in_c_depth;     // ceil(C / 4): channel depth of the source texture
  int out_c_depth;    // ceil(O / 4)
  int src_c_depth;    // ceil(C / 4): z extent of one source output channel
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Regroups a sliding-window weight payload {O, C, KH, KW} for the tap-packed
 * kernel: the destination texel (kx, ky, (o4 * C4 + ic4) * 4 + lane) carries,
 * in its components, w[oc = 4*o4 + comp][ic = 4*ic4 + lane][ky][kx].  The
 * source texture stores element (oc, ic, ky, kx) at texel
 * (kx, ky, oc * src_c_depth + ic4) lane ic % 4.  Output channels past the
 * edge are written as zeros.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.weight_sizes.w || pos.y >= uBlock.weight_sizes.z ||
      pos.z >= uBlock.out_c_depth * uBlock.in_c_depth * 4) {
    return;
  }

  // Decode the destination group indices: z selects (group, lane) where the
  // lane indexes the input channel within its group of four.
  const int lane = pos.z % 4;
  const int group = pos.z / 4;
  const int ic4 = group % uBlock.in_c_depth;
  const int o4 = group / uBlock.in_c_depth;

  vec4 out_t = vec4(0.0f);
  for (int comp = 0; comp < 4; ++comp) {
    const int oc = o4 * 4 + comp;
    if (oc < uBlock.weight_sizes.x) {
      const int src_z = oc * uBlock.src_c_depth + ic4;
      const vec4 src_tex =
          texelFetch(uInput, ivec3(pos.x, pos.y, src_z), 0);
      if (ic4 * 4 + lane < uBlock.weight_sizes.y) {
        out_t[comp] = src_tex[lane];
      }
    }
  }

  imageStore(uOutput, pos, out_t);
}

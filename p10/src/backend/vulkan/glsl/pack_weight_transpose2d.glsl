#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 weight_sizes; // (C, O, KH, KW) logical source sizes
  int out_c_depth;    // ceil(O / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Regroups a transposed-convolution weight payload {C, O, KH, KW}: the
 * destination texel (ci * KW + kx, o4 * KH + ky, 0) carries the four output
 * channels w[ci][4 * o4 + comp][ky][kx] in its components.  The source
 * texture stores element (ci, o, ky, kx) at texel (kx, ky, ci * ceil(O / 4) +
 * o / 4) in lane o % 4.  Padding channels and output lanes are zeros.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int packed_width =
      ((uBlock.weight_sizes.x + 3) / 4) * 4 * uBlock.weight_sizes.w;
  const int packed_height = uBlock.out_c_depth * uBlock.weight_sizes.z;

  if (pos.x >= packed_width || pos.y >= packed_height || pos.z != 0) {
    return;
  }

  const int ci = pos.x / uBlock.weight_sizes.w;
  const int kx = pos.x % uBlock.weight_sizes.w;
  const int o4 = pos.y / uBlock.weight_sizes.z;
  const int ky = pos.y % uBlock.weight_sizes.z;

  vec4 out_t = vec4(0.0f);
  for (int comp = 0; comp < 4; ++comp) {
    const int o = o4 * 4 + comp;
    if (ci < uBlock.weight_sizes.x && o < uBlock.weight_sizes.y) {
      out_t[comp] = texelFetch(
          uInput,
          ivec3(kx, ky, ci * uBlock.out_c_depth + o4),
          0)[comp];
    }
  }

  imageStore(uOutput, ivec3(pos.x, pos.y, 0), out_t);
}

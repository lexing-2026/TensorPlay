#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define OP(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uOther;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the output
  ivec4 out_sizes;
  int c_depth;
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Elementwise comparison: the result is a Bool payload in a signed-byte
 * texture, one 0/1 byte per lane.  Both operands share the output's texel
 * geometry, so any broadcasting was resolved by the caller into equal
 * shapes.  OP returns a GLSL bool vector, mixed into 0.0/1.0 floats and
 * stored as the byte texture.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.c_depth) {
    return;
  }

  const vec4 a = texelFetch(uInput, pos, 0);
  const vec4 b = texelFetch(uOther, pos, 0);
  const vec4 r = mix(vec4(0.0f), vec4(1.0f), OP(a, b));
  imageStore(uOutput, pos, r);
}

#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
#define DTYPE ${DTYPE}

#define OP(X, P0, P1) ${OPERATOR}
// clang-format on

layout(std430) buffer;

$if DTYPE == "int":
  // Signed-word planes; the two parameters ride the block as int words.
  layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly iimage3D uOutput;
  layout(set = 0, binding = 1, FORMAT) uniform PRECISION restrict readonly iimage3D uInput;
  layout(set = 0, binding = 2) uniform PRECISION restrict Block {
    ivec4 extents;
    int p0;
    int p1;
  }
  uBlock;
$else:
  layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
  layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
  layout(set = 0, binding = 2) uniform PRECISION restrict Block {
    ivec4 extents;
    float p0;
    float p1;
  }
  uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Parameterized pointwise activation: the formula receives the input texel
 * plus the two scalar parameters carried by the block (one may be unused).
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, uBlock.extents.xyz))) {
    return;
  }

  // clang-format off
  $if DTYPE == "int":
    const ivec4 x = imageLoad(uInput, pos);
    const ivec4 p0 = ivec4(uBlock.p0);
    const ivec4 p1 = ivec4(uBlock.p1);
    imageStore(uOutput, pos, OP(x, p0, p1));
  $else:
    const vec4 x = texelFetch(uInput, pos, 0);
    const vec4 p0 = vec4(uBlock.p0);
    const vec4 p1 = vec4(uBlock.p1);
    imageStore(uOutput, pos, OP(x, p0, p1));
  // clang-format on
}

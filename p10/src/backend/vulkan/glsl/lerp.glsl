#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define TENSOR_WEIGHT ${TENSOR_WEIGHT}
#define INPLACE ${INPLACE}
// clang-format on

layout(std430) buffer;

// clang-format off
$if not INPLACE:
  $if TENSOR_WEIGHT:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
    layout(set = 0, binding = 2) uniform PRECISION sampler3D uEnd;
    layout(set = 0, binding = 3) uniform PRECISION sampler3D uWeight;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
    layout(set = 0, binding = 2) uniform PRECISION sampler3D uEnd;
$else:
  $if TENSOR_WEIGHT:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION sampler3D uEnd;
    layout(set = 0, binding = 2) uniform PRECISION sampler3D uWeight;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION sampler3D uEnd;
// clang-format on

layout(set = 0, binding = ${BLOCK_BINDING}) uniform PRECISION restrict Block {
  ivec4 extents;
  // nonzero when the end tensor is a scalar (read texel (0, 0, 0))
  int scalar_end;
  // nonzero when the weight tensor is a scalar (tensor-weight variants)
  int scalar_weight;
  float weight;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Linear interpolation self + weight * (end - self).  The end tensor and a
 * tensor-valued weight are broadcastable: scalars read texel (0, 0, 0),
 * same-shape tensors read the aligned texel.  A scalar weight arrives
 * through the parameter block instead of a texture.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, uBlock.extents.xyz))) {
    return;
  }

  const vec4 end = (uBlock.scalar_end != 0)
      ? texelFetch(uEnd, ivec3(0, 0, 0), 0).xxxx
      : texelFetch(uEnd, pos, 0);

  // clang-format off
  $if INPLACE:
    vec4 out_texel = imageLoad(uOutput, pos);
  $else:
    const vec4 out_texel = texelFetch(uInput, pos, 0);
  // clang-format on

  // clang-format off
  $if TENSOR_WEIGHT:
    const vec4 weight = (uBlock.scalar_weight != 0)
        ? texelFetch(uWeight, ivec3(0, 0, 0), 0).xxxx
        : texelFetch(uWeight, pos, 0);
    imageStore(uOutput, pos, out_texel + weight * (end - out_texel));
  $else:
    imageStore(uOutput, pos, out_texel + uBlock.weight * (end - out_texel));
  // clang-format on
}

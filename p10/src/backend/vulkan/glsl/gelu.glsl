#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define GELU_TANH ${GELU_TANH}
// clang-format on

layout(std430) buffer;

#include "math_ext.h"

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 extents;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Gaussian-error linear unit.  The exact form evaluates the normal CDF
 * through the erf approximation above; the tanh form follows the standard
 * approximation 0.5 * x * (1 + tanh(beta * (x + kappa * x^3))) with
 * beta = sqrt(2/pi) and kappa = 0.044715.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, uBlock.extents.xyz))) {
    return;
  }

  const vec4 x = texelFetch(uInput, pos, 0);

  // clang-format off
  $if GELU_TANH:
    const float kBeta = 0.7978845608028654f;
    const float kKappa = 0.044715f;
    const vec4 inner = kBeta * (x + kKappa * x * x * x);
    imageStore(uOutput, pos, 0.5f * x * (1.0f + tanh(inner)));
  $else:
    imageStore(uOutput, pos, 0.5f * x * (1.0f + erf_approx(x * 0.70710678118654752f)));
  // clang-format on
}

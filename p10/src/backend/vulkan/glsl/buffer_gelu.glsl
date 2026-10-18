#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
#define INPLACE ${INPLACE}

#define GELU_TANH ${GELU_TANH}
// clang-format on

layout(std430) buffer;

#include "math_ext.h"

// clang-format off
$if not INPLACE:
  layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
    float data[];
  }
  uOutput;
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
    float data[];
  }
  uInput;
  layout(set = 0, binding = 2) uniform PRECISION Block {
    uint buf_length;
  }
  uBlock;
$else:
  layout(set = 0, binding = 0) buffer PRECISION OutBuffer {
    float data[];
  }
  uOutput;
  // Numbered per variant; the params block takes the first free binding
  // because this variant rewrites its operand instead of reading a second one.
  layout(set = 0, binding = 1) uniform PRECISION Block {
    uint buf_length;
  }
  uBlock;
// clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Gaussian-error linear unit over a linear buffer.  The exact form
 * evaluates the normal CDF through the erf approximation; the tanh form
 * uses beta = sqrt(2/pi) and kappa = 0.044715.
 */
void main() {
  const uint idx = gl_GlobalInvocationID.x;

  if (idx >= uBlock.buf_length) {
    return;
  }

  // clang-format off
  $if not INPLACE:
    const float x = uInput.data[idx];
  $else:
    const float x = uOutput.data[idx];
  // clang-format on

  // clang-format off
  $if GELU_TANH:
    const float kBeta = 0.7978845608028654f;
    const float kKappa = 0.044715f;
    const float inner = kBeta * (x + kKappa * x * x * x);
    uOutput.data[idx] = 0.5f * x * (1.0f + tanh(inner));
  $else:
    uOutput.data[idx] = 0.5f * x * (1.0f + erf_approx(x * 0.70710678118654752f));
  // clang-format on
}

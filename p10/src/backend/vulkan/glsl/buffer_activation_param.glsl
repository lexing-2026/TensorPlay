#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define OP(X, P0, P1) ${OPERATOR}
// clang-format on

layout(std430) buffer;

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
    float p0;
    float p1;
  }
  uBlock;
$else:
  layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
    float data[];
  }
  uOutput;
  layout(set = 0, binding = 2) uniform PRECISION Block {
    uint buf_length;
    float p0;
    float p1;
  }
  uBlock;
// clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Parameterized pointwise activation over a linear buffer: the formula
 * receives each element plus the two scalar parameters from the block.
 */
void main() {
  const uint idx = gl_GlobalInvocationID.x;

  if (idx >= uBlock.buf_length) {
    return;
  }

  const float p0 = uBlock.p0;
  const float p1 = uBlock.p1;

  // clang-format off
  $if not INPLACE:
    uOutput.data[idx] = OP(uInput.data[idx], p0, p1);
  $else:
    uOutput.data[idx] = OP(uOutput.data[idx], p0, p1);
  // clang-format on
}

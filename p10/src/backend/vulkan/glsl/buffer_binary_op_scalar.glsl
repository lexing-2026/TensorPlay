#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define OP(X, Y, A) ${OPERATOR}
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
  layout(set = 0, binding = 3) uniform PRECISION Block {
    uint buf_length;
    uint fill0;
    float other;
  }
  uBlock;
$else:
  layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
    float data[];
  }
  uOutput;
  layout(set = 0, binding = 3) uniform PRECISION Block {
    uint buf_length;
    uint fill0;
    float other;
  }
  uBlock;
// clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Element-wise operation between a linear buffer and a broadcast scalar:
 * out = OP(in, other, 1).  One invocation handles one element.
 */
void main() {
  const uint idx = gl_GlobalInvocationID.x;

  if (idx >= uBlock.buf_length) {
    return;
  }

  // clang-format off
  $if not INPLACE:
    uOutput.data[idx] = OP(uInput.data[idx], uBlock.other, 1.0f);
  $else:
    uOutput.data[idx] = OP(uOutput.data[idx], uBlock.other, 1.0f);
  // clang-format on
}

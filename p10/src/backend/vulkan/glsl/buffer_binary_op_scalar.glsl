#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
#define DTYPE ${DTYPE}

#define OP(X, Y, A) ${OPERATOR}
// clang-format on

layout(std430) buffer;

$if DTYPE == "int":
  // Signed-word storage buffers with an integer scalar operand.
  $if not INPLACE:
    layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
      int data[];
    }
    uOutput;
    layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
      int data[];
    }
    uInput;
    layout(set = 0, binding = 2) uniform PRECISION Block {
      uint buf_length;
      uint fill0;
      int alpha;
    }
    uBlock;
  $else:
    layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
      int data[];
    }
    uOutput;
    layout(set = 0, binding = 2) uniform PRECISION Block {
      uint buf_length;
      uint fill0;
      int alpha;
    }
    uBlock;
$else:
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
      uint fill0;
      float alpha;
    }
    uBlock;
  $else:
    layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
      float data[];
    }
    uOutput;
    layout(set = 0, binding = 2) uniform PRECISION Block {
      uint buf_length;
      uint fill0;
      float alpha;
    }
    uBlock;
  // clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Element-wise binary operation between a linear buffer and a broadcast
 * scalar: out = OP(in, alpha).  One invocation handles one element.
 */
void main() {
  const uint idx = gl_GlobalInvocationID.x;

  if (idx >= uBlock.buf_length) {
    return;
  }

  // clang-format off
  $if not INPLACE:
    $if DTYPE == "int":
      uOutput.data[idx] = OP(uInput.data[idx], uBlock.alpha, 1);
    $else:
      uOutput.data[idx] = OP(uInput.data[idx], uBlock.alpha, 1.0f);
  $else:
    $if DTYPE == "int":
      uOutput.data[idx] = OP(uOutput.data[idx], uBlock.alpha, 1);
    $else:
      uOutput.data[idx] = OP(uOutput.data[idx], uBlock.alpha, 1.0f);
  // clang-format on
}

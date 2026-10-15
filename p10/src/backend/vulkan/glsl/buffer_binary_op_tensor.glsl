#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
#define DTYPE ${DTYPE}

#define OP(X, Y, A) ${OPERATOR}
// clang-format on

layout(std430) buffer;

$if DTYPE == "int":
  // Signed-word storage buffers for the integer variants.  Data is laid out
  // as 16-byte vectors so the bulk of the work uses aligned 128-bit loads.
  #define VECTOR ivec4
  $if not INPLACE:
    layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
      VECTOR data[];
    }
    uOutput;
    layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
      VECTOR data[];
    }
    uInput;
    layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer2 {
      VECTOR data[];
    }
    uOther;
    layout(set = 0, binding = 3) uniform PRECISION Block {
      uint buf_length;
      uint fill0;
      int alpha;
    }
    uBlock;
  $else:
    layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
      VECTOR data[];
    }
    uOutput;
    layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer2 {
      VECTOR data[];
    }
    uOther;
    layout(set = 0, binding = 3) uniform PRECISION Block {
      uint buf_length;
      uint fill0;
      int alpha;
    }
    uBlock;
$else:
  // clang-format off
  #define VECTOR vec4
  $if not INPLACE:
    layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
      VECTOR data[];
    }
    uOutput;
    layout(set = 0, binding = 1) buffer PRECISION restrict readonly InBuffer {
      VECTOR data[];
    }
    uInput;
    layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer2 {
      VECTOR data[];
    }
    uOther;
    layout(set = 0, binding = 3) uniform PRECISION Block {
      uint buf_length;
      uint fill0;
      float alpha;
    }
    uBlock;
  $else:
    layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
      VECTOR data[];
    }
    uOutput;
    layout(set = 0, binding = 2) buffer PRECISION restrict readonly InBuffer2 {
      VECTOR data[];
    }
    uOther;
    layout(set = 0, binding = 3) uniform PRECISION Block {
      uint buf_length;
      uint fill0;
      float alpha;
    }
    uBlock;
  // clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Element-wise binary operation over two linear buffers:
 * out = OP(in, other, alpha).  Each invocation handles one 16-byte vector
 * (four elements); a small scalar tail covers lengths that are not a
 * multiple of four.  Buffer allocations are rounded up to 16 bytes so the
 * tail accesses never leave the allocation.
 */
void main() {
  const uint idx = gl_GlobalInvocationID.x;
  const uint n = uBlock.buf_length;
  const uint nvec = n >> 2u;
  const uint rem = n & 3u;

  if (idx < nvec) {
    // clang-format off
    $if not INPLACE:
      uOutput.data[idx] =
          OP(uInput.data[idx], uOther.data[idx], uBlock.alpha);
    $else:
      uOutput.data[idx] =
          OP(uOutput.data[idx], uOther.data[idx], uBlock.alpha);
    // clang-format on
  } else if (idx < nvec + rem) {
    const uint i = (nvec << 2u) + (idx - nvec);
    const uint j = i >> 2u;
    const uint k = i & 3u;
    // clang-format off
    $if not INPLACE:
      uOutput.data[j][k] =
          OP(uInput.data[j][k], uOther.data[j][k], uBlock.alpha);
    $else:
      uOutput.data[j][k] =
          OP(uOutput.data[j][k], uOther.data[j][k], uBlock.alpha);
    // clang-format on
  }
}
#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
#define DTYPE ${DTYPE}

#define OP(X) ${OPERATOR}
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
    layout(set = 0, binding = 2) uniform PRECISION Block {
      uint buf_length;
    }
    uBlock;
  $else:
    layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
      VECTOR data[];
    }
    uOutput;
    // Every variant's bindings must be numbered contiguously from zero: the
    // descriptor set layout is rebuilt from the ordered binding list, so a
    // reserved slot for the absent input buffer would shift the uniform block
    // out of the position the arguments are bound to.
    layout(set = 0, binding = 1) uniform PRECISION Block {
      uint buf_length;
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
    layout(set = 0, binding = 2) uniform PRECISION Block {
      uint buf_length;
    }
    uBlock;
  $else:
    layout(set = 0, binding = 0) buffer PRECISION restrict OutBuffer {
      VECTOR data[];
    }
    uOutput;
    // See the integer branch: the uniform block takes the first free binding
    // of this variant, since no input buffer is bound in place.
    layout(set = 0, binding = 1) uniform PRECISION Block {
      uint buf_length;
    }
    uBlock;
  // clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Applies a unary operation to every element of a linear buffer.  Each
 * invocation handles one 16-byte vector (four elements); a small scalar
 * tail covers lengths that are not a multiple of four.  Buffer allocations
 * are rounded up to 16 bytes so the tail accesses never leave the
 * allocation.
 */
void main() {
  const uint idx = gl_GlobalInvocationID.x;
  const uint n = uBlock.buf_length;
  const uint nvec = n >> 2u;
  const uint rem = n & 3u;

  if (idx < nvec) {
    // Apply the scalar operator to each 16-byte-vector component.  The
    // operator expressions are authored for scalar elements (some integer
    // variants cast X to float), so expand per component and reassemble.
    // clang-format off
    $if not INPLACE:
      uOutput.data[idx] = VECTOR(OP(uInput.data[idx].x), OP(uInput.data[idx].y), OP(uInput.data[idx].z), OP(uInput.data[idx].w));
    $else:
      uOutput.data[idx] = VECTOR(OP(uOutput.data[idx].x), OP(uOutput.data[idx].y), OP(uOutput.data[idx].z), OP(uOutput.data[idx].w));
    // clang-format on
  } else if (idx < nvec + rem) {
    const uint i = (nvec << 2u) + (idx - nvec);
    const uint j = i >> 2u;
    const uint k = i & 3u;
    // clang-format off
    $if not INPLACE:
      uOutput.data[j][k] = OP(uInput.data[j][k]);
    $else:
      uOutput.data[j][k] = OP(uOutput.data[j][k]);
    // clang-format on
  }
}
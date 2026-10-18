#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define DTYPE ${DTYPE}
#define INPLACE ${INPLACE}
// clang-format on

layout(std430) buffer;

$if DTYPE == "int":
  // Signed-word storage buffers: the bounds travel as int words and the clip
  // uses the integer min/max.
  #define VECTOR ivec4
  #define SCALAR int
$else:
  #define VECTOR vec4
  #define SCALAR float

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
    uint buf_length; // elements in the payload
    uint fill0;      // alignment
    SCALAR lo;       // lower bound
    SCALAR hi;       // upper bound
  }
  uBlock;
$else:
  layout(set = 0, binding = 0) buffer PRECISION OutBuffer {
    VECTOR data[];
  }
  uOutput;
  // This variant has no input buffer, so the params block takes the first free
  // binding number; the descriptor set layout is built from the ordered
  // binding list, so a reserved slot would misalign the arguments.
  layout(set = 0, binding = 1) uniform PRECISION Block {
    uint buf_length; // elements in the payload
    uint fill0;      // alignment
    SCALAR lo;       // lower bound
    SCALAR hi;       // upper bound
  }
  uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Clips every element of a linear buffer to the closed range [lo, hi].  One
 * invocation handles a 16-byte vector, so the bulk of the payload moves through
 * aligned 128-bit accesses; a scalar tail covers lengths that are not a multiple
 * of four.  Buffer allocations are rounded up to 16 bytes, so the tail never
 * reads or writes outside the allocation.
 *
 * A bound is meaningful only when it is present, so the caller folds the missing
 * side out of the range before dispatching; the kernel always receives a finite
 * pair.
 */
void main() {
  const uint idx = gl_GlobalInvocationID.x;
  const uint n = uBlock.buf_length;
  const uint nvec = n >> 2u;
  const uint rem = n & 3u;

  // clang-format off
  $if not INPLACE:
    #define SRC uInput
  $else:
    #define SRC uOutput
  // clang-format on

  if (idx < nvec) {
    const VECTOR x = SRC.data[idx];
    uOutput.data[idx] = VECTOR(
        min(max(x.x, uBlock.lo), uBlock.hi),
        min(max(x.y, uBlock.lo), uBlock.hi),
        min(max(x.z, uBlock.lo), uBlock.hi),
        min(max(x.w, uBlock.lo), uBlock.hi));
  } else if (idx < nvec + rem) {
    const uint i = (nvec << 2u) + (idx - nvec);
    const uint j = i >> 2u;
    const uint k = i & 3u;
    uOutput.data[j][k] = min(max(SRC.data[j][k], uBlock.lo), uBlock.hi);
  }
}

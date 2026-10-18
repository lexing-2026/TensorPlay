#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

#define FOUR 4

layout(std430) buffer;

/*
 * First pass of the decode product over a weight matrix held in buffer
 * storage, which is what every weight too wide or too tall for the device's
 * 3-D image axis limit ends up as.  One invocation owns one output column and
 * one chunk of the reduction axis.
 *
 * Two weight orientations reach this kernel.  A (K x N) payload is the
 * transposed form a matrix product asks for: four consecutive reduction
 * elements then sit N words apart and the packed activation texel folds them
 * into one accumulator step.  A (N x K) payload is the row-major form a linear
 * layer stores, where one output column reads a whole contiguous weight row
 * and no relayout of a multi-gigabyte operand is worth paying for.  Rows of
 * such a payload are walked four words at a time whenever the reduction length
 * divides by four: one fetch then moves sixteen bytes per lane instead of four,
 * and the block a lane lands in is finished by the same lane instead of being
 * revisited once per element.
 *
 * The activation operand rides a width-packed texture for as long as its own
 * row fits an image; past that point it too streams from a buffer.  Partial
 * results land in a linear arena addressed by (split, row, column); the fold
 * pass sums the split rows of a column back together.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Partials {
  float data[];
}
uPartials;
// clang-format off
$if LHS_BUFFER == 1:
  layout(set = 0, binding = 1) buffer PRECISION restrict readonly Mat1 {
    float data[];
  }
  uMat1;
$else:
  layout(set = 0, binding = 1) uniform PRECISION sampler3D uMat1;
// clang-format on
// clang-format off
$if RHS_NK == 1 and RHS_VEC4 == 1:
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly Mat2 {
    vec4 data[];
  }
  uMat2;
$else:
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly Mat2 {
    float data[];
  }
  uMat2;
// clang-format on
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_sizes; // (W=N, H=M, C=1, N=1) logical product sizes
  int k_elements;  // length of the reduction axis
  int chunk;       // reduction elements covered by one split
  int splits;
  int stride_n;    // element step to the next weight column
  int stride_k;    // element step to the next weight row
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// clang-format off
$if LHS_BUFFER == 1:
  float lhs_at(const int row, const int k) {
    return uMat1.data[row * uBlock.k_elements + k];
  }
  vec4 lhs_vec(const int row, const int j) {
    const int base = row * uBlock.k_elements + FOUR * j;
    return vec4(
        uMat1.data[base],
        uMat1.data[base + 1],
        uMat1.data[base + 2],
        uMat1.data[base + 3]);
  }
$else:
  float lhs_at(const int row, const int k) {
    const vec4 group = texelFetch(uMat1, ivec3(k / FOUR, row, 0), 0);
    const int lane = k - FOUR * (k / FOUR);
    return lane == 0
        ? group.x
        : (lane == 1 ? group.y : (lane == 2 ? group.z : group.w));
  }
  // A width-packed row holds one reduction group per texel, so the four
  // elements a vectorized weight step needs are already one fetch apart.
  vec4 lhs_vec(const int row, const int j) {
    return texelFetch(uMat1, ivec3(j, row, 0), 0);
  }
// clang-format on

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int col = pos.x;
  const int split = pos.y;
  const int row = pos.z;

  const int n = uBlock.out_sizes.x;
  const int m = uBlock.out_sizes.y;

  if (col >= n || split >= uBlock.splits || row >= m) {
    return;
  }

  const int k_begin = split * uBlock.chunk;
  const int k_end = min(k_begin + uBlock.chunk, uBlock.k_elements);

  float acc = 0.0;

  // clang-format off
  $if RHS_NK == 1 and RHS_VEC4 == 1:
    // Chunks open on a group boundary, since the host sizes them in multiples
    // of four, so both ends of the span sit inside one weight row.
    const int groups = uBlock.k_elements / FOUR;
    for (int j = k_begin / FOUR; j < k_end / FOUR; ++j) {
      acc += dot(lhs_vec(row, j), uMat2.data[col * groups + j]);
    }
  $elif RHS_NK == 1 or LHS_BUFFER == 1:
    for (int k = k_begin; k < k_end; ++k) {
      acc += lhs_at(row, k) *
          uMat2.data[col * uBlock.stride_n + k * uBlock.stride_k];
    }
  $else:
    // Chunks open on a texel boundary, so k_begin always starts a packed
    // group.  The far-edge guards drop the lanes a reduction length that is
    // not a multiple of four leaves in the last texel of the last chunk;
    // without them the tail would index weight rows that do not exist.
    const int j_begin = k_begin / FOUR;
    const int j_end = (k_end + FOUR - 1) / FOUR;
    for (int j = j_begin; j < j_end; ++j) {
      const vec4 a = texelFetch(uMat1, ivec3(j, row, 0), 0);
      const int k0 = FOUR * j;
      if (k0 < k_end) {
        acc += a.x * uMat2.data[k0 * n + col];
      }
      if (k0 + 1 < k_end) {
        acc += a.y * uMat2.data[(k0 + 1) * n + col];
      }
      if (k0 + 2 < k_end) {
        acc += a.z * uMat2.data[(k0 + 2) * n + col];
      }
      if (k0 + 3 < k_end) {
        acc += a.w * uMat2.data[(k0 + 3) * n + col];
      }
    }
  // clang-format on

  uPartials.data[(split * m + row) * n + col] = acc;
}

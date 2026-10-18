#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define LHS_BUFFER ${LHS_BUFFER}
#define RHS_NK ${RHS_NK}
#define RHS_VEC4 ${RHS_VEC4}
// clang-format on

#define FOUR 4

layout(std430) buffer;

/*
 * First pass of the half-precision decode product over a weight held in buffer
 * storage.  The weight has no texture form because its extents pass the device's
 * 3-D image axis limit, and its payload keeps two lanes in every 32-bit word,
 * low lane first, so every read goes through an unpack while rows stay
 * addressed by element.
 *
 * One invocation owns one output column and one chunk of the reduction axis, and
 * the accumulation stays in single precision: partials travel to the fold pass
 * as floats, so a long reduction never narrows between terms.  The two weight
 * orientations differ only in which step a fetch takes; a row-major (N x K)
 * payload whose length divides by four moves eight bytes per fetch, because one
 * aligned pair of words already carries the four elements a step consumes.
 *
 * The activation operand arrives either as its own linear payload or as a
 * width-packed image, and a half image hands its four lanes back as floats, so
 * the texel path needs no unpacking.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Partials {
  float data[];
}
uPartials;
// clang-format off
$if LHS_BUFFER == 1:
  $if RHS_VEC4 == 1:
    layout(set = 0, binding = 1) buffer PRECISION restrict readonly Mat1 {
      uvec2 data[];
    }
    uMat1;
  $else:
    layout(set = 0, binding = 1) buffer PRECISION restrict readonly Mat1 {
      uint data[];
    }
    uMat1;
$else:
  layout(set = 0, binding = 1) uniform PRECISION sampler3D uMat1;
// clang-format on
// clang-format off
$if RHS_VEC4 == 1:
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly Mat2 {
    uvec2 data[];
  }
  uMat2;
$else:
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly Mat2 {
    uint data[];
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
$if RHS_VEC4 == 1:
  // Four reduction elements land in one aligned pair of words, in order, so a
  // step reads eight bytes and unpacks them into one accumulator vector.
  vec4 weight_vec(const int col, const int j) {
    const int groups = uBlock.k_elements / FOUR;
    const uvec2 packed = uMat2.data[col * groups + j];
    return vec4(unpackHalf2x16(packed.x), unpackHalf2x16(packed.y));
  }
$else:
  float lane_of(const in uint word, const in uint lane) {
    const vec2 pair = unpackHalf2x16(word);
    return lane == 0u ? pair.x : pair.y;
  }
  float weight_at(const in int index) {
    return lane_of(uMat2.data[index >> 1], uint(index) & 1u);
  }
// clang-format on

// clang-format off
$if LHS_BUFFER == 1:
  $if RHS_VEC4 == 1:
    // A row opens on a group boundary, because the reduction length that
    // reaches this build divides by four.
    vec4 lhs_vec(const int row, const int j) {
      return vec4(
          unpackHalf2x16(uMat1.data[row * (uBlock.k_elements / FOUR) + j].x),
          unpackHalf2x16(uMat1.data[row * (uBlock.k_elements / FOUR) + j].y));
    }
  $else:
    float lhs_at(const int row, const int k) {
      const int index = row * uBlock.k_elements + k;
      return lane_of(uMat1.data[index >> 1], uint(index) & 1u);
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
  // elements a vectorized weight step needs are one fetch apart.
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
    for (int j = k_begin / FOUR; j < k_end / FOUR; ++j) {
      acc += dot(lhs_vec(row, j), weight_vec(col, j));
    }
  $elif RHS_NK == 1 or LHS_BUFFER == 1:
    for (int k = k_begin; k < k_end; ++k) {
      acc += lhs_at(row, k) *
          weight_at(col * uBlock.stride_n + k * uBlock.stride_k);
    }
  $else:
    // Chunks open on a texel boundary, so k_begin always starts a packed
    // activation group.  The far-edge guards drop the lanes a reduction length
    // that is not a multiple of four leaves in the last group of the last
    // chunk; without them the tail would index weight rows that do not exist.
    const int j_begin = k_begin / FOUR;
    const int j_end = (k_end + FOUR - 1) / FOUR;
    for (int j = j_begin; j < j_end; ++j) {
      const vec4 a = texelFetch(uMat1, ivec3(j, row, 0), 0);
      const int k0 = FOUR * j;
      if (k0 < k_end) {
        acc += a.x * weight_at(k0 * n + col);
      }
      if (k0 + 1 < k_end) {
        acc += a.y * weight_at((k0 + 1) * n + col);
      }
      if (k0 + 2 < k_end) {
        acc += a.z * weight_at((k0 + 2) * n + col);
      }
      if (k0 + 3 < k_end) {
        acc += a.w * weight_at((k0 + 3) * n + col);
      }
    }
  // clang-format on

  uPartials.data[(split * m + row) * n + col] = acc;
}

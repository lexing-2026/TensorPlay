#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * Fused root-mean-square normalization of the trailing axis over linear
 * storage.  A payload whose token axis is longer than the device's image axis
 * limit has no texture form, and the composite (square, mean, add, rsqrt,
 * multiply, multiply) would move the row through memory five times.  Here one
 * workgroup of 256 invocations owns one row: it folds the squares in a
 * coalesced stride, tree-reduces them in shared memory (about 1 KB per
 * workgroup, so a compute unit hosts many rows), and writes the scaled row
 * back from the values it already holds.  Rows of up to 1024 elements are
 * served entirely from registers; longer rows re-read their tail.
 *
 * The row index sits on the two workgroup axes beyond x, because the row count
 * is not bounded by a single dispatch axis.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  float data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  float data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int span;      // elements of the normalized trailing axis
  int rows;      // rows to normalize
  int rows_axis; // rows named by the second axis before the third takes over
  float inv_span; // 1 / span
  float eps;
  int pad0;
  int pad1;
  int pad2;
}
uBlock;
// clang-format off
$if WEIGHT_MODE == 1:
  layout(set = 0, binding = 3) buffer PRECISION restrict readonly Weight {
    float data[];
  }
  uWeight;
$elif WEIGHT_MODE == 2:
  layout(set = 0, binding = 3) uniform PRECISION sampler3D uWeight;
// clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const int kLocal = 256;
shared float sSq[kLocal];
shared float sRstd;

// The affine parameter is rank-1 along the normalized axis: a short weight
// stays an image and is read per position, a wide one is a flat array, and
// without a parameter every position scales by one.
float weight_at(const in int i) {
  // clang-format off
  $if WEIGHT_MODE == 1:
    return uWeight.data[i];
  $elif WEIGHT_MODE == 2:
    return texelFetch(uWeight, ivec3(i, 0, 0), 0).x;
  $else:
    return 1.0f;
  // clang-format on
}

void main() {
  const int x0 = int(gl_LocalInvocationID.x);
  // The workgroup, not the invocation, names the row: the x axis carries the
  // 256 lanes that stride along it.
  const int row =
      int(gl_WorkGroupID.y) + int(gl_WorkGroupID.z) * uBlock.rows_axis;
  if (row >= uBlock.rows) {
    return;
  }
  const int span = uBlock.span;
  const int base = row * span;

  // Constant-indexed locals keep the four strided positions out of private
  // memory: they cover a row of up to 1024 elements in a single fetch.
  const int i1 = x0 + kLocal;
  const int i2 = x0 + 2 * kLocal;
  const int i3 = x0 + 3 * kLocal;
  float k0 = 0.0f;
  float k1 = 0.0f;
  float k2 = 0.0f;
  float k3 = 0.0f;
  if (x0 < span) {
    k0 = uInput.data[base + x0];
  }
  if (i1 < span) {
    k1 = uInput.data[base + i1];
  }
  if (i2 < span) {
    k2 = uInput.data[base + i2];
  }
  if (i3 < span) {
    k3 = uInput.data[base + i3];
  }
  float acc = k0 * k0 + k1 * k1 + k2 * k2 + k3 * k3;
  for (int i = x0 + 4 * kLocal; i < span; i += kLocal) {
    const float v = uInput.data[base + i];
    acc += v * v;
  }

  sSq[x0] = acc;
  barrier();
  for (int stride = kLocal / 2; stride > 0; stride >>= 1) {
    if (x0 < stride) {
      sSq[x0] += sSq[x0 + stride];
    }
    barrier();
  }
  if (x0 == 0) {
    sRstd = inversesqrt(sSq[0] * uBlock.inv_span + uBlock.eps);
  }
  barrier();

  const float rstd = sRstd;
  if (x0 < span) {
    uOutput.data[base + x0] = k0 * rstd * weight_at(x0);
  }
  if (i1 < span) {
    uOutput.data[base + i1] = k1 * rstd * weight_at(i1);
  }
  if (i2 < span) {
    uOutput.data[base + i2] = k2 * rstd * weight_at(i2);
  }
  if (i3 < span) {
    uOutput.data[base + i3] = k3 * rstd * weight_at(i3);
  }
  for (int i = x0 + 4 * kLocal; i < span; i += kLocal) {
    uOutput.data[base + i] = uInput.data[base + i] * rstd * weight_at(i);
  }
}

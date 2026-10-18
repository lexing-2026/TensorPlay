#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define GREATER ${GREATER}
// clang-format on

layout(std430) buffer;

/*
 * Single-axis extremum-index reduction over buffer storage, for shapes
 * whose extents exceed the 3-D image axis limit.  Each sort row decomposes
 * into an outer block and an inner run; the 256 lanes of one workgroup walk
 * strided slices of the row, then a shared-memory reduction picks the
 * extremum value together with the smallest winning axis position.  Strict
 * comparisons implement the first-occurrence tie-break, which is what makes
 * greedy decoding pick the lowest index of a tied set.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
  int data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Input {
  float data[];
}
uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  int axis_size;
  int inner;
  int rows;
  int greater;
}
uBlock;

shared float s_value[256];
shared int s_index[256];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  // The workgroup, not the invocation, names the row: the x axis carries the
  // 256 lanes that stride along it, and the row index rides the y axis.
  const int row = int(gl_WorkGroupID.y);
  const int tid = int(gl_LocalInvocationIndex);
  if (row >= uBlock.rows) {
    return;
  }
  const int outer = row / uBlock.inner;
  const int inner_pos = row - outer * uBlock.inner;
  const int base = outer * (uBlock.axis_size * uBlock.inner);

  // The first lane position is the first candidate; lanes past the axis keep
  // an empty sentinel that loses every comparison (all-equal rows still pick
  // the smallest real position because its index is smaller than the sentinel).
  float best = -3.4e38;
  int best_index = tid;
  if (tid < uBlock.axis_size) {
    best = uInput.data[base + tid * uBlock.inner + inner_pos];
  }
  for (int i = tid + 256; i < uBlock.axis_size; i += 256) {
    const float value = uInput.data[base + i * uBlock.inner + inner_pos];
    const bool take = uBlock.greater != 0 ? value > best : value < best;
    if (take) {
      best = value;
      best_index = i;
    }
  }
  s_value[tid] = best;
  s_index[tid] = best_index;
  barrier();

  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const float a = s_value[tid];
      const float b = s_value[tid + stride];
      const int ia = s_index[tid];
      const int ib = s_index[tid + stride];
      // Equal values fall back to the smaller axis position.
      const bool b_wins = uBlock.greater != 0
          ? (b > a || (b == a && ib < ia))
          : (b < a || (b == a && ib < ia));
      if (b_wins) {
        s_value[tid] = b;
        s_index[tid] = ib;
      }
    }
    barrier();
  }

  if (tid == 0) {
    uOutput.data[row] = s_index[0];
  }
}
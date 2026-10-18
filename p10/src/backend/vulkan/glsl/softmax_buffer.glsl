#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define LOG_MODE ${LOG_MODE}
// clang-format on

layout(std430) buffer;

/*
 * Row-wise softmax (or log-softmax) over buffer storage, for shapes whose
 * extents exceed the 3-D image axis limit.  Each row decomposes into an
 * outer block and an inner run, and the whole 256-invocation workgroup
 * cooperates on the row: the local maxima are reduced through shared memory,
 * the max-subtracted exponentials are summed the same way, then every lane
 * normalizes the slice it owns.  The shared arrays match the 256-lane local
 * size the dispatcher always pairs with this shader.
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
  int axis_size;
  int inner;
  int rows;
}
uBlock;

shared float s_max[256];
shared float s_sum[256];

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

  float mx = -3.4e38;
  for (int i = tid; i < uBlock.axis_size; i += 256) {
    mx = max(mx, uInput.data[base + i * uBlock.inner + inner_pos]);
  }
  s_max[tid] = mx;
  barrier();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_max[tid] = max(s_max[tid], s_max[tid + stride]);
    }
    barrier();
  }
  const float row_mx = s_max[0];

  float total = 0.0;
  for (int i = tid; i < uBlock.axis_size; i += 256) {
    total += exp(uInput.data[base + i * uBlock.inner + inner_pos] - row_mx);
  }
  s_sum[tid] = total;
  barrier();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
    }
    barrier();
  }
  const float row_total = s_sum[0];

  for (int i = tid; i < uBlock.axis_size; i += 256) {
    const int index = base + i * uBlock.inner + inner_pos;
    const float shifted = uInput.data[index] - row_mx;
    // clang-format off
    $if LOG_MODE == 1:
      uOutput.data[index] = shifted - log(row_total);
    $else:
      uOutput.data[index] = exp(shifted) / row_total;
    // clang-format on
  }
}
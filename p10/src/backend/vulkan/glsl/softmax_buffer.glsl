#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define LOG_MODE ${LOG_MODE}
// clang-format on

layout(std430) buffer;

// Row-wise softmax (or log-softmax) over buffer storage, for shapes whose
// extents exceed the 3-D image axis limit.  Each row decomposes into an
// outer block and an inner run, and one invocation walks the softmax axis
// three times: maximum logit, exponential sum, then the normalized write.
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

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int row = int(gl_GlobalInvocationID.x);
  if (row >= uBlock.rows) {
    return;
  }
  const int outer = row / uBlock.inner;
  const int inner_pos = row - outer * uBlock.inner;
  const int base = outer * (uBlock.axis_size * uBlock.inner);

  float mx = uInput.data[base + inner_pos];
  for (int i = 1; i < uBlock.axis_size; ++i) {
    mx = max(mx, uInput.data[base + i * uBlock.inner + inner_pos]);
  }

  float total = 0.0;
  for (int i = 0; i < uBlock.axis_size; ++i) {
    total += exp(uInput.data[base + i * uBlock.inner + inner_pos] - mx);
  }

  for (int i = 0; i < uBlock.axis_size; ++i) {
    const int index = base + i * uBlock.inner + inner_pos;
    const float shifted = uInput.data[index] - mx;
    // clang-format off
    $if LOG_MODE == 1:
      uOutput.data[index] = shifted - log(total);
    $else:
      uOutput.data[index] = exp(shifted) / total;
    // clang-format on
  }
}
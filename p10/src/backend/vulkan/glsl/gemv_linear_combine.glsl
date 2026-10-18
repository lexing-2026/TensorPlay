#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

/*
 * Second pass of the decode product over buffer storage: fold the split
 * partial rows of one output column and apply the epilogue.  The addend
 * broadcasts over singleton rows or columns; the result keeps whichever
 * storage the product shape selected, so the same template serves a linear
 * output and an image output.
 */
// clang-format off
$if OUTPUT_BUFFER == 1:
  layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Output {
    float data[];
  }
  uOutput;
$else:
  layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
// clang-format on
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Partials {
  float data[];
}
uPartials;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 out_sizes;  // (W=N, H=M, C=1, N=1) logical product sizes
  ivec4 bias_sizes; // (W, H, C, N) logical sizes of the addend
  int splits;       // partial rows folded per output element
  int fill;
  float alpha;
  float beta;
}
uBlock;
// clang-format off
$if HAS_BIAS == 1:
  $if OUTPUT_BUFFER == 1:
    layout(set = 0, binding = 3) buffer PRECISION restrict readonly Bias {
      float data[];
    }
    uBias;
  $else:
    layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
// clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int col = pos.x;
  const int row = pos.y;

  const int n = uBlock.out_sizes.x;
  const int m = uBlock.out_sizes.y;

  if (col >= n || row >= m) {
    return;
  }

  float acc = 0.0;
  for (int s = 0; s < uBlock.splits; ++s) {
    acc += uPartials.data[(s * m + row) * n + col];
  }
  acc = uBlock.alpha * acc;

  // clang-format off
  $if HAS_BIAS == 1:
    const int col_eff = (uBlock.bias_sizes.x == 1) ? 0 : col;
    const int row_eff = (uBlock.bias_sizes.y == 1) ? 0 : row;
    $if OUTPUT_BUFFER == 1:
      acc += uBlock.beta *
          uBias.data[row_eff * uBlock.bias_sizes.x + col_eff];
    $else:
      acc += uBlock.beta * texelFetch(uBias, ivec3(col_eff, row_eff, 0), 0).x;
  // clang-format on

  // clang-format off
  $if OUTPUT_BUFFER == 1:
    uOutput.data[row * n + col] = acc;
  $else:
    imageStore(uOutput, ivec3(col, row, 0), vec4(acc));
  // clang-format on
}

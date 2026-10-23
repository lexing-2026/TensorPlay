#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// The weight operand rides the same texel lanes as the activation: four
// consecutive reduction elements per texel, one output column per image row.
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uMat1;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uMat2;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 out_sizes; // (W=N, H=M, C=1, N=1) logical product sizes
  ivec4 bias_sizes; // (W, H, C, N) logical sizes of the addend
  int k_texels; // reduction texels per output lane: ceil(K / 4)
  int fill;
  float alpha;
  float beta;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// A workgroup holds one block of output columns, each owned by a row of lanes
// spread across the reduction axis.  Columns per workgroup times lanes never
// exceeds 256.
shared vec4 lane_partials[256];

/*
 * Decode-shape product with the reduction axis running along texture width.
 *
 * A few output rows against a tall weight make the reduction the whole cost:
 * every weight texel is read exactly once, and nothing is reused, so the
 * kernel is a memory stream.  A stream needs its fetches contiguous, which the
 * width-packed operand supplies -- consecutive lanes take consecutive texels of
 * one column, and the walk to the next texel group steps by the whole workgroup
 * row, so a lane's fetches never leave its own column.
 *
 * Each lane accumulates one output column over all product rows, keeping a
 * partial per row in a texel lane of its own accumulator.  The column's partial
 * then folds across the lanes of its row in shared memory, and the workgroup's
 * first lane applies the epilogue and stores.  A column beyond the output width
 * reads zero texels and stores nothing, which covers a width that is not a
 * multiple of the workgroup's column block.
 */
void main() {
  const int lane = int(gl_LocalInvocationID.x);
  const int lanes = int(gl_WorkGroupSize.x);
  const int column = int(gl_GlobalInvocationID.y);
  const int rows = uBlock.out_sizes.y;

  vec4 acc = vec4(0.0);

  const int j_step = lanes * int(gl_NumWorkGroups.x);
  for (int j = int(gl_GlobalInvocationID.x); j < uBlock.k_texels;
      j += j_step) {
    const vec4 w = texelFetch(uMat2, ivec3(j, column, 0), 0);
    for (int m = 0; m < rows; ++m) {
      acc[m] += dot(w, texelFetch(uMat1, ivec3(j, m, 0), 0));
    }
  }

  lane_partials[lane + lanes * int(gl_LocalInvocationID.y)] = acc;
  barrier();

  if (lane != 0) {
    return;
  }

  const int block = int(gl_LocalInvocationID.y);
  vec4 total = lane_partials[lanes * block];
  for (int i = 1; i < lanes; ++i) {
    total += lane_partials[i + lanes * block];
  }

  if (column >= uBlock.out_sizes.x) {
    return;
  }

  // The addend broadcasts over a singleton column or row; a zero beta term
  // skips the fetch, which is how a bias-free product uses this pass (the
  // binding then carries an operand texture that is never read).
  const int col_eff = (uBlock.bias_sizes.x == 1) ? 0 : column;
  for (int m = 0; m < rows; ++m) {
    float bias_val = 0.0f;
    if (uBlock.beta != 0.0f) {
      const int row_eff = (uBlock.bias_sizes.y == 1) ? 0 : m;
      bias_val = texelFetch(uBias, ivec3(col_eff, row_eff, 0), 0).x;
    }
    imageStore(
        uOutput,
        ivec3(column, m, 0),
        vec4(uBlock.beta * bias_val + uBlock.alpha * total[m]));
  }
}

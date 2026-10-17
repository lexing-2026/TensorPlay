#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

#define FOUR 4

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uPartials;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uBias;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_sizes;  // (W=N, H=M, C=1, N=1) logical sizes
  ivec4 bias_sizes; // (W, H, C, N) logical sizes of the addend
  int splits;       // number of partial rows folded per output
  float alpha;
  float beta;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Second pass of the two-pass decode matrix-vector product.  One invocation
 * owns one output row and one four-wide column texel; it folds the split
 * partials straight out of the width-packed arena and applies the product
 * epilogue.  The addend broadcasts over singleton rows/columns the same way
 * the wide product path handles it, and a zero beta term skips the fetch,
 * which is how the bias-free product reaches this pass (the binding then
 * carries a stand-in texture that is never read).
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (FOUR * pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w) {
    return;
  }

  vec4 acc = vec4(0.0);
  for (int s = 0; s < uBlock.splits; ++s) {
    acc += texelFetch(
        uPartials,
        ivec3(pos.x, s * uBlock.out_sizes.y + pos.y, 0),
        0);
  }

  for (int c = 0; c < FOUR; ++c) {
    const int col = FOUR * pos.x + c;
    if (col >= uBlock.out_sizes.x) {
      break;
    }
    float bias_val = 0.0f;
    if (uBlock.beta != 0.0f) {
      const int col_eff = (uBlock.bias_sizes.x == 1) ? 0 : col;
      const int row_eff = (uBlock.bias_sizes.y == 1) ? 0 : pos.y;
      bias_val = texelFetch(uBias, ivec3(col_eff, row_eff, 0), 0).x;
    }
    imageStore(
        uOutput,
        ivec3(col, pos.y, pos.z),
        vec4(uBlock.beta * bias_val + uBlock.alpha * acc[c]));
  }
}

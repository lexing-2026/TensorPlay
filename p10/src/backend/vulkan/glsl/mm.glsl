#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uMat1;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uMat2;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_sizes; // {M, N} as (W=N, H=M, C=1, N=1)
  ivec4 in1_sizes; // {M, K} as (W=K, H=M, C=1, N=1)
  ivec4 in2_sizes; // {K, N} as (W=N, H=K, C=1, N=1)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Naive matrix product: one invocation computes one output element by
 * walking the shared K axis.  Both operands and the result carry a single
 * channel, so each element lives in lane x of its texel.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.z) {
    return;
  }

  const int K = uBlock.in1_sizes.x;
  const int m = pos.y;
  const int n = pos.x;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc += texelFetch(uMat1, ivec3(k, m, 0), 0).x *
        texelFetch(uMat2, ivec3(n, k, 0), 0).x;
  }

  imageStore(uOutput, pos, vec4(acc));
}

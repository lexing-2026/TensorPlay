#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// True Int8 GEMM with per-output-channel requantization:
//   out[m, n] = clamp(roundEven((input_scale * w_scale[n] * acc + bias[n])
//                               / out_scale) + out_zero_point,
//                     quant_min, quant_max)
// where acc = sum_k (x[m,k] - x_zp) * (w[n,k] - w_zp[n]) accumulates in
// int32.
//
// Quantized tensors are stored channel-packed, which for a 2-D [M, K]
// tensor yields image extents {K, M, 1}: texel (k, m) holds element
// [m, k] in its first lane (the remaining lanes are padding).  Weights
// [N, K] are {K, N, 1}; the per-channel params texture is {N, 3, 1} with
// scale / zero point / bias in rows 0/1/2.  One invocation computes one
// output element.
layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION isampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uParams;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  int out_m;
  int out_n;
  int k;
  float input_scale;
  int input_zero_point;
  float out_inv_scale; // 1 / out_scale
  int out_zero_point;
  int quant_min;
  int quant_max;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
  const int n = pos.x;
  const int m = pos.y;

  if (n >= uBlock.out_n || m >= uBlock.out_m) {
    return;
  }

  // The params texture is width-packed along N (four output channels per
  // texel), matching the host upload layout.
  const float w_scale = texelFetch(uParams, ivec3(n >> 2, 0, 0), 0)[n & 3];
  const int w_zp = int(texelFetch(uParams, ivec3(n >> 2, 1, 0), 0)[n & 3]);
  const float bias = texelFetch(uParams, ivec3(n >> 2, 2, 0), 0)[n & 3];

  int acc = 0;
  for (int k = 0; k < uBlock.k; ++k) {
    const int x = texelFetch(uInput, ivec3(k, m, 0), 0).x;
    const int w = texelFetch(uWeight, ivec3(k, n, 0), 0).x;
    acc += (x - uBlock.input_zero_point) * (w - w_zp);
  }

  const float out_f =
      uBlock.input_scale * w_scale * float(acc) + bias;
  const int q = clamp(
      int(roundEven(out_f * uBlock.out_inv_scale)) +
          uBlock.out_zero_point,
      uBlock.quant_min,
      uBlock.quant_max);
  imageStore(uOutput, ivec3(n, m, 0), ivec4(q, 0, 0, 0));
}
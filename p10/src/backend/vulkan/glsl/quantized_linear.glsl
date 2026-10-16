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
// int32.  Activations [M, K] and weights [N, K] are signed-byte textures in
// the width-packed layout: texel (k4, row) holds four consecutive K
// elements in its lanes.  Per-channel scale / zero point / bias ride in a
// 3-row float texture indexed by output channel, also width-packed.
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

/*
 * One invocation computes a 4x4 block of the output: four rows of M times
 * four columns of N.  For every packed K texel it fetches four activation
 * texels and four weight texels, so the reduction axis is amortized over
 * sixteen output elements.  A scalar tail handles K not divisible by four
 * (the packed texture zero-fills the remainder only if the uploader did,
 * which is not guaranteed, so the tail is accumulated explicitly).
 */
void main() {
  const ivec2 block = ivec2(gl_GlobalInvocationID.xy);
  const int n4 = block.x;
  const int m4 = block.y;

  const int n_texels = (uBlock.out_n + 3) / 4;
  const int m_texels = (uBlock.out_m + 3) / 4;
  if (n4 >= n_texels || m4 >= m_texels) {
    return;
  }

  const int k_texels = uBlock.k / 4;
  const int k_rem = uBlock.k & 3;

  const int m_base = 4 * m4;
  const int n_base = 4 * n4;

  const vec4 w_scale4 = texelFetch(uParams, ivec3(n4, 0, 0), 0);
  const ivec4 w_zp4 = ivec4(texelFetch(uParams, ivec3(n4, 1, 0), 0));
  const vec4 bias4 = texelFetch(uParams, ivec3(n4, 2, 0), 0);

  int acc[4][4];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      acc[r][c] = 0;
    }
  }

  for (int k4 = 0; k4 < k_texels; ++k4) {
    ivec4 x[4];
    ivec4 w[4];
    for (int r = 0; r < 4; ++r) {
      x[r] = texelFetch(uInput, ivec3(k4, m_base + r, 0), 0);
    }
    for (int c = 0; c < 4; ++c) {
      w[c] = texelFetch(uWeight, ivec3(k4, n_base + c, 0), 0);
    }
    for (int r = 0; r < 4; ++r) {
      const ivec4 xq = x[r] - ivec4(uBlock.input_zero_point);
      for (int c = 0; c < 4; ++c) {
        const ivec4 wq = w[c] - ivec4(w_zp4[c]);
        acc[r][c] += xq.x * wq.x + xq.y * wq.y + xq.z * wq.z + xq.w * wq.w;
      }
    }
  }

  if (k_rem > 0) {
    const int k4 = k_texels;
    for (int r = 0; r < 4; ++r) {
      const ivec4 x = texelFetch(uInput, ivec3(k4, m_base + r, 0), 0);
      for (int c = 0; c < 4; ++c) {
        const ivec4 w = texelFetch(uWeight, ivec3(k4, n_base + c, 0), 0);
        for (int t = 0; t < k_rem; ++t) {
          const int xq = x[t] - uBlock.input_zero_point;
          const int wq = w[t] - w_zp4[c];
          acc[r][c] += xq * wq;
        }
      }
    }
  }

  for (int r = 0; r < 4; ++r) {
    const int m = m_base + r;
    if (m >= uBlock.out_m) {
      continue;
    }
    ivec4 out4;
    for (int c = 0; c < 4; ++c) {
      const int n = n_base + c;
      if (n >= uBlock.out_n) {
        out4[c] = 0;
        continue;
      }
      const float out_f =
          uBlock.input_scale * w_scale4[c] * float(acc[r][c]) + bias4[c];
      out4[c] = clamp(
          int(roundEven(out_f * uBlock.out_inv_scale)) +
              uBlock.out_zero_point,
          uBlock.quant_min,
          uBlock.quant_max);
    }
    imageStore(uOutput, ivec3(n4, m, 0), out4);
  }
}
#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// Fused Int8 GEMM with per-channel weight requantization: activations [M, K]
// and weights [N, K] are signed-byte textures (one byte per lane), the
// per-output-channel scale / zero point / bias triple rides in a 3-row float
// texture, and the result is Float32 [M, N].
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION isampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uParams;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  int out_m;
  int out_n;
  int k;
  float input_scale;
  int input_zero_point;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * One invocation computes one output element.  Activation (m, k) sits at
 * texel (k, m) and weight (n, k) at texel (n, k); the params texture rows
 * carry weight scale (row 0), weight zero point (row 1) and bias (row 2)
 * indexed by the output channel n.  The dot product accumulates in the
 * dequantized domain, so the sum is a float with the usual reassociation
 * tolerance.  out[m, n] = input_scale * weight_scales[n] *
 * Σ_k (x[m,k] - x_zp) * (w[n,k] - w_zp[n]) + bias[n].
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_n || pos.y >= uBlock.out_m || pos.z > 0) {
    return;
  }

  const int m = pos.y;
  const int n = pos.x;

  float acc = 0.0f;
  for (int k = 0; k < uBlock.k; ++k) {
    const float x = float(texelFetch(uInput, ivec3(k, m, 0), 0).x) -
        float(uBlock.input_zero_point);
    const float w = float(texelFetch(uWeight, ivec3(k, n, 0), 0).x) -
        texelFetch(uParams, ivec3(n, 1, 0), 0).x;
    acc += x * w;
  }

  const float w_scale = texelFetch(uParams, ivec3(n, 0, 0), 0).x;
  const float bias = texelFetch(uParams, ivec3(n, 2, 0), 0).x;

  imageStore(
      uOutput, pos, vec4(uBlock.input_scale * w_scale * acc + bias));
}

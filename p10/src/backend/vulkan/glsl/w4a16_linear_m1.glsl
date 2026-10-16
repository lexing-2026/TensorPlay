#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Single-row W4A16 linear for the decode shape (M == 1): one thread per
// output channel streams the width-packed FP16 activation row (two
// texels = 8 halves per iteration) against the packed 4-bit weight codes
// (one texel = 8 nibbles), accumulating in FP32.  This avoids the tiled
// GEMM's shared-memory staging and its 8x8 workgroup startup for rows with
// a single token.
layout(set = 0, binding = 0, rgba16f) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uMat1;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uMat2;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uParams;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  int k;     // logical reduction width
  int out_n; // output channels
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

int nib0(int byte) {
  const uint u = uint(byte);
  const int v = int(u & 0xFu);
  return (v >= 8) ? (v - 16) : v;
}

int nib1(int byte) {
  const uint u = uint(byte);
  const int v = int((u >> 4u) & 0xFu);
  return (v >= 8) ? (v - 16) : v;
}

void main() {
  const int n = int(gl_GlobalInvocationID.x);
  if (n >= uBlock.out_n) {
    return;
  }

  float acc = 0.0;
  const int k_texels_w = (uBlock.k + 7) / 8;
  const int k_texels_act = (uBlock.k + 3) / 4;
  for (int k8 = 0; k8 < k_texels_w; ++k8) {
    const int jtex = k8 * 2;
    const vec4 a0 = texelFetch(uMat1, ivec3(jtex, 0, 0), 0);
    const vec4 a1 = (jtex + 1 < k_texels_act)
        ? texelFetch(uMat1, ivec3(jtex + 1, 0, 0), 0)
        : vec4(0.0);
    const ivec4 b = texelFetch(uMat2, ivec3(n, k8, 0), 0);
    acc += a0.x * float(nib0(b.x)) + a0.y * float(nib1(b.x)) +
        a0.z * float(nib0(b.y)) + a0.w * float(nib1(b.y)) +
        a1.x * float(nib0(b.z)) + a1.y * float(nib1(b.z)) +
        a1.z * float(nib0(b.w)) + a1.w * float(nib1(b.w));
  }

  const float w_scale = texelFetch(uParams, ivec3(n >> 2, 0, 0), 0)[n & 3];
  const float bias = texelFetch(uParams, ivec3(n >> 2, 2, 0), 0)[n & 3];
  imageStore(uOutput, ivec3(n, 0, 0), vec4(w_scale * acc + bias, 0, 0, 0));
}
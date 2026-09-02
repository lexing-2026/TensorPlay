#version 450 core
// clang-format off
#define PRECISION ${PRECISION}

#define HAS_BIAS ${HAS_BIAS}
// clang-format on

layout(std430) buffer;

// Fused quantized 2D convolution: signed-byte input and weight textures,
// a float bias texture, and a signed-byte output texture.  Operands are
// dequantized on read, the convolution accumulates in the float domain, and
// the output is requantized with round-to-nearest-even into [-128, 127].
layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uInput;
layout(set = 0, binding = 2, rgba8i) uniform PRECISION restrict readonly iimage3D uWeight;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;
layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  ivec4 in_sizes;      // (W, H, C, N) logical sizes
  ivec4 out_sizes;     // (OW, OH, O, N) logical sizes
  ivec4 weight_sizes;  // .x = O, .y = C, .z = KH, .w = KW
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
  int in_c_depth;      // ceil(C / 4)
  int out_c_depth;     // ceil(O / 4)
  int weight_c_depth;  // ceil(C / 4)
  float in_scale;
  int in_zero_point;
  float weight_scale;
  int weight_zero_point;
  float inv_out_scale;
  int out_zero_point;
}
uBlock;

#include "param_fetch.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * One invocation computes the four output channels of one texel: the kernel
 * window walks width/height and the input channel axis, each byte is
 * dequantized as (q - zero_point) * scale, the products accumulate into the
 * float accumulator, and the bias (already in the float domain) is added
 * before the requantization.  Weight addressing follows the float conv2d
 * shader: weight texel z is (output channel * weight_c_depth + input
 * channel group) and the lane picks the input channel.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  const int ow = pos.x;
  const int oh = pos.y;
  const int n = pos.z / uBlock.out_c_depth;
  const int o4 = pos.z % uBlock.out_c_depth;

  vec4 acc = vec4(0.0f);

  for (int ky = 0; ky < uBlock.weight_sizes.z; ++ky) {
    const int ih =
        oh * uBlock.stride.y - uBlock.padding.y + ky * uBlock.dilation.y;
    if (ih < 0 || ih >= uBlock.in_sizes.y) {
      continue;
    }
    for (int kx = 0; kx < uBlock.weight_sizes.w; ++kx) {
      const int iw =
          ow * uBlock.stride.x - uBlock.padding.x + kx * uBlock.dilation.x;
      if (iw < 0 || iw >= uBlock.in_sizes.x) {
        continue;
      }
      for (int ci = 0; ci < uBlock.in_sizes.z; ++ci) {
        const float v = (float(imageLoad(
                             uInput,
                             ivec3(iw, ih, n * uBlock.in_c_depth + ci / 4))[ci % 4]) -
                         float(uBlock.in_zero_point)) *
            uBlock.in_scale;

        const int o = o4 * 4;
        vec4 w;
        for (int lane = 0; lane < 4; ++lane) {
          w[lane] =
              (float(imageLoad(
                   uWeight,
                   ivec3(
                       kx,
                       ky,
                       (o + lane) * uBlock.weight_c_depth + ci / 4))[ci % 4]) -
               float(uBlock.weight_zero_point)) *
              uBlock.weight_scale;
        }
        acc += v * w;
      }
    }
  }

  // clang-format off
  $if HAS_BIAS:
    acc += param_vec(uBias, o4, uBlock.out_sizes.z);
  // clang-format on

  const ivec4 rounded =
      ivec4(roundEven(acc * uBlock.inv_out_scale)) + ivec4(uBlock.out_zero_point);
  imageStore(uOutput, pos, clamp(rounded, ivec4(-128), ivec4(127)));
}

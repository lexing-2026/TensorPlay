#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}

#define HAS_BIAS ${HAS_BIAS}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION sampler3D uKernel;
layout(set = 0, binding = 3) uniform PRECISION sampler3D uBias;

layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  int in_length;
  int kernel_size;
  int strides;
  int padding;
  int dilation;
  int in_group_size;
  int out_group_size;
  int batch_size;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Grouped (single group) 1D convolution.
 *
 * input = (N, in_C, in_L), output = (N, out_C, out_L), kernel = K:
 * weight = (out_C, in_C, K), bias = (out_C,).
 *
 * One invocation computes the rolling kernel of the length dimension for
 * each batch, i.e. out_L * N results for one output channel.  The 3d tensors
 * are channel-packed along the batch dimension, so four batches are read and
 * written at a time.  The kernel texture is width-packed along the length
 * dimension, so four consecutive taps cost one fetch:
 *   texel (k / 4, in_c, out_c) carries w[out_c][in_c][4 * (k / 4) .. + 3].
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const int in_length = uBlock.in_length;
  const int kernel_size = uBlock.kernel_size;
  const int strides = uBlock.strides;
  const int padding = uBlock.padding;
  const int dilation = uBlock.dilation;
  const int in_group_size = uBlock.in_group_size;
  const int out_group_size = uBlock.out_group_size;
  const int batch_size = uBlock.batch_size;

  // "out_c" is the output's channel index where we write our result.
  // Across shader invocations, this is the only value that varies.
  int out_c = pos.y;
  vec4 bias = texelFetch(uBias, ivec3(out_c, 0, 0), 0);

  // "in_c" tracks the input's channel start index.  We iterate over the
  // input group that corresponds to the output group.
  int c_start = (out_c / out_group_size) * in_group_size;
  int c_end = c_start + in_group_size;

  // "in_l" tracks the input's length start index for our input-kernel
  // overlay region.
  int l_start = -padding;
  int l_end = in_length + padding - dilation * (kernel_size - 1);

  // Since the input/output tensors are channel-packed, which is along the
  // batch dimension, we can batch-read/write four elements at a time.
  for (int n = 0; n < batch_size; n += 4) {
    // "out_l" tracks the output's length index where we write our result.
    int out_l = 0;

    for (int in_l = l_start; in_l < l_end; in_l += strides, ++out_l) {
      vec4 sum = vec4(0, 0, 0, 0);

      for (int in_c = c_start; in_c < c_end; ++in_c) {
        // "k" tracks the kernel's index for our input-kernel computation.
        // It reads out-of-bound zeros, but trying to avoid them complicates
        // for-loop conditions, which results in worse performance.
        for (int k = 0; k < kernel_size; k += 4) {
          // Since the weight tensor is width-packed, which is along the
          // length dimension, we can batch-read four elements at a time.
          const ivec3 w_pos = ivec3(k / 4, in_c % in_group_size, out_c);
          const vec4 weight = texelFetch(uKernel, w_pos, 0);

          const ivec3 in_pos_0 = ivec3(in_l + k * dilation, in_c, n / 4);
          sum = fma(weight.xxxx, texelFetch(uInput, in_pos_0, 0), sum);

          const ivec3 in_pos_1 = ivec3(in_l + (k + 1) * dilation, in_c, n / 4);
          sum = fma(weight.yyyy, texelFetch(uInput, in_pos_1, 0), sum);

          const ivec3 in_pos_2 = ivec3(in_l + (k + 2) * dilation, in_c, n / 4);
          sum = fma(weight.zzzz, texelFetch(uInput, in_pos_2, 0), sum);

          const ivec3 in_pos_3 = ivec3(in_l + (k + 3) * dilation, in_c, n / 4);
          sum = fma(weight.wwww, texelFetch(uInput, in_pos_3, 0), sum);
        }
      }

      ivec3 out_pos = ivec3(out_l, out_c, n / 4);
      // clang-format off
      $if HAS_BIAS:
        imageStore(uOutput, out_pos, sum + bias.x);
      $else:
        imageStore(uOutput, out_pos, sum);
      // clang-format on
    }
  }
}

#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // input sizes (W, H, C, N); the scan axis runs over W
  ivec4 in_sizes;
  int fill;
  int fill1;
  int fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Inclusive prefix product along the width axis.  Each invocation
 * accumulates the input row up to and including its own position; the
 * small tensors targeted by this backend keep the linear walk affordable.
 * Channels and batches scan independently through their texel lanes.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.in_sizes.x || pos.y >= uBlock.in_sizes.y ||
      pos.z >= uBlock.in_sizes.w * ((uBlock.in_sizes.z + 3) / 4)) {
    return;
  }

  vec4 acc = vec4(1.0f);
  for (int x = 0; x <= pos.x; ++x) {
    acc *= texelFetch(uInput, ivec3(x, pos.y, pos.z), 0);
  }
  imageStore(uOutput, pos, acc);
}

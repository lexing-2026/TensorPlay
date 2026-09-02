#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  ivec4 in_sizes; // (W, H, C, N) of the input
  ivec4 out_sizes; // (W, H, C, N) of the output
  int axis; // 0: W, 1: H, 2: C, 3: N
  int in_c_depth; // ceil(in C / 4)
  int out_c_depth; // ceil(out C / 4)
  float scale; // multiplied into the accumulator (1 for plain sum)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Single-axis reduction.  The output carries the reduced axis with length
 * one (or the equivalent collapsed layout for a trailing squeeze), so every
 * invocation aggregates a full line of the input axis.  The accumulator is
 * a vec4: for width/height/batch reductions lanes reduce independently,
 * while a channel reduction collapses all lanes into one value that is then
 * replicated across the output lanes.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  vec4 acc = vec4(0.0f);

  if (uBlock.axis == 0) {
    // Width: (0, y, z) aggregates x = 0 .. in_W-1; channel and batch are
    // untouched so the texel z coordinate carries over directly.
    for (int x = 0; x < uBlock.in_sizes.x; ++x) {
      acc += texelFetch(uInput, ivec3(x, pos.y, pos.z), 0);
    }
    imageStore(uOutput, pos, acc * uBlock.scale);
  } else if (uBlock.axis == 1) {
    // Height: (x, 0, z) aggregates y = 0 .. in_H-1.
    for (int y = 0; y < uBlock.in_sizes.y; ++y) {
      acc += texelFetch(uInput, ivec3(pos.x, y, pos.z), 0);
    }
    imageStore(uOutput, pos, acc * uBlock.scale);
  } else if (uBlock.axis == 2) {
    // Channel: (x, y, n) aggregates every channel of batch n.  Lanes beyond
    // the channel count are masked, the lane results are collapsed, and the
    // total is replicated so later stages see a consistent value per lane.
    const int n = pos.z;
    const vec4 lane_mask = vec4(lessThan(
        ivec4(0, 1, 2, 3), ivec4(uBlock.in_sizes.z)));
    for (int c4 = 0; c4 < uBlock.in_c_depth; ++c4) {
      acc += lane_mask *
          texelFetch(uInput, ivec3(pos.x, pos.y, n * uBlock.in_c_depth + c4),
                     0);
    }
    const float total =
        (acc.x + acc.y) + (acc.z + acc.w);
    imageStore(uOutput, ivec3(pos.x, pos.y, n), vec4(total * uBlock.scale));
  } else {
    // Batch: (x, y, c4) aggregates n = 0 .. in_N-1; one batch step moves
    // in_c_depth texels along z.  Lanes reduce independently.
    for (int n = 0; n < uBlock.in_sizes.w; ++n) {
      acc += texelFetch(
          uInput, ivec3(pos.x, pos.y, n * uBlock.in_c_depth + pos.z), 0);
    }
    imageStore(uOutput, pos, acc * uBlock.scale);
  }
}

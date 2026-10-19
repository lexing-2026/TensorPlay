#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// Apply pass of a scatter that moves a payload, over image storage: one
// invocation owns one whole texel of the destination and rewrites it in place,
// so the lanes no mark names keep their values and no invocation races another
// over a shared texel.  A mark holds the position of the entry that named its
// destination, and the payload is read at that position: the source is walked
// in the same flat order as the index list, so it rides linear storage whatever
// its own shape has an image form of.
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uSelf;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Marks {
  int data[];
}
uMarks;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Src {
  vec4 data[];
}
uSrc;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 self_sizes; // (width, height, channels, batch) of the destination
  int planes_per_batch; // channel groups of four to a texel, stacked per batch
  int fill;
  int fill2;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x >= uBlock.self_sizes.x || pos.y >= uBlock.self_sizes.y) {
    return;
  }

  const int batch = pos.z / uBlock.planes_per_batch;
  const int channel_base = (pos.z % uBlock.planes_per_batch) * 4;

  vec4 value = imageLoad(uSelf, pos);
  bool touched = false;
  for (int lane = 0; lane < 4; ++lane) {
    const int channel = channel_base + lane;
    if (channel >= uBlock.self_sizes.z) {
      continue;
    }
    // The flat element this lane stands for, in the row-major order of the
    // destination: the depth axis stacks each batch's channel groups, the two
    // texel axes carry the rest of the shape.
    const int element =
        ((batch * uBlock.self_sizes.z + channel) * uBlock.self_sizes.y +
         pos.y) *
            uBlock.self_sizes.x +
        pos.x;
    const int entry = uMarks.data[element];
    if (entry > 0) {
      const int source = entry - 1;
      value[lane] = uSrc.data[source >> 2][source & 3];
      touched = true;
    }
  }
  if (touched) {
    imageStore(uSelf, pos, value);
  }
}

#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

// Apply pass of a scatter that moves a half payload, over image storage.  Two
// halves share a word of the source, so a lane reaches into one of them without
// disturbing the other, and the value it finds is stored into the texel field
// of the same width.  Reading the pair and taking one half of it costs one
// unpack per lane rather than a pass that rebuilds the whole source first.
layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uSelf;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Marks {
  int data[];
}
uMarks;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly Src {
  uint data[];
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
    const int element =
        ((batch * uBlock.self_sizes.z + channel) * uBlock.self_sizes.y +
         pos.y) *
            uBlock.self_sizes.x +
        pos.x;
    const int entry = uMarks.data[element];
    if (entry > 0) {
      const int source = entry - 1;
      value[lane] =
          unpackHalf2x16(uSrc.data[source >> 1])[source & 1];
      touched = true;
    }
  }
  if (touched) {
    imageStore(uSelf, pos, value);
  }
}

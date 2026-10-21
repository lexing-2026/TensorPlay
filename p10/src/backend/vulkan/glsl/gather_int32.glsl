#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Row gather over image storage of a payload that counts: the same walk the
// single-precision one makes, with signed texels because a code is a word and
// not a measure.  One invocation still produces one output texel and resolves
// all four of its channel lanes before the store, since an image has no
// read-modify-write and a lane left to another invocation would be lost.
//
// The gathered axis is one of the two texel axes (the width or the height), so a
// code only moves the fetch along that axis; a gather of the channel or batch
// axis runs over linear storage.
layout(set = 0, binding = 0, rgba32i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION isampler3D uSelf;
layout(set = 0, binding = 2) uniform PRECISION isampler3D uIndices;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_sizes;            // (width, height, channels, batch) of the output
  int self_planes_per_batch; // ceil(self channels / 4)
  int gather_width;          // the gathered axis is the width, not the height
  int self_rows;             // entries available along the gathered axis
  int fill;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// The flat element a channel of this texel stands for, in the row-major order of
// the index payload: the depth axis stacks the channel groups of every batch
// (four channels to a texel, ceil(channels / 4) planes to a batch), and the two
// texel axes carry the remaining shape.
int flat_element(int batch, int channel, ivec3 pos) {
  const int planes = int(ceil(uBlock.out_sizes.z / 4.0));
  return ((batch * uBlock.out_sizes.z + channel) * uBlock.out_sizes.y + pos.y) *
      uBlock.out_sizes.x + pos.x;
}

// One lane of the output texel.  A code naming no row of the source contributes
// a zero lane, the same way the row lookups of this backend answer an
// out-of-range request.
#define GATHER_LANE(LANE, COMP)                                                  \
  if (channel_base + LANE < uBlock.out_sizes.z) {                                \
    const int code = texelFetch(                                                 \
        uIndices, ivec3(flat_element(batch, channel_base + LANE, pos), 0, 0), 0) \
        .x;                                                                      \
    if (code >= 0 && code < uBlock.self_rows) {                                  \
      const ivec3 source = uBlock.gather_width == 1                              \
          ? ivec3(code, pos.y, self_z)                                           \
          : ivec3(pos.x, code, self_z);                                          \
      value.COMP = texelFetch(uSelf, source, 0).COMP;                            \
    }                                                                            \
  }

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y) {
    return;
  }

  const int planes = int(ceil(uBlock.out_sizes.z / 4.0));
  const int batch = pos.z / planes;
  const int channel_base = (pos.z % planes) * 4;
  // The source plane of the same batch and channel group, counted with the
  // source's own number of groups per batch.
  const int self_z = batch * uBlock.self_planes_per_batch + pos.z % planes;

  ivec4 value = ivec4(0, 0, 0, 0);
  GATHER_LANE(0, x)
  GATHER_LANE(1, y)
  GATHER_LANE(2, z)
  GATHER_LANE(3, w)
  imageStore(uOutput, pos, value);
}

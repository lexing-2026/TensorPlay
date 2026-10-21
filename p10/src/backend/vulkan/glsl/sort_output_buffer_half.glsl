#version 450 core
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly HalfBuffer { uint data[]; } values;
layout(set = 0, binding = 1) buffer restrict writeonly IntBuffer { int data[]; } indices;
layout(set = 0, binding = 2) buffer restrict readonly Input { uvec2 data[]; } src;
layout(set = 0, binding = 3) uniform highp restrict Block {
  ivec4 in_sizes;
  ivec4 out_sizes;
  int axis_size;
  int inner;
  int rows;
  int tiles;
  int descending;
  int run;
  int count;
  int start;
} uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Final gather for a half payload kept in linear storage.  Two neighbouring
 * elements share one word, so an invocation owns a whole word: it resolves
 * both of its elements through the (outer, slot, inner) split the float
 * variant uses, reads their sorted pairs, and packs the two widenings back
 * into narrow lanes with a single write.  A word past the last element of an
 * odd-length payload leaves its upper lane zero, which no reader reaches.
 */
uvec2 pair_at(const in int lin) {
  int outer = lin / (uBlock.count * uBlock.inner);
  int rem = lin - outer * (uBlock.count * uBlock.inner);
  int slot = rem / uBlock.inner;
  int row = outer * uBlock.inner + (rem - slot * uBlock.inner);
  return src.data[row * uBlock.axis_size + slot + uBlock.start];
}

void main() {
  const int total = uBlock.out_sizes.x * uBlock.out_sizes.y *
      uBlock.out_sizes.z * uBlock.out_sizes.w;
  const int first = int(gl_GlobalInvocationID.x) * 2;
  if (first >= total) return;
  vec2 lanes = vec2(0.0);
  for (int lane = 0; lane < 2; ++lane) {
    const int lin = first + lane;
    if (lin >= total) break;
    uvec2 value = pair_at(lin);
    lanes[lane] = uintBitsToFloat(value.x);
    indices.data[lin] = int(value.y);
  }
  values.data[first / 2] = packHalf2x16(lanes);
}

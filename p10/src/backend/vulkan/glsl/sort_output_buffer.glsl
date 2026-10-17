#version 450 core
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly FloatBuffer { float data[]; } values;
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
 * Final gather over buffer (contiguous) output storage.  Each invocation
 * writes one output element: a (row, slot) pair maps to the contiguous
 * position (row / inner) * count * inner + slot * inner + row % inner, and
 * reads the sorted pair at row * axis_size + slot + start.
 */
void main() {
  int lin = int(gl_GlobalInvocationID.x);
  if (lin >= uBlock.out_sizes.x * uBlock.out_sizes.y * uBlock.out_sizes.z *
          uBlock.out_sizes.w) {
    return;
  }
  int inner = uBlock.inner;
  int outer = lin / (uBlock.count * inner);
  int rem = lin - outer * (uBlock.count * inner);
  int slot = rem / inner;
  int inner_pos = rem - slot * inner;
  int row = outer * inner + inner_pos;
  int index = slot + uBlock.start;
  uvec2 value = src.data[row * uBlock.axis_size + index];
  values.data[lin] = uintBitsToFloat(value.x);
  indices.data[lin] = int(value.y);
}

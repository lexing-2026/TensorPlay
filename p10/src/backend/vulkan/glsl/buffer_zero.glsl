#version 450 core

layout(std430) buffer;

/*
 * Output Buffer
 */
layout(set = 0, binding = 0) buffer restrict writeonly OutBuffer {
  float data[];
}
uOutput;

/*
 * Params Buffer
 *
 * The descriptor set layout is rebuilt from the ordered binding list, so the
 * arguments a dispatch passes map onto consecutive binding numbers.  This
 * kernel has no second buffer, hence the params block sits at one rather than
 * reusing the slot the two-buffer kernels reserve for their input.
 */
layout(set = 0, binding = 1) uniform Block {
  uint buf_length;
}
uBlock;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Returns a buffer filled with zeros.  One invocation handles one element.
 */
void main() {
  const uint idx = gl_GlobalInvocationID.x;

  if (idx >= uBlock.buf_length) {
    return;
  }

  uOutput.data[idx] = 0.0f;
}

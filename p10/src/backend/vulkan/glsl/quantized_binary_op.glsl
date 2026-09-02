#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define OPERATOR(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

// Signed-byte textures in, signed-byte texture out.
layout(set = 0, binding = 0, rgba8i) uniform PRECISION restrict writeonly iimage3D uOutput;
layout(set = 0, binding = 1, rgba8i) uniform PRECISION restrict readonly iimage3D uInputA;
layout(set = 0, binding = 2, rgba8i) uniform PRECISION restrict readonly iimage3D uInputB;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  ivec4 out_sizes; // (W, H, C, N) sizes of the output
  ivec4 a_sizes;   // (W, H, C, N) sizes of operand A
  ivec4 b_sizes;   // (W, H, C, N) sizes of operand B
  float a_scale;
  int a_zero_point;
  float b_scale;
  int b_zero_point;
  float inv_out_scale;
  int out_zero_point;
  int a_c_depth; // ceil(A C / 4)
  int b_c_depth; // ceil(B C / 4)
  int out_c_depth;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Quantized broadcast arithmetic: each output coordinate wraps onto the
 * operands through modulo on width/height/batch (singleton axes repeat), and
 * the channel axis broadcasts at lane granularity so a single-channel
 * operand repeats its byte across every output channel.  Each operand byte
 * is dequantized as (q - zero_point) * scale, the float operation is applied
 * lane-wise, and the result is requantized with round-to-nearest-even into
 * [-128, 127] under the output qparams.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  const int n_out = pos.z / uBlock.out_c_depth;
  const int c4 = pos.z % uBlock.out_c_depth;

  vec4 y;
  for (int i = 0; i < 4; ++i) {
    const int c = c4 * 4 + i; // output channel in the global C axis

    // Operand A: wrap every broadcast axis, remap the channel to its texel.
    const int c_a = c % uBlock.a_sizes.z;
    const int n_a = n_out % uBlock.a_sizes.w;
    const ivec4 qa = imageLoad(
        uInputA,
        ivec3(
            pos.x % uBlock.a_sizes.x,
            pos.y % uBlock.a_sizes.y,
            n_a * uBlock.a_c_depth + c_a / 4));
    const float xa =
        (float(qa[c_a % 4]) - float(uBlock.a_zero_point)) * uBlock.a_scale;

    // Operand B: same wrap against its own sizes.
    const int c_b = c % uBlock.b_sizes.z;
    const int n_b = n_out % uBlock.b_sizes.w;
    const ivec4 qb = imageLoad(
        uInputB,
        ivec3(
            pos.x % uBlock.b_sizes.x,
            pos.y % uBlock.b_sizes.y,
            n_b * uBlock.b_c_depth + c_b / 4));
    const float xb =
        (float(qb[c_b % 4]) - float(uBlock.b_zero_point)) * uBlock.b_scale;

    y[i] = OPERATOR(xa, xb);
  }

  const ivec4 rounded =
      ivec4(roundEven(y * uBlock.inv_out_scale)) + ivec4(uBlock.out_zero_point);
  imageStore(uOutput, pos, clamp(rounded, ivec4(-128), ivec4(127)));
}

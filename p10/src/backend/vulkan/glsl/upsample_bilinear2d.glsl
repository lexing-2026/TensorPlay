#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
layout(set = 0, binding = 2) uniform PRECISION restrict Block {
  // (W, H, C, N) sizes of the input
  ivec4 in_sizes;
  // (W, H, C, N) sizes of the output
  ivec4 out_sizes;
  float rwidth; // source step per output column (precomputed on the host)
  float rheight;
  int align_corners;
  int out_c_depth; // ceil(out C / 4)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Bilinear 2D upsampling.  The host passes the area-pixel scale for each
 * spatial axis; the source coordinate is scale * dst for corner-aligned
 * sampling and scale * (dst + 0.5) - 0.5 otherwise, with negative sources
 * clamped to the border for the half-pixel form.  The four neighboring
 * texels are fetched and blended with the fractional weights.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos.x >= uBlock.out_sizes.x || pos.y >= uBlock.out_sizes.y ||
      pos.z >= uBlock.out_sizes.w * uBlock.out_c_depth) {
    return;
  }

  float src_x = uBlock.align_corners != 0
      ? uBlock.rwidth * float(pos.x)
      : max(uBlock.rwidth * (float(pos.x) + 0.5f) - 0.5f, 0.0f);
  float src_y = uBlock.align_corners != 0
      ? uBlock.rheight * float(pos.y)
      : max(uBlock.rheight * (float(pos.y) + 0.5f) - 0.5f, 0.0f);

  const int x0 = int(src_x);
  const int y0 = int(src_y);
  const float fx = src_x - float(x0);
  const float fy = src_y - float(y0);
  const int x1 = min(x0 + 1, uBlock.in_sizes.x - 1);
  const int y1 = min(y0 + 1, uBlock.in_sizes.y - 1);

  const vec4 v00 = texelFetch(uInput, ivec3(x0, y0, pos.z), 0);
  const vec4 v01 = texelFetch(uInput, ivec3(x1, y0, pos.z), 0);
  const vec4 v10 = texelFetch(uInput, ivec3(x0, y1, pos.z), 0);
  const vec4 v11 = texelFetch(uInput, ivec3(x1, y1, pos.z), 0);

  const vec4 top = mix(v00, v01, vec4(fx));
  const vec4 bottom = mix(v10, v11, vec4(fx));
  imageStore(uOutput, pos, mix(top, bottom, vec4(fy)));
}

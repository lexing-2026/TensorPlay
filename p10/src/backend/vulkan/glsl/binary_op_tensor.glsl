#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
#define DTYPE ${DTYPE}

#define OP(X, Y, A) ${OPERATOR}
// clang-format on

layout(std430) buffer;

#include "texel_access.h"
$if DTYPE == "int":
  // Signed-word planes: iimage loads on both sides (integer formats cannot
  // bind a sampler, so broadcast fetches expand lanes inline).
  $if not INPLACE:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly iimage3D uOutput;
    layout(set = 0, binding = 1, FORMAT) uniform PRECISION restrict readonly iimage3D uInput;
    layout(set = 0, binding = 2, FORMAT) uniform PRECISION restrict readonly iimage3D uOther;
    layout(set = 0, binding = 3) uniform PRECISION restrict Block {
      ivec4 output_sizes;
      ivec4 input_sizes;
      ivec4 other_sizes;
      int alpha;
    }
    uArgs;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict iimage3D uOutput;
    layout(set = 0, binding = 1, FORMAT) uniform PRECISION restrict readonly iimage3D uOther;
    layout(set = 0, binding = 2) uniform PRECISION restrict Block {
      ivec4 output_sizes;
      ivec4 other_sizes;
      int alpha;
    }
    uArgs;
  // texture-branch declarations end
$else:
  $if not INPLACE:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
    layout(set = 0, binding = 2) uniform PRECISION sampler3D uOther;
    layout(set = 0, binding = 3) uniform PRECISION restrict Block {
      ivec4 output_sizes;
      ivec4 input_sizes;
      ivec4 other_sizes;
      float alpha;
    }
    uArgs;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION sampler3D uOther;
    layout(set = 0, binding = 2) uniform PRECISION restrict Block {
      ivec4 output_sizes;
      ivec4 other_sizes;
      float alpha;
    }
    uArgs;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  ivec3 output_extents;
  output_extents.xy = uArgs.output_sizes.xy;
  output_extents.z =
      uArgs.output_sizes.w * int(ceil(uArgs.output_sizes.z / 4.0));

  if (any(greaterThanEqual(pos, output_extents.xyz))) {
    return;
  }

  $if DTYPE == "int":
    ivec3 other_pos =
        map_output_pos_to_input_pos(pos, uArgs.output_sizes, uArgs.other_sizes);
    const ivec4 other_raw = imageLoad(uOther, other_pos);
    ivec4 other_texel = (uArgs.output_sizes.z != uArgs.other_sizes.z)
        ? ivec4(other_raw.x)
        : other_raw;

    $if IS_DIV:
      // Channel padding: neutralize pad lanes against a divisor of one so
      // the integer division stays defined (quotient discarded anyway).
      const int c_index = (pos.z % ((uArgs.output_sizes.z + 3) / 4)) * 4;
      if (uArgs.other_sizes.z != 1 && c_index + 3 >= uArgs.output_sizes.z) {
        ivec4 c_ind = ivec4(c_index) + ivec4(0, 1, 2, 3);
        ivec4 mask = ivec4(lessThan(c_ind, ivec4(uArgs.output_sizes.z)));
        other_texel = other_texel * mask + ivec4(1, 1, 1, 1) - mask;
      }

    $if not INPLACE:
      ivec3 input_pos =
          map_output_pos_to_input_pos(pos, uArgs.output_sizes, uArgs.input_sizes);
      const ivec4 in_raw = imageLoad(uInput, input_pos);
      const ivec4 in_texel = (uArgs.output_sizes.z != uArgs.input_sizes.z)
          ? ivec4(in_raw.x)
          : in_raw;

      imageStore(uOutput, pos, OP(in_texel, other_texel, uArgs.alpha));
    $else:
      const ivec4 in_texel = imageLoad(uOutput, pos);
      imageStore(uOutput, pos, OP(in_texel, other_texel, uArgs.alpha));
    // int-branch statements end
  $else:
    ivec3 other_pos =
        map_output_pos_to_input_pos(pos, uArgs.output_sizes, uArgs.other_sizes);
    vec4 other_texel =
        load_texel(other_pos, uArgs.output_sizes, uArgs.other_sizes, uOther);

    $if IS_DIV:
      // Zero padding is added to the channels dimension when tensors are
      // stored as image textures. This will cause a divide-by-zero when
      // performing division. For division, apply an extra step of detecting
      // which elements are zero padding to avoid division by zero.
      const int c_index = (pos.z % ((uArgs.output_sizes.z + 3) / 4)) * 4;
      if (uArgs.other_sizes.z != 1 && c_index + 3 >= uArgs.output_sizes.z) {
        ivec4 c_ind = ivec4(c_index) + ivec4(0, 1, 2, 3);
        vec4 mask = vec4(lessThan(c_ind, ivec4(uArgs.output_sizes.z)));
        other_texel = other_texel * mask + vec4(1, 1, 1, 1) - mask;
      }

    $if not INPLACE:
      ivec3 input_pos =
          map_output_pos_to_input_pos(pos, uArgs.output_sizes, uArgs.input_sizes);
      const vec4 in_texel =
          load_texel(input_pos, uArgs.output_sizes, uArgs.input_sizes, uInput);

      imageStore(uOutput, pos, OP(in_texel, other_texel, uArgs.alpha));
    $else:
      const vec4 in_texel = imageLoad(uOutput, pos);
      imageStore(uOutput, pos, OP(in_texel, other_texel, uArgs.alpha));
}

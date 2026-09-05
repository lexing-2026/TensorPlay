#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
#define DTYPE ${DTYPE}

#define OP(X, Y) ${OPERATOR}
// clang-format on

layout(std430) buffer;

$if DTYPE == "int":
  // Signed-word planes with an integer scalar broadcast to all lanes.
  $if not INPLACE:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly iimage3D uOutput;
    layout(set = 0, binding = 1, FORMAT) uniform PRECISION restrict readonly iimage3D uInput;
    layout(set = 0, binding = 2) uniform PRECISION restrict Block {
      ivec4 extents;
      // scalar argument
      int other;
    }
    uArgs;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict iimage3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION restrict Block {
      ivec4 extents;
      // scalar argument
      int other;
    }
    uArgs;
$else:
  // clang-format off
  $if not INPLACE:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
    layout(set = 0, binding = 2) uniform PRECISION restrict Block {
      ivec4 extents;
      // scalar argument
      float other;
    }
    uArgs;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION restrict Block {
      ivec4 extents;
      // scalar argument
      float other;
    }
    uArgs;
  // clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/**
 * Performs a binary elementwise operation between uInput and uArgs.other,
 * writing the output to uOutput.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, uArgs.extents.xyz))) {
    return;
  }

  // clang-format off
  $if DTYPE == "int":
    ivec4 v_other = ivec4(uArgs.other);
    $if not INPLACE:
      ivec4 v = imageLoad(uInput, pos);
      ivec4 out_texel = OP(v, v_other);
    $else:
      ivec4 out_texel = imageLoad(uOutput, pos);
      out_texel = OP(out_texel, v_other);
  $else:
    vec4 v_other = vec4(uArgs.other);
    $if not INPLACE:
      vec4 v = texelFetch(uInput, pos, 0);
      vec4 out_texel = OP(v, v_other);
    $else:
      vec4 out_texel = imageLoad(uOutput, pos);
      out_texel = OP(out_texel, v_other);
  // clang-format on

  imageStore(uOutput, pos, out_texel);
}

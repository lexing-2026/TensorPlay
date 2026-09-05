#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
#define DTYPE ${DTYPE}

#define OP(X) ${OPERATOR}
// clang-format on

layout(std430) buffer;

$if DTYPE == "int":
  // Integer planes bind as signed-word storage images: samplers cannot read
  // integer formats, so both directions go through iimage loads.
  $if not INPLACE:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly iimage3D uOutput;
    layout(set = 0, binding = 1, FORMAT) uniform PRECISION restrict readonly iimage3D uInput;
    layout(set = 0, binding = 2) uniform PRECISION restrict Block {
      ivec4 extents;
    }
    uBlock;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict iimage3D uInput;
    layout(set = 0, binding = 1) uniform PRECISION restrict Block {
      ivec4 extents;
    }
    uBlock;
$else:
  // clang-format off
  $if not INPLACE:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uOutput;
    layout(set = 0, binding = 1) uniform PRECISION sampler3D uInput;
    layout(set = 0, binding = 2) uniform PRECISION restrict Block {
      ivec4 extents;
    }
    uBlock;
  $else:
    layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict image3D uInput;
    layout(set = 0, binding = 1) uniform PRECISION restrict Block {
      ivec4 extents;
    }
    uBlock;
  // clang-format on

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, uBlock.extents.xyz))) {
    return;
  }

  // clang-format off
  $if DTYPE == "int":
    $if not INPLACE:
      const ivec4 v = imageLoad(uInput, pos);
      imageStore(uOutput, pos, OP(v));
    $else:
      ivec4 v = imageLoad(uInput, pos);
      imageStore(uInput, pos, OP(v));
  $else:
    $if not INPLACE:
      const vec4 v = texelFetch(uInput, pos, 0);
      imageStore(uOutput, pos, OP(v));
    $else:
      vec4 v = imageLoad(uInput, pos);
      imageStore(uInput, pos, OP(v));
  // clang-format on
}

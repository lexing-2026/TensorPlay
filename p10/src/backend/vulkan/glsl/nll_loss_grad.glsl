#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
#define FORMAT ${FORMAT}
// clang-format on

layout(std430) buffer;

/*
 * Gradient of a likelihood loss.  Each invocation owns one element of the
 * gradient plane and answers the same question: does the class this row's
 * target name sit at this position?  Only that one element per row carries a
 * gradient, so the rest of the plane is written as zero rather than left to a
 * clearing pass, and every texel or word has exactly one writer.
 *
 * The gradient of a row is the class weight times the incoming gradient of the
 * row's loss, negated.  A reduced loss hands one gradient to every row and, for
 * the averaged form, the rows share it divided by the weight total the forward
 * pass measured; an unreduced loss hands each row its own.  Both the incoming
 * gradient and that total arrive as one-element payloads of their own, so no
 * value crosses through the host.  A row whose target names no class, or names
 * the code the caller asked to skip, has no gradient to hand out.
 */
// clang-format off
$if SELF_PACKED == 1:
  // One class per texel along the width, so the texel position is the answer to
  // the question this invocation asks.
  layout(set = 0, binding = 0, FORMAT) uniform PRECISION restrict writeonly image3D uGrad;
$else:
  layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Grad {
    float data[];
  }
  uGrad;
// clang-format on
layout(set = 0, binding = 1) buffer PRECISION restrict readonly Target {
  uvec2 data[];
}
uTarget;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly GradIn {
  float data[];
}
uGradIn;
layout(set = 0, binding = 3) buffer PRECISION restrict readonly WeightTotal {
  float data[];
}
uWeightTotal;
// Without class weights the pass folds their factor away; the target is already
// bound read-only, so naming it twice cannot alias a write.
layout(set = 0, binding = 4) buffer PRECISION restrict readonly ClassWeight {
  float data[];
}
uClassWeight;
layout(set = 0, binding = 5) uniform PRECISION restrict Block {
  int rows;
  int classes;
  // The code to skip, split into the words a wide target stores it in
  int skip_low;
  int skip_high;
  // 1: divide the shared gradient by the weight total
  int mean_mode;
  // 1: each row owns an entry of the incoming gradient
  int per_row_grad;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  // clang-format off
  $if SELF_PACKED == 1:
    const ivec3 pos = ivec3(gl_GlobalInvocationID);
    const int class_at = pos.x;
    const int row = pos.y;
    if (row >= uBlock.rows) {
      return;
    }
  $else:
    const int index = int(gl_GlobalInvocationID.x);
    const int row = index / uBlock.classes;
    const int class_at = index - row * uBlock.classes;
    if (index >= uBlock.rows * uBlock.classes) {
      return;
    }
  // clang-format on

  float value = 0.0;
  const uvec2 code = uTarget.data[row];
  const int class_code = int(code.x);
  const bool skipped =
      class_code == uBlock.skip_low && int(code.y) == uBlock.skip_high;
  if (!skipped && int(code.y) == 0 && class_code >= 0 &&
      class_code < uBlock.classes && class_code == class_at) {
    float weight = 1.0;
    // clang-format off
    $if HAS_WEIGHT == 1:
      weight = uClassWeight.data[class_code];
    // clang-format on
    float incoming =
        uBlock.per_row_grad == 1 ? uGradIn.data[row] : uGradIn.data[0];
    if (uBlock.mean_mode == 1) {
      const float total = uWeightTotal.data[0];
      if (total > 0.0) {
        incoming /= total;
      }
    }
    value = -weight * incoming;
  }
  // clang-format off
  $if SELF_PACKED == 1:
    imageStore(uGrad, pos, vec4(value, 0.0, 0.0, 0.0));
  $else:
    uGrad.data[index] = value;
  // clang-format on
}

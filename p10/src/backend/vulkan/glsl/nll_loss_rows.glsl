#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

/*
 * One pass over the rows of a likelihood loss.  Each invocation owns one row:
 * it reads the class its target names, fetches that class's log-likelihood, and
 * writes the row's loss together with the class weight it contributes to the
 * normalizer.  Nothing accumulates here, so a row neither reads nor races any
 * other row's word.
 *
 * A wide target keeps one pair of words per element, the low word first, so a
 * class is readable only when the high word is empty and the low word lands
 * inside the class axis.  A code that names no class, and the code the caller
 * asked to skip, contribute a zero row and a zero weight - the same answer the
 * row lookups of this backend already give to a request outside their rows.  The
 * skipped code arrives as the pair of words it is stored in, so a count wider
 * than a word is compared without ordering it.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly Loss {
  float data[];
}
uLoss;
layout(set = 0, binding = 1) buffer PRECISION restrict writeonly Weights {
  float data[];
}
uWeights;
// clang-format off
$if SELF_PACKED == 1:
  // A payload with an image form carries one class per texel along its width,
  // so a class code moves the fetch along x and the row rides on y.
  layout(set = 0, binding = 2) uniform PRECISION sampler3D uSelf;
$else:
  layout(set = 0, binding = 2) buffer PRECISION restrict readonly Self {
    float data[];
  }
  uSelf;
// clang-format on
layout(set = 0, binding = 3) buffer PRECISION restrict readonly Target {
  uvec2 data[];
}
uTarget;
// Without class weights the pass folds their factor away, and the binding still
// needs a readable payload of its own; the target is already bound read-only, so
// naming it twice cannot alias a write.
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
  int fill;
  int fill1;
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int row = int(gl_GlobalInvocationID.x);
  if (row >= uBlock.rows) {
    return;
  }
  const uvec2 code = uTarget.data[row];
  const int class_code = int(code.x);
  const bool skipped =
      class_code == uBlock.skip_low && int(code.y) == uBlock.skip_high;
  if (!skipped && int(code.y) == 0 && class_code >= 0 &&
      class_code < uBlock.classes) {
    float weight = 1.0;
    // clang-format off
    $if HAS_WEIGHT == 1:
      weight = uClassWeight.data[class_code];
    // clang-format on
    // clang-format off
    $if SELF_PACKED == 1:
      const float picked =
          texelFetch(uSelf, ivec3(class_code, row, 0), 0).x;
    $else:
      const float picked = uSelf.data[row * uBlock.classes + class_code];
    // clang-format on
    uLoss.data[row] = -picked * weight;
    uWeights.data[row] = weight;
  } else {
    uLoss.data[row] = 0.0;
    uWeights.data[row] = 0.0;
  }
}

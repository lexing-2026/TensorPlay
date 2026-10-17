#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

layout(std430) buffer;

// Inverse-CDF categorical sampling over buffer-stored probability rows, for
// category counts beyond the 3-D image axis limit.  One thread per row
// computes the row's total weight, then draws num_samples indices by
// scanning the cumulative weights against uniform noise.  Rows whose total
// mass is not positive yield index 0, matching the serial reference
// implementation.  The Int64 output is written as two uint32 halves (lo,
// hi) since Vulkan textures do not carry 64-bit integer channels.
layout(set = 0, binding = 0) buffer restrict writeonly Output {
  uvec2 data[];
} uOut;
layout(set = 0, binding = 1) buffer restrict readonly Prob {
  float data[];
} uProb;
layout(set = 0, binding = 2) buffer restrict readonly Noise {
  float data[];
} uNoise;
layout(set = 0, binding = 3) uniform PRECISION restrict Block {
  int cols;         // categories per row
  int num_samples;  // samples per row
  int total;        // rows * num_samples (flat output element count)
}
uBlock;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int r = int(gl_GlobalInvocationID.x);
  const int rows = uBlock.total / max(uBlock.num_samples, 1);
  if (r >= rows) {
    return;
  }
  const int base = r * uBlock.cols;

  float total = 0.0;
  for (int c = 0; c < uBlock.cols; ++c) {
    total += uProb.data[base + c];
  }

  if (!(total > 0.0)) {
    for (int s = 0; s < uBlock.num_samples; ++s) {
      uOut.data[r * uBlock.num_samples + s] = uvec2(0u, 0u);
    }
    return;
  }

  for (int s = 0; s < uBlock.num_samples; ++s) {
    const float noise = uNoise.data[r * uBlock.num_samples + s];
    const float target = noise * total;
    float cum = 0.0;
    int idx = uBlock.cols - 1;
    for (int c = 0; c < uBlock.cols; ++c) {
      cum += uProb.data[base + c];
      if (target < cum) {
        idx = c;
        break;
      }
    }
    uOut.data[r * uBlock.num_samples + s] = uvec2(uint(idx), 0u);
  }
}
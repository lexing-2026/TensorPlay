#pragma once

#include <cstdint>

namespace tensorplay {

constexpr int32_t num_sdp_backends = 5;

// Routing targets for scaled dot product attention.  The integer values are
// the stable encoding shared with the backend selector and the priority
// order list.
enum class SDPBackend : int32_t {
  error = -1,
  math = 0,
  flash_attention = 1,
  efficient_attention = 2,
  cudnn_attention = 3,
  overrideable = 4
};

} // namespace tensorplay

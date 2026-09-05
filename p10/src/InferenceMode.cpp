#include "InferenceMode.h"

namespace tensorplay {

namespace {
thread_local bool tp_inference_mode_enabled = false;
} // namespace

bool InferenceMode::is_enabled() { return tp_inference_mode_enabled; }
void InferenceMode::set_enabled(bool enabled) { tp_inference_mode_enabled = enabled; }

} // namespace tensorplay

#include "InferenceMode.h"

namespace tensorplay {

// Default mirrors PyTorch: inference mode starts disabled.
thread_local bool InferenceMode::enabled_ = false;

} // namespace tensorplay

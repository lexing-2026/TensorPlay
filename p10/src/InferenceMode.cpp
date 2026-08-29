#include "InferenceMode.h"

namespace tensorplay {

thread_local bool InferenceMode::enabled_ = false;

} // namespace tensorplay

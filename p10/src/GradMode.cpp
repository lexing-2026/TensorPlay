#include "GradMode.h"

namespace tensorplay {

thread_local bool GradMode::enabled_ = true;

} // namespace tensorplay

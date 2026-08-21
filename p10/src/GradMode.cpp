#include "GradMode.h"

namespace tensorplay {

// Default mirrors PyTorch: autograd recording starts enabled.
thread_local bool GradMode::enabled_ = true;

} // namespace tensorplay

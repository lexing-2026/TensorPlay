#include "store/Store.h"

#include <thread>

namespace tensorplay {
namespace distributed {

Store::~Store() = default;

bool Store::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  if (check(keys)) {
    return true;
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    if (check(keys)) {
      return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

} // namespace distributed
} // namespace tensorplay

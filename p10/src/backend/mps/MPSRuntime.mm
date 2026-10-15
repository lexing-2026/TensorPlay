#include "backend/mps/MPSRuntime.h"

#include "Exception.h"

#include <string>

namespace tensorplay {
namespace mps {

#ifdef USE_MPS

void checkMPS(void* error, const char* operation) {
  if (error) {
    NSError* nsError = (NSError*)error;
    TP_THROW(RuntimeError,
             std::string(operation), " failed: ",
             [[nsError localizedDescription] UTF8String]);
  }
}

int currentDevice() {
  // The MPS backend exposes a single device.
  return 0;
}

int deviceCount() {
  return is_available() ? 1 : 0;
}

MPSGuard::MPSGuard(int device_index) : original_device_(currentDevice()) {
  // Single-device backend: nothing to switch, but normalize negative indexes
  // to the valid slot for callers that pass -1 ("current device").
  if (device_index < 0) {
    device_index = currentDevice();
  }
  if (device_index != original_device_) {
    TP_THROW(RuntimeError,
             "MPS device index ", device_index, " is out of range for ",
             deviceCount(), " visible device(s)");
  }
  changed_ = false;
}

MPSGuard::MPSGuard(const Device& device) : MPSGuard(static_cast<int>(device.index())) {}

MPSGuard::~MPSGuard() noexcept = default;

Stream getCurrentStream(int device_index) {
  return getCurrentMPSStream()->unwrap();
}

MPSStreamGuard::MPSStreamGuard(const Stream& stream)
    : original_device_(currentDevice()),
      stream_device_(static_cast<int>(stream.device_index())) {
  if (stream.device_type() != DeviceType::MPS) {
    active_ = false;
    return;
  }
  original_stream_ = getCurrentMPSStream();
  setCurrentMPSStream(getStreamFromStream(stream));
  active_ = true;
}

MPSStreamGuard::~MPSStreamGuard() noexcept {
  if (active_) {
    setCurrentMPSStream(original_stream_);
  }
}

OptionalMPSGuard::OptionalMPSGuard(const Device& device) {
  if (device.is_mps()) {
    guard_ = std::make_unique<MPSGuard>(device);
  }
}

OptionalMPSGuard::~OptionalMPSGuard() = default;

OptionalMPSStreamGuard::OptionalMPSStreamGuard(const std::optional<Stream>& stream) {
  if (stream.has_value() && stream->device_type() == DeviceType::MPS) {
    guard_.emplace(*stream);
  }
}

#endif // USE_MPS

} // namespace mps
} // namespace tensorplay
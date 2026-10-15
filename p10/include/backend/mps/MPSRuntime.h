#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "Device.h"
#include "Macros.h"
#include "Stream.h"
#include "backend/mps/MPSStream.h"

namespace tensorplay {
namespace mps {

#ifdef USE_MPS

// Throws a TensorPlay RuntimeError with the Metal error text, if any. The
// error is passed as an opaque pointer (an NSError* in Objective-C++ TUs)
// so this header stays usable from plain C++.
P10_API void checkMPS(void* error, const char* operation);
// The MPS backend exposes a single device (index 0).
P10_API int currentDevice();
P10_API int deviceCount();

class P10_API MPSGuard {
public:
    explicit MPSGuard(int device_index);
    explicit MPSGuard(const Device& device);
    ~MPSGuard() noexcept;

    MPSGuard(const MPSGuard&) = delete;
    MPSGuard& operator=(const MPSGuard&) = delete;

private:
    int original_device_ = -1;
    bool changed_ = false;
};

// The ambient MPS stream of the device as a backend-agnostic Stream.
P10_API Stream getCurrentStream(int device_index = -1);

class P10_API MPSStreamGuard {
public:
    explicit MPSStreamGuard(const Stream& stream);
    ~MPSStreamGuard() noexcept;

    MPSStreamGuard(const MPSStreamGuard&) = delete;
    MPSStreamGuard& operator=(const MPSStreamGuard&) = delete;

private:
    int original_device_ = -1;
    int stream_device_ = -1;
    MPSStream* original_stream_ = nullptr;
    bool active_ = false;
};

// Device guard that is a no-op for CPU devices. Generated dispatch wrappers
// use this so tensors on mps:0 execute correctly even when another device is
// current in the calling thread.
class P10_API OptionalMPSGuard {
public:
    explicit OptionalMPSGuard(const Device& device);
    ~OptionalMPSGuard();

    OptionalMPSGuard(const OptionalMPSGuard&) = delete;
    OptionalMPSGuard& operator=(const OptionalMPSGuard&) = delete;

private:
    std::unique_ptr<MPSGuard> guard_;
};

// Switches the current thread's MPS stream while the guard is alive, and
// restores the previous stream on destruction. It is a no-op when the
// optional stream is empty or not on an MPS device.
class P10_API OptionalMPSStreamGuard {
public:
    explicit OptionalMPSStreamGuard(const std::optional<Stream>& stream);

    OptionalMPSStreamGuard(const OptionalMPSStreamGuard&) = delete;
    OptionalMPSStreamGuard& operator=(const OptionalMPSStreamGuard&) = delete;

private:
    std::optional<MPSStreamGuard> guard_;
};

#else

class OptionalMPSGuard {
public:
    explicit OptionalMPSGuard(const Device&) {}
};

class OptionalMPSStreamGuard {
public:
    explicit OptionalMPSStreamGuard(const std::optional<Stream>&) {}

    OptionalMPSStreamGuard(const OptionalMPSStreamGuard&) = delete;
    OptionalMPSStreamGuard& operator=(const OptionalMPSStreamGuard&) = delete;
};

#endif // USE_MPS

} // namespace mps
} // namespace tensorplay
#pragma once

// Global runtime context, mirroring the subset of at::Context that governs
// process-wide behavior: default dtype/device for factory functions,
// deterministic-algorithms flags, and float32 matmul precision.
//
// PyTorch references:
//   - aten/src/ATen/Context.h   (deterministic, Float32MatmulPrecision)
//   - c10/core/DefaultDType.h   (get_default_dtype)

#include "Macros.h"
#include "Device.h"
#include "DType.h"

#include <optional>
#include <string>
#include <vector>

namespace tensorplay {

enum class P10_API Float32MatmulPrecision { HIGHEST, HIGH, MEDIUM };

std::string P10_API Float32MatmulPrecisionToString(Float32MatmulPrecision precision);
Float32MatmulPrecision P10_API Float32MatmulPrecisionFromString(const std::string& s);

bool P10_API isFloatingPoint(DType dtype);

class P10_API Context {
public:
    // -- Default dtype ---------------------------------------------------
    // Mirrors c10::get_default_dtype / torch._C._set_default_dtype: only
    // floating point dtypes are accepted as the default dtype.
    void setDefaultDType(DType dtype);
    DType defaultDType() const { return default_dtype_; }

    // -- Default device --------------------------------------------------
    // Thread-local so that set_default_device() in one thread does not leak
    // into others (PyTorch keeps this state on a thread-local mode stack).
    void setDefaultDevice(std::optional<Device> device) { default_device_ = device; }
    void clearDefaultDevice() { default_device_.reset(); }
    std::optional<Device> getDefaultDeviceOverride() const { return default_device_; }
    // The device factory functions allocate on; "cpu" when no override is set.
    Device defaultDevice() const {
        if (default_device_.has_value()) return *default_device_;
        return Device(DeviceType::CPU);
    }
    // Scoped override backing `with tensorplay.device(...):`, mirroring
    // torch.utils._device.DeviceContext (thread-local, nestable).
    void pushDefaultDevice(Device device);
    void popDefaultDevice();

    // -- Deterministic algorithms ----------------------------------------
    // Mirrors at::Context::setDeterministicAlgorithms. Individual kernels
    // consult deterministicAlgorithms() and call alertNotDeterministic() when
    // they can only provide a nondeterministic implementation.
    void setDeterministicAlgorithms(bool mode, bool warn_only = false) {
        deterministic_algorithms_ = mode;
        deterministic_algorithms_warn_only_ = warn_only;
    }
    bool deterministicAlgorithms() const { return deterministic_algorithms_; }
    bool deterministicAlgorithmsWarnOnly() const { return deterministic_algorithms_warn_only_; }

    // Throws (or warns when warn_only) if called under
    // use_deterministic_algorithms(True), mirroring
    // at::globalContext().alertNotDeterministic.
    void alertNotDeterministic(const std::string& caller) const;

    // -- Float32 matmul precision ----------------------------------------
    // Mirrors at::Context::setFloat32MatmulPrecision ("highest"/"high"/"medium").
    void setFloat32MatmulPrecision(const std::string& p) { float32_matmul_precision_ = Float32MatmulPrecisionFromString(p); }
    void setFloat32MatmulPrecision(Float32MatmulPrecision p) { float32_matmul_precision_ = p; }
    Float32MatmulPrecision float32MatmulPrecision() const { return float32_matmul_precision_; }
    std::string getFloat32MatmulPrecisionStr() const {
        return Float32MatmulPrecisionToString(float32_matmul_precision_);
    }

    // Equivalent of torch.backends.cuda.matmul.allow_tf32: enabled whenever
    // the matmul precision is not "highest".
    bool allowTF32CuBLAS() const {
        return float32_matmul_precision_ != Float32MatmulPrecision::HIGHEST;
    }
    // Mirrors Context::setAllowTF32CuBLAS: setting the flag is equivalent to
    // switching float32 matmul precision between "high" and "highest".
    void setAllowTF32CuBLAS(bool b);

    // Mirrors torch.backends.cudnn.allow_tf32 (independent of the matmul
    // precision; defaults to True like PyTorch).
    bool allowTF32CuDNN() const { return allow_tf32_cudnn_; }
    void setAllowTF32CuDNN(bool b) { allow_tf32_cudnn_ = b; }

    // Mirrors torch.backends.cudnn.benchmark: when enabled, convolution
    // algorithm selection times the candidate algorithms on first use of a
    // shape and caches the fastest (instead of trusting the heuristic).
    // Defaults to False like PyTorch.
    bool cudnnBenchmark() const { return cudnn_benchmark_; }
    void setCudnnBenchmark(bool b) { cudnn_benchmark_ = b; }

private:
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context() = default;
    friend P10_API Context& globalContext();

    DType default_dtype_ = DType::Float32;
    inline static thread_local std::optional<Device> default_device_;
    inline static thread_local std::vector<std::optional<Device>> device_stack_;
    bool deterministic_algorithms_ = false;
    bool deterministic_algorithms_warn_only_ = false;
    Float32MatmulPrecision float32_matmul_precision_ = Float32MatmulPrecision::HIGHEST;
    bool allow_tf32_cudnn_ = true;
    bool cudnn_benchmark_ = false;
};

P10_API Context& globalContext();

} // namespace tensorplay

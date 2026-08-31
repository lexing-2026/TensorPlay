#pragma once

// process-wide behavior: default dtype/device for factory functions,
// deterministic-algorithms flags, and float32 matmul precision.
//

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
    // floating point dtypes are accepted as the default dtype.
    void setDefaultDType(DType dtype);
    DType defaultDType() const { return default_dtype_; }

    // -- Default device --------------------------------------------------
    // Thread-local so that set_default_device() in one thread does not leak
    void setDefaultDevice(std::optional<Device> device) { default_device_ = device; }
    void clearDefaultDevice() { default_device_.reset(); }
    std::optional<Device> getDefaultDeviceOverride() const { return default_device_; }
    // The device factory functions allocate on; "cpu" when no override is set.
    Device defaultDevice() const {
        if (default_device_.has_value()) return *default_device_;
        return Device(DeviceType::CPU);
    }
    // Scoped override backing `with tensorplay.device(...):`,
    void pushDefaultDevice(Device device);
    void popDefaultDevice();

    // -- Deterministic algorithms ----------------------------------------
    // consult deterministicAlgorithms() and call alertNotDeterministic() when
    // they can only provide a nondeterministic implementation.
    void setDeterministicAlgorithms(bool mode, bool warn_only = false) {
        deterministic_algorithms_ = mode;
        deterministic_algorithms_warn_only_ = warn_only;
    }
    bool deterministicAlgorithms() const { return deterministic_algorithms_; }
    bool deterministicAlgorithmsWarnOnly() const { return deterministic_algorithms_warn_only_; }

    // Throws (or warns when warn_only) when deterministic algorithms are
    // enabled.
    void alertNotDeterministic(const std::string& caller) const;

    // -- Float32 matmul precision ----------------------------------------
    void setFloat32MatmulPrecision(const std::string& p) { float32_matmul_precision_ = Float32MatmulPrecisionFromString(p); }
    void setFloat32MatmulPrecision(Float32MatmulPrecision p) { float32_matmul_precision_ = p; }
    Float32MatmulPrecision float32MatmulPrecision() const { return float32_matmul_precision_; }
    std::string getFloat32MatmulPrecisionStr() const {
        return Float32MatmulPrecisionToString(float32_matmul_precision_);
    }

    // the matmul precision is not "highest".
    bool allowTF32CuBLAS() const {
        return float32_matmul_precision_ != Float32MatmulPrecision::HIGHEST;
    }
    // Set the legacy TF32 flag by selecting the corresponding float32
    // matmul precision mode.
    void setAllowTF32CuBLAS(bool b);

    bool allowTF32CuDNN() const { return allow_tf32_cudnn_; }
    void setAllowTF32CuDNN(bool b) { allow_tf32_cudnn_ = b; }

    // algorithm selection times the candidate algorithms on first use of a
    // shape and caches the fastest (instead of trusting the heuristic).
    bool cudnnBenchmark() const { return cudnn_benchmark_; }
    void setCudnnBenchmark(bool b) { cudnn_benchmark_ = b; }

    // Master switches for the oneDNN and NNPACK convolution backends; both
    // default to enabled and are consulted before a backend claims a call.
    bool userEnabledMkldnn() const { return enabled_mkldnn_; }
    void setUserEnabledMkldnn(bool e) { enabled_mkldnn_ = e; }
    bool userEnabledNNPACK() const { return enabled_nnpack_; }
    void setUserEnabledNNPACK(bool e) { enabled_nnpack_ = e; }

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
    bool enabled_mkldnn_ = true;
    bool enabled_nnpack_ = true;
};

P10_API Context& globalContext();

} // namespace tensorplay

#pragma once

// process-wide behavior: default dtype/device for factory functions,
// deterministic-algorithms flags, and float32 matmul precision.

#include "Macros.h"
#include "BlasBackend.h"
#include "Device.h"
#include "DType.h"
#include "LinalgBackend.h"
#include "SDPBackend.h"

#include <array>
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
    // The complex dtype paired with the default dtype: ComplexHalf/Float/
    // Double for a Half/Float/Double default, ComplexFloat otherwise.
    DType defaultComplexDType() const {
        switch (default_dtype_) {
            case DType::Float16: return DType::ComplexHalf;
            case DType::Float64: return DType::ComplexDouble;
            default: return DType::ComplexFloat;
        }
    }

    // -- Default device --------------------------------------------------
    // Thread-local so that set_default_device() in one thread does not
    // leak into other threads. The TLS slots live inside the library:
    // thread-storage objects cannot carry a dll interface on Windows, so
    // the exported surface keeps plain accessors.
    void setDefaultDevice(std::optional<Device> device);
    void clearDefaultDevice();
    std::optional<Device> getDefaultDeviceOverride() const;
    // The device factory functions allocate on; "cpu" when no override is set.
    Device defaultDevice() const {
        if (getDefaultDeviceOverride().has_value())
            return *getDefaultDeviceOverride();
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

    // -- GPU linear algebra / BLAS backend selection ----------------------
    // The linalg preference routes dense factorizations; "magma" is
    // rejected because this build ships no MAGMA backend. The blas
    // preference chooses between the classic cuBLAS API and cuBLASLt for
    // matrix products; "default" lets each call site apply its own
    // heuristic.
    void setLinalgPreferredBackend(LinalgBackend b);
    LinalgBackend linalgPreferredBackend() const { return linalg_preferred_backend_; }

    void setBlasPreferredBackend(BlasBackend b) { blas_preferred_backend_ = b; }
    BlasBackend blasPreferredBackend() const { return blas_preferred_backend_; }

    // -- Scaled dot product attention backend controls -------------------
    // Every fused kernel family owns an enable switch; the priority order
    // ranks the backends when the caller lets the library route.  Both are
    // consumed by the Python ``sdpa_kernel`` context manager and the
    // ``can_use_*`` eligibility gates.
    void setSDPUseFlash(bool e) { enabled_flash_sdp_ = e; }
    bool userEnabledFlashSDP() const { return enabled_flash_sdp_; }

    void setSDPUseFA3(bool e) { enabled_fa3_sdp_ = e; }
    bool userEnabledFA3SDP() const { return enabled_fa3_sdp_; }

    void setSDPUseFA4(bool e) { enabled_fa4_sdp_ = e; }
    bool userEnabledFA4SDP() const { return enabled_fa4_sdp_; }

    void setSDPUseMemEfficient(bool e) { enabled_mem_efficient_sdp_ = e; }
    bool userEnabledMemEfficientSDP() const { return enabled_mem_efficient_sdp_; }

    void setSDPUseMath(bool e) { enabled_math_sdp_ = e; }
    bool userEnabledMathSDP() const { return enabled_math_sdp_; }

    void setSDPUseCuDNN(bool e) { enabled_cudnn_sdp_ = e; }
    bool userEnabledCuDNNSDP() const { return enabled_cudnn_sdp_; }

    void setSDPUseOverrideable(bool e) { enabled_overrideable_sdp_ = e; }
    bool userEnabledOverrideableSDP() const { return enabled_overrideable_sdp_; }

    void setAllowFP16BF16ReductionMathSDP(bool e) {
        allow_fp16_bf16_reduction_math_sdp_ = e;
    }
    bool allowFP16BF16ReductionMathSDP() const {
        return allow_fp16_bf16_reduction_math_sdp_;
    }

    // Every backend must appear exactly once; entries are validated by the
    // setter, and the routing loop reads them back in user-supplied order.
    void setSDPPriorityOrder(const std::vector<int64_t>& order);
    std::array<SDPBackend, num_sdp_backends> sdpPriorityOrder() const {
        return sdp_priority_order_;
    }

    // Optional SM carve-out hint forwarded to registered flash kernels.
    void setSMCarveout(std::optional<int32_t> val) { sm_carveout_ = val; }
    std::optional<int32_t> smCarveout() const { return sm_carveout_; }

private:
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context() = default;
    friend P10_API Context& globalContext();

    DType default_dtype_ = DType::Float32;
    bool deterministic_algorithms_ = false;
    bool deterministic_algorithms_warn_only_ = false;
    Float32MatmulPrecision float32_matmul_precision_ = Float32MatmulPrecision::HIGHEST;
    bool allow_tf32_cudnn_ = true;
    bool cudnn_benchmark_ = false;
    bool enabled_mkldnn_ = true;
    bool enabled_nnpack_ = true;
    LinalgBackend linalg_preferred_backend_ = LinalgBackend::Default;
    BlasBackend blas_preferred_backend_ = BlasBackend::Default;

    bool enabled_flash_sdp_ = true;
    bool enabled_fa3_sdp_ = false;
    bool enabled_fa4_sdp_ = false;
    bool enabled_mem_efficient_sdp_ = true;
    bool enabled_math_sdp_ = true;
    bool enabled_cudnn_sdp_ = true;
    bool enabled_overrideable_sdp_ = true;
    bool allow_fp16_bf16_reduction_math_sdp_ = false;
    std::array<SDPBackend, num_sdp_backends> sdp_priority_order_ = {
        SDPBackend::flash_attention,
        SDPBackend::efficient_attention,
        SDPBackend::math,
        SDPBackend::cudnn_attention,
        SDPBackend::overrideable};
    std::optional<int32_t> sm_carveout_ = std::nullopt;
};

P10_API Context& globalContext();

} // namespace tensorplay

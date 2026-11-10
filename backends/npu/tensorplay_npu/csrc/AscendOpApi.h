#pragma once

// aclnn launch plumbing: element type mapping, aclTensor and aclScalar
// views over device tensors, and the two-phase call with the executor
// cache the operator library keeps per thread.

#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <cstdint>
#include <vector>

#include "AscendRuntime.h"
#include "Exception.h"
#include "Scalar.h"
#include "Tensor.h"

namespace tensorplay {
namespace npu {

constexpr int kMaxAclDims = 8;

// acl element type for ``dtype``; false when the device has none.
bool acl_dtype_of(DType dtype, aclDataType* data_type);

// The floating types the native math kernels compute in.
inline bool is_native_float(DType dtype) {
    return dtype == DType::Float32 || dtype == DType::Float16 || dtype == DType::BFloat16;
}

// Element type two operands compute in: the dimensioned operand's type wins
// over a 0-d one of the same kind, otherwise the usual promotion.
DType binary_result_dtype(const Tensor& a, const Tensor& b);

// ``t`` (a device tensor) converted to ``dtype`` on its device; ``t``
// itself when it already has that dtype.
Tensor cast_on_device(const Tensor& t, DType dtype);

// ``t`` in the compute type, packed so an operator reads it directly: a
// contiguous device tensor of ``dtype``.
Tensor compute_operand(const Tensor& t, DType dtype);

// Views of one launch's arguments.  Each carries what phase one sees, so
// the executor cache can tell two calls apart without rebuilding anything.
struct AclTensorRef {
    aclTensor* ptr = nullptr;
    const void* data = nullptr;
    DType dtype = DType::Float32;
    int64_t sizes[kMaxAclDims] = {};
    int64_t strides[kMaxAclDims] = {};
    int32_t ndim = 0;
    operator aclTensor*() const { return ptr; }
};

struct AclScalarRef {
    aclScalar* ptr = nullptr;
    uint64_t bits = 0;
    DType dtype = DType::Float32;
    operator aclScalar*() const { return ptr; }
};

struct AclIntArrayRef {
    aclIntArray* ptr = nullptr;
    const int64_t* values = nullptr;
    int64_t size = 0;
    operator aclIntArray*() const { return ptr; }
};

// Owns the acl descriptors created for one launch.
class AclArgs {
public:
    AclArgs() = default;
    AclArgs(const AclArgs&) = delete;
    AclArgs& operator=(const AclArgs&) = delete;
    ~AclArgs();

    // ``t`` must live on a device with a supported rank and type.  The view
    // is described as it is: real strides and storage extent, so an operand
    // that is a transposed or sliced view is read without a repack.
    AclTensorRef tensor(const Tensor& t);
    // ``value`` narrowed to ``dtype``.
    AclScalarRef scalar(const Scalar& value, DType dtype);
    // ``values`` must outlive the launch (a temporary is fine: the view is
    // read before the call returns).
    AclIntArrayRef ints(const std::vector<int64_t>& values);

private:
    std::vector<aclTensor*> tensors_;
    std::vector<aclScalar*> scalars_;
    std::vector<aclIntArray*> arrays_;
};

namespace detail {

template <typename T>
struct acl_arg {
    using type = T;
};
template <>
struct acl_arg<AclTensorRef> {
    using type = aclTensor*;
};
template <>
struct acl_arg<AclScalarRef> {
    using type = aclScalar*;
};
template <>
struct acl_arg<AclIntArrayRef> {
    using type = aclIntArray*;
};

template <typename A>
typename acl_arg<A>::type native_arg(A value) {
    return static_cast<typename acl_arg<A>::type>(value);
}

inline void hash_mix(uint64_t& h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
}

inline void hash_arg(uint64_t& h, const AclTensorRef& t) {
    hash_mix(h, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(t.data)));
    hash_mix(h, static_cast<uint64_t>(t.dtype));
    hash_mix(h, static_cast<uint64_t>(t.ndim));
    for (int32_t i = 0; i < t.ndim; ++i) {
        hash_mix(h, static_cast<uint64_t>(t.sizes[i]));
        hash_mix(h, static_cast<uint64_t>(t.strides[i]));
    }
}

inline void hash_arg(uint64_t& h, const AclScalarRef& s) {
    hash_mix(h, s.bits);
    hash_mix(h, static_cast<uint64_t>(s.dtype));
}

inline void hash_arg(uint64_t& h, const AclIntArrayRef& a) {
    hash_mix(h, static_cast<uint64_t>(a.size));
    for (int64_t i = 0; i < a.size; ++i) hash_mix(h, static_cast<uint64_t>(a.values[i]));
}

inline void hash_arg(uint64_t& h, int64_t v) { hash_mix(h, static_cast<uint64_t>(v)); }
inline void hash_arg(uint64_t& h, int32_t v) { hash_mix(h, static_cast<uint64_t>(v)); }
inline void hash_arg(uint64_t& h, bool v) { hash_mix(h, v ? 1u : 0u); }
inline void hash_arg(uint64_t& h, aclDataType v) { hash_mix(h, static_cast<uint64_t>(v)); }
inline void hash_arg(uint64_t& h, const char* v) {
    for (; v != nullptr && *v != '\0'; ++v) hash_mix(h, static_cast<unsigned char>(*v));
}

// The operator library's per-thread executor cache.
struct ExecCacheApi {
    void (*init)() = nullptr;
    void (*uninit)() = nullptr;
    void (*set_key)(uint64_t) = nullptr;
    aclOpExecutor* (*get)(uint64_t, uint64_t*) = nullptr;
    bool (*usable)(const char*) = nullptr;
};

const ExecCacheApi& exec_cache_api();

// Grow-only device scratch reused by every launch of this thread.
void* workspace_for(int index, uint64_t nbytes);

inline const char* acl_error() {
    const char* message = aclGetRecentErrMsg();
    return message != nullptr ? message : "(no runtime message)";
}

} // namespace detail

// One aclnn operator, resolved from the operator library on first use.
// ``run`` takes the arguments in call order: tensor and scalar views, or
// plain values such as dimensions.  A cache hit skips phase one entirely.
class OpApi {
public:
    explicit OpApi(const char* name);

    bool valid() const { return get_workspace_ != nullptr && execute_ != nullptr; }

    template <typename... A>
    void run(int index, A... args) const {
        DeviceGuard guard(index);
        aclrtStream stream = npu::stream(index);
        uint64_t key = 0xcbf29ce484222325ull;
        for (const char* p = name_; *p != '\0'; ++p) {
            detail::hash_mix(key, static_cast<unsigned char>(*p));
        }
        (detail::hash_arg(key, args), ...);
        detail::hash_mix(key, static_cast<uint64_t>(index));
        detail::hash_mix(key, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream)));

        const detail::ExecCacheApi& cache = detail::exec_cache_api();
        const bool cacheable = cacheable_ && cache.get != nullptr && cache.init != nullptr &&
                               cache.set_key != nullptr;
        uint64_t workspace_size = 0;
        aclOpExecutor* executor = nullptr;
        bool cache_live = false;
        if (cacheable) {
            cache.init();
            cache_live = true;
            cache.set_key(key);
            executor = cache.get(key, &workspace_size);
        }
        struct Release {
            const detail::ExecCacheApi* cache;
            bool live;
            ~Release() {
                if (live) cache->uninit();
            }
        } release{&cache, cache_live};

        if (executor == nullptr) {
            using Phase1 = aclnnStatus (*)(typename detail::acl_arg<A>::type..., uint64_t*,
                                           aclOpExecutor**);
            const aclnnStatus status = reinterpret_cast<Phase1>(get_workspace_)(
                detail::native_arg(args)..., &workspace_size, &executor);
            TP_CHECK(status == ACL_SUCCESS, "aclnn ", name_,
                     " workspace query failed: ", detail::acl_error());
            TP_CHECK(executor != nullptr, "aclnn ", name_, " produced no executor");
        }
        void* workspace = workspace_size == 0 ? nullptr
                                              : detail::workspace_for(index, workspace_size);
        using Phase2 = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
        const aclnnStatus status =
            reinterpret_cast<Phase2>(execute_)(workspace, workspace_size, executor, stream);
        TP_CHECK(status == ACL_SUCCESS, "aclnn ", name_, " failed: ", detail::acl_error());
    }

private:
    const char* name_;
    void* get_workspace_ = nullptr;
    void* execute_ = nullptr;
    bool cacheable_ = false;
};

} // namespace npu
} // namespace tensorplay

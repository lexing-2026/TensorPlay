#include "AscendOpApi.h"

#include <dlfcn.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include "TypePromotion.h"

namespace tensorplay {
namespace npu {

namespace {

void* open_library(const char* name) {
    if (void* handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL)) return handle;
    const char* roots[] = {std::getenv("ASCEND_HOME_PATH"), std::getenv("ASCEND_TOOLKIT_HOME"),
                           "/usr/local/Ascend/ascend-toolkit/latest"};
    for (const char* root : roots) {
        if (root == nullptr) continue;
        const std::string path = std::string(root) + "/lib64/" + name;
        if (void* handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL)) return handle;
    }
    return nullptr;
}

struct SymbolTable {
    std::mutex lock;
    std::unordered_map<std::string, void*> resolved;
    std::vector<void*> libraries;
};

SymbolTable& symbol_table() {
    static SymbolTable table;
    return table;
}

} // namespace

void* op_api_symbol(const char* name) {
    SymbolTable& table = symbol_table();
    std::lock_guard<std::mutex> guard(table.lock);
    if (const auto it = table.resolved.find(name); it != table.resolved.end()) return it->second;
    if (table.libraries.empty()) {
        for (const char* library : {"libopapi.so", "libopapi_math.so", "libopapi_nn.so",
                                    "libopapi_cv.so", "libopapi_transformer.so"}) {
            if (void* handle = open_library(library)) table.libraries.push_back(handle);
        }
    }
    void* address = nullptr;
    for (void* handle : table.libraries) {
        address = dlsym(handle, name);
        if (address != nullptr) break;
    }
    table.resolved.emplace(name, address);
    return address;
}

namespace detail {

const ExecCacheApi& exec_cache_api() {
    static const ExecCacheApi api = [] {
        ExecCacheApi resolved;
        resolved.init = reinterpret_cast<void (*)()>(op_api_symbol("InitPTACacheThreadLocal"));
        resolved.uninit = reinterpret_cast<void (*)()>(op_api_symbol("UnInitPTACacheThreadLocal"));
        resolved.set_key = reinterpret_cast<void (*)(uint64_t)>(op_api_symbol("SetPTAHashKey"));
        resolved.get = reinterpret_cast<aclOpExecutor* (*)(uint64_t, uint64_t*)>(
            op_api_symbol("PTAGetExecCache"));
        resolved.usable = reinterpret_cast<bool (*)(const char*)>(op_api_symbol("CanUsePTACache"));
        return resolved;
    }();
    return api;
}

void* workspace_for(int index, uint64_t nbytes) {
    struct Scratch {
        void* ptr = nullptr;
        size_t size = 0;
    };
    thread_local std::array<Scratch, 64> scratch;
    TP_CHECK(index >= 0 && index < static_cast<int>(scratch.size()), "device index ", index,
             " out of range");
    Scratch& entry = scratch[static_cast<size_t>(index)];
    if (static_cast<uint64_t>(entry.size) >= nbytes) return entry.ptr;
    // Whatever the old buffer still feeds has to drain before it is freed.
    synchronize(index);
    void* grown = nullptr;
    TP_ACL_CHECK(aclrtMalloc(&grown, static_cast<size_t>(nbytes), ACL_MEM_MALLOC_HUGE_FIRST));
    if (entry.ptr != nullptr) TP_ACL_CHECK(aclrtFree(entry.ptr));
    entry.ptr = grown;
    entry.size = static_cast<size_t>(nbytes);
    return entry.ptr;
}

} // namespace detail

bool acl_dtype_of(DType dtype, aclDataType* data_type) {
    switch (dtype) {
        case DType::Float32: *data_type = ACL_FLOAT; return true;
        case DType::Float16: *data_type = ACL_FLOAT16; return true;
        case DType::BFloat16: *data_type = ACL_BF16; return true;
        case DType::Float64: *data_type = ACL_DOUBLE; return true;
        case DType::Int64: *data_type = ACL_INT64; return true;
        case DType::Int32: *data_type = ACL_INT32; return true;
        case DType::Int16: *data_type = ACL_INT16; return true;
        case DType::Int8: *data_type = ACL_INT8; return true;
        case DType::UInt8: *data_type = ACL_UINT8; return true;
        case DType::Bool: *data_type = ACL_BOOL; return true;
        default: return false;
    }
}

DType binary_result_dtype(const Tensor& a, const Tensor& b) {
    if (a.dtype() == b.dtype()) return a.dtype();
    const bool a_scalar = a.dim() == 0;
    const bool b_scalar = b.dim() == 0;
    if (a_scalar != b_scalar) {
        const Tensor& full = a_scalar ? b : a;
        const Tensor& scalar = a_scalar ? a : b;
        const bool same_kind = isFloatingType(full.dtype()) == isFloatingType(scalar.dtype()) &&
                               (full.dtype() == DType::Bool) == (scalar.dtype() == DType::Bool);
        if (same_kind || (isFloatingType(full.dtype()) && !isFloatingType(scalar.dtype()))) {
            return full.dtype();
        }
    }
    return promoteTypes(a.dtype(), b.dtype());
}

Tensor cast_on_device(const Tensor& t, DType dtype) {
    if (t.dtype() == dtype) return t;
    static const OpApi api("aclnnCast");
    TP_CHECK(api.valid(), "aclnnCast is missing from the operator library");
    aclDataType target = ACL_DT_UNDEFINED;
    TP_CHECK(acl_dtype_of(dtype, &target), "no acl element type for ", dtype);
    const Tensor source = contiguous_on_device(t);
    Tensor out = empty_on(source.sizes().vec(), dtype, index_of(t.device()));
    if (out.numel() == 0) return out;
    AclArgs args;
    const AclTensorRef self = args.tensor(source);
    const AclTensorRef result = args.tensor(out);
    api.run(index_of(t.device()), self, target, result);
    return out;
}

Tensor compute_operand(const Tensor& t, DType dtype) {
    if (t.dtype() != dtype) return cast_on_device(t, dtype);
    return t.is_contiguous() ? t : contiguous_on_device(t);
}

AclArgs::~AclArgs() {    for (aclTensor* tensor : tensors_) aclDestroyTensor(tensor);
    for (aclScalar* scalar : scalars_) aclDestroyScalar(scalar);
    for (aclIntArray* array : arrays_) aclDestroyIntArray(array);
}

AclTensorRef AclArgs::tensor(const Tensor& t) {
    TP_CHECK(t.device().is_privateuse1(), "aclnn operand is not a device tensor");
    TP_CHECK(t.is_contiguous(), "aclnn operand is not contiguous");
    TP_CHECK(t.dim() <= kMaxAclDims, "rank ", t.dim(), " exceeds the acl tensor limit");
    aclDataType data_type = ACL_DT_UNDEFINED;
    TP_CHECK(acl_dtype_of(t.dtype(), &data_type), "no acl element type for ", t.dtype());
    int64_t dims[kMaxAclDims] = {};
    int64_t strides[kMaxAclDims] = {};
    int64_t ndim = t.dim();
    if (ndim == 0) {
        // A 0-d tensor reads as one element: the operator sees a 1-d view
        // of the same memory.
        dims[0] = 1;
        strides[0] = 1;
        ndim = 1;
    } else {
        const std::vector<int64_t> sizes = t.sizes().vec();
        const std::vector<int64_t> tensor_strides = t.strides();
        for (int64_t i = 0; i < ndim; ++i) {
            dims[i] = sizes[static_cast<size_t>(i)];
            strides[i] = tensor_strides[static_cast<size_t>(i)];
        }
    }
    aclTensor* created =
        aclCreateTensor(dims, static_cast<uint64_t>(ndim), data_type, strides, 0, ACL_FORMAT_ND,
                        dims, static_cast<uint64_t>(ndim), const_cast<void*>(t.data_ptr()));
    TP_CHECK(created != nullptr, "aclCreateTensor failed for ", t.dtype(), " rank ", t.dim());
    tensors_.push_back(created);
    AclTensorRef view;
    view.ptr = created;
    view.data = t.data_ptr();
    view.dtype = t.dtype();
    view.ndim = static_cast<int32_t>(ndim);
    std::memcpy(view.sizes, dims, sizeof(int64_t) * static_cast<size_t>(ndim));
    return view;
}

AclScalarRef AclArgs::scalar(const Scalar& value, DType dtype) {
    aclDataType data_type = ACL_DT_UNDEFINED;
    TP_CHECK(acl_dtype_of(dtype, &data_type), "no acl element type for ", dtype);
    alignas(8) unsigned char buffer[8] = {};
    switch (dtype) {
        case DType::Float32: {
            const float narrowed = value.to<float>();
            std::memcpy(buffer, &narrowed, sizeof(narrowed));
            break;
        }
        case DType::Float16: {
            const Half narrowed(value.to<float>());
            std::memcpy(buffer, &narrowed, sizeof(narrowed));
            break;
        }
        case DType::BFloat16: {
            const BFloat16 narrowed(value.to<float>());
            std::memcpy(buffer, &narrowed, sizeof(narrowed));
            break;
        }
        case DType::Float64: {
            const double narrowed = value.to<double>();
            std::memcpy(buffer, &narrowed, sizeof(narrowed));
            break;
        }
        case DType::Int64: {
            const int64_t narrowed = value.to<int64_t>();
            std::memcpy(buffer, &narrowed, sizeof(narrowed));
            break;
        }
        case DType::Int32: {
            const int32_t narrowed = value.to<int32_t>();
            std::memcpy(buffer, &narrowed, sizeof(narrowed));
            break;
        }
        case DType::Int16: {
            const int16_t narrowed = value.to<int16_t>();
            std::memcpy(buffer, &narrowed, sizeof(narrowed));
            break;
        }
        case DType::Int8: {
            const int8_t narrowed = value.to<int8_t>();
            std::memcpy(buffer, &narrowed, sizeof(narrowed));
            break;
        }
        case DType::UInt8: {
            const uint8_t narrowed = value.to<uint8_t>();
            std::memcpy(buffer, &narrowed, sizeof(narrowed));
            break;
        }
        case DType::Bool: {
            const bool narrowed = value.to<bool>();
            std::memcpy(buffer, &narrowed, sizeof(narrowed));
            break;
        }
        default:
            TP_CHECK(false, "no acl scalar element type for ", dtype);
    }
    aclScalar* created = aclCreateScalar(buffer, data_type);
    TP_CHECK(created != nullptr, "aclCreateScalar failed");
    scalars_.push_back(created);
    AclScalarRef view;
    view.ptr = created;
    std::memcpy(&view.bits, buffer, sizeof(view.bits));
    view.dtype = dtype;
    return view;
}

AclIntArrayRef AclArgs::ints(const std::vector<int64_t>& values) {
    aclIntArray* created = aclCreateIntArray(values.data(), values.size());
    TP_CHECK(created != nullptr, "aclCreateIntArray failed");
    arrays_.push_back(created);
    AclIntArrayRef view;
    view.ptr = created;
    view.values = values.data();
    view.size = static_cast<int64_t>(values.size());
    return view;
}

OpApi::OpApi(const char* name) : name_(name) {
    const std::string workspace_name = std::string(name) + "GetWorkspaceSize";
    get_workspace_ = op_api_symbol(workspace_name.c_str());
    execute_ = op_api_symbol(name);
    const detail::ExecCacheApi& cache = detail::exec_cache_api();
    cacheable_ = cache.get != nullptr && cache.init != nullptr && cache.set_key != nullptr &&
                 (cache.usable == nullptr || cache.usable(name));
}

} // namespace npu
} // namespace tensorplay

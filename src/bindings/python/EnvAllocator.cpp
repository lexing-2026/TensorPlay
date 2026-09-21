// Native environment allocator for the C FFI used by JIT-loaded kernels.
//
// Kernel-side allocation requests (Tensor::FromEnvAlloc against the
// engine's env-alloc entry point) are fulfilled here with native
// tensorplay allocations. No Python object crosses the path, so
// requests are served from any thread without the interpreter lock.
//
// The engine is an optional runtime dependency: its C entry points are
// resolved with dlopen on demand and nothing links against it.

#include "python_bindings.h"
#include "dlpack_types.h"
#include "DLPackConvert.h"

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

using AllocatorFn = DLPackManagedTensorAllocatorFunction;
using SetAllocatorFn = int (*)(AllocatorFn, int, AllocatorFn*);
using GetAllocatorFn = AllocatorFn (*)();

// Keeps the tensor — and with it the storage, sizes and strides that
// the DLTensor points into — alive for as long as the consumer holds
// the versioned wrapper. The deleter is the only legal way to release
// it.
struct EnvAllocContext {
    Tensor handle;
    DLManagedTensorVersioned tensor{};
};

void env_alloc_deleter(DLManagedTensorVersioned* self) {
    delete static_cast<EnvAllocContext*>(self->manager_ctx);
}

int env_allocate(DLTensor* prototype, DLManagedTensorVersioned** out,
                 void* error_ctx,
                 void (*set_error)(void*, const char*, const char*)) {
    try {
        DType dtype = from_dlpack_dtype(prototype->dtype);
        Device device = from_dlpack_device(prototype->device);
        std::vector<int64_t> size(prototype->shape,
                                  prototype->shape + prototype->ndim);

        auto ctx = std::make_unique<EnvAllocContext>();
        ctx->handle = Tensor::empty(size, dtype, device);

        DLManagedTensorVersioned* wrapper = &ctx->tensor;
        wrapper->version.major = 1;
        wrapper->version.minor = 0;
        wrapper->manager_ctx = ctx.get();
        wrapper->deleter = env_alloc_deleter;
        wrapper->flags = 0;

        DLTensor& dl = wrapper->dl_tensor;
        auto impl = ctx->handle.unsafeGetTensorImpl();
        dl.data = ctx->handle.data_ptr();
        dl.device = to_dlpack_device(device);
        dl.ndim = static_cast<int32_t>(impl->dim());
        dl.dtype = to_dlpack_dtype(ctx->handle.dtype());
        dl.shape = const_cast<int64_t*>(impl->sizes().data());
        dl.strides = const_cast<int64_t*>(impl->strides().data());
        dl.byte_offset = 0;

        *out = &(ctx.release()->tensor);
        return 0;
    } catch (const std::exception& exc) {
        if (set_error != nullptr) {
            set_error(error_ctx, "RuntimeError", exc.what());
        }
        return -1;
    }
}

#ifndef _WIN32

// Install status: 0 = installed, 1 = a host allocator was already
// present (it wins), 2 = the engine could not be used.
int install_ffi_env_allocator(const char* engine_library_path) {
    static std::once_flag once;
    static int status = 2;
    std::call_once(once, [&] {
        void* lib = dlopen(engine_library_path, RTLD_LAZY | RTLD_LOCAL);
        if (lib == nullptr) return;
        void* set_sym = dlsym(lib, "TVMFFIEnvSetDLPackManagedTensorAllocator");
        void* get_sym = dlsym(lib, "TVMFFIEnvGetDLPackManagedTensorAllocator");
        if (set_sym == nullptr || get_sym == nullptr) return;
        auto get = reinterpret_cast<GetAllocatorFn>(get_sym);
        if (get() != nullptr) {
            status = 1;
            return;
        }
        auto set = reinterpret_cast<SetAllocatorFn>(set_sym);
        if (set(&env_allocate, 1, nullptr) != 0) return;
        status = 0;
    });
    return status;
}

#else

int install_ffi_env_allocator(const char* engine_library_path) {
    (void)engine_library_path;
    return 2;  // dynamic engine loading is not supported on this platform
}

#endif

}  // namespace

void init_env_allocator(py::module_& m) {
    m.def("_install_ffi_env_allocator", &install_ffi_env_allocator,
          py::arg("engine_library_path"),
          "Install the native tensor allocator for the FFI engine.\n\n"
          "Returns 0 when installed, 1 when a host allocator is already\n"
          "present, 2 when the engine could not be used.");
}

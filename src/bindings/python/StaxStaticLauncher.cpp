// Static launch surface for recorded Triton kernel binaries.
//
// The Stax fast-launch path replays a recorded CompiledKernel dispatch with
// pre-bound geometry. This module goes one level lower: a small object type
// pre-binds the CUfunction handle, block geometry, shared-memory size and
// the scalar type string of the live kernel parameters, so one launch is
// one vectorcall that packs arguments into stack slots and calls
// cuLaunchKernel directly -- no per-call binder, no metadata build, no
// re-parsing through the generic launcher.
//
// Call-shape compatibility: the object accepts the same positional layout
// the replay path uses -- (grid_x, grid_y, grid_z, stream, function_handle,
// packed_metadata, launch_metadata, enter_hook, exit_hook, *kernel_args) --
// ignoring the five middle slots it has already pre-bound. A recorded
// record tuple can swap its callable for this object without touching the
// call site.
//
// Parameter packing: the caller passes the FULL kernel argument list in
// declaration order, including positions that were compiled away
// (``constexpr`` parameters and scalars specialized to a fixed value).
// ``live_indices`` maps each packed parameter to its position in that full
// list; dead positions are never read. Trailing scratch parameters required
// by the current kernel ABI (global / profile scratch pointers) are
// pre-zeroed NULLs: kernels that actually need scratch storage are rejected
// at construction time, so the NULL slots only satisfy the ABI's fixed
// trailing parameters.
//
// Lifetime: the constructor stores a reference to the ``keepalive`` object
// (the owning compiled kernel); the CUfunction handle stays valid as long
// as this object lives. Shared-memory opt-in above the 48 KB static limit
// is configured once when the kernel handles are initialized, so the
// launch passes the recorded shared-memory size unmodified.
//
// Thread safety: the per-call slot storage lives in the object and is
// written with the GIL held; cuLaunchKernel copies the argument values
// synchronously before returning, so nothing escapes the call. Not safe
// under free-threaded Python.

#include "python_bindings.h"

#include <Python.h>

#include <cstdint>
#include <cstring>

#ifdef USE_CUDA

#include <cuda.h>

#include "CPythonBridge.h"
#include "backend/cuda/DriverApi.h"

namespace {

constexpr int kMaxArgs = 128;

struct StaxFastLauncherObject {
    PyObject_HEAD
    vectorcallfunc vectorcall;
    CUfunction function;
    uint32_t num_warps;
    uint32_t shared_bytes;
    int num_live;       // packed kernel parameters
    int num_scratch;    // trailing ABI scratch slots (pre-zeroed NULLs)
    int num_call_args;  // full Python argument count (live + dead)
    char types[kMaxArgs + 1];
    int live_index[kMaxArgs];
    uint64_t storage[kMaxArgs];
    void* slots[kMaxArgs];
    PyObject* keepalive;
};

// Device pointer of one pointer-typed argument: an int is taken as the raw
// address, None is a NULL, a plain tensor wrapper is read through the C++
// value holder, and anything else falls back to a ``data_ptr()`` call.
CUdeviceptr device_pointer(PyObject* obj) {
    if (PyLong_Check(obj)) {
        unsigned long long raw = PyLong_AsUnsignedLongLong(obj);
        if (raw == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
            return 0;
        }
        return static_cast<CUdeviceptr>(raw);
    }
    if (Py_IsNone(obj)) {
        return 0;
    }
    long long version = 0;
    if (tensorplay::python_c::tpx_tensor_guard_probe(obj, &version) >= 0) {
        const tensorplay::Tensor& tensor =
            tensorplay::python_c::tpx_py_tensor_cref(obj);
        return reinterpret_cast<CUdeviceptr>(tensor.data_ptr());
    }
    PyObject* method = PyObject_GetAttrString(obj, "data_ptr");
    if (method == nullptr) {
        return 0;
    }
    PyObject* raw = PyObject_CallNoArgs(method);
    Py_DECREF(method);
    if (raw == nullptr) {
        return 0;
    }
    unsigned long long address = PyLong_AsUnsignedLongLong(raw);
    Py_DECREF(raw);
    if (address == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
        return 0;
    }
    return static_cast<CUdeviceptr>(address);
}

int64_t as_int64(PyObject* obj) {
    long long value = PyLong_AsLongLong(obj);
    if (value == -1 && PyErr_Occurred()) {
        return 0;
    }
    return static_cast<int64_t>(value);
}

uint64_t as_uint64(PyObject* obj) {
    unsigned long long value = PyLong_AsUnsignedLongLong(obj);
    if (value == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
        return 0;
    }
    return static_cast<uint64_t>(value);
}

// The hot path -- one call per recorded kernel launch.
// args layout: [grid_x, grid_y, grid_z, stream, function_handle,
// packed_metadata, launch_metadata, enter_hook, exit_hook, *kernel_args];
// slots 4..8 duplicate what the object pre-binds and are ignored.

namespace {

using CuCtxGetCurrentFn = CUresult (*)(CUcontext*);
using CuInitFn = CUresult (*)(unsigned int);
using CuDeviceGetFn = CUresult (*)(CUdevice*, int);
using CuDevicePrimaryCtxRetainFn = CUresult (*)(CUcontext*, CUdevice);
using CuCtxSetCurrentFn = CUresult (*)(CUcontext);
using CuLaunchKernelFn =
    CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int,
                 unsigned int, unsigned int, unsigned int, unsigned int,
                 CUstream, void**, void**);

// Driver entry points are resolved lazily so the wheel imports on machines
// without the driver; the launcher reports a missing driver at call time.
struct StaxDriverApi {
    StaxDriverApi() {
        cu_ctx_get_current = tensorplay::cuda::driver::resolve_symbol<
            CuCtxGetCurrentFn>("cuCtxGetCurrent");
        cu_init = tensorplay::cuda::driver::resolve_symbol<CuInitFn>("cuInit");
        cu_device_get =
            tensorplay::cuda::driver::resolve_symbol<CuDeviceGetFn>(
                "cuDeviceGet");
        cu_device_primary_ctx_retain =
            tensorplay::cuda::driver::resolve_symbol<
                CuDevicePrimaryCtxRetainFn>("cuDevicePrimaryCtxRetain");
        cu_ctx_set_current =
            tensorplay::cuda::driver::resolve_symbol<CuCtxSetCurrentFn>(
                "cuCtxSetCurrent");
        cu_launch_kernel =
            tensorplay::cuda::driver::resolve_symbol<CuLaunchKernelFn>(
                "cuLaunchKernel");
    }
    CuCtxGetCurrentFn cu_ctx_get_current;
    CuInitFn cu_init;
    CuDeviceGetFn cu_device_get;
    CuDevicePrimaryCtxRetainFn cu_device_primary_ctx_retain;
    CuCtxSetCurrentFn cu_ctx_set_current;
    CuLaunchKernelFn cu_launch_kernel;
};

StaxDriverApi& stax_driver_api() {
    static StaxDriverApi api;
    return api;
}

}  // namespace

PyObject* stax_fast_launcher_vectorcall(
    PyObject* callable,
    PyObject* const* args,
    size_t nargsf,
    PyObject* kwnames) {
    auto* self = reinterpret_cast<StaxFastLauncherObject*>(callable);
    if (kwnames != nullptr && PyTuple_GET_SIZE(kwnames) != 0) {
        PyErr_SetString(PyExc_TypeError,
                        "_StaxFastLauncher: keyword arguments are not"
                        " supported");
        return nullptr;
    }
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    Py_ssize_t expected = 9 + static_cast<Py_ssize_t>(self->num_call_args);
    if (nargs != expected) {
        PyErr_Format(PyExc_TypeError,
                     "_StaxFastLauncher: expected %zd arguments, got %zd",
                     expected, nargs);
        return nullptr;
    }
    long long grid_x = PyLong_AsLongLong(args[0]);
    long long grid_y = PyLong_AsLongLong(args[1]);
    long long grid_z = PyLong_AsLongLong(args[2]);
    if (PyErr_Occurred()) {
        return nullptr;
    }
    if (grid_x <= 0 || grid_y <= 0 || grid_z <= 0) {
        Py_RETURN_NONE;
    }
    uint64_t stream = as_uint64(args[3]);
    if (PyErr_Occurred()) {
        return nullptr;
    }

    for (int k = 0; k < self->num_live; ++k) {
        PyObject* item = args[9 + self->live_index[k]];
        void* slot = static_cast<void*>(&self->storage[k]);
        switch (self->types[k]) {
            case 'O': {
                CUdeviceptr pointer = device_pointer(item);
                if (PyErr_Occurred()) {
                    return nullptr;
                }
                *reinterpret_cast<CUdeviceptr*>(slot) = pointer;
                break;
            }
            case 'b':
                *reinterpret_cast<int8_t*>(slot) =
                    static_cast<int8_t>(as_int64(item));
                break;
            case 'h':
                *reinterpret_cast<int16_t*>(slot) =
                    static_cast<int16_t>(as_int64(item));
                break;
            case 'i':
                *reinterpret_cast<int32_t*>(slot) =
                    static_cast<int32_t>(as_int64(item));
                break;
            case 'l':
                *reinterpret_cast<int64_t*>(slot) = as_int64(item);
                break;
            case 'B':
                *reinterpret_cast<uint8_t*>(slot) =
                    static_cast<uint8_t>(as_uint64(item));
                break;
            case 'H':
                *reinterpret_cast<uint16_t*>(slot) =
                    static_cast<uint16_t>(as_uint64(item));
                break;
            case 'I':
                *reinterpret_cast<uint32_t*>(slot) =
                    static_cast<uint32_t>(as_uint64(item));
                break;
            case 'K':
                *reinterpret_cast<uint64_t*>(slot) = as_uint64(item);
                break;
            case 'f':
                *reinterpret_cast<float*>(slot) =
                    static_cast<float>(PyFloat_AsDouble(item));
                break;
            case 'd':
                *reinterpret_cast<double*>(slot) = PyFloat_AsDouble(item);
                break;
            default:
                PyErr_Format(PyExc_TypeError,
                             "_StaxFastLauncher: unknown argument type '%c'",
                             self->types[k]);
                return nullptr;
        }
        if (PyErr_Occurred()) {
            return nullptr;
        }
    }

    StaxDriverApi& driver_api = stax_driver_api();
    if (driver_api.cu_ctx_get_current == nullptr ||
        driver_api.cu_init == nullptr || driver_api.cu_device_get == nullptr ||
        driver_api.cu_device_primary_ctx_retain == nullptr ||
        driver_api.cu_ctx_set_current == nullptr ||
        driver_api.cu_launch_kernel == nullptr) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_StaxFastLauncher: the CUDA driver is not available");
        return nullptr;
    }
    // Kernel handles are initialized on the current context; a process that
    // never touched CUDA before starts from cuInit, then acquires the
    // device's primary context.
    CUcontext context = nullptr;
    CUresult context_result = driver_api.cu_ctx_get_current(&context);
    if (context_result == CUDA_ERROR_NOT_INITIALIZED) {
        if (driver_api.cu_init(0) != CUDA_SUCCESS) {
            PyErr_SetString(PyExc_RuntimeError,
                            "_StaxFastLauncher: cuInit failed");
            return nullptr;
        }
        context_result = driver_api.cu_ctx_get_current(&context);
    }
    if (context_result != CUDA_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError,
                        "_StaxFastLauncher: cuCtxGetCurrent failed");
        return nullptr;
    }
    if (context == nullptr) {
        CUdevice device = 0;
        if (driver_api.cu_device_get(&device, 0) != CUDA_SUCCESS ||
            driver_api.cu_device_primary_ctx_retain(&context, device) !=
                CUDA_SUCCESS ||
            driver_api.cu_ctx_set_current(context) != CUDA_SUCCESS) {
            PyErr_SetString(PyExc_RuntimeError,
                            "_StaxFastLauncher: failed to acquire a CUDA"
                            " context");
            return nullptr;
        }
    }
    CUresult result = driver_api.cu_launch_kernel(
        self->function,
        static_cast<uint32_t>(grid_x),
        static_cast<uint32_t>(grid_y),
        static_cast<uint32_t>(grid_z),
        32u * self->num_warps,
        1,
        1,
        self->shared_bytes,
        reinterpret_cast<CUstream>(stream),
        self->slots,
        nullptr);
    if (result != CUDA_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError,
                     "_StaxFastLauncher: cuLaunchKernel failed with code %d",
                     static_cast<int>(result));
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* StaxFastLauncher_new(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwds) {
    if (kwds != nullptr && PyDict_GET_SIZE(kwds) != 0) {
        PyErr_SetString(PyExc_TypeError,
                        "_StaxFastLauncher: keyword arguments are not"
                        " supported");
        return nullptr;
    }
    unsigned long long function_handle = 0;
    int num_warps = 0;
    int shared_bytes = 0;
    const char* live_types = nullptr;
    PyObject* live_indices = nullptr;
    int num_scratch = 0;
    int num_call_args = 0;
    PyObject* keepalive = nullptr;
    if (!PyArg_ParseTuple(
            args,
            "KiisOiiO",
            &function_handle,
            &num_warps,
            &shared_bytes,
            &live_types,
            &live_indices,
            &num_scratch,
            &num_call_args,
            &keepalive)) {
        return nullptr;
    }
    if (keepalive == nullptr) {
        PyErr_SetString(PyExc_ValueError,
                        "_StaxFastLauncher: keepalive object is required");
        return nullptr;
    }
    Py_ssize_t num_live_seq = PySequence_Size(live_indices);
    if (num_live_seq < 0) {
        return nullptr;
    }
    size_t num_live = static_cast<size_t>(strlen(live_types));
    if (num_live != static_cast<size_t>(num_live_seq)) {
        PyErr_Format(PyExc_ValueError,
                     "_StaxFastLauncher: %zu live types but %zd indices",
                     num_live, num_live_seq);
        return nullptr;
    }
    if (num_live + static_cast<size_t>(num_scratch) >
        static_cast<size_t>(kMaxArgs)) {
        PyErr_Format(PyExc_ValueError,
                     "_StaxFastLauncher: too many arguments (%zu > %d)",
                     num_live + static_cast<size_t>(num_scratch), kMaxArgs);
        return nullptr;
    }
    auto* self = reinterpret_cast<StaxFastLauncherObject*>(
        type->tp_alloc(type, 0));
    if (self == nullptr) {
        return nullptr;
    }
    self->function = reinterpret_cast<CUfunction>(function_handle);
    self->num_warps = static_cast<uint32_t>(num_warps);
    self->shared_bytes = static_cast<uint32_t>(shared_bytes);
    self->num_live = static_cast<int>(num_live);
    self->num_scratch = num_scratch;
    self->num_call_args = num_call_args;
    self->keepalive = Py_NewRef(keepalive);
    std::memcpy(self->types, live_types, num_live);
    for (size_t k = num_live;
         k < num_live + static_cast<size_t>(num_scratch);
         ++k) {
        self->types[k] = 'O';
    }
    self->types[num_live + static_cast<size_t>(num_scratch)] = '\0';
    std::memset(self->storage, 0, sizeof(self->storage));
    for (int k = 0; k < kMaxArgs; ++k) {
        self->slots[k] = &self->storage[k];
    }
    for (size_t k = 0; k < num_live; ++k) {
        PyObject* item = PySequence_GetItem(live_indices, k);
        if (item == nullptr) {
            Py_DECREF(self);
            return nullptr;
        }
        long long index = PyLong_AsLongLong(item);
        Py_DECREF(item);
        if (index == -1 && PyErr_Occurred()) {
            Py_DECREF(self);
            return nullptr;
        }
        if (index < 0 || index >= num_call_args) {
            PyErr_Format(PyExc_ValueError,
                         "_StaxFastLauncher: live index %lld outside the"
                         " %d-argument call",
                         index, num_call_args);
            Py_DECREF(self);
            return nullptr;
        }
        self->live_index[k] = static_cast<int>(index);
    }
    self->vectorcall = stax_fast_launcher_vectorcall;
    return reinterpret_cast<PyObject*>(self);
}

void StaxFastLauncher_dealloc(PyObject* self) {
    auto* launcher = reinterpret_cast<StaxFastLauncherObject*>(self);
    Py_XDECREF(launcher->keepalive);
    Py_TYPE(self)->tp_free(self);
}

PyTypeObject StaxFastLauncherType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "tensorplay._C._StaxFastLauncher",  // tp_name
    sizeof(StaxFastLauncherObject),     // tp_basicsize
    0,                                  // tp_itemsize
    StaxFastLauncher_dealloc,           // tp_dealloc
    offsetof(StaxFastLauncherObject, vectorcall),  // tp_vectorcall_offset
    nullptr,                            // tp_getattr
    nullptr,                            // tp_setattr
    nullptr,                            // tp_as_async
    nullptr,                            // tp_repr
    nullptr,                            // tp_as_number
    nullptr,                            // tp_as_sequence
    nullptr,                            // tp_as_mapping
    nullptr,                            // tp_hash
    PyVectorcall_Call,                  // tp_call
    nullptr,                            // tp_str
    nullptr,                            // tp_getattro
    nullptr,                            // tp_setattro
    nullptr,                            // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    "Pre-bound direct launcher for recorded Triton kernels (vectorcall)",
    nullptr,                            // tp_traverse
    nullptr,                            // tp_clear
    nullptr,                            // tp_richcompare
    0,                                  // tp_weaklistoffset
    nullptr,                            // tp_iter
    nullptr,                            // tp_iternext
    nullptr,                            // tp_methods
    nullptr,                            // tp_members
    nullptr,                            // tp_getset
    nullptr,                            // tp_base
    nullptr,                            // tp_dict
    nullptr,                            // tp_descr_get
    nullptr,                            // tp_descr_set
    0,                                  // tp_dictoffset
    nullptr,                            // tp_init
    nullptr,                            // tp_alloc
    StaxFastLauncher_new,               // tp_new
};

}  // namespace

void init_stax_static_launcher(pybind11::module_& m) {
    if (PyType_Ready(&StaxFastLauncherType) < 0) {
        throw pybind11::error_already_set();
    }
    if (PyModule_AddType(m.ptr(), &StaxFastLauncherType) < 0) {
        throw pybind11::error_already_set();
    }
}

#else  // !USE_CUDA

void init_stax_static_launcher(pybind11::module_& m) {
    (void)m;  // No CUDA surface: the type stays absent and callers fall back.
}

#endif  // USE_CUDA

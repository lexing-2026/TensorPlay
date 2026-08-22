#include "python_bindings.h"
#include "tensorplay/ops/TensorBindingsGenerated.h"
#include "utils.h"
#include "dlpack_types.h"
#include "TensorImpl.h" // For unsafeGetTensorImpl
#include "Autograd.h" // tpx autograd helpers; Tensor is the p10 tensor type
#include "Storage.h"
#include "DataPtr.h"
#include "Node.h" // For grad_fn
#include <mutex>
#include <pybind11/functional.h>
#include <cstring>
#include <cstdio>

#ifdef USE_CUDA
#include "CUDARuntime.h"
#include <cuda_runtime.h>
#endif

using namespace tensorplay::python;

// Using TPX Tensor as the main Tensor exposed to Python
using Tensor = tensorplay::tpx::Tensor; 
using Tensor = tensorplay::Tensor;

// --- DLPack Helpers ---

static DLDataType to_dlpack_dtype(DType dtype) {
    DLDataType dt;
    dt.lanes = 1;
    switch (dtype) {
        case DType::Float32: dt.code = kDLFloat; dt.bits = 32; break;
        case DType::Float64: dt.code = kDLFloat; dt.bits = 64; break;
        case DType::Float16: dt.code = kDLFloat; dt.bits = 16; break;
        case DType::BFloat16: dt.code = kDLBfloat; dt.bits = 16; break;
        case DType::ComplexHalf: dt.code = kDLComplex; dt.bits = 32; break;
        case DType::ComplexFloat: dt.code = kDLComplex; dt.bits = 64; break;
        case DType::ComplexDouble: dt.code = kDLComplex; dt.bits = 128; break;
        // DLPack has no bfloat16-complex code. Keep this explicit instead of
        // silently advertising an incompatible component type.
        case DType::BComplex32:
            TP_THROW(RuntimeError, "BComplex32 is not supported by DLPack");
        case DType::Int32:   dt.code = kDLInt;   dt.bits = 32; break;
        case DType::Int64:   dt.code = kDLInt;   dt.bits = 64; break;
        case DType::Int8:    dt.code = kDLInt;   dt.bits = 8;  break;
        case DType::Int16:   dt.code = kDLInt;   dt.bits = 16; break;
        case DType::UInt8:   dt.code = kDLUInt;  dt.bits = 8;  break;
        case DType::UInt16:  dt.code = kDLUInt;  dt.bits = 16; break;
        case DType::UInt32:  dt.code = kDLUInt;  dt.bits = 32; break;
        case DType::UInt64:  dt.code = kDLUInt;  dt.bits = 64; break;
        case DType::Bool:    dt.code = kDLBool;  dt.bits = 8;  break;
        default: TP_THROW(RuntimeError, "Unsupported DType for DLPack");
    }
    return dt;
}

static DType from_dlpack_dtype(DLDataType dt) {
    if (dt.lanes != 1) TP_THROW(RuntimeError, "DLPack: Unsupported lanes != 1");
    if (dt.code == kDLFloat) {
        if (dt.bits == 32) return DType::Float32;
        if (dt.bits == 64) return DType::Float64;
        if (dt.bits == 16) return DType::Float16;
    } else if (dt.code == kDLBfloat) {
        if (dt.bits == 16) return DType::BFloat16;
    } else if (dt.code == kDLComplex) {
        if (dt.bits == 32) return DType::ComplexHalf;
        if (dt.bits == 64) return DType::ComplexFloat;
        if (dt.bits == 128) return DType::ComplexDouble;
    } else if (dt.code == kDLInt) {
        if (dt.bits == 32) return DType::Int32;
        if (dt.bits == 64) return DType::Int64;
        if (dt.bits == 8)  return DType::Int8;
        if (dt.bits == 16) return DType::Int16;
    } else if (dt.code == kDLUInt) {
        if (dt.bits == 8)  return DType::UInt8;
        if (dt.bits == 16) return DType::UInt16;
        if (dt.bits == 32) return DType::UInt32;
        if (dt.bits == 64) return DType::UInt64;
    } else if (dt.code == kDLBool) {
        if (dt.bits == 8) return DType::Bool;
    }
    TP_THROW(RuntimeError, "Unsupported DLPack dtype");
}

static DLDevice to_dlpack_device(Device device) {
    DLDevice d;
    d.device_id = device.index();
    switch (device.type()) {
        case DeviceType::CPU: d.device_type = kDLCPU; break;
        case DeviceType::CUDA: d.device_type = kDLCUDA; break;
        default: TP_THROW(RuntimeError, "Unsupported Device for DLPack");
    }
    return d;
}

static Device from_dlpack_device(DLDevice d) {
    DeviceType type;
    switch (d.device_type) {
        case kDLCPU: type = DeviceType::CPU; break;
        case kDLCUDA: type = DeviceType::CUDA; break;
        case kDLCUDAHost: type = DeviceType::CPU; break; // Treat CUDA Host as CPU
        default: TP_THROW(RuntimeError, "Unsupported DLPack device type");
    }
    return Device(type, d.device_id);
}

// Simple thread-safe object pool for DLManagedTensor
struct DLManagedTensorPool {
    std::vector<DLManagedTensor*> pool;
    std::mutex mutex;
    
    ~DLManagedTensorPool() {
        for (auto* p : pool) delete p;
    }
    
    DLManagedTensor* allocate() {
        std::lock_guard<std::mutex> lock(mutex);
        if (pool.empty()) {
            return new DLManagedTensor();
        }
        DLManagedTensor* p = pool.back();
        pool.pop_back();
        return p;
    }
    
    void deallocate(DLManagedTensor* p) {
        std::lock_guard<std::mutex> lock(mutex);
        pool.push_back(p);
    }
};

static DLManagedTensorPool global_dlpack_pool;

// DLPack Deleter (C-compatible)
static void dlpack_deleter(DLManagedTensor* tensor) {
    if (tensor->manager_ctx) {
        // Decrement refcount of the tensorplay Tensor
        delete static_cast<Tensor*>(tensor->manager_ctx);
    }
    // Return to pool instead of delete
    global_dlpack_pool.deallocate(tensor);
}

// Optimized deleter for PyObject-managed DLPack
static void dlpack_pyobject_deleter(DLManagedTensor* managed) {
    if (managed->manager_ctx) {
        py::gil_scoped_acquire gil;
        Py_DECREF(static_cast<PyObject*>(managed->manager_ctx));
    }
    // Return to pool instead of delete
    managed->manager_ctx = nullptr; // Clear ctx
    global_dlpack_pool.deallocate(managed);
}

// Capsule Destructor
static void dlpack_capsule_destructor(PyObject* cap) {
    // If the capsule is still named "dltensor", it means it wasn't consumed.
    // We must clean up the DLManagedTensor.
    const char* name = PyCapsule_GetName(cap);
    if (name && strcmp(name, "dltensor") == 0) {
        DLManagedTensor* managed = (DLManagedTensor*)PyCapsule_GetPointer(cap, "dltensor");
        if (managed) {
            managed->deleter(managed);
        }
    }
}

static py::capsule to_dlpack(py::object self_obj, std::optional<int64_t> stream = std::nullopt) {
    const Tensor& self = py::cast<const Tensor&>(self_obj);
    // Use pool
    DLManagedTensor* managed = global_dlpack_pool.allocate();
    
    // Optimization: Keep the Python object alive instead of copying the C++ Tensor.
    // This avoids one heap allocation (new Tensor) and one copy constructor.
    PyObject* ptr = self_obj.ptr();
    Py_INCREF(ptr);
    managed->manager_ctx = ptr;
    managed->deleter = dlpack_pyobject_deleter;
    
    DLTensor& dl = managed->dl_tensor;
    dl.data = self.data_ptr();
    dl.byte_offset = 0;
    dl.ndim = static_cast<int>(self.dim());
    
    // We need persistent pointers for shape and strides.
    // The Python object owns the C++ Tensor, which owns the TensorImpl, which owns the vectors.
    // So as long as self_obj is alive, these pointers are valid.
    auto impl = self.unsafeGetTensorImpl();
    
    dl.shape = const_cast<int64_t*>(impl->sizes().data());
    dl.strides = const_cast<int64_t*>(impl->strides().data());
    
    dl.dtype = to_dlpack_dtype(self.dtype());
    dl.device = to_dlpack_device(self.device());
    
    PyObject* cap = PyCapsule_New(managed, "dltensor", dlpack_capsule_destructor);
    return py::reinterpret_steal<py::capsule>(cap);
}

// Capture-less deleter for DataPtr contexts that own a PyObject reference
// (zero-copy NumPy wraps, shared-memory handles). Acquires the GIL because
// it may run on any thread during teardown.
static void pyobject_deleter(void* ctx) {
    if (!ctx) return;
    py::gil_scoped_acquire gil;
    Py_DECREF(static_cast<PyObject*>(ctx));
}

// Capture-less deleter for DLPack tensors: forwards to the tensor's own
// deleter, which releases the manager context (often an owned PyObject).
static void dlpack_managed_deleter(void* ctx) {
    auto* managed = static_cast<DLManagedTensor*>(ctx);
    if (managed && managed->deleter) {
        managed->deleter(managed);
    }
}

static Tensor from_dlpack(py::object o) {
    PyObject* cap_ptr;
    bool is_capsule = PyCapsule_CheckExact(o.ptr());
    py::capsule cap_holder; // Holds reference if we created a new capsule
    
    if (is_capsule) {
        cap_ptr = o.ptr();
    } else {
        // Optimization: Use C-API to call __dlpack__ directly.
        // faster than py::hasattr + o.attr()()
        static PyObject* dlpack_str = PyUnicode_InternFromString("__dlpack__");
        PyObject* res = PyObject_CallMethodObjArgs(o.ptr(), dlpack_str, nullptr);
        if (!res) {
             PyErr_Clear(); // Clear AttributeError
             TP_THROW(TypeError, "Object is not a DLPack capsule and does not have __dlpack__ method");
        }
        cap_holder = py::reinterpret_steal<py::capsule>(res);
        cap_ptr = res;
    }
    
    // Check name
    const char* name = PyCapsule_GetName(cap_ptr);
    if (strcmp(name, "dltensor") != 0) {
        TP_THROW(ValueError, "DLPack capsule is invalid or already consumed");
    }
    
    DLManagedTensor* managed = (DLManagedTensor*)PyCapsule_GetPointer(cap_ptr, "dltensor");
    if (!managed) TP_THROW(ValueError, "Invalid DLPack capsule pointer");
    
    // Rename to mark as consumed
    PyCapsule_SetName(cap_ptr, "used_dltensor");
    
    // Extract metadata
    DLTensor& dl = managed->dl_tensor;
    DType dtype = from_dlpack_dtype(dl.dtype);
    Device device = from_dlpack_device(dl.device);
    
    std::vector<int64_t> shape(dl.shape, dl.shape + dl.ndim);
    std::vector<int64_t> strides;
    if (dl.strides) {
        strides.assign(dl.strides, dl.strides + dl.ndim);
    } else {
        // Assume contiguous
        strides.resize(dl.ndim);
        int64_t stride = 1;
        for (int i = dl.ndim - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }
    
    // Create DataPtr with custom deleter: state (the DLManagedTensor*)
    // travels through the DataPtr context so the deleter itself is a
    // capture-less function pointer.

    // Calculate total bytes roughly for storage info (optional but good for tracking)
    size_t nbytes = 1;
    for(auto s : shape) nbytes *= s;
    nbytes *= (dl.dtype.bits / 8);
    
    // Data pointer
    void* data_ptr_raw = static_cast<char*>(dl.data) + dl.byte_offset;
    
    tensorplay::DataPtr ptr(data_ptr_raw, managed, &dlpack_managed_deleter, device);
    tensorplay::Storage storage(std::move(ptr), nbytes); // Using wrapper constructor
    
    // Create Tensor directly with strides (Optimization: Avoid as_strided overhead)
    auto impl = std::make_shared<tensorplay::TensorImpl>(storage, shape, strides, dtype);
    return Tensor(impl);
}

// Helper function implementation
Tensor create_tensor(py::object data, std::optional<DType> dtype, std::optional<Device> device) {
    Tensor t;
    
    // 0. Fast Path: Check for NumPy array directly using type comparison
    // This is faster than strcmp and avoids string operations
    static PyTypeObject* numpy_array_type = []() -> PyTypeObject* {
        PyObject* np = PyImport_ImportModule("numpy");
        if (!np) { PyErr_Clear(); return nullptr; }
        PyObject* type = PyObject_GetAttrString(np, "ndarray");
        Py_DECREF(np);
        return (PyTypeObject*)type; // Leak reference intentionally to keep type alive
    }();

    if (numpy_array_type && Py_TYPE(data.ptr()) == numpy_array_type) {
         // Zero-copy wrap of the NumPy array via the buffer protocol
         py::array array = py::array::ensure(data);
         
         size_t ndim = array.ndim();
         std::vector<int64_t> shape(ndim);
         for (size_t i = 0; i < ndim; ++i) {
             shape[i] = array.shape(i);
         }
         
         // Map NumPy dtype (kind + itemsize) to TensorPlay DType
         py::dtype ndt = array.dtype();
         char kind = ndt.kind();
         size_t bits = ndt.itemsize() * 8;
         DType inferred_dtype = DType::Undefined;
         
         if (kind == 'f' && bits == 16) inferred_dtype = DType::Float16;
         else if (kind == 'f' && bits == 32) inferred_dtype = DType::Float32;
         else if (kind == 'f' && bits == 64) inferred_dtype = DType::Float64;
         else if (kind == 'c' && bits == 64) inferred_dtype = DType::ComplexFloat;
         else if (kind == 'c' && bits == 128) inferred_dtype = DType::ComplexDouble;
         else if (kind == 'i' && bits == 8) inferred_dtype = DType::Int8;
         else if (kind == 'i' && bits == 16) inferred_dtype = DType::Int16;
         else if (kind == 'i' && bits == 32) inferred_dtype = DType::Int32;
         else if (kind == 'i' && bits == 64) inferred_dtype = DType::Int64;
         else if (kind == 'u' && bits == 8) inferred_dtype = DType::UInt8;
         else if (kind == 'u' && bits == 16) inferred_dtype = DType::UInt16;
         else if (kind == 'u' && bits == 32) inferred_dtype = DType::UInt32;
         else if (kind == 'u' && bits == 64) inferred_dtype = DType::UInt64;
         else if (kind == 'b') inferred_dtype = DType::Bool;
         else {
             TP_THROW(TypeError, "Unsupported NumPy dtype for Tensor creation");
         }
         
         DType final_dtype = dtype.value_or(inferred_dtype);
         
         // Calculate element-wise strides for TensorPlay
         // (NumPy strides are in bytes, TensorPlay strides are in elements)
         size_t itemsize = bits / 8;
         std::vector<int64_t> strides;
         for (size_t i = 0; i < ndim; ++i) {
             strides.push_back(array.strides(i) / (int64_t)itemsize);
         }
         
         // Zero-Copy Path conditions:
         // 1. dtypes match
         // 2. target device is CPU (since numpy is on CPU)
         // 3. no explicit device move requested to non-CPU
         bool is_cpu = !device.has_value() || device->type() == DeviceType::CPU;
         
         if (final_dtype == inferred_dtype && is_cpu) {
             // ZERO-COPY IMPLEMENTATION
             // We use the raw pointer from numpy and keep the numpy object alive via deleter
             
             PyObject* py_obj = data.ptr();
             // Increment refcount to keep numpy array alive; the raw PyObject*
             // travels as the DataPtr context.
             Py_INCREF(py_obj);

             // Calculate total size for info
             size_t numel = 1;
             for(auto s : shape) numel *= s;
             size_t nbytes = numel * itemsize;

             tensorplay::DataPtr ptr(array.mutable_data(), py_obj,
                                     &pyobject_deleter, Device(DeviceType::CPU));
             tensorplay::Storage storage(std::move(ptr), nbytes);
             
             // Create Tensor with specific strides directly (Optimization: Avoid as_strided overhead)
            auto impl = std::make_shared<tensorplay::TensorImpl>(storage, shape, strides, final_dtype);
            t = Tensor(impl);
            
        } else {
             // Copy Path (Casting or Device Move)
             // Use Tensor directly for intermediate
             Tensor p10_t(shape, final_dtype, device.value_or(Device(DeviceType::CPU)));
             t = Tensor(p10_t);
             
             size_t numel = 1;
             for(auto s : shape) numel *= s;
             size_t total_bytes = numel * itemsize; 
             
             if (final_dtype == inferred_dtype) {
                 std::memcpy(t.data_ptr(), array.data(), total_bytes);
             } else {
                 Tensor src(shape, inferred_dtype, Device(DeviceType::CPU));
                 std::memcpy(src.data_ptr(), array.data(), total_bytes);
                 t.copy_(src);
             }
         }
    }
    // 1. Check for DLPack support (fast path for interop other than NumPy)
    // Use C-API with interned string for maximum speed
    else if ([](PyObject* ptr) {
        static PyObject* dlpack_attr_name = PyUnicode_InternFromString("__dlpack__");
        return PyObject_HasAttr(ptr, dlpack_attr_name);
    }(data.ptr())) {
        t = from_dlpack(data);
    } else if (py::isinstance<py::list>(data) || py::isinstance<py::tuple>(data)) {
        t = list_to_tensor(data.ptr(), dtype, device);
    } else if (py::isinstance<py::int_>(data) || py::isinstance<py::float_>(data) ||
               py::isinstance<py::bool_>(data) || PyComplex_Check(data.ptr())) {
          py::list l;
          l.append(data);
         t = list_to_tensor(l.ptr(), dtype, device);
         t = t.reshape({});
    } else {
         if (data.is_none()) {
            // Return undefined tensor (default constructor)
            return Tensor();
        }
        try {
             py::list l(data);
             t = list_to_tensor(l.ptr(), dtype, device);
         } catch (...) {
             TP_THROW(TypeError, "Unsupported data type for Tensor creation");
         }
    }
    
    // Handle dtype conversion if needed
    if (dtype.has_value() && t.dtype() != *dtype) {
        Tensor new_t(static_cast<std::vector<int64_t>>(t.shape()), *dtype, device.value_or(Device(DeviceType::CPU)));
        Tensor new_t_wrapper(new_t);
        convert_tensor_data(t, new_t_wrapper);
        t = new_t_wrapper;
    }
    
    // Handle device movement if needed
    // Note: list_to_tensor returns CPU tensor.
    if (device.has_value() && t.device() != *device) {
        Tensor new_t(static_cast<std::vector<int64_t>>(t.shape()), t.dtype(), *device);
        new_t.copy_(t);
        t = Tensor(new_t);
    }

    return t;
}

static void set_storage_from_shm(Tensor& self, py::object shm, size_t nbytes) {
    py::object buf_obj = shm.attr("buf");
    
    Py_buffer view;
    if (PyObject_GetBuffer(buf_obj.ptr(), &view, PyBUF_SIMPLE) != 0) {
        throw py::error_already_set();
    }
    
    void* shm_ptr = view.buf;
    // We release the buffer view because SharedMemory keeps the buffer alive
    PyBuffer_Release(&view);
    
    // Copy data to shared memory
    // Note: self.data_ptr() might be on different device, but here we only support CPU sharing
    std::memcpy(shm_ptr, self.data_ptr(), nbytes);
    
    // Own a reference on the shm object; it travels as the DataPtr context
    // and is released by pyobject_deleter.
    PyObject* shm_owned = shm.ptr();
    Py_INCREF(shm_owned);

    // Create DataPtr with shared memory pointer
    tensorplay::DataPtr data_ptr(shm_ptr, shm_owned, &pyobject_deleter, Device(DeviceType::CPU));

    // Create Storage
    tensorplay::Storage new_storage(std::move(data_ptr), nbytes, nullptr);
    
    // Replace Tensor storage
    self.unsafeGetTensorImpl()->set_storage(new_storage);
}


// Normalize a Python slice against a length (mirrors nanobind's slice::compute)
static std::tuple<int64_t, int64_t, int64_t, int64_t> compute_slice(py::slice s, int64_t length) {
    ssize_t start, stop, step, slicelength;
    if (!s.compute((ssize_t)length, &start, &stop, &step, &slicelength)) {
        throw py::error_already_set();
    }
    return {start, stop, step, slicelength};
}

static std::pair<Tensor, py::dict> setstate_helper(py::tuple state) {
    // Check for shared memory tag
    if (state.size() == 8) {
        // Try-catch block not needed for simple cast, but safe
        bool is_shm = false;
        try {
            std::string tag = py::cast<std::string>(state[0]);
            if (tag == "shm") is_shm = true;
        } catch (...) {}
        
        if (is_shm) {
            py::object shm = py::cast<py::object>(state[1]);
            std::vector<int64_t> shape = py::cast<std::vector<int64_t>>(state[2]);
            std::vector<int64_t> strides = py::cast<std::vector<int64_t>>(state[3]);
            DType dtype = (DType)py::cast<int>(state[4]);
            DeviceType device_type = (DeviceType)py::cast<int>(state[5]);
            int device_index = py::cast<int>(state[6]);
            bool requires_grad = py::cast<bool>(state[7]);
            
            // SharedMemory object is already attached/opened by pickle
            
            size_t nbytes = 0;
            {
                Device device(device_type, device_index);
                // Use contiguous constructor to get itemsize/numel info safely
                Tensor temp_t(shape, dtype, device); 
                nbytes = temp_t.numel() * temp_t.itemsize();
            }
            
            py::object buf_obj = shm.attr("buf");
            Py_buffer view;
            if (PyObject_GetBuffer(buf_obj.ptr(), &view, PyBUF_SIMPLE) != 0) throw py::error_already_set();
            void* shm_ptr = view.buf;
            PyBuffer_Release(&view);
            
            PyObject* shm_owned = shm.ptr();
            Py_INCREF(shm_owned);

            tensorplay::DataPtr data_ptr(shm_ptr, shm_owned, &pyobject_deleter, Device(DeviceType::CPU));
            tensorplay::Storage new_storage(std::move(data_ptr), nbytes, nullptr);
            
            Tensor final_p10(new_storage, shape, strides, dtype);
            
            Tensor t(final_p10);
            tensorplay::tpx::impl::set_requires_grad(t, requires_grad);
            
            py::dict extra;
            extra["_shared_memory"] = shm;
            return {std::move(t), std::move(extra)};
        }
    }
    
    if (state.size() != 7) {
            throw std::runtime_error("Invalid state for Tensor unpickling");
    }
    
    py::bytes data_bytes = py::cast<py::bytes>(state[0]);
    std::vector<int64_t> shape = py::cast<std::vector<int64_t>>(state[1]);
    std::vector<int64_t> strides = py::cast<std::vector<int64_t>>(state[2]);
    DType dtype = (DType)py::cast<int>(state[3]);
    DeviceType device_type = (DeviceType)py::cast<int>(state[4]);
    int device_index = py::cast<int>(state[5]);
    bool requires_grad = py::cast<bool>(state[6]);
    
    Device device(device_type, device_index);
    
    Tensor p10_t(shape, dtype, device);
    
    size_t nbytes = p10_t.numel() * p10_t.itemsize();
    if (py::len(data_bytes) != nbytes) {
            throw std::runtime_error("Tensor pickle data size mismatch");
    }
    
    std::memcpy(p10_t.data_ptr(), PyBytes_AsString(data_bytes.ptr()), nbytes);
    
    Tensor t(p10_t);
    tensorplay::tpx::impl::set_requires_grad(t, requires_grad);
    
    return {std::move(t), py::dict()};
}

// Optimized Vision to Tensor: HWC(uint8) -> CHW(float32) / 255.0
Tensor vision_to_tensor(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img) {
    size_t H = img.shape(0);
    size_t W = img.shape(1);
    size_t C = img.shape(2);
    
    // Output: C, H, W (float32)
    size_t numel = H * W * C;
    float* data = new float[numel];
    
    const uint8_t* in_ptr = img.data();
    float* out_ptr = data;
    
    size_t stride_c = H * W;
    
    // Optimized loop: HWC -> CHW
    // Cache friendly? Not really for writes (strided writes), but sequential reads.
    // Given typical image sizes, this is faster than Python overhead + multiple passes.
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            size_t in_offset = (y * W + x) * C;
            size_t out_offset_base = y * W + x;
            
            for (size_t c = 0; c < C; ++c) {
                out_ptr[c * stride_c + out_offset_base] = static_cast<float>(in_ptr[in_offset + c]) * (1.0f / 255.0f);
            }
        }
    }
    
    auto deleter = [](void* p) { delete[] static_cast<float*>(p); };
    tensorplay::DataPtr ptr(data, deleter, Device(DeviceType::CPU, 0));
    tensorplay::Storage storage(std::move(ptr), numel * sizeof(float));
    
    std::vector<int64_t> out_shape = {static_cast<int64_t>(C), static_cast<int64_t>(H), static_cast<int64_t>(W)};
    std::vector<int64_t> out_strides = {static_cast<int64_t>(stride_c), static_cast<int64_t>(W), 1};
    
    auto impl = std::make_shared<tensorplay::TensorImpl>(storage, out_shape, out_strides, DType::Float32);
    return Tensor(impl);
}

// Optimized Audio to Tensor:
// 1. Transpose: (Time, Channels) -> (Channels, Time)
// 2. Normalize (if int16): x / 32768.0
Tensor audio_to_tensor(py::object obj) {
    // We expect a numpy array
    py::array array = py::array::ensure(obj);
    if (!array) {
        TP_THROW(TypeError, "audio_to_tensor: expected a numpy array");
    }
    
    // Check dimensions
    size_t ndim = array.ndim();
    if (ndim != 1 && ndim != 2) {
        TP_THROW(RuntimeError, "audio_to_tensor: input must be 1D or 2D array");
    }

    size_t time_steps = array.shape(0);
    size_t channels = (ndim == 2) ? array.shape(1) : 1;
    
    // Output shape: (Channels, Time)
    std::vector<int64_t> out_shape = {static_cast<int64_t>(channels), static_cast<int64_t>(time_steps)};
    std::vector<int64_t> out_strides = {static_cast<int64_t>(time_steps), 1}; // Contiguous CHW (here C, T)
    
    size_t numel = channels * time_steps;
    float* data = new float[numel];
    
    py::dtype dt = array.dtype();
    char kind = dt.kind();
    size_t bits = dt.itemsize() * 8;
    
    // Dispatch based on input type
    // int16 -> normalize
    if (kind == 'i' && bits == 16) {
        const int16_t* in_ptr = static_cast<const int16_t*>(array.data());
        float* out_ptr = data;
        
        // Parallelize if large enough? For now sequential is likely fast enough compared to python
        if (ndim == 1) {
            for (size_t t = 0; t < time_steps; ++t) {
                out_ptr[t] = static_cast<float>(in_ptr[t]) * (1.0f / 32768.0f);
            }
        } else {
            // Transpose loop: (T, C) -> (C, T)
            // Input stride: C (usually)
            // Output stride: T
            
            // Access: in[t * C + c] -> out[c * T + t]
            for (size_t t = 0; t < time_steps; ++t) {
                for (size_t c = 0; c < channels; ++c) {
                    float val = static_cast<float>(in_ptr[t * channels + c]) * (1.0f / 32768.0f);
                    out_ptr[c * time_steps + t] = val;
                }
            }
        }
    } 
    // float32 -> copy transpose
    else if (kind == 'f' && bits == 32) {
        const float* in_ptr = static_cast<const float*>(array.data());
        float* out_ptr = data;
        
        if (ndim == 1) {
            std::memcpy(out_ptr, in_ptr, numel * sizeof(float));
        } else {
            for (size_t t = 0; t < time_steps; ++t) {
                for (size_t c = 0; c < channels; ++c) {
                    out_ptr[c * time_steps + t] = in_ptr[t * channels + c];
                }
            }
        }
    }
    // int32 -> normalize (rare but possible, div 2147483648.0)
    else if (kind == 'i' && bits == 32) {
        const int32_t* in_ptr = static_cast<const int32_t*>(array.data());
        float* out_ptr = data;
        
        if (ndim == 1) {
            for (size_t t = 0; t < time_steps; ++t) {
                 out_ptr[t] = static_cast<float>(in_ptr[t]) * (1.0f / 2147483648.0f);
            }
        } else {
            for (size_t t = 0; t < time_steps; ++t) {
                for (size_t c = 0; c < channels; ++c) {
                    float val = static_cast<float>(in_ptr[t * channels + c]) * (1.0f / 2147483648.0f);
                    out_ptr[c * time_steps + t] = val;
                }
            }
        }
    }
    // uint8 -> normalize (0-255 -> -1, 1? usually audio is signed. If unsigned 8bit, usually 0-255 map to -1..1. (x-128)/128.0)
    else if (kind == 'u' && bits == 8) {
         const uint8_t* in_ptr = static_cast<const uint8_t*>(array.data());
         float* out_ptr = data;
         
         if (ndim == 1) {
             for (size_t t = 0; t < time_steps; ++t) {
                 out_ptr[t] = (static_cast<float>(in_ptr[t]) - 128.0f) * (1.0f / 128.0f);
             }
         } else {
             for (size_t t = 0; t < time_steps; ++t) {
                 for (size_t c = 0; c < channels; ++c) {
                     float val = (static_cast<float>(in_ptr[t * channels + c]) - 128.0f) * (1.0f / 128.0f);
                     out_ptr[c * time_steps + t] = val;
                 }
             }
         }
    }
    else {
        delete[] data;
        TP_THROW(TypeError, "audio_to_tensor: unsupported input dtype. Expected int16, int32, uint8 or float32.");
    }

    auto deleter = [](void* p) { delete[] static_cast<float*>(p); };
    tensorplay::DataPtr ptr(data, deleter, Device(DeviceType::CPU, 0));
    tensorplay::Storage storage(std::move(ptr), numel * sizeof(float));
    
    auto impl = std::make_shared<tensorplay::TensorImpl>(storage, out_shape, out_strides, DType::Float32);
    return Tensor(impl);
}

py::object as_tensor(py::object data, std::optional<DType> dtype, std::optional<Device> device) {
    if (py::isinstance<Tensor>(data)) {
        Tensor t = py::cast<Tensor>(data);
        
        DType target_dtype = dtype.has_value() ? *dtype : t.dtype();
        Device target_device = device.has_value() ? *device : t.device();
        
        if (t.dtype() == target_dtype && t.device() == target_device) {
            return data;
        }
        
        // Use .to() logic if possible, but here we can't easily call .to() of the object unless we cast to object
        // Calling python .to() is easiest to ensure consistency
        py::dict kwargs;
        if (dtype.has_value()) kwargs["dtype"] = *dtype;
        if (device.has_value()) kwargs["device"] = *device;
        
        return data.attr("to")(**kwargs);
    }
    
    return py::cast(create_tensor(data, dtype, device));
}

void init_tensor(py::module_& m) {
    // from_numpy — direct port of torch/csrc/utils/tensor_numpy.cpp
    // tensor_from_numpy(): zero-copy from_blob view; non-writable arrays warn
    // once instead of failing; byte-stride divisibility, negative strides and
    // foreign byte order are rejected with torch's exact messages.
    static bool warned_numpy_not_writeable = false;
    m.def("from_numpy", [](py::object obj) -> Tensor {
        py::array array = py::array::ensure(obj);
        if (!array) {
            TP_THROW(TypeError, "expected np.ndarray");
        }

        const int64_t ndim = array.ndim();
        std::vector<int64_t> sizes(ndim), strides(ndim);
        for (int64_t i = 0; i < ndim; ++i) {
            sizes[i] = array.shape(i);
            strides[i] = array.strides(i);
        }
        const int64_t element_size_in_bytes = (std::max<py::ssize_t>(array.itemsize(), 1));
        if (!(element_size_in_bytes > 0)) {
            TP_THROW(ValueError, "element_size must be 0");
        }
        for (auto& stride : strides) {
            if (stride % element_size_in_bytes != 0) {
                TP_THROW(ValueError,
                         "given numpy array strides not a multiple of the element byte size. "
                         "Copy the numpy array to reallocate the memory.");
            }
            stride /= element_size_in_bytes;
        }
        for (const auto& stride : strides) {
            if (stride < 0) {
                TP_THROW(ValueError,
                         "At least one stride in the given numpy array is negative, "
                         "and tensors with negative strides are not currently supported. "
                         "(You can probably work around this by making a copy of your array "
                         " with array.copy().) ");
            }
        }
        // Byte-order check (tensor_numpy.cpp): only native-order arrays are
        // accepted, matching ATen's PyArray_EquivByteorders gate.
        {
            static char native_order = []() {
                int one = 1;
                return (*reinterpret_cast<char*>(&one) == 1) ? '<' : '>';
            }();
            py::dtype dt = array.dtype();
            py::object bo = py::reinterpret_steal<py::object>(
                PyObject_GetAttrString(dt.ptr(), "byteorder"));
            if (bo.ptr() != nullptr) {
                char c = bo.cast<char>();
                if (c != '=' && c != '|' && c != native_order) {
                    TP_THROW(ValueError,
                             "given numpy array has byte order different from the native byte order. "
                             "Conversion between byte orders is currently not supported.");
                }
            }
        }

        if (!array.writeable() && !warned_numpy_not_writeable) {
            PyErr_WarnEx(PyExc_UserWarning,
                         "The given NumPy array is not writable, and TensorPlay does "
                         "not support non-writable tensors. This means writing to this tensor "
                         "would result in undefined behavior. You may want to copy the array "
                         "to protect its data or make it writable before converting it to a "
                         "tensor. This type of warning will be suppressed for the rest of this "
                         "program.", 1);
            warned_numpy_not_writeable = true;
        }

        // Route through create_tensor's numpy branch: dtype mapping + zero-copy
        // DataPtr (keeps the array alive) are shared with tp.tensor().
        // create_tensor uses mutable_data(); for read-only arrays fall back to
        // const data cast — writes are UB exactly as torch documents above.
        Tensor t = create_tensor(std::move(obj), std::nullopt, std::nullopt);
        return t;
    }, "ndarray"_a);

    // frombuffer — direct port of torch/csrc/utils/tensor_new.cpp
    // tensor_frombuffer(): buffer-protocol view, zero-copy, writable preferred
    // with the same non-writable warn-once fallback and value checks.
    static bool warned_non_writable = false;
    m.def("frombuffer", [warned_non_writable](py::object buffer, DType dtype,
                                              int64_t count, int64_t offset,
                                              bool requires_grad) mutable -> Tensor {
        const size_t elsize = tensorplay::elementSize(dtype);
        Py_buffer view;
        if (PyObject_GetBuffer(buffer.ptr(), &view, PyBUF_WRITABLE) < 0) {
            if (PyObject_GetBuffer(buffer.ptr(), &view, PyBUF_SIMPLE) < 0) {
                TP_THROW(ValueError, "could not retrieve buffer from object");
            }
            if (!warned_non_writable) {
                PyErr_WarnEx(PyExc_UserWarning,
                             "The given buffer is not writable, and TensorPlay does "
                             "not support non-writable tensors. This means you can write to the "
                             "underlying (supposedly non-writable) buffer using the tensor. "
                             "You may want to copy the buffer to protect its data or make it writable "
                             "before converting it to a tensor. This type of warning will be "
                             "suppressed for the rest of this program.", 1);
                warned_non_writable = true;
            }
            PyErr_Clear();
        }

        PyObject* view_obj = view.obj;
        Py_INCREF(view_obj);

        const int64_t len = static_cast<int64_t>(view.len);
        void* buf = view.buf;
        PyBuffer_Release(&view);
        (void)buf;

        if (!(len > 0 && count != 0)) {
            Py_DECREF(view_obj);
            TP_THROW(ValueError, "both buffer length and count must not be 0");
        }
        if (!(offset >= 0 && offset < len)) {
            Py_DECREF(view_obj);
            TP_THROW(ValueError, "offset must be non-negative and no greater than buffer length minus 1");
        }
        if (!(count > 0 || (len - offset) % static_cast<int64_t>(elsize) == 0)) {
            Py_DECREF(view_obj);
            TP_THROW(ValueError, "buffer length after offset must be a multiple of element size");
        }

        size_t actual_count;
        if (count < 0) {
            actual_count = static_cast<size_t>((len - offset) / static_cast<int64_t>(elsize));
        } else {
            actual_count = static_cast<size_t>(count);
        }
        if (static_cast<size_t>(offset) + actual_count * elsize > static_cast<size_t>(len)) {
            Py_DECREF(view_obj);
            TP_THROW(ValueError, "requested buffer length must not be greater than actual buffer length");
        }

        auto* offset_buf = static_cast<char*>(buf) + offset;

        // Zero-copy: the DataPtr keeps the buffer's owner alive via DECREF.
        tensorplay::DataPtr ptr(offset_buf, view_obj, &pyobject_deleter, Device(DeviceType::CPU));
        tensorplay::Storage storage(std::move(ptr), actual_count * elsize);

        std::vector<int64_t> shape{static_cast<int64_t>(actual_count)};
        std::vector<int64_t> strides{1};
        auto impl = std::make_shared<tensorplay::TensorImpl>(storage, shape, strides, dtype);
        Tensor t(impl);
        tensorplay::tpx::impl::set_requires_grad(t, requires_grad);
        return t;
    }, "buffer"_a, "dtype"_a = DType::Float32, "count"_a = -1, "offset"_a = 0,
       "requires_grad"_a = false);

    // Expose from_dlpack as a module function
    m.def("from_dlpack", &from_dlpack, "obj"_a);
    m.def("to_dlpack", &to_dlpack, "obj"_a, "stream"_a = py::none());

    // from_numpy: zero-copy view over the numpy array's memory when dtypes
    // match (same contract as ATen torch.from_numpy; the array is kept alive
    // by the DataPtr deleter, and negative strides are rejected like torch).
    m.def("from_numpy", [](py::array array) -> Tensor {
        if (array.ndim() > 0) {
            for (size_t i = 0; i < (size_t)array.ndim(); ++i) {
                if (array.strides(i) < 0) {
                    TP_THROW(ValueError,
                             "from_numpy: negative strides are not supported; "
                             "use np.ascontiguousarray(arr)");
                }
            }
        }
        return create_tensor(py::object(std::move(array)), std::nullopt, std::nullopt);
    }, "array"_a);
    
    // Expose as_tensor
    m.def("as_tensor", &as_tensor, "data"_a, "dtype"_a = py::none(), "device"_a = py::none(),
          "Converts data into a tensor, sharing data and preserving autograd history if possible.");

    // Expose vision optimization
    m.def("vision_to_tensor", &vision_to_tensor, "image"_a, "Optimized conversion from HWC uint8 image to CHW float32 tensor (div 255)");

    // Expose audio optimization
    m.def("audio_to_tensor", &audio_to_tensor, "audio"_a, "Optimized conversion for audio: (Time, Channels) -> (Channels, Time) with normalization");

    py::class_<Tensor> tensor(m, "TensorBase", py::dynamic_attr());
    tensor.attr("__module__") = "tensorplay._C";
    
    tensor
        .def(py::init<>())
        // Constructor from data (torch.tensor equivalent)
        .def("__init__", [](Tensor* self, py::object data, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
            Tensor t = create_tensor(data, dtype, device);
            tensorplay::tpx::impl::set_requires_grad(t, requires_grad);
            new (self) Tensor(std::move(t));
        }, "data"_a, "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a = false)
        
        // Properties
        .def_property_readonly("_impl_id", [](const Tensor& self) {
             return (uintptr_t)self.unsafeGetTensorImpl().get();
        })
        .def_property_readonly("shape", &Tensor::shape)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("device", &Tensor::device)
        .def_property_readonly("ndim", &Tensor::dim)
        .def("dim", &Tensor::dim)
        .def("numel", &Tensor::numel)
        .def("itemsize", &Tensor::itemsize)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("is_complex", [](const Tensor& t) { 
             return tensorplay::isComplexType(t.dtype());
        })
        .def("is_floating_point", [](const Tensor& t) {
             return tensorplay::isFloatingType(t.dtype());
        })
        .def("t", [](const Tensor& t) {
            // Simple transpose for 2D. For ND we need generic transpose.
            if (t.dim() != 2) throw std::runtime_error("t() expects a 2D tensor, but self is " + std::to_string(t.dim()) + "D");
            // Generic transpose: swap strides and shape
            std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(t.shape());
            std::vector<int64_t> strides = t.strides();
            std::swap(sizes[0], sizes[1]);
            std::swap(strides[0], strides[1]);
            return t.as_strided(sizes, strides);
        })
        .def_property_readonly("is_sparse", &Tensor::is_sparse)
        .def("is_coalesced", &Tensor::is_coalesced)
        .def("sparse_dim", &Tensor::sparse_dim)
        .def("dense_dim", &Tensor::dense_dim)
        .def("_indices", &Tensor::_indices)
        .def("_values", &Tensor::_values)
        .def("coalesce", &Tensor::coalesce)
        .def("sparse_mask",
             static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::sparse_mask),
             "mask"_a)
        .def_property_readonly("strides", [](const Tensor& self) {
            return py::tuple(py::cast(self.strides()));
        })
        .def("stride", [](const Tensor& self) {
            return py::tuple(py::cast(self.strides()));
        })
        .def("stride", [](const Tensor& self, int64_t dim) {
            return self.stride(dim);
        })
        .def_property("requires_grad", &Tensor::requires_grad, [](Tensor& self, bool r) {
            tensorplay::tpx::impl::set_requires_grad(self, r);
        })
        .def_property_readonly("_version", [](const Tensor& self) {
            return self.unsafeGetTensorImpl()->version();
        })
        .def_property_readonly("is_leaf", [](const Tensor& self) { return tensorplay::tpx::impl::is_leaf(self); })
        .def_property_readonly("retains_grad", &Tensor::retains_grad)
        .def_property_readonly("grad_fn", [](const Tensor& self) { return tensorplay::tpx::impl::grad_fn(self); })
        .def("_set_grad_fn", [](Tensor& self, std::shared_ptr<tensorplay::tpx::Node> node, int output_nr) {
            tensorplay::tpx::impl::set_grad_fn(self, std::move(node), output_nr);
        }, "node"_a, "output_nr"_a = 0)
        .def_property_readonly("_output_nr", [](const Tensor& self) {
            return tensorplay::tpx::impl::output_nr(self);
        })
        .def_property_readonly("_accumulate_grad_node", [](const Tensor& self) -> std::shared_ptr<tensorplay::tpx::Node> {
            return tensorplay::tpx::impl::grad_accumulator(self);
        })
        .def_property_readonly("is_cuda", [](const Tensor& self) { return self.device().type() == DeviceType::CUDA; })
        .def("pin_memory", [](const Tensor& self) {
             Tensor result(self.pin_memory());
             tensorplay::tpx::impl::set_requires_grad(result, self.requires_grad());
             return result;
        })
        .def("is_pinned", [](const Tensor& self) {
             return self.is_pinned();
        })
#ifdef USE_CUDA
        .def("record_stream", [](const Tensor& self, py::object stream_object) {
             // Accept both tensorplay.cuda.Stream and the underlying
             // tensorplay._C._cuda._CudaStream, as PyTorch does for its
             // public Tensor.record_stream API.
             py::object core_stream = py::hasattr(stream_object, "_stream")
                 ? stream_object.attr("_stream")
                 : stream_object;
             const auto& stream = core_stream.cast<const tensorplay::cuda::CUDAStream&>();
             if (!self.device().is_cuda()) {
                 TP_THROW(RuntimeError, "record_stream expects a CUDA tensor");
             }
             if (self.device().index() != stream.device_index()) {
                 TP_THROW(DeviceMismatchError,
                          "tensor is on " + self.device().toString() +
                          " but stream is on " + stream.device().toString());
             }
             auto impl = self.unsafeGetTensorImpl();
             tensorplay::cuda::recordStream(impl->storage().data(), stream);
        }, "stream"_a)
#endif
        .def_property("grad", 
            [](const Tensor& self) -> std::optional<Tensor> {
                Tensor g = self.grad();
                if (g.defined()) return g;
                return std::nullopt;
            },
            [](Tensor& self, const Tensor* grad) {
                if (grad) {
                    self.set_grad(*grad);
                } else {
                    self.set_grad(Tensor());
                }
            }
        )
        .def("retain_grad", [](Tensor& self) { tensorplay::tpx::impl::retain_grad(self); })
        .def("backward", [](Tensor& self, std::optional<Tensor> gradient, std::optional<bool> retain_graph, bool create_graph) {
             bool keep_graph = retain_graph.value_or(create_graph);
             // The engine may run Python-backed nodes on worker threads that
             // need the GIL; the initiating thread must not hold it while it
             // waits for the graph to drain.
             py::gil_scoped_release release;
             if (gradient) {
                 tensorplay::tpx::backward(self, *gradient, keep_graph, create_graph);
             } else {
                 tensorplay::tpx::backward(self, Tensor(), keep_graph, create_graph);
             }
        }, "gradient"_a = py::none(), "retain_graph"_a = py::none(), "create_graph"_a = false)
        .def_property("data", 
            [](const Tensor& self) { return self.detach(); },
            [](Tensor& self, const Tensor& other) {
                if (!self.defined() || !other.defined()) {
                    self = other;
                    return;
                }
                // Update underlying TensorImpl data/metadata in-place
                // This ensures other references (like p.grad) see the change
                self.unsafeGetTensorImpl()->copy_metadata_from(*other.unsafeGetTensorImpl());
            }
        )
        .def("detach", &Tensor::detach)
        .def("detach_", [](py::object self_obj) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            self.set_requires_grad(false);
            
            return self_obj;
        })
        .def("clone", &Tensor::clone)
        .def("requires_grad_", [](py::object self_obj, bool requires_grad) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            tensorplay::tpx::impl::set_requires_grad(self, requires_grad);
            return self_obj;
        }, "requires_grad"_a = true)
        
        // Methods
        .def("size", [](const Tensor& self) {
            return self.shape();
        })
        .def("size", [](const Tensor& self, int64_t dim) {
            return self.size(dim);
        })
        .def("view", [](const Tensor& self, py::args args) {
            std::vector<int64_t> shape;
            if (args.size() == 1 && (py::isinstance<py::list>(args[0]) || py::isinstance<py::tuple>(args[0]) || py::isinstance<Size>(args[0]))) {
                py::object obj = args[0];
                for (auto item : obj) {
                    shape.push_back(py::cast<int64_t>(item));
                }
            } else {
                for (auto item : args) {
                    shape.push_back(py::cast<int64_t>(item));
                }
            }
            return tensorplay::tpx::ops::view(self, shape);
        })
        .def("reshape", [](const Tensor& self, py::args args) {
            std::vector<int64_t> shape;
            if (args.size() == 1 && (py::isinstance<py::list>(args[0]) || py::isinstance<py::tuple>(args[0]) || py::isinstance<Size>(args[0]))) {
                py::object obj = args[0];
                for (auto item : obj) {
                    shape.push_back(py::cast<int64_t>(item));
                }
            } else {
                for (auto item : args) {
                    shape.push_back(py::cast<int64_t>(item));
                }
            }
            return tensorplay::tpx::ops::reshape(self, shape);
        })
        .def("expand", [](const Tensor& self, const std::vector<int64_t>& size, bool /*implicit*/) {
            return tensorplay::tpx::expand(self, size);
        }, "size"_a, "implicit"_a = false)
        .def("as_strided", [](const Tensor& self,
                              const std::vector<int64_t>& size,
                              const std::vector<int64_t>& stride,
                              std::optional<int64_t> storage_offset) {
            return tensorplay::tpx::as_strided(self, size, stride, storage_offset);
        }, "size"_a, "stride"_a, "storage_offset"_a = py::none())
        .def("select", [](const Tensor& self, int64_t dim, int64_t index) {
            return tensorplay::tpx::select(self, dim, index);
        }, "dim"_a, "index"_a)
        .def("slice", [](const Tensor& self, int64_t dim, int64_t start, int64_t end, int64_t step) {
            return tensorplay::tpx::slice(self, dim, start, end, step);
        }, "dim"_a, "start"_a, "end"_a, "step"_a = 1)
        .def("narrow", [](const Tensor& self, int64_t dim, int64_t start, int64_t length) {
            return tensorplay::tpx::narrow(self, dim, start, length);
        }, "dim"_a, "start"_a, "length"_a)
        .def("copy_", [](py::object self_obj, const Tensor& src, bool non_blocking) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            self.copy_(src, non_blocking);
            return self_obj;
        }, "src"_a, "non_blocking"_a = false)
        .def("fill_", [](py::object self_obj, Scalar value) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            self.fill_(value);
            return self_obj;
        }, "value"_a)
        .def("zero_", [](py::object self_obj) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            self.zero_();
            return self_obj;
        })
        // In-place random sampling
        .def("bernoulli_", [](py::object self_obj) {
             py::cast<Tensor&>(self_obj).bernoulli_();
             return self_obj;
        })
        .def("cauchy_", [](py::object self_obj, double median, double sigma) {
             py::cast<Tensor&>(self_obj).cauchy_(median, sigma);
             return self_obj;
        }, "median"_a = 0.0, "sigma"_a = 1.0)
        .def("exponential_", [](py::object self_obj, double lambd) {
             py::cast<Tensor&>(self_obj).exponential_(lambd);
             return self_obj;
        }, "lambd"_a = 1.0)
        .def("geometric_", [](py::object self_obj, double p) {
             py::cast<Tensor&>(self_obj).geometric_(p);
             return self_obj;
        }, "p"_a)
        .def("log_normal_", [](py::object self_obj, double mean, double std) {
             py::cast<Tensor&>(self_obj).log_normal_(mean, std);
             return self_obj;
        }, "mean"_a = 1.0, "std"_a = 2.0)
        .def("normal_", [](py::object self_obj, double mean, double std) {
             py::cast<Tensor&>(self_obj).normal_(mean, std);
             return self_obj;
        }, "mean"_a = 0.0, "std"_a = 1.0)
        .def("random_", [](py::object self_obj, int64_t low, int64_t high) {
             py::cast<Tensor&>(self_obj).random_(low, high);
             return self_obj;
        }, "low"_a = 0, "high"_a = 0)
        .def("uniform_", [](py::object self_obj, double from, double to) {
             py::cast<Tensor&>(self_obj).uniform_(from, to);
             return self_obj;
        }, "from"_a = 0.0, "to"_a = 1.0)
        


// ...

        .def_static("_load_file_segment", [](std::string filename, size_t offset, size_t nbytes, std::vector<int64_t> shape, DType dtype, std::optional<Device> device) {
            FILE* f = fopen(filename.c_str(), "rb");
            if (!f) {
                throw std::runtime_error("Could not open file: " + filename);
            }
            
            if (fseek(f, (long)offset, SEEK_SET) != 0) {
                fclose(f);
                throw std::runtime_error("Could not seek to offset " + std::to_string(offset) + " in file " + filename);
            }

            Device target_device = device.value_or(Device(DeviceType::CPU));
            Tensor p10_t(shape, dtype, target_device);
            
            size_t expected_bytes = p10_t.numel() * p10_t.itemsize();
            if (nbytes != expected_bytes) {
                fclose(f);
                throw std::runtime_error("Requested bytes " + std::to_string(nbytes) + " does not match tensor size " + std::to_string(expected_bytes));
            }

            if (target_device.is_cpu()) {
                size_t read = fread(p10_t.data_ptr(), 1, nbytes, f);
                fclose(f);
                
                if (read != nbytes) {
                    throw std::runtime_error("Read failed: expected " + std::to_string(nbytes) + " bytes, got " + std::to_string(read));
                }
            } else if (target_device.is_cuda()) {
#ifdef USE_CUDA
                // Read to host buffer then copy to device
                std::vector<char> buffer(nbytes);
                size_t read = fread(buffer.data(), 1, nbytes, f);
                fclose(f);
                
                if (read != nbytes) {
                    throw std::runtime_error("Read failed: expected " + std::to_string(nbytes) + " bytes, got " + std::to_string(read));
                }
                
                cudaError_t err = cudaMemcpy(p10_t.data_ptr(), buffer.data(), nbytes, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    throw std::runtime_error("CUDA Copy Error: " + std::string(cudaGetErrorString(err)));
                }
#else
                fclose(f);
                throw std::runtime_error("Loading to CUDA device but USE_CUDA not enabled");
#endif
            } else {
                fclose(f);
                throw std::runtime_error("Unsupported device type for loading");
            }
            
            return Tensor(p10_t);
        }, "filename"_a, "offset"_a, "nbytes"_a, "shape"_a, "dtype"_a, "device"_a = py::none())

        .def_static("_load_file_segments", [](std::string filename, std::vector<std::tuple<Tensor, int64_t, int64_t>> segments) {
            FILE* f = fopen(filename.c_str(), "rb");
            if (!f) {
                throw std::runtime_error("Could not open file: " + filename);
            }

            std::vector<char> buffer; // Reusable buffer for CUDA reads

            for (auto& seg : segments) {
                Tensor& t = std::get<0>(seg);
                int64_t offset = std::get<1>(seg);
                int64_t length = std::get<2>(seg);
                
                // Get internal P10 tensor
                tensorplay::Tensor& p10_t = t;
                
                if (!p10_t.is_contiguous()) {
                     fclose(f);
                     throw std::runtime_error("Tensor must be contiguous");
                }
                
                if (fseek(f, (long)offset, SEEK_SET) != 0) {
                     fclose(f);
                     throw std::runtime_error("Seek failed for offset " + std::to_string(offset));
                }
                
                if (p10_t.device().is_cpu()) {
                    size_t read_count = fread(p10_t.data_ptr(), 1, length, f);
                    if (read_count != length) {
                         fclose(f);
                         throw std::runtime_error("Read failed or unexpected EOF. Expected " + std::to_string(length) + ", got " + std::to_string(read_count));
                    }
                } else if (p10_t.device().is_cuda()) {
#ifdef USE_CUDA
                    if (buffer.size() < length) buffer.resize(length);
                    
                    size_t read_count = fread(buffer.data(), 1, length, f);
                    if (read_count != length) {
                         fclose(f);
                         throw std::runtime_error("Read failed or unexpected EOF. Expected " + std::to_string(length) + ", got " + std::to_string(read_count));
                    }
                    
                    cudaError_t err = cudaMemcpy(p10_t.data_ptr(), buffer.data(), length, cudaMemcpyHostToDevice);
                    if (err != cudaSuccess) {
                        fclose(f);
                        throw std::runtime_error("CUDA Copy Error: " + std::string(cudaGetErrorString(err)));
                    }
#else
                    fclose(f);
                    throw std::runtime_error("Loading to CUDA device but USE_CUDA not enabled");
#endif
                } else {
                    fclose(f);
                    throw std::runtime_error("Unsupported device type for loading segments");
                }
            }
            fclose(f);
        }, "filename"_a, "segments"_a)

        .def_static("_save_file_segments", [](std::string filename, std::vector<Tensor> tensors) {
             FILE* f = fopen(filename.c_str(), "ab"); // Append binary
             if (!f) {
                 throw std::runtime_error("Could not open file for appending: " + filename);
             }
             
             for (const auto& t : tensors) {
                 // Ensure CPU and contiguous
                 Tensor t_cpu = t;
                 if (t_cpu.device().type() != DeviceType::CPU) {
                     t_cpu = t_cpu.to(Device(DeviceType::CPU));
                 }
                 if (!t_cpu.is_contiguous()) {
                     t_cpu = t_cpu.clone(); 
                 }
                 
                 const tensorplay::Tensor& p10_t = t_cpu;
                 size_t nbytes = p10_t.numel() * p10_t.itemsize();
                 
                 size_t written = fwrite(p10_t.data_ptr(), 1, nbytes, f);
                 if (written != nbytes) {
                     fclose(f);
                     throw std::runtime_error("Write failed");
                 }
             }
             fclose(f);
        }, "filename"_a, "tensors"_a)

        .def_static("_from_bytes", [](py::bytes data, std::vector<int64_t> shape, DType dtype) {
             size_t nbytes = py::len(data);
             
             // Create empty tensor
             Tensor p10_t(shape, dtype, Device(DeviceType::CPU));
             
             size_t expected_bytes = p10_t.numel() * p10_t.itemsize();
             if (nbytes != expected_bytes) {
                 throw std::runtime_error("Tensor data size mismatch: expected " + std::to_string(expected_bytes) + ", got " + std::to_string(nbytes));
             }
             
             // Copy data
             std::memcpy(p10_t.data_ptr(), PyBytes_AsString(data.ptr()), nbytes);
             
             return Tensor(p10_t);
        }, "data"_a, "shape"_a, "dtype"_a);

        // Bind generated methods
        bind_generated_tensor_methods(tensor);

        // torch.Tensor.movedim accepts Union[int, int[]]; the generated
        // binding covers the list form, so add the scalar-int overloads here.
        tensor.def("movedim", [](const Tensor& self, int64_t source, int64_t destination) {
            return tensorplay::tpx::ops::movedim(self, {source}, {destination});
        }, "source"_a, "destination"_a)
        .def("sum", [](const Tensor& self, std::optional<DType> dtype) {
            return self.sum(dtype.value_or(DType::Undefined));
        }, py::kw_only(), "dtype"_a = py::none())
        .def("sum", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, std::optional<DType> dtype) {
            return self.sum(dim, keepdim, dtype.value_or(DType::Undefined));
        }, "dim"_a, "keepdim"_a = false, py::kw_only(), "dtype"_a = py::none())
        .def("sum", [](const Tensor& self, int64_t dim, bool keepdim, std::optional<DType> dtype) {
            return self.sum({dim}, keepdim, dtype.value_or(DType::Undefined));
        }, "dim"_a, "keepdim"_a = false, py::kw_only(), "dtype"_a = py::none())

        .def("mean", [](const Tensor& self, std::optional<DType> dtype) {
            return self.mean(dtype.value_or(DType::Undefined));
        }, py::kw_only(), "dtype"_a = py::none())
        .def("mean", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, std::optional<DType> dtype) {
            return self.mean(dim, keepdim, dtype.value_or(DType::Undefined));
        }, "dim"_a, "keepdim"_a = false, py::kw_only(), "dtype"_a = py::none())
        .def("mean", [](const Tensor& self, int64_t dim, bool keepdim, std::optional<DType> dtype) {
            return self.mean({dim}, keepdim, dtype.value_or(DType::Undefined));
        }, "dim"_a, "keepdim"_a = false, py::kw_only(), "dtype"_a = py::none())

        .def("prod", [](const Tensor& self, std::optional<DType> dtype) {
            return self.prod(dtype.value_or(DType::Undefined));
        }, py::kw_only(), "dtype"_a = py::none())
        .def("prod", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim, std::optional<DType> dtype) {
            return self.prod(dim, keepdim, dtype.value_or(DType::Undefined));
        }, "dim"_a, "keepdim"_a = false, py::kw_only(), "dtype"_a = py::none())
        .def("prod", [](const Tensor& self, int64_t dim, bool keepdim, std::optional<DType> dtype) {
            return self.prod({dim}, keepdim, dtype.value_or(DType::Undefined));
        }, "dim"_a, "keepdim"_a = false, py::kw_only(), "dtype"_a = py::none())

        .def("all", [](const Tensor& self) { return self.all(); })
        .def("all", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
            return self.all(dim, keepdim);
        }, "dim"_a, "keepdim"_a = false)
        .def("all", [](const Tensor& self, int64_t dim, bool keepdim) {
            return self.all({dim}, keepdim);
        }, "dim"_a, "keepdim"_a = false)

        .def("any", [](const Tensor& self) { return self.any(); })
        .def("any", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
            return self.any(dim, keepdim);
        }, "dim"_a, "keepdim"_a = false)
        .def("any", [](const Tensor& self, int64_t dim, bool keepdim) {
            return self.any({dim}, keepdim);
        }, "dim"_a, "keepdim"_a = false)

        .def("argmax", [](const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
            return self.argmax(dim, keepdim);
        }, "dim"_a = py::none(), "keepdim"_a = false)
        .def("argmin", [](const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
            return self.argmin(dim, keepdim);
        }, "dim"_a = py::none(), "keepdim"_a = false)

        .def("var", [](const Tensor& self, int64_t correction) {
            return self.var(correction);
        }, "correction"_a = 1)
        .def("var", [](const Tensor& self, const std::vector<int64_t>& dim, int64_t correction, bool keepdim) {
            return self.var(dim, correction, keepdim);
        }, "dim"_a, "correction"_a = 1, "keepdim"_a = false)
        .def("var", [](const Tensor& self, int64_t dim, int64_t correction, bool keepdim) {
            return self.var({dim}, correction, keepdim);
        }, "dim"_a, "correction"_a = 1, "keepdim"_a = false)

        .def("std", [](const Tensor& self, int64_t correction) {
            return self.std(correction);
        }, "correction"_a = 1)
        .def("std", [](const Tensor& self, const std::vector<int64_t>& dim, int64_t correction, bool keepdim) {
            return self.std(dim, correction, keepdim);
        }, "dim"_a, "correction"_a = 1, "keepdim"_a = false)
        .def("std", [](const Tensor& self, int64_t dim, int64_t correction, bool keepdim) {
            return self.std({dim}, correction, keepdim);
        }, "dim"_a, "correction"_a = 1, "keepdim"_a = false)

        .def("norm", [](const Tensor& self, double p) {
            return self.norm(p);
        }, "p"_a = 2.0)
        .def("norm", [](const Tensor& self, const std::vector<int64_t>& dim, double p, bool keepdim) {
            return self.norm(dim, p, keepdim);
        }, "dim"_a, "p"_a = 2.0, "keepdim"_a = false)
        .def("norm", [](const Tensor& self, int64_t dim, double p, bool keepdim) {
            return self.norm({dim}, p, keepdim);
        }, "dim"_a, "p"_a = 2.0, "keepdim"_a = false)

        .def("max", [](const Tensor& self) { return self.max(); })
        .def("max", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
            return self.max(dim, keepdim);
        }, "dim"_a, "keepdim"_a = false)
        .def("max", [](const Tensor& self, int64_t dim, bool keepdim) {
            return self.max({dim}, keepdim);
        }, "dim"_a, "keepdim"_a = false)

        .def("min", [](const Tensor& self) { return self.min(); })
        .def("min", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
            return self.min(dim, keepdim);
        }, "dim"_a, "keepdim"_a = false)
        .def("min", [](const Tensor& self, int64_t dim, bool keepdim) {
            return self.min({dim}, keepdim);
        }, "dim"_a, "keepdim"_a = false)

        .def("pow", [](const Tensor& self, Scalar exponent) {
            return self.pow(exponent);
        }, "exponent"_a)
        .def("sqrt", [](const Tensor& self) { return self.sqrt(); })
        .def("abs", [](const Tensor& self) { return self.abs(); })

        .def("max", [](const Tensor& self) {
            return self.max();
        })
        .def("max", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
            return self.max(dim, keepdim);
        }, "dim"_a, "keepdim"_a = false)
        .def("max", [](const Tensor& self, int64_t dim, bool keepdim) {
            return self.max({dim}, keepdim);
        }, "dim"_a, "keepdim"_a = false)

        .def("min", [](const Tensor& self) {
            return self.min();
        })
        .def("min", [](const Tensor& self, const std::vector<int64_t>& dim, bool keepdim) {
            return self.min(dim, keepdim);
        }, "dim"_a, "keepdim"_a = false)
        .def("min", [](const Tensor& self, int64_t dim, bool keepdim) {
            return self.min({dim}, keepdim);
        }, "dim"_a, "keepdim"_a = false)

        // Manual overloads using lambdas
        .def("to", [](const Tensor& self, DType dtype, bool non_blocking, bool copy) {
            return tensorplay::tpx::to(self, dtype, non_blocking, copy);
        }, "dtype"_a, "non_blocking"_a = false, "copy"_a = false)
        .def("to", [](const Tensor& self, Device device, bool non_blocking, bool copy) {
            return tensorplay::tpx::to(self, device, non_blocking, copy);
        }, "device"_a, "non_blocking"_a = false, "copy"_a = false)
        .def("to", [](const Tensor& self, Device device, DType dtype, bool non_blocking, bool copy) {
            return tensorplay::tpx::to(self, device, dtype, non_blocking, copy);
        }, "device"_a, "dtype"_a, "non_blocking"_a = false, "copy"_a = false)

        .def("__array__", [](py::object self_obj, py::object dtype, bool copy) {
            try {
                py::module_ np = py::module_::import("numpy");
                // Delegate to from_dlpack which is zero-copy and efficient
                py::object arr = np.attr("from_dlpack")(self_obj);
                if (copy) {
                    arr = arr.attr("copy")();
                }
                if (!dtype.is_none()) {
                    return arr.attr("astype")(dtype, "copy"_a = false);
                }
                return arr;
            } catch (const std::exception&) {
                TP_THROW(RuntimeError, "numpy is not installed or cannot be imported.");
            }
        }, "dtype"_a = py::none(), "copy"_a = true)

        .def("numpy", [](py::object self_obj) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            if (self.requires_grad()) {
                TP_THROW(RuntimeError, "Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.");
            }
            if (self.device().type() != DeviceType::CPU) {
                TP_THROW(RuntimeError, "Can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.");
            }
            
            DType dtype = self.dtype();

            if (dtype == DType::BFloat16) {
                // NumPy has no native bfloat16: convert element-wise to float32.
                size_t numel = self.numel();
                std::vector<float> buf(numel);
                const tensorplay::BFloat16* src = self.data_ptr<tensorplay::BFloat16>();
                for (size_t i = 0; i < numel; ++i) buf[i] = static_cast<float>(src[i]);
                // Copy into a NumPy-owned buffer, then reshape to the tensor's
                // shape.  Constructing array_t explicitly is important here:
                // py::cast(vector<float>) may choose NumPy's default float64.
                py::array_t<float> f32(numel);
                std::memcpy(f32.mutable_data(), buf.data(), numel * sizeof(float));
                auto sizes = self.shape();
                std::vector<py::ssize_t> shape(sizes.begin(), sizes.end());
                return py::array(f32.attr("reshape")(py::cast(shape)));
            }

            if (dtype == DType::ComplexHalf || dtype == DType::BComplex32) {
                // NumPy has no complex-half or complex-bfloat16 dtype. Match
                // torch's practical interop behavior by widening to complex64.
                size_t numel = self.numel();
                std::vector<std::complex<float>> buf(numel);
                auto sizes = self.shape();
                if (dtype == DType::ComplexHalf) {
                    const auto* src = self.data_ptr<std::complex<tensorplay::Half>>();
                    for (size_t i = 0; i < numel; ++i) {
                        buf[i] = {static_cast<float>(src[i].real()),
                                  static_cast<float>(src[i].imag())};
                    }
                } else {
                    const auto* src = self.data_ptr<std::complex<tensorplay::BFloat16>>();
                    for (size_t i = 0; i < numel; ++i) {
                        buf[i] = {static_cast<float>(src[i].real()),
                                  static_cast<float>(src[i].imag())};
                    }
                }
                py::array_t<std::complex<float>> c64(numel);
                std::memcpy(c64.mutable_data(), buf.data(),
                            numel * sizeof(std::complex<float>));
                std::vector<py::ssize_t> shape(sizes.begin(), sizes.end());
                return py::array(c64.attr("reshape")(py::cast(shape)));
            }

            std::string fmt;
            switch (dtype) {
                case DType::Float32: fmt = "f4"; break;
                case DType::Float64: fmt = "f8"; break;
                case DType::Float16: fmt = "f2"; break;
                case DType::ComplexFloat: fmt = "c8"; break;
                case DType::ComplexDouble: fmt = "c16"; break;
                case DType::Int32:   fmt = "i4"; break;
                case DType::Int64:   fmt = "i8"; break;
                case DType::Int8:    fmt = "i1"; break;
                case DType::Int16:   fmt = "i2"; break;
                case DType::UInt8:   fmt = "u1"; break;
                case DType::UInt16:  fmt = "u2"; break;
                case DType::UInt32:  fmt = "u4"; break;
                case DType::UInt64:  fmt = "u8"; break;
                case DType::Bool:    fmt = "?";  break;
                default: TP_THROW(RuntimeError, "Unsupported DType for NumPy conversion");
            }

            auto sizes = self.shape();
            std::vector<py::ssize_t> shape(sizes.begin(), sizes.end());
            std::vector<int64_t> strides_int64 = self.strides();
            
            // NumPy strides are in bytes, TensorPlay strides are in elements
            size_t itemsize = self.itemsize();
            std::vector<py::ssize_t> strides_bytes;
            strides_bytes.reserve(strides_int64.size());
            for (auto s : strides_int64) {
                strides_bytes.push_back(s * (py::ssize_t)itemsize);
            }
            
            // Zero-copy: wrap the tensor's memory, keeping the tensor alive as the base.
            return py::array(py::dtype(fmt), shape, strides_bytes, self.data_ptr(), self_obj);
        })
        
        .def("data_ptr", [](const Tensor& self) {
            return reinterpret_cast<uintptr_t>(self.data_ptr());
        })
        .def("item", [](const Tensor& self) -> py::object {
            switch (self.dtype()) {
                case DType::Float32: return py::float_(self.item().to<float>());
                case DType::Float64: return py::float_(self.item().to<double>());
                case DType::Float16: return py::float_(self.item().to<float>());
                case DType::BFloat16: return py::float_(self.item().to<float>());
                case DType::Int32: return py::int_(self.item().to<int32_t>());
                case DType::Int64: return py::int_(self.item().to<int64_t>());
                case DType::Int8: return py::int_(self.item().to<int8_t>());
                case DType::Int16: return py::int_(self.item().to<int16_t>());
                case DType::UInt8: return py::int_(self.item().to<uint8_t>());
                case DType::UInt16: return py::int_(self.item().to<uint16_t>());
                case DType::UInt32: return py::int_(self.item().to<uint32_t>());
                case DType::UInt64: return py::int_(self.item().to<uint64_t>());
                case DType::ComplexHalf:
                case DType::ComplexFloat:
                    return py::cast(self.item().to<std::complex<float>>());
                case DType::ComplexDouble:
                case DType::BComplex32:
                    return py::cast(self.item().to<std::complex<double>>());
                case DType::Bool: return py::bool_(self.item().to<bool>());
                default: TP_THROW(NotImplementedError, "item() not implemented for this dtype");
            }
        })
        
        // Indexing
        .def("tolist", [](const Tensor& self) -> py::object {
            if (self.device().type() != DeviceType::CPU) {
                 TP_THROW(RuntimeError, "tolist() is only supported on CPU tensors");
            }

            auto get_dtype_size = [](DType dtype) -> size_t {
                return tensorplay::elementSize(dtype);
            };

            auto scalar_to_python = [](const void* ptr, DType dtype) -> py::object {
                switch (dtype) {
                    case DType::Float32: return py::float_(*static_cast<const float*>(ptr));
                    case DType::Float64: return py::float_(*static_cast<const double*>(ptr));
                    case DType::Float16: return py::float_(static_cast<float>(*static_cast<const tensorplay::Half*>(ptr)));
                    case DType::BFloat16: return py::float_(static_cast<float>(*static_cast<const tensorplay::BFloat16*>(ptr)));
                    case DType::Int8: return py::int_(static_cast<int64_t>(*static_cast<const int8_t*>(ptr)));
                    case DType::Int16: return py::int_(static_cast<int64_t>(*static_cast<const int16_t*>(ptr)));
                    case DType::Int32: return py::int_(static_cast<int64_t>(*static_cast<const int32_t*>(ptr)));
                    case DType::Int64: return py::int_(*static_cast<const int64_t*>(ptr));
                    case DType::UInt8: return py::int_(static_cast<uint64_t>(*static_cast<const uint8_t*>(ptr)));
                    case DType::UInt16: return py::int_(static_cast<uint64_t>(*static_cast<const uint16_t*>(ptr)));
                    case DType::UInt32: return py::int_(static_cast<uint64_t>(*static_cast<const uint32_t*>(ptr)));
                    case DType::UInt64: return py::int_(*static_cast<const uint64_t*>(ptr));
                    case DType::Bool: return py::bool_(*static_cast<const bool*>(ptr));
                    case DType::ComplexHalf: {
                        const auto& value = *static_cast<const std::complex<tensorplay::Half>*>(ptr);
                        return py::cast(std::complex<float>(static_cast<float>(value.real()), static_cast<float>(value.imag())));
                    }
                    case DType::BComplex32: {
                        const auto& value = *static_cast<const std::complex<tensorplay::BFloat16>*>(ptr);
                        return py::cast(std::complex<float>(static_cast<float>(value.real()), static_cast<float>(value.imag())));
                    }
                    case DType::ComplexFloat: return py::cast(*static_cast<const std::complex<float>*>(ptr));
                    case DType::ComplexDouble: return py::cast(*static_cast<const std::complex<double>*>(ptr));
                    default: TP_THROW(NotImplementedError, "tolist() not implemented for this dtype");
                }
            };
            
            if (self.dim() == 0) {
                return scalar_to_python(self.data_ptr(), self.dtype());
            }

            // Recursive helper lambda
            auto recurse = [&](auto&& self_recurse, const void* data, int64_t ndim, const int64_t* sizes, const int64_t* strides, DType dtype) -> py::object {
                int64_t size = sizes[0];
                int64_t stride = strides[0];
                py::list result;
                size_t itemsize = get_dtype_size(dtype);

                if (ndim == 1) {
                    for (int64_t i = 0; i < size; ++i) {
                        const char* ptr = static_cast<const char*>(data) + i * stride * itemsize;
                        result.append(scalar_to_python(ptr, dtype));
                    }
                } else {
                    for (int64_t i = 0; i < size; ++i) {
                        const char* ptr = static_cast<const char*>(data) + i * stride * itemsize;
                        result.append(self_recurse(self_recurse, ptr, ndim - 1, sizes + 1, strides + 1, dtype));
                    }
                }
                return result;
            };

            std::vector<int64_t> shape_vec = static_cast<std::vector<int64_t>>(self.shape());
            std::vector<int64_t> strides_vec = self.strides();
            
            return recurse(recurse, self.data_ptr(), self.dim(), shape_vec.data(), strides_vec.data(), self.dtype());
        })

        .def("__getitem__", [](const Tensor& self, py::object index) -> Tensor {
            if (py::isinstance<Tensor>(index)) {
            Tensor idx = py::cast<Tensor>(index);
            if (idx.dtype() == DType::Bool) {
                return self.masked_select(idx);
            }
        }
            if (py::isinstance<py::tuple>(index)) {
                 py::tuple indices = py::cast<py::tuple>(index);
                 Tensor result = self;
                 int64_t target_dim = 0;
                 for (size_t i = 0; i < indices.size(); ++i) {
                     py::object idx = indices[i];
                     if (py::isinstance<py::int_>(idx)) {
                         int64_t val = py::cast<int64_t>(idx);
                         // Route indexing through the autograd-aware wrapper;
                         // calling the raw Tensor view would sever gradients
                         // for common RoPE/decoder slicing patterns.
                         result = tensorplay::tpx::select(result, target_dim, val);
                     } else if (py::isinstance<py::slice>(idx)) {
                         py::slice s = py::cast<py::slice>(idx);
                         auto [start, stop, step, slicelength] = compute_slice(s, result.size(target_dim));
                         result = tensorplay::tpx::slice(result, target_dim, start, stop, step);
                         target_dim++;
                     } else {
                         TP_THROW(TypeError, "Unsupported index type in tuple");
                     }
                 }
                 return result;
            } else if (py::isinstance<py::int_>(index)) {
                return tensorplay::tpx::select(self, 0, py::cast<int64_t>(index));
            } else if (py::isinstance<py::slice>(index)) {
                py::slice s = py::cast<py::slice>(index);
                auto [start, stop, step, slicelength] = compute_slice(s, self.size(0));
                return tensorplay::tpx::slice(self, 0, start, stop, step);
            }
            TP_THROW(TypeError, "Unsupported index type");
        })
        .def("__setitem__", [](Tensor& self, py::object index, py::object value) {
            Tensor target;
            if (py::isinstance<py::tuple>(index)) {
                 py::tuple indices = py::cast<py::tuple>(index);
                 target = self;
                 int64_t target_dim = 0;
                 for (size_t i = 0; i < indices.size(); ++i) {
                     py::object idx = indices[i];
                     if (py::isinstance<py::int_>(idx)) {
                         int64_t val = py::cast<int64_t>(idx);
                         target = target.select(target_dim, val);
                     } else if (py::isinstance<py::slice>(idx)) {
                         py::slice s = py::cast<py::slice>(idx);
                         auto [start, stop, step, slicelength] = compute_slice(s, target.size(target_dim));
                         target = target.slice(target_dim, start, stop, step);
                         target_dim++;
                     } else {
                         TP_THROW(TypeError, "Unsupported index type in tuple");
                     }
                 }
            } else if (py::isinstance<py::int_>(index)) {
                target = tensorplay::tpx::select(self, 0, py::cast<int64_t>(index));
            } else if (py::isinstance<py::slice>(index)) {
                py::slice s = py::cast<py::slice>(index);
                auto [start, stop, step, slicelength] = compute_slice(s, self.size(0));
                target = tensorplay::tpx::slice(self, 0, start, stop, step);
            } else {
                TP_THROW(TypeError, "Unsupported index type");
            }

            if (py::isinstance<Tensor>(value)) {
                target.copy_(py::cast<Tensor>(value));
            } else {
                try {
                    // Try to cast to scalar (float/int/bool)
                    if (py::isinstance<py::float_>(value) || py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value)) {
                         double v = py::cast<double>(value);
                         target.fill_(Scalar(v));
                    } else {
                         TP_THROW(TypeError, "Unsupported value type for setitem");
                    }
                } catch (...) {
                    TP_THROW(TypeError, "Unsupported value type for setitem");
                }
            }
        })
        
        // Operators
        .def("__neg__", [](const Tensor& t) { return tensorplay::tpx::ops::neg(t); })
        .def("__add__", [](const Tensor& a, const Tensor& b) { return tensorplay::tpx::ops::add(a, b); })
        .def("__sub__", [](const Tensor& a, const Tensor& b) { return tensorplay::tpx::ops::sub(a, b); })
        .def("__mul__", [](const Tensor& a, const Tensor& b) { return tensorplay::tpx::ops::mul(a, b); })
        .def("__truediv__", [](const Tensor& a, const Tensor& b) { return tensorplay::tpx::ops::div(a, b); })
        .def("__add__", [](const Tensor& t, double s) { return tensorplay::tpx::ops::add(t, Scalar(s)); })
        .def("__sub__", [](const Tensor& t, double s) { return tensorplay::tpx::ops::sub(t, Scalar(s)); })
        .def("__mul__", [](const Tensor& t, double s) { return tensorplay::tpx::ops::mul(t, Scalar(s)); })
        .def("__truediv__", [](const Tensor& t, double s) { return tensorplay::tpx::ops::div(t, Scalar(s)); })
        .def("__radd__", [](const Tensor& t, double s) { return tensorplay::tpx::ops::add(t, Scalar(s)); })
        .def("__rsub__", [](const Tensor& t, double s) {
            Tensor s_t = Tensor::full({}, Scalar(s), t.dtype(), t.device());
            return tensorplay::tpx::ops::sub(s_t, t);
        })
        .def("__rmul__", [](const Tensor& t, double s) { return tensorplay::tpx::ops::mul(t, Scalar(s)); })
        .def("__rtruediv__", [](const Tensor& t, double s) {
            Tensor s_t = Tensor::full({}, Scalar(s), t.dtype(), t.device());
            return tensorplay::tpx::ops::div(s_t, t);
        })
        .def("__iadd__", [](Tensor& self, const Tensor& other) {
            tensorplay::tpx::ops::add_(self, other);
            return self;
        })
        .def("__isub__", [](Tensor& self, const Tensor& other) {
            tensorplay::tpx::ops::sub_(self, other);
            return self;
        })
        .def("__imul__", [](Tensor& self, const Tensor& other) {
            tensorplay::tpx::ops::mul_(self, other);
            return self;
        })
        .def("__itruediv__", [](Tensor& self, const Tensor& other) {
            tensorplay::tpx::ops::div_(self, other);
            return self;
        })
        .def("__iadd__", [](Tensor& self, double s) {
            tensorplay::tpx::ops::add_(self, Scalar(s));
            return self;
        })
        .def("__isub__", [](Tensor& self, double s) {
            tensorplay::tpx::ops::sub_(self, Scalar(s));
            return self;
        })
        .def("__imul__", [](Tensor& self, double s) {
            tensorplay::tpx::ops::mul_(self, Scalar(s));
            return self;
        })
        .def("__itruediv__", [](Tensor& self, double s) {
            tensorplay::tpx::ops::div_(self, Scalar(s));
            return self;
        })
        
        // Explicit arithmetic
        .def("add", [](const Tensor& self, const Tensor& other, std::optional<Scalar> alpha) {
            return tensorplay::tpx::ops::add(self, other, alpha.value_or(Scalar(1)));
        }, "other"_a, "alpha"_a = py::none())
        .def("add", [](const Tensor& self, Scalar other, std::optional<Scalar> alpha) {
            return tensorplay::tpx::ops::add(self, other, alpha.value_or(Scalar(1)));
        }, "other"_a, "alpha"_a = py::none())
        .def("add_", [](py::object self_obj, const Tensor& other, std::optional<Scalar> alpha) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            tensorplay::tpx::ops::add_(self, other, alpha.value_or(Scalar(1)));
            return self_obj;
        }, "other"_a, "alpha"_a = py::none())
        .def("add_", [](py::object self_obj, Scalar other, std::optional<Scalar> alpha) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            tensorplay::tpx::ops::add_(self, other, alpha.value_or(Scalar(1)));
            return self_obj;
        }, "other"_a, "alpha"_a = py::none())
        .def("sub", [](const Tensor& self, const Tensor& other, std::optional<Scalar> alpha) {
            return tensorplay::tpx::ops::sub(self, other, alpha.value_or(Scalar(1)));
        }, "other"_a, "alpha"_a = py::none())
        .def("sub", [](const Tensor& self, Scalar other, std::optional<Scalar> alpha) {
            return tensorplay::tpx::ops::sub(self, other, alpha.value_or(Scalar(1)));
        }, "other"_a, "alpha"_a = py::none())
        .def("sub_", [](py::object self_obj, const Tensor& other, std::optional<Scalar> alpha) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            tensorplay::tpx::ops::sub_(self, other, alpha.value_or(Scalar(1)));
            return self_obj;
        }, "other"_a, "alpha"_a = py::none())
        .def("sub_", [](py::object self_obj, Scalar other, std::optional<Scalar> alpha) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            tensorplay::tpx::ops::sub_(self, other, alpha.value_or(Scalar(1)));
            return self_obj;
        }, "other"_a, "alpha"_a = py::none())
        .def("mul", [](const Tensor& self, const Tensor& other) {
            return tensorplay::tpx::ops::mul(self, other);
        }, "other"_a)
        .def("mul", [](const Tensor& self, Scalar other) {
            return tensorplay::tpx::ops::mul(self, other);
        }, "other"_a)
        .def("mul_", [](py::object self_obj, const Tensor& other) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            tensorplay::tpx::ops::mul_(self, other);
            return self_obj;
        }, "other"_a)
        .def("mul_", [](py::object self_obj, Scalar other) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            tensorplay::tpx::ops::mul_(self, other);
            return self_obj;
        }, "other"_a)
        .def("div", [](const Tensor& self, const Tensor& other) {
            return tensorplay::tpx::ops::div(self, other);
        }, "other"_a)
        .def("div", [](const Tensor& self, Scalar other) {
            return tensorplay::tpx::ops::div(self, other);
        }, "other"_a)
        .def("div_", [](py::object self_obj, const Tensor& other) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            tensorplay::tpx::ops::div_(self, other);
            return self_obj;
        }, "other"_a)
        .def("div_", [](py::object self_obj, Scalar other) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            tensorplay::tpx::ops::div_(self, other);
            return self_obj;
        }, "other"_a)
        .def("bernoulli", static_cast<Tensor(Tensor::*)() const>(&Tensor::bernoulli))
        .def("poisson", static_cast<Tensor(Tensor::*)() const>(&Tensor::poisson))
        .def("mm", [](const Tensor& self, const Tensor& other) { return tensorplay::tpx::ops::mm(self, other); }, "other"_a)
        .def("matmul", [](const Tensor& self, const Tensor& other) { return tensorplay::tpx::ops::matmul(self, other); }, "other"_a)
        .def("__matmul__", [](const Tensor& self, const Tensor& other) { return tensorplay::tpx::ops::matmul(self, other); }, "other"_a)

        // Comparison operators
        .def("__hash__", [](const Tensor& self) { return (intptr_t)&self; })
        .def("__eq__", [](const Tensor& self, const Tensor& other) { return self.eq(other); })
        .def("__eq__", [](const Tensor& self, Scalar other) { return self.eq(other); })
        .def("__ne__", [](const Tensor& self, const Tensor& other) { return self.ne(other); })
        .def("__ne__", [](const Tensor& self, Scalar other) { return self.ne(other); })
        .def("__lt__", [](const Tensor& self, const Tensor& other) { return self.lt(other); })
        .def("__lt__", [](const Tensor& self, Scalar other) { return self.lt(other); })
        .def("__le__", [](const Tensor& self, const Tensor& other) { return self.le(other); })
        .def("__le__", [](const Tensor& self, Scalar other) { return self.le(other); })
        .def("__gt__", [](const Tensor& self, const Tensor& other) { return self.gt(other); })
        .def("__gt__", [](const Tensor& self, Scalar other) { return self.gt(other); })
        .def("__ge__", [](const Tensor& self, const Tensor& other) { return self.ge(other); })
        .def("__ge__", [](const Tensor& self, Scalar other) { return self.ge(other); })

        // Pointwise ops
        .def("abs", [](const Tensor& self) { return tensorplay::tpx::ops::abs(self); })
        .def("acos", [](const Tensor& self) { return tensorplay::tpx::ops::acos(self); })
        .def("acosh", [](const Tensor& self) { return tensorplay::tpx::ops::acosh(self); })
        .def("angle", [](const Tensor& self) { return tensorplay::tpx::ops::angle(self); })
        .def("asin", [](const Tensor& self) { return tensorplay::tpx::ops::asin(self); })
        .def("asinh", [](const Tensor& self) { return tensorplay::tpx::ops::asinh(self); })
        .def("atan", [](const Tensor& self) { return tensorplay::tpx::ops::atan(self); })
        .def("atanh", [](const Tensor& self) { return tensorplay::tpx::ops::atanh(self); })
        .def("ceil", [](const Tensor& self) { return tensorplay::tpx::ops::ceil(self); })
        .def("clamp", [](const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
            return tensorplay::tpx::ops::clamp(self, min, max);
        }, "min"_a = py::none(), "max"_a = py::none())
        .def("cos", [](const Tensor& self) { return tensorplay::tpx::ops::cos(self); })
        .def("cosh", [](const Tensor& self) { return tensorplay::tpx::ops::cosh(self); })
        .def("exp", [](const Tensor& self) { return tensorplay::tpx::ops::exp(self); })
        .def("floor", [](const Tensor& self) { return tensorplay::tpx::ops::floor(self); })
        .def("gelu", [](const Tensor& self) { return tensorplay::tpx::ops::gelu(self); })
        .def("lerp", [](const Tensor& self, const Tensor& end, Scalar weight) {
            return tensorplay::tpx::ops::lerp(self, end, weight);
        }, "end"_a, "weight"_a)
        .def("lerp", [](const Tensor& self, const Tensor& end, const Tensor& weight) {
            return tensorplay::tpx::ops::lerp(self, end, weight);
        }, "end"_a, "weight"_a)
        .def("log", [](const Tensor& self) { return tensorplay::tpx::ops::log(self); })
        .def("neg", [](const Tensor& self) { return tensorplay::tpx::ops::neg(self); })
        .def("pow", [](const Tensor& self, Scalar exponent) { return tensorplay::tpx::ops::pow(self, exponent); }, "exponent"_a)
        .def("pow", [](const Tensor& self, const Tensor& exponent) { return tensorplay::tpx::ops::pow(self, exponent); }, "exponent"_a)
        .def("__pow__", [](const Tensor& self, Scalar exponent) { return tensorplay::tpx::ops::pow(self, exponent); }, "exponent"_a)
        .def("__pow__", [](const Tensor& self, const Tensor& exponent) { return tensorplay::tpx::ops::pow(self, exponent); }, "exponent"_a)
        .def("__rpow__", [](const Tensor& self, Scalar base) {
            Tensor base_t = Tensor::full({}, base, self.dtype(), self.device());
            return tensorplay::tpx::ops::pow(base_t, self);
        })
        .def("relu", [](const Tensor& self) { return tensorplay::tpx::ops::relu(self); })
        .def("round", [](const Tensor& self) { return tensorplay::tpx::ops::round(self); })
        .def("rsqrt", [](const Tensor& self) { return tensorplay::tpx::ops::rsqrt(self); })
        .def("sigmoid", [](const Tensor& self) { return tensorplay::tpx::ops::sigmoid(self); })
        .def("silu", [](const Tensor& self) { return tensorplay::tpx::ops::silu(self); })
        .def("sign", [](const Tensor& self) { return tensorplay::tpx::ops::sign(self); })
        .def("sin", [](const Tensor& self) { return tensorplay::tpx::ops::sin(self); })
        .def("sinh", [](const Tensor& self) { return tensorplay::tpx::ops::sinh(self); })
        .def("softmax", [](const Tensor& self, int64_t dim, DType dtype) {
            return tensorplay::tpx::ops::softmax(self, dim, dtype);
        }, "dim"_a, "dtype"_a = DType::Undefined)
        .def("sqrt", [](const Tensor& self) { return tensorplay::tpx::ops::sqrt(self); })
        .def("square", [](const Tensor& self) { return tensorplay::tpx::ops::square(self); })
        .def("tan", [](const Tensor& self) { return tensorplay::tpx::ops::tan(self); })
        .def("tanh", [](const Tensor& self) { return tensorplay::tpx::ops::tanh(self); })

        // DLPack
        .def("__dlpack__", [](py::object self_obj, std::optional<int64_t> stream) {
            return to_dlpack(self_obj, stream);
        }, "stream"_a = py::none())
        .def("__dlpack_device__", [](const Tensor& self) {
            DLDevice d = to_dlpack_device(self.device());
            return py::make_tuple(d.device_type, d.device_id);
        })
        
        // Multiprocessing shared memory support
        .def("share_memory_", [](py::object self_obj) {
             Tensor& self = py::cast<Tensor&>(self_obj);
             if (self.device().type() != DeviceType::CPU) return self_obj;
             if (py::hasattr(self_obj, "_shared_memory")) return self_obj;

             if (!self.is_contiguous()) {
                 TP_THROW(RuntimeError, "share_memory_() currently only supports contiguous tensors. Call .contiguous() before sharing.");
             }

             size_t nbytes = self.numel() * self.itemsize();
             py::object shm_cls = py::module_::import("multiprocessing.shared_memory").attr("SharedMemory");
             // create=True, size=nbytes
             py::object shm = shm_cls(py::arg("create")=true, py::arg("size")=nbytes);
             
             // Use helper to set storage and copy data
             set_storage_from_shm(self, shm, nbytes);
             
             py::setattr(self_obj, "_shared_memory", shm);
             return self_obj;
        })
        .def("is_shared", [](py::object self_obj) {
             return py::hasattr(self_obj, "_shared_memory");
        })

        // Pickling support
        .def(py::pickle(
            [](py::object self_obj) -> py::tuple {

            Tensor& self = py::cast<Tensor&>(self_obj);
            if (self.device().type() != DeviceType::CPU) {
                 TP_THROW(RuntimeError, "Pickling of non-CPU tensors is not yet supported");
            }
            
            // Check for shared memory
            if (py::hasattr(self_obj, "_shared_memory")) {
                py::object shm = py::getattr(self_obj, "_shared_memory");
                // Pickle the shared memory object itself, not just the name.
                // This ensures that when unpickled, the SharedMemory object is properly
                // reconstructed and the handle is preserved/duplicated if necessary.
                
                return py::tuple(py::make_tuple(
                    py::str("shm"), // Tag
                    shm,   // SharedMemory object
                    static_cast<std::vector<int64_t>>(self.shape()), 
                    self.strides(),
                    (int)self.dtype(), 
                    (int)self.device().type(), 
                    self.device().index(),
                    self.requires_grad()
                ));
            }
            
            Tensor contig = self.is_contiguous() ? self : self.clone();
            size_t nbytes = contig.numel() * contig.itemsize();
            
            // Create bytes object from data
            py::bytes data_bytes((const char*)contig.data_ptr(), nbytes);
            
            return py::tuple(py::make_tuple(
                data_bytes,
                static_cast<std::vector<int64_t>>(contig.shape()), 
                contig.strides(),
                (int)contig.dtype(), 
                (int)contig.device().type(), 
                contig.device().index(),
                self.requires_grad()
            ));
        
            },
            [](py::tuple state) -> std::pair<Tensor, py::dict> {
                return setstate_helper(std::move(state));
            }
        ))

        // pybind11 auto-registers a __reduce__ that bypasses the getstate/setstate
        // above (it reconstructs via the default constructor, losing the data).
        // Route __reduce__ through the registered getstate/setstate so that
        // pickle and multiprocessing serialize tensors correctly.
        .def("__reduce__", [](py::object self_obj) -> py::tuple {
            py::object state = self_obj.attr("__getstate__")();
            py::module_ copyreg = py::module_::import("copyreg");
            py::object newobj = copyreg.attr("__newobj__");
            py::object cls = self_obj.attr("__class__");
            return py::make_tuple(newobj, py::make_tuple(cls), state);
        })

        // String repr
        .def("__repr__", &Tensor::toString)
        .def("__str__", &Tensor::toString);
}

#include "python_bindings.h"
#include "tensorplay/ops/TensorBindingsGenerated.h"
#include "tensorplay/ops/TensorCPythonGenerated.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
#include "utils.h"
#include "dlpack_types.h"
#include "TensorImpl.h" // For unsafeGetTensorImpl
#include "Autograd.h" // tpx autograd helpers; Tensor is the p10 tensor type
#include "Storage.h"
#include "DataPtr.h"
#include "Node.h" // For grad_fn
#include "AccumulateGrad.h" // lazy grad_accumulator creation
#include "Utils.h" // broadcast shape validation for indexed assignment
#include "TypePromotion.h" // complex weak-scalar reflected-op dtype rules
#include <mutex>
#include <pybind11/functional.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <limits>
#include <vector>

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
    // resolves "current device" (-1) the same way.
    int64_t index = device.index();
    if (index < 0) index = 0;
    d.device_id = index;
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
    PyObject* cap_ptr = o.ptr();
    py::object cap_holder; // Holds a reference to a capsule we produced

    if (!PyCapsule_CheckExact(cap_ptr)) {
        // Optimization: Use C-API to call __dlpack__ directly.
        // faster than py::hasattr + o.attr()()
        static PyObject* dlpack_str = PyUnicode_InternFromString("__dlpack__");
        PyObject* res = PyObject_CallMethodObjArgs(o.ptr(), dlpack_str, nullptr);
        if (!res) {
             PyErr_Clear(); // Clear AttributeError
             TP_THROW(TypeError, "Object is not a DLPack capsule and does not have __dlpack__ method");
        }
        // The reference is taken before validating: a permissive __getattr__
        // makes any object look like it implements the protocol, so whatever
        // came back must be owned (and released) even when it is not a capsule.
        cap_holder = py::reinterpret_steal<py::object>(res);
        cap_ptr = res;
    }

    // Validate before dereferencing: PyCapsule_GetName on a non-capsule
    // returns NULL with an exception set, and reading through that pointer
    // is a hard crash rather than a Python-level error.
    if (!PyCapsule_IsValid(cap_ptr, "dltensor")) {
        if (PyCapsule_IsValid(cap_ptr, "used_dltensor")) {
            TP_THROW(ValueError,
                     "DLPack capsule has already been consumed; a DLTensor can "
                     "only be turned into a tensor once");
        }
        TP_THROW(TypeError,
                 "expected a DLPack capsule named \"dltensor\", got ",
                 Py_TYPE(cap_ptr)->tp_name);
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
             
             // The NumPy buffer is host memory.  Directly memcpy'ing it into
             // `t` is invalid when the requested device is CUDA (and also
             // mishandles non-contiguous NumPy views).  Normalize the source
             // to a C-contiguous CPU array, then let copy_ select the correct
             // host-to-device and/or dtype-conversion path.
             py::array contiguous_array =
                 py::array::ensure(array, py::array::c_style);
             if (!contiguous_array) {
                 TP_THROW(ValueError,
                          "could not make NumPy array C-contiguous");
             }
             Tensor src(shape, inferred_dtype, Device(DeviceType::CPU));
             std::memcpy(src.data_ptr(), contiguous_array.data(), total_bytes);
             t.copy_(src);
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
        Tensor new_t(static_cast<std::vector<int64_t>>(t.shape()), *dtype,
                     device.value_or(tensorplay::globalContext().defaultDevice()));
        Tensor new_t_wrapper(new_t);
        convert_tensor_data(t, new_t_wrapper);
        t = new_t_wrapper;
    }
    
    // Handle device movement if needed
    // Note: list_to_tensor returns CPU tensor.
    if (!device.has_value()) {
        Device target = tensorplay::globalContext().defaultDevice();
        if (t.device() != target && target.type() != DeviceType::CPU) {
            t = t.to(target);
        }
    } else if (t.device() != *device) {
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


// Normalize a Python slice against a length (py::slice::compute, i.e. PySlice_GetIndicesEx)
static std::tuple<int64_t, int64_t, int64_t, int64_t> compute_slice(py::slice s, int64_t length) {
    ssize_t start, stop, step, slicelength;
    if (!s.compute((ssize_t)length, &start, &stop, &step, &slicelength)) {
        throw py::error_already_set();
    }
    return {start, stop, step, slicelength};
}

// Python's list/tensor indexing is advanced indexing, not a view operation.
// Keep the conversion here, at the C++ binding boundary, so all Python entry
// points (including DataLoader datasets) use the same device, bounds, and
// autograd-aware index_select path.  In particular, do not read a CUDA index
// through data_ptr() on the host: stage it explicitly before normalizing
// negative values and checking bounds.
struct PreparedTensorIndex {
    Tensor flat;
    std::vector<int64_t> shape;
    std::vector<int64_t> values;
    bool scalar = false;
};

// deprecation warning).  Keep that interpretation at this boundary instead
// of treating a byte mask as an integer index tensor.
static bool is_boolean_mask_dtype(DType dtype) {
    return dtype == DType::Bool || dtype == DType::UInt8;
}

static int64_t checked_index_dim(const Tensor& self, int64_t dim) {
    const int64_t ndim = self.dim();
    if (ndim == 0) {
        TP_THROW(IndexError, "too many indices for tensor of dimension 0");
    }
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, "Dimension out of range");
    }
    return dim;
}

static Tensor make_normalized_index(const Tensor& self, int64_t dim,
                                    std::vector<int64_t>& values) {
    dim = checked_index_dim(self, dim);
    const int64_t dim_size = self.size(dim);
    for (int64_t& value : values) {
        if (value < 0) value += dim_size;
        if (value < 0 || value >= dim_size) {
            TP_THROW(IndexError, "index out of range");
        }
    }

    // Tensor::tensor(vector, ...) is a CPU factory.  Constructing the small
    // index on CPU and moving it once is also the safe route for CUDA: the
    // CUDA index_select kernel must never be handed a host pointer.
    Tensor result = Tensor::tensor(values, DType::Int64);
    if (!self.device().is_cpu()) result = result.to(self.device());
    return result;
}

static PreparedTensorIndex prepare_integer_index(const Tensor& self,
                                                 int64_t dim,
                                                 const Tensor& raw_index) {
    dim = checked_index_dim(self, dim);
    if (raw_index.dtype() == DType::Bool ||
        !isIntegralType(raw_index.dtype(), /*includeBool=*/false)) {
        TP_THROW(TypeError,
                 "tensors used as indices must be long, int, short, byte or bool tensors");
    }

    PreparedTensorIndex prepared;
    prepared.shape = static_cast<std::vector<int64_t>>(raw_index.shape());
    prepared.scalar = raw_index.dim() == 0;

    Tensor index = raw_index.to(DType::Int64).contiguous();
    Tensor host_index = index;
    if (!host_index.device().is_cpu()) {
        host_index = host_index.to(Device(DeviceType::CPU));
    }
    host_index = host_index.contiguous();
    prepared.values.resize(static_cast<size_t>(host_index.numel()));
    if (!prepared.values.empty()) {
        std::memcpy(prepared.values.data(), host_index.data_ptr<int64_t>(),
                    prepared.values.size() * sizeof(int64_t));
    }
    prepared.flat = make_normalized_index(self, dim, prepared.values);
    return prepared;
}

static bool is_python_bool_vector(py::handle object) {
    if (!PyList_Check(object.ptr()) && !PyTuple_Check(object.ptr())) return false;
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    const Py_ssize_t length = sequence.size();
    // integer-list behavior here.
    if (length == 0) return false;
    for (Py_ssize_t i = 0; i < length; ++i) {
        if (!PyBool_Check(sequence[i].ptr())) return false;
    }
    return true;
}

static PreparedTensorIndex prepare_python_bool_index(const Tensor& self,
                                                     int64_t dim,
                                                     py::handle object) {
    dim = checked_index_dim(self, dim);
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    const int64_t length = static_cast<int64_t>(sequence.size());
    if (length != self.size(dim)) {
        TP_THROW(IndexError, "The shape of the mask does not match the indexed tensor");
    }

    PreparedTensorIndex prepared;
    for (int64_t i = 0; i < length; ++i) {
        if (PyObject_IsTrue(sequence[static_cast<Py_ssize_t>(i)].ptr())) {
            prepared.values.push_back(i);
        }
    }
    prepared.shape = {static_cast<int64_t>(prepared.values.size())};
    prepared.flat = make_normalized_index(self, dim, prepared.values);
    return prepared;
}

static PreparedTensorIndex prepare_bool_tensor_index(const Tensor& self,
                                                     int64_t dim,
                                                     const Tensor& raw_index) {
    dim = checked_index_dim(self, dim);
    if (raw_index.dim() != 1 || raw_index.size(0) != self.size(dim)) {
        TP_THROW(IndexError, "The shape of the mask does not match the indexed tensor");
    }

    Tensor host_mask = raw_index;
    if (!host_mask.device().is_cpu()) {
        host_mask = host_mask.to(Device(DeviceType::CPU));
    }
    host_mask = host_mask.contiguous();

    PreparedTensorIndex prepared;
    const bool* bool_mask = host_mask.dtype() == DType::Bool
                                ? host_mask.data_ptr<bool>()
                                : nullptr;
    const uint8_t* byte_mask = host_mask.dtype() == DType::UInt8
                                   ? host_mask.data_ptr<uint8_t>()
                                   : nullptr;
    for (int64_t i = 0; i < host_mask.numel(); ++i) {
        if (bool_mask ? bool_mask[i] : byte_mask[i]) {
            prepared.values.push_back(i);
        }
    }
    prepared.shape = {static_cast<int64_t>(prepared.values.size())};
    prepared.flat = make_normalized_index(self, dim, prepared.values);
    return prepared;
}

static PreparedTensorIndex prepare_python_integer_index(const Tensor& self,
                                                        int64_t dim,
                                                        py::handle object) {
    // list_to_tensor handles nested lists as well, so e.g. x[[[0, 1]]]
    // preserves the [1, 2] index shape after index_select.
    Tensor raw = list_to_tensor(object.ptr(), DType::Int64,
                                Device(DeviceType::CPU));
    return prepare_integer_index(self, dim, raw);
}

static Tensor apply_prepared_index(const Tensor& self, int64_t dim,
                                   const PreparedTensorIndex& prepared) {
    dim = checked_index_dim(self, dim);
    if (prepared.scalar) {
        return tensorplay::tpx::ops::select(self, dim, prepared.values.at(0));
    }

    Tensor selected = tensorplay::tpx::ops::index_select(self, dim, prepared.flat);
    if (prepared.shape.size() <= 1) return selected;

    std::vector<int64_t> output_shape = static_cast<std::vector<int64_t>>(self.shape());
    output_shape.erase(output_shape.begin() + dim);
    output_shape.insert(output_shape.begin() + dim,
                        prepared.shape.begin(), prepared.shape.end());
    return tensorplay::tpx::ops::reshape(selected, output_shape);
}

static Tensor prepare_setitem_value(const Tensor& self, py::object value) {
    if (py::isinstance<Tensor>(value)) {
        Tensor result = py::cast<Tensor>(value);
        if (result.dtype() != self.dtype() || result.device() != self.device()) {
            result = result.to(self.device(), self.dtype());
        }
        return result;
    }
    if (py::isinstance<py::array>(value)) {
        return create_tensor(std::move(value), self.dtype(), self.device());
    }
    if (py::isinstance<py::list>(value) ||
        py::isinstance<py::tuple>(value)) {
        return list_to_tensor(value.ptr(), self.dtype(), self.device());
    }
    // Preserve Python integer precision.  Converting an int through double
    // first silently rounds large int64 assignments (e.g. 2**63 - 1).
    if (py::isinstance<py::bool_>(value)) {
        return Tensor::full({}, Scalar(py::cast<bool>(value)), self.dtype(),
                            self.device());
    }
    if (py::isinstance<py::int_>(value)) {
        return Tensor::full({}, Scalar(py::cast<int64_t>(value)), self.dtype(),
                            self.device());
    }
    if (py::isinstance<py::float_>(value)) {
        return Tensor::full({}, Scalar(py::cast<double>(value)), self.dtype(),
                            self.device());
    }
    TP_THROW(TypeError, "Unsupported value type for setitem");
}

static Tensor prepare_setitem_value(const Tensor& self, py::object value,
                                    const std::vector<int64_t>& target_shape) {
    Tensor result = prepare_setitem_value(self, std::move(value));
    const auto result_shape = static_cast<std::vector<int64_t>>(result.shape());
    const auto broadcast_shape = tensorplay::broadcast_shapes(target_shape, result_shape);
    if (broadcast_shape != target_shape) {
        TP_THROW(RuntimeError, "shape mismatch: value cannot be broadcast to indexed result");
    }
    if (result_shape != target_shape) {
        result = result.expand(target_shape).contiguous();
    }
    return result;
}

static std::vector<int64_t> indexed_result_shape_dim0(
    const Tensor& self, const PreparedTensorIndex& prepared) {
    std::vector<int64_t> shape = prepared.shape;
    const auto self_shape = static_cast<std::vector<int64_t>>(self.shape());
    shape.insert(shape.end(), self_shape.begin() + 1, self_shape.end());
    return shape;
}

static void assign_prepared_index_dim0(Tensor& self,
                                       const PreparedTensorIndex& prepared,
                                       py::object value) {
    if (prepared.scalar) {
        Tensor target = self.select(0, prepared.values.at(0));
        Tensor rhs = prepare_setitem_value(self, std::move(value),
                                           static_cast<std::vector<int64_t>>(target.shape()));
        tensorplay::tpx::ops::copy_(target, rhs);
        return;
    }

    const std::vector<int64_t> indexed_shape =
        indexed_result_shape_dim0(self, prepared);
    Tensor rhs = prepare_setitem_value(self, std::move(value), indexed_shape);

    // index_copy_ consumes a one-dimensional index and a source whose
    // indexed dimension is the index length.  Flatten only the advanced
    // index shape; the remaining tensor dimensions retain their layout.
    std::vector<int64_t> source_shape;
    source_shape.reserve(indexed_shape.size() - prepared.shape.size() + 1);
    source_shape.push_back(static_cast<int64_t>(prepared.values.size()));
    const auto self_shape = static_cast<std::vector<int64_t>>(self.shape());
    source_shape.insert(source_shape.end(), self_shape.begin() + 1, self_shape.end());
    rhs = rhs.reshape(source_shape).contiguous();
    tensorplay::tpx::ops::index_copy_(self, 0, prepared.flat, rhs);
}

// A tuple can contain more than one advanced index.  Applying those indices
// one at a time is observably wrong for non-adjacent indices (x[[0, 1], :,
// [1, 2]] must pair the two index vectors, rather than form a cartesian
// product).  Keep a small native index planner for this less common path;
// the one-dimensional top-level path above stays on index_select/index_copy_
// for DataLoader-sized batches.
enum class NativeIndexKind {
    Integer,
    Slice,
    Advanced,
    NewAxis,
};

struct NativeIndexComponent {
    NativeIndexKind kind = NativeIndexKind::Slice;
    int64_t input_dim = -1;
    int64_t integer = 0;
    int64_t start = 0;
    int64_t step = 1;
    int64_t length = 0;
    std::vector<int64_t> values;
    std::vector<int64_t> shape;
};

static int64_t checked_shape_numel(const std::vector<int64_t>& shape) {
    int64_t result = 1;
    for (const int64_t size : shape) {
        if (size < 0 || (size != 0 &&
                         result > std::numeric_limits<int64_t>::max() / size)) {
            TP_THROW(RuntimeError, "invalid or overflowing indexed shape");
        }
        result *= size;
    }
    return result;
}

static NativeIndexComponent make_full_index_component(const Tensor& self,
                                                      int64_t input_dim) {
    NativeIndexComponent component;
    component.kind = NativeIndexKind::Slice;
    component.input_dim = input_dim;
    component.length = self.size(input_dim);
    return component;
}

static NativeIndexComponent make_slice_index_component(const Tensor& self,
                                                       int64_t input_dim,
                                                       py::slice slice) {
    auto [start, stop, step, length] = compute_slice(slice, self.size(input_dim));
    NativeIndexComponent component;
    component.kind = NativeIndexKind::Slice;
    component.input_dim = input_dim;
    component.start = start;
    component.step = step;
    component.length = length;
    (void)stop;
    return component;
}

static NativeIndexComponent make_integer_index_component(const Tensor& self,
                                                         int64_t input_dim,
                                                         int64_t value) {
    input_dim = checked_index_dim(self, input_dim);
    const int64_t size = self.size(input_dim);
    if (value < 0) value += size;
    if (value < 0 || value >= size) TP_THROW(IndexError, "index out of range");

    NativeIndexComponent component;
    component.kind = NativeIndexKind::Integer;
    component.input_dim = input_dim;
    component.integer = value;
    return component;
}

static NativeIndexComponent make_advanced_index_component(
    int64_t input_dim, PreparedTensorIndex prepared) {
    NativeIndexComponent component;
    component.kind = NativeIndexKind::Advanced;
    component.input_dim = input_dim;
    component.values = std::move(prepared.values);
    component.shape = std::move(prepared.shape);
    return component;
}

static std::vector<NativeIndexComponent> make_bool_tensor_components(
    const Tensor& self, int64_t first_dim, const Tensor& raw_mask) {
    first_dim = checked_index_dim(self, first_dim);
    const int64_t mask_dim = raw_mask.dim();
    if (mask_dim == 0 || first_dim + mask_dim > self.dim()) {
        TP_THROW(IndexError, "The shape of the mask does not match the indexed tensor");
    }
    for (int64_t d = 0; d < mask_dim; ++d) {
        if (raw_mask.size(d) != self.size(first_dim + d)) {
            TP_THROW(IndexError, "The shape of the mask does not match the indexed tensor");
        }
    }

    Tensor host_mask = raw_mask;
    if (!host_mask.device().is_cpu()) {
        host_mask = host_mask.to(Device(DeviceType::CPU));
    }
    host_mask = host_mask.contiguous();
    const int64_t mask_numel = host_mask.numel();
    const bool* bool_mask = host_mask.dtype() == DType::Bool
                                ? host_mask.data_ptr<bool>()
                                : nullptr;
    const uint8_t* byte_mask = host_mask.dtype() == DType::UInt8
                                   ? host_mask.data_ptr<uint8_t>()
                                   : nullptr;
    std::vector<std::vector<int64_t>> coordinates(
        static_cast<size_t>(mask_dim));
    for (int64_t linear = 0; linear < mask_numel; ++linear) {
        if (!(bool_mask ? bool_mask[linear] : byte_mask[linear])) continue;
        int64_t remainder = linear;
        std::vector<int64_t> current(static_cast<size_t>(mask_dim));
        for (int64_t d = mask_dim - 1; d >= 0; --d) {
            const int64_t size = raw_mask.size(d);
            current[static_cast<size_t>(d)] = size == 0 ? 0 : remainder % size;
            if (size != 0) remainder /= size;
        }
        for (int64_t d = 0; d < mask_dim; ++d) {
            coordinates[static_cast<size_t>(d)].push_back(
                current[static_cast<size_t>(d)]);
        }
    }

    std::vector<NativeIndexComponent> components;
    components.reserve(static_cast<size_t>(mask_dim));
    for (int64_t d = 0; d < mask_dim; ++d) {
        NativeIndexComponent component;
        component.kind = NativeIndexKind::Advanced;
        component.input_dim = first_dim + d;
        component.values = std::move(coordinates[static_cast<size_t>(d)]);
        component.shape = {static_cast<int64_t>(component.values.size())};
        components.push_back(std::move(component));
    }
    return components;
}

static int64_t consumed_index_dims(py::handle index) {
    if (index.is_none() || index.ptr() == Py_Ellipsis) return 0;
    if (py::isinstance<py::bool_>(index)) return 0;
    if (py::isinstance<py::int_>(index) ||
        py::isinstance<py::slice>(index) ||
        py::isinstance<py::list>(index)) {
        return 1;
    }
    if (py::isinstance<Tensor>(index)) {
        Tensor tensor_index = py::cast<Tensor>(index);
        if (is_boolean_mask_dtype(tensor_index.dtype())) {
            // A scalar bool is a separate, zero-width newaxis; non-scalar
            // masks consume one input dimension per mask dimension.
            return tensor_index.dim();
        }
        if (isIntegralType(tensor_index.dtype(), /*includeBool=*/false)) return 1;
        TP_THROW(TypeError,
                 "tensors used as indices must be long, int, short, byte or bool tensors");
    }
    TP_THROW(TypeError, "Unsupported index type in tuple");
}

static std::vector<NativeIndexComponent> expand_native_index_tuple(
    const Tensor& self, py::tuple indices) {
    const int64_t ndim = self.dim();
    int64_t consumed = 0;
    int64_t ellipsis = -1;
    for (size_t i = 0; i < indices.size(); ++i) {
        py::handle index = indices[i];
        if (index.ptr() == Py_Ellipsis) {
            if (ellipsis >= 0) {
                TP_THROW(IndexError, "an index can only have a single ellipsis");
            }
            ellipsis = static_cast<int64_t>(i);
            continue;
        }
        consumed += consumed_index_dims(index);
    }
    if (consumed > ndim) TP_THROW(IndexError, "too many indices for tensor");

    const int64_t ellipsis_fill = ndim - consumed;
    std::vector<NativeIndexComponent> components;
    components.reserve(static_cast<size_t>(ndim + indices.size()));
    int64_t input_dim = 0;

    auto append_index = [&](py::handle index) {
        if (index.is_none()) {
            NativeIndexComponent component;
            component.kind = NativeIndexKind::NewAxis;
            component.length = 1;
            components.push_back(std::move(component));
            return;
        }
        if (py::isinstance<py::bool_>(index)) {
            NativeIndexComponent component;
            component.kind = NativeIndexKind::NewAxis;
            component.length = py::cast<bool>(index) ? 1 : 0;
            components.push_back(std::move(component));
            return;
        }
        if (py::isinstance<py::int_>(index)) {
            components.push_back(make_integer_index_component(
                self, input_dim++, py::cast<int64_t>(index)));
            return;
        }
        if (py::isinstance<py::slice>(index)) {
            components.push_back(make_slice_index_component(
                self, input_dim++, py::cast<py::slice>(index)));
            return;
        }
        if (py::isinstance<py::list>(index)) {
            PreparedTensorIndex prepared =
                is_python_bool_vector(index)
                    ? prepare_python_bool_index(self, input_dim, index)
                    : prepare_python_integer_index(self, input_dim, index);
            components.push_back(make_advanced_index_component(
                input_dim++, std::move(prepared)));
            return;
        }
        if (py::isinstance<Tensor>(index)) {
            Tensor tensor_index = py::cast<Tensor>(index);
            if (is_boolean_mask_dtype(tensor_index.dtype())) {
                if (tensor_index.dim() == 0) {
                    NativeIndexComponent component;
                    component.kind = NativeIndexKind::NewAxis;
                    component.length = tensor_index.item<bool>() ? 1 : 0;
                    components.push_back(std::move(component));
                    return;
                }
                std::vector<NativeIndexComponent> mask_components =
                    make_bool_tensor_components(self, input_dim, tensor_index);
                input_dim += tensor_index.dim();
                for (auto& component : mask_components) {
                    components.push_back(std::move(component));
                }
                return;
            }
            if (!isIntegralType(tensor_index.dtype(), /*includeBool=*/false)) {
                TP_THROW(TypeError,
                         "tensors used as indices must be long, int, short, byte or bool tensors");
            }
            PreparedTensorIndex prepared =
                prepare_integer_index(self, input_dim, tensor_index);
            if (prepared.scalar) {
                components.push_back(make_integer_index_component(
                    self, input_dim++, prepared.values.at(0)));
            } else {
                components.push_back(make_advanced_index_component(
                    input_dim++, std::move(prepared)));
            }
            return;
        }
        TP_THROW(TypeError, "Unsupported index type in tuple");
    };

    for (size_t i = 0; i < indices.size(); ++i) {
        py::handle index = indices[i];
        if (index.ptr() == Py_Ellipsis) {
            for (int64_t d = 0; d < ellipsis_fill; ++d) {
                components.push_back(make_full_index_component(self, input_dim++));
            }
        } else {
            append_index(index);
        }
    }
    while (input_dim < ndim) {
        components.push_back(make_full_index_component(self, input_dim++));
    }
    return components;
}

static bool tuple_needs_native_index_plan(py::tuple indices) {
    int64_t advanced_count = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        py::handle index = indices[i];
        if (index.is_none()) return true;
        if (py::isinstance<py::bool_>(index)) return true;
        if (py::isinstance<py::list>(index)) {
            ++advanced_count;
            continue;
        }
        if (py::isinstance<Tensor>(index)) {
            Tensor tensor_index = py::cast<Tensor>(index);
            if (is_boolean_mask_dtype(tensor_index.dtype())) {
                if (tensor_index.dim() == 0 || tensor_index.dim() > 1) return true;
                if (tensor_index.dim() == 1) ++advanced_count;
            } else if (isIntegralType(tensor_index.dtype(), /*includeBool=*/false)) {
                // A scalar integer tensor is a basic index, but sending it
                // through the native planner keeps tuple getitem/setitem on
                // the same path (the old setter only handled Python ints).
                if (tensor_index.dim() == 0) return true;
                ++advanced_count;
            }
        }
    }
    return advanced_count > 1;
}

static bool tuple_contains_advanced_index(py::tuple indices) {
    for (size_t i = 0; i < indices.size(); ++i) {
        py::handle index = indices[i];
        if (py::isinstance<py::list>(index)) return true;
        if (!py::isinstance<Tensor>(index)) continue;
        Tensor tensor_index = py::cast<Tensor>(index);
        if (is_boolean_mask_dtype(tensor_index.dtype())) {
            if (tensor_index.dim() > 0) return true;
        } else if (isIntegralType(tensor_index.dtype(), /*includeBool=*/false) &&
                   tensor_index.dim() > 0) {
            return true;
        }
    }
    return false;
}

struct NativeIndexPlan {
    std::vector<int64_t> output_shape;
    std::vector<int64_t> linear_indices;
};

static NativeIndexPlan build_native_index_plan(
    const Tensor& self, const std::vector<NativeIndexComponent>& components) {
    std::vector<size_t> advanced_positions;
    std::vector<int64_t> advanced_shape;
    for (size_t i = 0; i < components.size(); ++i) {
        if (components[i].kind != NativeIndexKind::Advanced) continue;
        advanced_positions.push_back(i);
        if (advanced_shape.empty()) {
            advanced_shape = components[i].shape;
        } else {
            advanced_shape = tensorplay::broadcast_shapes(
                advanced_shape, components[i].shape);
        }
    }

    NativeIndexPlan plan;
    const bool has_advanced = !advanced_positions.empty();
    const size_t first_advanced = has_advanced ? advanced_positions.front()
                                               : components.size();
    const size_t last_advanced = has_advanced ? advanced_positions.back()
                                              : components.size();
    const bool advanced_contiguous =
        !has_advanced || last_advanced - first_advanced + 1 == advanced_positions.size();
    int64_t advanced_output_start = 0;
    std::vector<int64_t> component_output_axis(components.size(), -1);

    auto append_basic_shape = [&](size_t component_index) {
        const auto& component = components[component_index];
        if (component.kind == NativeIndexKind::NewAxis) {
            component_output_axis[component_index] =
                static_cast<int64_t>(plan.output_shape.size());
            plan.output_shape.push_back(component.length);
        } else if (component.kind == NativeIndexKind::Slice) {
            component_output_axis[component_index] =
                static_cast<int64_t>(plan.output_shape.size());
            plan.output_shape.push_back(component.length);
        }
    };

    if (!has_advanced) {
        for (size_t i = 0; i < components.size(); ++i) {
            append_basic_shape(i);
        }
    } else if (!advanced_contiguous) {
        plan.output_shape.insert(plan.output_shape.end(),
                                 advanced_shape.begin(), advanced_shape.end());
        for (size_t i = 0; i < components.size(); ++i) {
            if (components[i].kind != NativeIndexKind::Advanced) {
                append_basic_shape(i);
            }
        }
    } else {
        for (size_t i = 0; i < components.size(); ++i) {
            if (i == first_advanced) {
                advanced_output_start =
                    static_cast<int64_t>(plan.output_shape.size());
                plan.output_shape.insert(plan.output_shape.end(),
                                         advanced_shape.begin(), advanced_shape.end());
            }
            if (components[i].kind != NativeIndexKind::Advanced) {
                append_basic_shape(i);
            }
        }
    }

    const int64_t output_numel = checked_shape_numel(plan.output_shape);
    plan.linear_indices.resize(static_cast<size_t>(output_numel));
    const int64_t advanced_rank = static_cast<int64_t>(advanced_shape.size());
    auto advanced_value = [&](const NativeIndexComponent& component,
                              const std::vector<int64_t>& output_coords) {
        int64_t offset = 0;
        const int64_t component_rank = static_cast<int64_t>(component.shape.size());
        for (int64_t d = 0; d < component_rank; ++d) {
            const int64_t output_dim = advanced_output_start + advanced_rank -
                                       component_rank + d;
            const int64_t coordinate =
                component.shape[static_cast<size_t>(d)] == 1
                    ? 0
                    : output_coords[static_cast<size_t>(output_dim)];
            offset = offset * component.shape[static_cast<size_t>(d)] + coordinate;
        }
        return component.values[static_cast<size_t>(offset)];
    };

    // The planner is used for advanced tuples only.  Decode the output in
    // row-major order, then map each coordinate back to one source element.
    // order for both contiguous and non-contiguous advanced groups.
    for (int64_t linear = 0; linear < output_numel; ++linear) {
        std::vector<int64_t> output_coords(plan.output_shape.size(), 0);
        int64_t remainder = linear;
        for (int64_t d = static_cast<int64_t>(plan.output_shape.size()) - 1;
             d >= 0; --d) {
            const int64_t size = plan.output_shape[static_cast<size_t>(d)];
            output_coords[static_cast<size_t>(d)] = size == 0 ? 0 : remainder % size;
            if (size != 0) remainder /= size;
        }

        int64_t source_linear = 0;
        for (const auto& component : components) {
            if (component.kind == NativeIndexKind::NewAxis) continue;
            int64_t coordinate = 0;
            if (component.kind == NativeIndexKind::Integer) {
                coordinate = component.integer;
            } else if (component.kind == NativeIndexKind::Slice) {
                // Find the output dimension assigned to this basic component.
                // NewAxis and integer components do not consume an output slot.
                // component_output_axis was filled in the same order as the
                // output shape above.
                const size_t component_index = static_cast<size_t>(
                    &component - components.data());
                const int64_t output_dim = component_output_axis[component_index];
                coordinate = component.start + component.step *
                    output_coords[static_cast<size_t>(output_dim)];
            } else {
                coordinate = advanced_value(component, output_coords);
            }
            source_linear = source_linear * self.size(component.input_dim) + coordinate;
        }
        plan.linear_indices[static_cast<size_t>(linear)] = source_linear;
    }
    return plan;
}

static Tensor apply_native_index_plan(const Tensor& self,
                                      const NativeIndexPlan& plan) {
    Tensor index = Tensor::tensor(plan.linear_indices, DType::Int64);
    if (!self.device().is_cpu()) index = index.to(self.device());
    Tensor flat = self.reshape({self.numel()});
    Tensor selected = tensorplay::tpx::ops::index_select(flat, 0, index);
    return tensorplay::tpx::ops::reshape(selected, plan.output_shape);
}

static void assign_native_index_plan(Tensor& self, const NativeIndexPlan& plan,
                                     py::object value) {
    Tensor rhs = prepare_setitem_value(self, std::move(value), plan.output_shape);
    rhs = rhs.reshape({static_cast<int64_t>(plan.linear_indices.size())})
              .contiguous()
              .clone();
    if (plan.linear_indices.empty()) return;

    Tensor index = Tensor::tensor(plan.linear_indices, DType::Int64);
    if (!self.device().is_cpu()) index = index.to(self.device());
    // A single index tensor is deliberately used here: index_put_'s backend
    // interprets one index as a linear offset, which is exactly what the
    // planner produced.  Cloning rhs makes overlapping assignments (e.g.
    // The backend's linear writer is contiguous-only, however.  Stage through
    // a contiguous clone for a transposed/sliced destination, then copy the
    // logical result back through TensorIterator so the view's strides are
    // honored.
    Tensor target = self.is_contiguous() ? self : self.contiguous();
    tensorplay::tpx::ops::index_put_(
        target, std::vector<Tensor>{index}, rhs, false);
    if (!self.is_contiguous()) {
        tensorplay::tpx::ops::copy_(self, target);
    }
}

static Tensor apply_python_bool_scalar_index(const Tensor& self, bool value) {
    Tensor with_index_dim = tensorplay::tpx::ops::unsqueeze(self, 0);
    if (value) return with_index_dim;
    return tensorplay::tpx::ops::slice(with_index_dim, 0, 0, 0, 1);
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
    // item -> native Python number: the generated fastcall binding boxes the
    // result into a tp.Scalar object which the _tensor.py wrapper then
    // unboxes per dtype (with a per-call `import builtins`).  Returning raw
    // both extra layers.
    m.def("item_python", [](const Tensor& t) -> py::object {
        const Scalar v = t.item();
        if (v.isBoolean()) return py::bool_(v.to<bool>());
        if (v.isComplex()) {
            const auto c = v.to<std::complex<double>>();
            return py::reinterpret_steal<py::object>(
                PyComplex_FromDoubles(c.real(), c.imag()));
        }
        if (v.dtype() == DType::UInt64) {
            return py::reinterpret_steal<py::object>(PyLong_FromUnsignedLongLong(
                static_cast<unsigned long long>(v.to<uint64_t>())));
        }
        if (v.isFloatingPoint()) return py::float_(v.to<double>());
        return py::int_(v.to<int64_t>());
    });

    // tensor_from_numpy(): zero-copy from_blob view; non-writable arrays warn
    // once instead of failing; byte-stride divisibility, negative strides and
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
        Tensor t = create_tensor(std::move(obj), std::nullopt, std::nullopt);
        return t;
    }, "ndarray"_a);

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
        .def(py::init([](py::object data, std::optional<DType> dtype, std::optional<Device> device, bool requires_grad) {
            Tensor t = create_tensor(data, dtype, device);
            tensorplay::tpx::impl::set_requires_grad(t, requires_grad);
            return t;
        }), "data"_a, "dtype"_a = py::none(), "device"_a = py::none(), "requires_grad"_a = false)
        
        // Properties
        .def_property_readonly("_impl_id", [](const Tensor& self) {
             return (uintptr_t)self.unsafeGetTensorImpl().get();
        })
        .def_property_readonly(
            "shape",
            [](const Tensor& self) {
                return py::reinterpret_steal<py::object>(Size_New(self.shape()));
            })
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("device", &Tensor::device)
        .def_property_readonly(
            "ndim", static_cast<int64_t (Tensor::*)() const>(&Tensor::dim))
        // Transposed view; identity on 0-d inputs (matching the reference
        // semantics of a trailing-dimension swap).
        .def_property_readonly("T", [](const Tensor& self) {
            if (self.dim() == 0) {
                return self;
            }
            return tensorplay::tpx::ops::transpose(self, -2, -1);
        })
        // Conjugated transposed view; equals .T for non-complex dtypes.
        .def_property_readonly("H", [](const Tensor& self) {
            if (self.dim() == 0) {
                return self;
            }
            Tensor base = isComplexType(self.dtype()) ? tensorplay::tpx::ops::conj(self) : self;
            return tensorplay::tpx::ops::transpose(base, -2, -1);
        })
        .def("dim", static_cast<int64_t (Tensor::*)() const>(&Tensor::dim))
        // Ops that return an optional gradient slot (convolution_backward and
        // friends) hand back an undefined tensor for the slots the caller did
        // not ask for; this is how Python tells that apart from an empty one.
        .def("defined", &Tensor::defined)
        .def("numel", static_cast<int64_t (Tensor::*)() const>(&Tensor::numel))
        .def("itemsize", &Tensor::itemsize)
        .def("is_contiguous", [](const Tensor& t) { return t.is_contiguous(); })
        .def("is_contiguous", [](const Tensor& t, int64_t format) {
             return t.is_contiguous(static_cast<tensorplay::MemoryFormat>(format));
        }, "memory_format"_a)
        .def("memory_format", [](const Tensor& t) {
             return static_cast<int64_t>(t.memory_format());
        })
        .def("is_channels_last", &Tensor::is_channels_last)
        .def("is_channels_last_2d", &Tensor::is_channels_last_2d)
        .def("is_channels_last_3d", &Tensor::is_channels_last_3d)
                        .def("is_complex", [](const Tensor& t) {
             return tensorplay::isComplexType(t.dtype());
        })
        .def("is_floating_point", [](const Tensor& t) {
             return tensorplay::isFloatingType(t.dtype());
        })
                .def_property_readonly("is_sparse", &Tensor::is_sparse)
        .def("is_sparse_csr", &Tensor::is_sparse_csr)
        .def("is_coalesced", &Tensor::is_coalesced)
        .def("sparse_dim", &Tensor::sparse_dim)
        .def("dense_dim", &Tensor::dense_dim)
        .def("_indices", &Tensor::_indices)
        .def("_values", &Tensor::_values)
        .def("_crow_indices", &Tensor::_crow_indices)
        .def("_col_indices", &Tensor::_col_indices)
        .def("values", [](const Tensor& t) { return t._values(); })
        .def("crow_indices", &Tensor::_crow_indices)
        .def("col_indices", &Tensor::_col_indices)
        // 2 = strided (dense). Compare against tensorplay.sparse_coo etc.
        .def_property_readonly("layout", [](const Tensor& t) -> int64_t {
            if (!t.is_sparse()) return 2;
            return t.is_sparse_csr() ? 1 : 0;
        })
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
        .def("_bump_version", [](Tensor& self) {
            // must advance the version counter so saved-tensor checks and
            // double-backward see the mutation.
            self.unsafeGetTensorImpl()->bump_version();
        })
        .def_property_readonly("_output_nr", [](const Tensor& self) {
            return tensorplay::tpx::impl::output_nr(self);
        })
        .def_property_readonly("_accumulate_grad_node", [](const Tensor& self) -> std::shared_ptr<tensorplay::tpx::Node> {
            // AccumulateGrad node so leaf-only APIs (post-accumulate-grad
            // hooks, DDP reducer) work before the first graph use. The meta
            // only keeps a weak cache reference; callers that need the node
            // to outlive this call (DDP) must hold the returned shared_ptr.
            // NB: get_or_create -- a fresh leaf param has no meta until its
            // first autograd-aware op, so a null meta must not short-circuit.
            auto* meta = tensorplay::tpx::impl::get_or_create_autograd_meta(self);
            if (meta == nullptr) return nullptr;
            if (auto acc = meta->grad_accumulator()) return acc;
            // Hold the strong ref locally and transfer it to the caller
            // handing the sole owning reference to set_grad_accumulator
            // would destroy the node before the weak lock below.
            std::shared_ptr<tensorplay::tpx::Node> acc =
                std::make_shared<tensorplay::tpx::AccumulateGrad>(self);
            meta->set_grad_accumulator(acc);
            return acc;
        })
        .def("element_size", [](const Tensor& self) -> int64_t {
            return static_cast<int64_t>(self.itemsize());
        })
        .def("nbytes", [](const Tensor& self) -> int64_t {
            return static_cast<int64_t>(self.numel() * self.itemsize());
        })
        .def("storage_offset", [](const Tensor& self) -> int64_t {
            return static_cast<int64_t>(self.unsafeGetTensorImpl()->storage_offset());
        })
        .def("get_device", [](const Tensor& self) -> int64_t {
            const auto dev = self.device();
            return dev.is_cuda() || dev.type() != DeviceType::CPU ? dev.index() : -1;
        })
        .def("type_as", [](const Tensor& self, const Tensor& other) {
            if (self.dtype() == other.dtype()) return self;
            return self.to(other.dtype());
        }, py::arg("other"))
        .def("set_", [](Tensor& self, const Tensor& source,
                        std::optional<int64_t> storage_offset,
                        std::optional<std::vector<int64_t>> size,
                        std::optional<std::vector<int64_t>> stride) -> Tensor& {
            // storage.  Autograd metadata on the impl stays untouched.
            auto impl = self.unsafeGetTensorImpl();
            const auto& src_impl = *source.unsafeGetTensorImpl();
            std::vector<int64_t> ns = size.has_value()
                ? *size
                : static_cast<std::vector<int64_t>>(source.shape());
            std::vector<int64_t> nst = stride.has_value()
                ? *stride
                : (size.has_value()
                       ? std::vector<int64_t>{}
                       : static_cast<std::vector<int64_t>>(source.strides()));
            if (nst.empty() && !ns.empty()) {
                // sizes given without strides: fresh contiguous strides
                nst.assign(ns.size(), 1);
                for (int i = static_cast<int>(ns.size()) - 2; i >= 0; --i)
                    nst[i] = nst[i + 1] * ns[i + 1];
            }
            impl->set_storage(src_impl.storage());
            impl->set_sizes_and_strides(ns, nst);
            impl->set_storage_offset(storage_offset.has_value()
                ? static_cast<size_t>(*storage_offset)
                : src_impl.storage_offset());
            return self;
        },
             py::arg("source"), py::arg("storage_offset").none(true) = py::none(),
             py::arg("size").none(true) = py::none(),
             py::arg("stride").none(true) = py::none())
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
                // Route through tpx::impl::set_grad so autograd metadata is
                // lazily created -- assigning .grad to a leaf that never
                if (grad) {
                    tensorplay::tpx::impl::set_grad(self, *grad);
                } else {
                    tensorplay::tpx::impl::set_grad(self, Tensor());
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
            [](const Tensor& self) {
                // version counter, so in-place writes through it stay
                // invisible to mutation tracking on the original tensor.
                Tensor out = self.detach();
                out.unsafeGetTensorImpl()->set_version_counter(tensorplay::VariableVersion());
                return out;
            },
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
        .def("detach", static_cast<Tensor (Tensor::*)() const>(&Tensor::detach))
        .def("_is_view", [](const Tensor& self) {
            return self.defined() && self.unsafeGetTensorImpl()->is_view();
        })
        .def("detach_", [](py::object self_obj) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            // legal on non-views (a view's impl is shared with its base, so
            // stripping its autograd edge in place would corrupt the base's
            // graph bookkeeping).
            if (self.defined() && self.unsafeGetTensorImpl()->is_view()) {
                TP_THROW(RuntimeError,
                    "Can't detach views in-place. Use detach() instead. "
                    "If you are using DistributedDataParallel (DDP) for training, "
                    "and gradient_as_bucket_view is set as True, gradients are "
                    "views of DDP buckets, and hence detach_() cannot be called "
                    "on these gradients. To fix this error, call "
                    "Optimizer.zero_grad() as the solution.");
            }
            tensorplay::tpx::impl::set_requires_grad(self, false);
            tensorplay::tpx::impl::set_grad_fn(self, nullptr, 0);
            return self_obj;
        })
                .def("requires_grad_", [](py::object self_obj, bool requires_grad) {
            Tensor& self = py::cast<Tensor&>(self_obj);
            tensorplay::tpx::impl::set_requires_grad(self, requires_grad);
            return self_obj;
        }, "requires_grad"_a = true)
        
        // Methods
        .def("size", [](const Tensor& self) {
            return py::reinterpret_steal<py::object>(Size_New(self.shape()));
        })
        .def("size", [](const Tensor& self, int64_t dim) {
            return self.size(dim);
        })
        // expand: served by the generated METH_FASTCALL layer (dispatcher op).
        // ([8, 1] / Size / tuple), -1 inference, and a dtype reinterpret.
        // Route through tpx::ops::view (NOT Tensor::view): the generated
        // wrapper records ViewBackward; the raw method silently detaches.
        .def("view", [](const Tensor& self, py::args args) -> Tensor {
            if (args.size() == 1) {
                py::object spec = args[0];
                if (py::isinstance<DType>(spec)) {
                    return self.view_dtype(spec.cast<DType>());
                }
                try {
                    return tensorplay::tpx::ops::view(self, spec.cast<std::vector<int64_t>>());
                } catch (const py::cast_error&) {
                    // fall through to per-arg ints below (e.g. numpy scalars)
                }
            }
            std::vector<int64_t> shape;
            shape.reserve(args.size());
            for (auto a : args) shape.push_back(a.cast<int64_t>());
            return tensorplay::tpx::ops::view(self, shape);
        })
        // (CompositeImplicitAutograd -> reshape(other.shape)).
        .def("reshape_as", [](const Tensor& self, const Tensor& other) -> Tensor {
            return tensorplay::tpx::ops::reshape(
                self, static_cast<std::vector<int64_t>>(other.shape()));
        })
        .def("as_strided", [](const Tensor& self,
                              const std::vector<int64_t>& size,
                              const std::vector<int64_t>& stride,
                              std::optional<int64_t> storage_offset) {
            return tensorplay::tpx::as_strided(self, size, stride, storage_offset);
        }, "size"_a, "stride"_a, "storage_offset"_a = py::none())
                                                        // In-place random sampling
                                                                        


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
                     t_cpu = t_cpu.contiguous();
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

        // binding covers the list form, so add the scalar-int overloads here.
        tensor                        
                        
                        
                        
                        
                
                        
                        
                        
                        
                        
                        
                        
                        
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
            // to .numpy() so both share one conversion path (grad/device/
            // reduced-dtype handling included), then optionally cast.
            py::object arr = self_obj.attr("numpy")();
            if (!dtype.is_none()) {
                return arr.attr("astype")(dtype, "copy"_a = false);
            }
            if (copy) {
                arr = arr.attr("copy")();
            }
            return arr;
        }, "dtype"_a = py::none(), "copy"_a = false)

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

        .def("__iter__", [](const Tensor& self) {
            if (self.dim() == 0) {
                TP_THROW(TypeError, "iteration over a 0-d tensor");
            }
            return py::iter(py::cast(tensorplay::tpx::ops::unbind(self, 0)));
        })

        .def("__getitem__", [](const Tensor& self, py::object index) -> Tensor {
            if (py::isinstance<Tensor>(index)) {
                Tensor idx = py::cast<Tensor>(index);
                if (is_boolean_mask_dtype(idx.dtype())) {
                    if (idx.dim() == 0) {
                        return apply_python_bool_scalar_index(self, idx.item<bool>());
                    }
                    std::vector<NativeIndexComponent> components =
                        make_bool_tensor_components(self, 0, idx);
                    for (int64_t dim = idx.dim(); dim < self.dim(); ++dim) {
                        components.push_back(make_full_index_component(self, dim));
                    }
                    return apply_native_index_plan(
                        self, build_native_index_plan(self, components));
                }
                if (isIntegralType(idx.dtype(), /*includeBool=*/false)) {
                    return apply_prepared_index(
                        self, 0, prepare_integer_index(self, 0, idx));
                }
                TP_THROW(TypeError,
                         "tensors used as indices must be long, int, short, byte or bool tensors");
            } else if (py::isinstance<py::list>(index)) {
                PreparedTensorIndex prepared =
                    is_python_bool_vector(index)
                        ? prepare_python_bool_index(self, 0, index)
                        : prepare_python_integer_index(self, 0, index);
                return apply_prepared_index(self, 0, prepared);
            }

            if (py::isinstance<py::tuple>(index)) {
                 py::tuple indices = py::cast<py::tuple>(index);
                 if (tuple_needs_native_index_plan(indices)) {
                     const auto components = expand_native_index_tuple(self, indices);
                     return apply_native_index_plan(
                         self, build_native_index_plan(self, components));
                 }
                 Tensor result = self;
                 bool ellipsis_seen = false;
                 int64_t target_dim = 0;
                 for (size_t i = 0; i < indices.size(); ++i) {
                     py::object idx = indices[i];
                     if (py::isinstance<py::int_>(idx)) {
                         int64_t val = py::cast<int64_t>(idx);
                         // Route indexing through the autograd-aware wrapper;
                         // calling the raw Tensor view would sever gradients
                         // for common RoPE/decoder slicing patterns.
                         result = tensorplay::tpx::ops::select(result, target_dim, val);
                     } else if (py::isinstance<py::slice>(idx)) {
                         py::slice s = py::cast<py::slice>(idx);
                         auto [start, stop, step, slicelength] = compute_slice(s, result.size(target_dim));
                         result = tensorplay::tpx::ops::slice(result, target_dim, start, stop, step);
                         target_dim++;
                     } else if (py::isinstance<py::list>(idx)) {
                         PreparedTensorIndex prepared =
                             is_python_bool_vector(idx)
                                 ? prepare_python_bool_index(result, target_dim, idx)
                                 : prepare_python_integer_index(result, target_dim, idx);
                         result = apply_prepared_index(result, target_dim, prepared);
                         if (!prepared.scalar) {
                             target_dim += static_cast<int64_t>(prepared.shape.size());
                         }
                     } else if (py::isinstance<Tensor>(idx)) {
                         Tensor tensor_index = py::cast<Tensor>(idx);
                         PreparedTensorIndex prepared;
                         if (is_boolean_mask_dtype(tensor_index.dtype())) {
                             prepared = prepare_bool_tensor_index(result, target_dim,
                                                                   tensor_index);
                         } else if (isIntegralType(tensor_index.dtype(),
                                                   /*includeBool=*/false)) {
                             prepared = prepare_integer_index(result, target_dim,
                                                              tensor_index);
                         } else {
                             TP_THROW(TypeError,
                                      "tensors used as indices must be long, int, short, byte or bool tensors");
                         }
                         result = apply_prepared_index(result, target_dim, prepared);
                         if (!prepared.scalar) {
                             target_dim += static_cast<int64_t>(prepared.shape.size());
                         }
                      } else if (idx.ptr() == Py_Ellipsis) {
                         // every dimension not covered by the other indices.
                         if (ellipsis_seen) {
                             TP_THROW(IndexError, "an index can only have a single ellipsis");
                         }
                         ellipsis_seen = true;
                         const int64_t remaining =
                             static_cast<int64_t>(indices.size() - i - 1);
                         const int64_t absorbed = result.dim() - target_dim - remaining;
                         if (absorbed < 0) {
                             TP_THROW(IndexError, "too many indices for tensor");
                         }
                         target_dim += absorbed;
                      } else {
                         TP_THROW(TypeError, "Unsupported index type in tuple");
                     }
                 }
                 return result;
            } else if (py::isinstance<py::bool_>(index)) {
                return apply_python_bool_scalar_index(
                    self, py::cast<bool>(index));
            } else if (py::isinstance<py::int_>(index)) {
                return tensorplay::tpx::ops::select(self, 0, py::cast<int64_t>(index));
            } else if (py::isinstance<py::slice>(index)) {
                py::slice s = py::cast<py::slice>(index);
                auto [start, stop, step, slicelength] = compute_slice(s, self.size(0));
                return tensorplay::tpx::ops::slice(self, 0, start, stop, step);
            }
            TP_THROW(TypeError, "Unsupported index type");
        })
        .def("__setitem__", [](Tensor& self, py::object index, py::object value) {
            Tensor target;
            if (py::isinstance<py::tuple>(index)) {
                 py::tuple indices = py::cast<py::tuple>(index);
                 if (tuple_contains_advanced_index(indices) ||
                     tuple_needs_native_index_plan(indices)) {
                     const auto components = expand_native_index_tuple(self, indices);
                     assign_native_index_plan(
                         self, build_native_index_plan(self, components),
                         std::move(value));
                     return;
                 }
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
                      } else if (idx.ptr() == Py_Ellipsis) {
                          int64_t absorbed = static_cast<int64_t>(self.dim())
                                             - static_cast<int64_t>(indices.size()) + 1;
                          if (absorbed > 0) target_dim += absorbed;
                      } else {
                          TP_THROW(TypeError, "Unsupported index type in tuple");
                      }
                  }
            } else if (index.ptr() == Py_Ellipsis) {
                target = self;
            } else if (py::isinstance<py::list>(index) ||
                       py::isinstance<Tensor>(index)) {
                PreparedTensorIndex prepared;
                if (py::isinstance<Tensor>(index)) {
                    Tensor tensor_index = py::cast<Tensor>(index);
                    if (is_boolean_mask_dtype(tensor_index.dtype())) {
                        if (tensor_index.dim() == 0) {
                            if (!tensor_index.item<bool>()) return;
                            Tensor rhs = prepare_setitem_value(
                                self, std::move(value),
                                static_cast<std::vector<int64_t>>(self.shape()));
                            tensorplay::tpx::ops::copy_(self, rhs);
                            return;
                        }
                        std::vector<NativeIndexComponent> components =
                            make_bool_tensor_components(self, 0, tensor_index);
                        for (int64_t dim = tensor_index.dim(); dim < self.dim(); ++dim) {
                            components.push_back(make_full_index_component(self, dim));
                        }
                        assign_native_index_plan(
                            self, build_native_index_plan(self, components),
                            std::move(value));
                        return;
                    }
                    prepared = prepare_integer_index(self, 0, tensor_index);
                } else {
                    prepared = is_python_bool_vector(index)
                                   ? prepare_python_bool_index(self, 0, index)
                                   : prepare_python_integer_index(self, 0, index);
                }
                if (prepared.scalar) {
                    target = self.select(0, prepared.values.at(0));
                } else {
                    assign_prepared_index_dim0(self, prepared, std::move(value));
                    return;
                }
            } else if (py::isinstance<py::bool_>(index)) {
                if (!py::cast<bool>(index)) return;
                Tensor rhs = prepare_setitem_value(
                    self, std::move(value),
                    static_cast<std::vector<int64_t>>(self.shape()));
                tensorplay::tpx::ops::copy_(self, rhs);
                return;
            } else if (py::isinstance<py::int_>(index)) {
                target = tensorplay::tpx::ops::select(self, 0, py::cast<int64_t>(index));
            } else if (py::isinstance<py::slice>(index)) {
                py::slice s = py::cast<py::slice>(index);
                auto [start, stop, step, slicelength] = compute_slice(s, self.size(0));
                target = tensorplay::tpx::ops::slice(self, 0, start, stop, step);
            } else {
                TP_THROW(TypeError, "Unsupported index type");
            }

            Tensor rhs = prepare_setitem_value(
                self, std::move(value),
                static_cast<std::vector<int64_t>>(target.shape()));
            tensorplay::tpx::ops::copy_(target, rhs);
        })
        
        // Operators
        .def("__neg__", [](const Tensor& t) { return tensorplay::tpx::ops::neg(t); })
        .def("__add__", [](const Tensor& a, const Tensor& b) { return tensorplay::tpx::ops::add(a, b); })
        .def("__sub__", [](const Tensor& a, const Tensor& b) { return tensorplay::tpx::ops::sub(a, b); })
        .def("__mul__", [](const Tensor& a, const Tensor& b) { return tensorplay::tpx::ops::mul(a, b); })
        .def("__truediv__", [](const Tensor& a, const Tensor& b) { return tensorplay::tpx::ops::div(a, b); })
        // as int64 scalars); the double overloads below handle real scalars.
        .def("__add__", [](const Tensor& t, int64_t s) { return tensorplay::tpx::ops::add(t, Scalar(s)); })
        .def("__sub__", [](const Tensor& t, int64_t s) { return tensorplay::tpx::ops::sub(t, Scalar(s)); })
        .def("__mul__", [](const Tensor& t, int64_t s) { return tensorplay::tpx::ops::mul(t, Scalar(s)); })
        .def("__add__", [](const Tensor& t, double s) { return tensorplay::tpx::ops::add(t, Scalar(s)); })
        .def("__sub__", [](const Tensor& t, double s) { return tensorplay::tpx::ops::sub(t, Scalar(s)); })
        .def("__mul__", [](const Tensor& t, double s) { return tensorplay::tpx::ops::mul(t, Scalar(s)); })
        .def("__truediv__", [](const Tensor& t, double s) { return tensorplay::tpx::ops::div(t, Scalar(s)); })
        // the weak-scalar promotion rules in the kernels.
        .def("__add__", [](const Tensor& t, std::complex<double> s) { return tensorplay::tpx::ops::add(t, Scalar(s)); })
        .def("__sub__", [](const Tensor& t, std::complex<double> s) { return tensorplay::tpx::ops::sub(t, Scalar(s)); })
        .def("__mul__", [](const Tensor& t, std::complex<double> s) { return tensorplay::tpx::ops::mul(t, Scalar(s)); })
        .def("__truediv__", [](const Tensor& t, std::complex<double> s) { return tensorplay::tpx::ops::div(t, Scalar(s)); })
        .def("__radd__", [](const Tensor& t, int64_t s) { return tensorplay::tpx::ops::add(t, Scalar(s)); })
        .def("__rmul__", [](const Tensor& t, int64_t s) { return tensorplay::tpx::ops::mul(t, Scalar(s)); })
        .def("__radd__", [](const Tensor& t, double s) { return tensorplay::tpx::ops::add(t, Scalar(s)); })
        .def("__rsub__", [](const Tensor& t, double s) {
            Tensor s_t = Tensor::full({}, Scalar(s), t.dtype(), t.device());
            return tensorplay::tpx::ops::sub(s_t, t);
        })
        .def("__rmul__", [](const Tensor& t, double s) { return tensorplay::tpx::ops::mul(t, Scalar(s)); })
        .def("__radd__", [](const Tensor& t, std::complex<double> s) { return tensorplay::tpx::ops::add(t, Scalar(s)); })
        // Weak-scalar rule for the reflected operand's storage width:
        // complex tensor keeps its dtype; float32 -> complex64,
        // float64 -> complex128; integral -> complex64.
        .def("__rsub__", [](const Tensor& t, std::complex<double> s) {
            DType sdt = isComplexType(t.dtype())
                ? t.dtype()
                : (isFloatingType(t.dtype()) ? promoteTypes(toComplexType(t.dtype()), DType::ComplexFloat)
                                             : DType::ComplexFloat);
            Tensor s_t = Tensor::full({}, Scalar(s), sdt, t.device());
            return tensorplay::tpx::ops::sub(s_t, t);
        })
        .def("__rmul__", [](const Tensor& t, std::complex<double> s) { return tensorplay::tpx::ops::mul(t, Scalar(s)); })
        // NOTE: the double overload must precede the std::complex<double> one.
        // pybind tries overloads in registration order and a python float is
        // convertible to complex<double>; otherwise every `1.0 / real_tensor`
        // silently promoted to complex (broke linalg.pinv et al).
        .def("__rtruediv__", [](const Tensor& t, double s) {
            Tensor s_t = Tensor::full({}, Scalar(s), t.dtype(), t.device());
            return tensorplay::tpx::ops::div(s_t, t);
        })
        .def("__rtruediv__", [](const Tensor& t, std::complex<double> s) {
            DType sdt = isComplexType(t.dtype())
                ? t.dtype()
                : (isFloatingType(t.dtype()) ? promoteTypes(toComplexType(t.dtype()), DType::ComplexFloat)
                                             : DType::ComplexFloat);
            Tensor s_t = Tensor::full({}, Scalar(s), sdt, t.device());
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
                                                                                                                .def("__pow__", [](const Tensor& self, Scalar exponent) { return tensorplay::tpx::ops::pow(self, exponent); }, "exponent"_a)
        .def("__pow__", [](const Tensor& self, const Tensor& exponent) { return tensorplay::tpx::ops::pow(self, exponent); }, "exponent"_a)
        .def("__rpow__", [](const Tensor& self, Scalar base) {
            Tensor base_t = Tensor::full({}, base, self.dtype(), self.device());
            return tensorplay::tpx::ops::pow(base_t, self);
        })        // DLPack
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
            
            Tensor contig = self.is_contiguous() ? self : self.contiguous();
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
        .def("__str__", &Tensor::toString)
        // pybind's default always-true object truthiness (which made a 0-d
        // Bool tensor bool(t) == True even when t.item() == False).
        .def("__bool__", [](const Tensor& self) -> bool {
            const int64_t n = self.numel();
            if (n == 0)
                TP_THROW(RuntimeError, "Boolean value of Tensor with no values is ambiguous");
            if (n != 1)
                TP_THROW(RuntimeError, "Boolean value of Tensor with more than one value is ambiguous");
            return self.item().to<bool>();
        })
        // Scalar/float conversion: Tensor::item() has the C++ side; without
        // this, float(t) on the raw extension type raises TypeError.
        .def("__float__", [](const Tensor& self) { return self.item().to<double>(); })
        .def("__int__", [](const Tensor& self) { return static_cast<int64_t>(self.item().to<double>()); })
        // len(t) == t.size(0); a 0-d tensor has no length.
        .def("__len__", [](const Tensor& self) -> int64_t {
            if (self.dim() == 0)
                TP_THROW(TypeError, "len() of a 0-d tensor");
            return self.size(0);
        });

    // FASTCALL method layer goes in LAST: it fills names nothing above
    // bound, and must never shadow a hand-written pybind overload (e.g.
    // sum's scalar/list forms).
    if (tensorplay::python_c::register_generated_cpython_methods(tensor.ptr()) != 0) {
        throw py::error_already_set();
    }
}

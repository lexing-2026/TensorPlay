#include "python_bindings.h"
#include "tensorplay/ops/Config.h"
#include "Context.h"
#include "Device.h" // For Device class and cuda namespace declarations

#ifdef USE_CUDA
#include "CUDARuntime.h"
#include <cuda_runtime.h>

struct CudaDeviceProperties {
    std::string name;
    int major;
    int minor;
    size_t total_memory;
    int multi_processor_count;
};
#endif

void init_device(py::module_& m) {
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA);

    py::class_<Device>(m, "Device")
        .def(py::init<DeviceType, int64_t>(), "type"_a, "index"_a = -1)
        .def(py::init<const std::string&>(), "device"_a)
        .def(py::init<const std::string&, int64_t>(), "type"_a, "index"_a)
        .def_property_readonly("type", [](const Device& d) {
            std::string s = d.toString();
            size_t colon = s.find(':');
            if (colon != std::string::npos) {
                return s.substr(0, colon);
            }
            return s;
        })
        .def_property_readonly("index", &Device::index)
        .def("is_cpu", &Device::is_cpu)
        .def("is_cuda", &Device::is_cuda)
        .def("__repr__", &Device::toString)
        .def("__str__", &Device::toString)
        .def(py::self == py::self)
        .def(py::self != py::self)
        // Device is used as a dict key by e.g. the optimizer's
        // _group_tensors_by_device_and_dtype; __eq__ without __hash__ would
        // make it unhashable.
        .def("__hash__", [](const Device& d) {
            return std::hash<std::string>()(d.toString());
        })
        // Mirrors torch/csrc/Device.cpp THPDevice_enter/exit: entering a
        // device object scopes the default device for factory functions, so
        // `with tensorplay.device('cuda'):` allocates on that device.
        .def("__enter__", [](py::object self) {
            tensorplay::globalContext().pushDefaultDevice(self.cast<const Device&>());
            return self;
        })
        .def("__exit__", [](py::object /*self*/, const py::object&, const py::object&, const py::object&) {
            tensorplay::globalContext().popDefaultDevice();
            return py::bool_(false);  // never suppress exceptions
        });

    py::implicitly_convertible<std::string, Device>();
        
    // CUDA submodule
    py::module_ cuda = m.def_submodule("_cuda", "CUDA computation backend");
    
#ifdef USE_CUDA
    py::class_<CudaDeviceProperties>(cuda, "_CudaDeviceProperties")
        .def_readonly("name", &CudaDeviceProperties::name)
        .def_readonly("major", &CudaDeviceProperties::major)
        .def_readonly("minor", &CudaDeviceProperties::minor)
        .def_readonly("total_memory", &CudaDeviceProperties::total_memory)
        .def_readonly("multi_processor_count", &CudaDeviceProperties::multi_processor_count)
        .def("__repr__", [](const CudaDeviceProperties& p) {
            return "_CudaDeviceProperties(name='" + p.name + "', major=" + std::to_string(p.major) + ", minor=" + std::to_string(p.minor) + ", total_memory=" + std::to_string(p.total_memory) + ", multi_processor_count=" + std::to_string(p.multi_processor_count) + ")";
        });

    py::class_<tensorplay::cuda::CUDAEvent>(cuda, "_CudaEvent")
        .def(py::init<bool, bool, bool>(),
             "enable_timing"_a = false, "blocking"_a = false,
             "interprocess"_a = false)
        .def("record", [](tensorplay::cuda::CUDAEvent& event,
                           const std::optional<tensorplay::cuda::CUDAStream>& stream) {
            if (stream) event.record(*stream);
            else event.record();
        }, "stream"_a = py::none())
        .def("wait", [](const tensorplay::cuda::CUDAEvent& event,
                         const std::optional<tensorplay::cuda::CUDAStream>& stream) {
            event.block(stream.value_or(tensorplay::cuda::getCurrentCUDAStream()));
        }, "stream"_a = py::none())
        .def("query", &tensorplay::cuda::CUDAEvent::query)
        .def("synchronize", &tensorplay::cuda::CUDAEvent::synchronize,
             py::call_guard<py::gil_scoped_release>())
        .def("elapsed_time", &tensorplay::cuda::CUDAEvent::elapsed_time, "end_event"_a)
        .def_property_readonly("device", [](const tensorplay::cuda::CUDAEvent& event) -> py::object {
            if (event.device_index() < 0) return py::none();
            return py::cast(Device(DeviceType::CUDA, event.device_index()));
        })
        .def_property_readonly("cuda_event", &tensorplay::cuda::CUDAEvent::id)
        .def("__repr__", [](const tensorplay::cuda::CUDAEvent& event) {
            return "<tensorplay.cuda.Event device=" +
                   (event.device_index() < 0 ? std::string("None")
                                             : std::to_string(event.device_index())) + ">";
        });

    py::class_<tensorplay::cuda::CUDAStream>(cuda, "_CudaStream")
        .def(py::init([](int device, int priority) {
            return tensorplay::cuda::getStreamFromPool(priority, device);
        }), "device"_a = -1, "priority"_a = 0)
        .def_property_readonly("device", [](const tensorplay::cuda::CUDAStream& stream) {
            return stream.device();
        })
        .def_property_readonly("device_index", &tensorplay::cuda::CUDAStream::device_index)
        .def_property_readonly("cuda_stream", &tensorplay::cuda::CUDAStream::id)
        .def_property_readonly("priority", &tensorplay::cuda::CUDAStream::priority)
        .def("query", &tensorplay::cuda::CUDAStream::query)
        .def("synchronize", &tensorplay::cuda::CUDAStream::synchronize,
             py::call_guard<py::gil_scoped_release>())
        .def("wait_event", [](const tensorplay::cuda::CUDAStream& stream,
                              const tensorplay::cuda::CUDAEvent& event) {
            event.block(stream);
        }, "event"_a)
        .def("wait_stream", [](const tensorplay::cuda::CUDAStream& stream,
                               const tensorplay::cuda::CUDAStream& other) {
            tensorplay::cuda::CUDAEvent event;
            event.record(other);
            event.block(stream);
        }, "stream"_a)
        .def("record_event", [](const tensorplay::cuda::CUDAStream& stream,
                                std::optional<tensorplay::cuda::CUDAEvent> event) {
            tensorplay::cuda::CUDAEvent result = event.value_or(tensorplay::cuda::CUDAEvent());
            result.record(stream);
            return result;
        }, "event"_a = py::none())
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__repr__", [](const tensorplay::cuda::CUDAStream& stream) {
            return "<tensorplay.cuda.Stream device=cuda:" +
                   std::to_string(stream.device_index()) + " cuda_stream=" +
                   std::to_string(stream.id()) + ">";
        });
#endif

    cuda.def("get_version", []() {
#ifdef USE_CUDA
        int ver = 0;
        cudaError_t err = cudaRuntimeGetVersion(&ver);
        if (err != cudaSuccess) return 0;
        return ver;
#else
        return 0;
#endif
    });

    cuda.def("get_driver_version", []() {
#ifdef USE_CUDA
        int ver = 0;
        cudaError_t err = cudaDriverGetVersion(&ver);
        if (err != cudaSuccess) return 0;
        return ver;
#else
        return 0;
#endif
    });

    cuda.def("is_available", []() {
#ifdef USE_CUDA
        int count = 0;
        cudaError_t error = cudaGetDeviceCount(&count);
        if (error != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        return count > 0;
#else
        return false;
#endif
    });

    cuda.def("device_count", []() {
#ifdef USE_CUDA
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess) return 0;
        return count;
#else
        return 0;
#endif
    });

    cuda.def("current_device", []() {
#ifdef USE_CUDA
        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
        }
        return device;
#else
        throw std::runtime_error("CUDA is not available");
#endif
    });

    cuda.def("set_device", [](int device) {
#ifdef USE_CUDA
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) {
             throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
        }
#else
        throw std::runtime_error("CUDA is not available");
#endif
    }, "device"_a);

    cuda.def("get_device_name", [](int device) {
#ifdef USE_CUDA
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
        }
        return std::string(prop.name);
#else
        throw std::runtime_error("CUDA is not available");
#endif
    }, "device"_a = 0);

    cuda.def("get_device_capability", [](int device) {
#ifdef USE_CUDA
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
        }
        return std::make_pair(prop.major, prop.minor);
#else
        throw std::runtime_error("CUDA is not available");
#endif
    }, "device"_a = 0);

    cuda.def("get_device_properties", [](int device) {
#ifdef USE_CUDA
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
        }
        CudaDeviceProperties p;
        p.name = prop.name;
        p.major = prop.major;
        p.minor = prop.minor;
        p.total_memory = prop.totalGlobalMem;
        p.multi_processor_count = prop.multiProcessorCount;
        return p;
#else
        throw std::runtime_error("CUDA is not available");
#endif
    }, "device"_a = 0);

    cuda.def("synchronize", [](int device) {
#ifdef USE_CUDA
        tensorplay::cuda::CUDAGuard guard(device);
        tensorplay::cuda::checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
#else
        throw std::runtime_error("CUDA is not available");
#endif
    }, "device"_a = -1);

    // Memory functions
    cuda.def("memory_allocated", [](int device) {
#ifdef USE_CUDA
        return tensorplay::cuda::memory_allocated(device);
#else
        return 0;
#endif
    }, "device"_a = -1);

    cuda.def("memory_reserved", [](int device) {
#ifdef USE_CUDA
        return tensorplay::cuda::memory_reserved(device);
#else
        return 0;
#endif
    }, "device"_a = -1);

    cuda.def("max_memory_allocated", [](int device) {
#ifdef USE_CUDA
        return tensorplay::cuda::max_memory_allocated(device);
#else
        return 0;
#endif
    }, "device"_a = -1);

    cuda.def("max_memory_reserved", [](int device) {
#ifdef USE_CUDA
        return tensorplay::cuda::max_memory_reserved(device);
#else
        return 0;
#endif
    }, "device"_a = -1);

    cuda.def("memory_stats", [](int device) -> py::dict {
        py::dict out;
#ifdef USE_CUDA
        for (const auto& [key, value] :
             tensorplay::cuda::memory_stats(device)) {
            out[key.c_str()] = py::cast(value);
        }
#endif
        return out;
    }, "device"_a = -1,
       "Fragmentation-aware allocator accounting (allocated/reserved/peaks, "
       "segment and free-block counts, largest free block, pending bytes, "
       "graph pools)");

    cuda.def("reset_max_memory_allocated", [](int device) {
#ifdef USE_CUDA
        tensorplay::cuda::reset_max_memory_allocated(device);
#endif
    }, "device"_a = -1);

    cuda.def("reset_peak_memory_stats", [](int device) {
#ifdef USE_CUDA
        tensorplay::cuda::reset_peak_memory_stats(device);
#endif
    }, "device"_a = -1);
    
    cuda.def("empty_cache", []() {
#ifdef USE_CUDA
        tensorplay::cuda::empty_cache();
#endif
    });

    cuda.def("manual_seed", [](uint64_t seed) {
#ifdef USE_CUDA
        tensorplay::cuda::manual_seed(seed);
#endif
    }, "seed"_a);

    cuda.def("manual_seed_all", [](uint64_t seed) {
#ifdef USE_CUDA
        tensorplay::cuda::manual_seed_all(seed);
#endif
    }, "seed"_a);

#ifdef USE_CUDA
    cuda.def("current_stream", [](int device) {
        return tensorplay::cuda::getCurrentCUDAStream(device);
    }, "device"_a = -1);

    cuda.def("default_stream", [](int device) {
        return tensorplay::cuda::getDefaultCUDAStream(device);
    }, "device"_a = -1);

    cuda.def("set_stream", [](const tensorplay::cuda::CUDAStream& stream) {
        tensorplay::cuda::setCurrentCUDAStream(stream);
    }, "stream"_a);

    cuda.def("get_stream_from_pool", [](int priority, int device) {
        return tensorplay::cuda::getStreamFromPool(priority, device);
    }, "priority"_a = 0, "device"_a = -1);

    cuda.def("get_stream_priority_range", []() {
        int least = 0;
        int greatest = 0;
        tensorplay::cuda::checkCuda(
            cudaDeviceGetStreamPriorityRange(&least, &greatest),
            "cudaDeviceGetStreamPriorityRange");
        return std::make_pair(least, greatest);
    });

    cuda.def("_sleep", &tensorplay::cuda::sleep, "cycles"_a);
#endif
}

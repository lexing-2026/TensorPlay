#include "python_bindings.h"
#include "tensorplay/ops/Config.h"
#include "Device.h" // For Device class and cuda namespace declarations

#ifdef USE_CUDA
#include <cuda_runtime.h>

struct CudaDeviceProperties {
    std::string name;
    int major;
    int minor;
    size_t total_memory;
    int multi_processor_count;
};
#endif

void init_device(nb::module_& m) {
    nb::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .export_values();

    nb::class_<Device>(m, "Device")
        .def(nb::init<DeviceType, int64_t>(), "type"_a, "index"_a = -1)
        .def(nb::init<const std::string&>(), "device"_a)
        .def(nb::init<const std::string&, int64_t>(), "type"_a, "index"_a)
        .def_prop_ro("type", [](const Device& d) {
            std::string s = d.toString();
            size_t colon = s.find(':');
            if (colon != std::string::npos) {
                return s.substr(0, colon);
            }
            return s;
        })
        .def_prop_ro("index", &Device::index)
        .def("is_cpu", &Device::is_cpu)
        .def("is_cuda", &Device::is_cuda)
        .def("__repr__", &Device::toString)
        .def("__str__", &Device::toString)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self);

    nb::implicitly_convertible<std::string, Device>();
        
    // CUDA submodule
    nb::module_ cuda = m.def_submodule("_cuda", "CUDA computation backend");
    
#ifdef USE_CUDA
    nb::class_<CudaDeviceProperties>(cuda, "_CudaDeviceProperties")
        .def_ro("name", &CudaDeviceProperties::name)
        .def_ro("major", &CudaDeviceProperties::major)
        .def_ro("minor", &CudaDeviceProperties::minor)
        .def_ro("total_memory", &CudaDeviceProperties::total_memory)
        .def_ro("multi_processor_count", &CudaDeviceProperties::multi_processor_count)
        .def("__repr__", [](const CudaDeviceProperties& p) {
            return "_CudaDeviceProperties(name='" + p.name + "', major=" + std::to_string(p.major) + ", minor=" + std::to_string(p.minor) + ", total_memory=" + std::to_string(p.total_memory) + ", multi_processor_count=" + std::to_string(p.multi_processor_count) + ")";
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

    cuda.def("is_available", []() {
#ifdef USE_CUDA
        return true;
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
        // Ignoring device arg for now as cudaDeviceSynchronize acts on current device
        // Ideally we should switch device, sync, then switch back
        int current_device;
        cudaGetDevice(&current_device);
        if (device != -1 && device != current_device) {
             cudaSetDevice(device);
        }
        cudaError_t err = cudaDeviceSynchronize();
        if (device != -1 && device != current_device) {
             cudaSetDevice(current_device);
        }
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
        }
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
    }, "device"_a = 0);

    cuda.def("max_memory_allocated", [](int device) {
#ifdef USE_CUDA
        return tensorplay::cuda::max_memory_allocated(device);
#else
        return 0;
#endif
    }, "device"_a = 0);

    cuda.def("reset_max_memory_allocated", [](int device) {
#ifdef USE_CUDA
        tensorplay::cuda::reset_max_memory_allocated(device);
#endif
    }, "device"_a = 0);
    
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
}

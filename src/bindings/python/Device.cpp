#include "python_bindings.h"
#include "tensorplay/ops/Config.h"
#include "Context.h"
#include "Device.h" // For Device class and cuda namespace declarations
#include "cpu/vec/intrinsics.h"

#include <optional>

#if defined(__linux__) && defined(__x86_64__) && !defined(__ANDROID__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#ifdef USE_CUDA
#include "CUDARuntime.h"
#include "CUDAGenerator.h"
#include <cuda_runtime.h>

// What a device is asked about is not a fixed list of five numbers.  A tile
// kernel is bounded by the registers a thread has and by the shared memory a
// block may ask for; a persistent launch is capped by how many
// multiprocessors there are; a group of tiles is chosen so that programs
// running together share an operand, which depends on the second-level cache.
// Every consumer that needs one of those and cannot read it has to invent a
// number, and an invented number is how a constraint stops constraining
// anything.  So the figures are read once, here, and every consumer reads the
// same ones.
struct CudaDeviceProperties {
    std::string name;
    int major;
    int minor;
    size_t total_memory;
    int multi_processor_count;
    int warp_size;
    int max_threads_per_multi_processor;
    int max_threads_per_block;
    int regs_per_multiprocessor;
    int shared_memory_per_block;
    int shared_memory_per_block_optin;
    int shared_memory_per_multiprocessor;
    int memory_bus_width;
    int memory_clock_rate;
    int clock_rate;
    size_t L2_cache_size;
};
#endif

void init_device(py::module_& m) {
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .value("Vulkan", DeviceType::Vulkan)
        .value("META", DeviceType::Meta)
        .value("Unknown", DeviceType::Unknown);

    py::class_<Device>(m, "Device")
        // One raw-args __init__ instead of typed pybind overloads: a bad
        // argument raises a one-line error naming the accepted forms, instead
        // of an aggregate dump of constructor signatures.
        .def("__init__", [](Device& self, py::args args, py::kwargs kwargs) {
            if (kwargs.size() > 0) {
                throw std::invalid_argument(
                    "device() got unexpected keyword arguments");
            }
            std::string spelling;
            if (args.size() == 1) {
                py::object spec = args[0];
                if (py::isinstance<Device>(spec)) {
                    spelling = spec.cast<const Device&>().toString();
                } else if (py::isinstance<py::tuple>(spec)
                           || py::isinstance<py::list>(spec)) {
                    if (py::len(spec) != 2) {
                        throw std::invalid_argument(
                            "device(): (type, index) pair must have 2 "
                            "elements");
                    }
                    const py::sequence pair = spec.cast<py::sequence>();
                    if (!PyUnicode_Check(pair[0].ptr())
                        || !PyIndex_Check(pair[1].ptr())) {
                        throw std::invalid_argument(
                            "device(): (type, index) pair must be a str and "
                            "an integer");
                    }
                    spelling = pair[0].cast<std::string>() + ":"
                               + std::to_string(pair[1].cast<int64_t>());
                } else if (py::isinstance<DeviceType>(spec)) {
                    new (&self) Device(spec.cast<DeviceType>(), -1);
                    return;
                } else if (PyUnicode_Check(spec.ptr())) {
                    spelling = spec.cast<std::string>();
                } else {
                    throw std::invalid_argument(
                        "device() expects a str, Device, DeviceType, or (type, "
                        "index) pair, not "
                        + std::string(Py_TYPE(spec.ptr())->tp_name));
                }
            } else if (args.size() == 2) {
                if (py::isinstance<DeviceType>(args[0])
                    && PyIndex_Check(args[1].ptr())) {
                    new (&self) Device(args[0].cast<DeviceType>(),
                                       args[1].cast<int64_t>());
                    return;
                }
                if (!PyUnicode_Check(args[0].ptr())
                    || !PyIndex_Check(args[1].ptr())) {
                    throw std::invalid_argument(
                        "device() expects a device type spelling and an "
                        "integer index");
                }
                spelling = args[0].cast<std::string>() + ":"
                           + std::to_string(args[1].cast<int64_t>());
            } else {
                throw std::invalid_argument(
                    "device() takes a str, a Device, a (type, index) pair, or "
                    "a type and index");
            }
            new (&self) Device(spelling);
        })
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
        .def("is_vulkan", &Device::is_vulkan)
        .def("is_meta", &Device::is_meta)
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
        // Picklable (and so copy/deepcopy-able): rebuilt from its spelling,
        // "cpu" or "cuda:1".
        .def("__reduce__", [](const Device& d) {
            return py::make_tuple(py::type::of<Device>(),
                                  py::make_tuple(d.toString()));
        })
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
        .def_readonly("warp_size", &CudaDeviceProperties::warp_size)
        .def_readonly("max_threads_per_multi_processor", &CudaDeviceProperties::max_threads_per_multi_processor)
        .def_readonly("max_threads_per_block", &CudaDeviceProperties::max_threads_per_block)
        .def_readonly("regs_per_multiprocessor", &CudaDeviceProperties::regs_per_multiprocessor)
        .def_readonly("shared_memory_per_block", &CudaDeviceProperties::shared_memory_per_block)
        .def_readonly("shared_memory_per_block_optin", &CudaDeviceProperties::shared_memory_per_block_optin)
        .def_readonly("shared_memory_per_multiprocessor", &CudaDeviceProperties::shared_memory_per_multiprocessor)
        .def_readonly("memory_bus_width", &CudaDeviceProperties::memory_bus_width)
        .def_readonly("memory_clock_rate", &CudaDeviceProperties::memory_clock_rate)
        .def_readonly("clock_rate", &CudaDeviceProperties::clock_rate)
        .def_readonly("L2_cache_size", &CudaDeviceProperties::L2_cache_size)
        .def("__repr__", [](const CudaDeviceProperties& p) {
            return "_CudaDeviceProperties(name='" + p.name + "', major=" + std::to_string(p.major) + ", minor=" + std::to_string(p.minor) + ", total_memory=" + std::to_string(p.total_memory) + ", multi_processor_count=" + std::to_string(p.multi_processor_count) + ", warp_size=" + std::to_string(p.warp_size) + ", regs_per_multiprocessor=" + std::to_string(p.regs_per_multiprocessor) + ", shared_memory_per_block_optin=" + std::to_string(p.shared_memory_per_block_optin) + ", L2_cache_size=" + std::to_string(p.L2_cache_size) + ")";
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

#ifdef USE_CUDA
    // The caching allocator reuses a freed block for allocations on its
    // allocation stream; a block whose storage is also used on another
    // stream must be marked so its reuse waits until that stream's
    // recorded work drains.
    cuda.def("record_stream",
             [](Tensor& self, const tensorplay::cuda::CUDAStream& stream) {
                 if (self.device().type() != DeviceType::CUDA) {
                     TP_THROW(RuntimeError,
                              "record_stream: expected a CUDA tensor");
                 }
                 tensorplay::cuda::recordStream(
                     self.impl()->storage().data(), stream);
             },
             "tensor"_a, "stream"_a);
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

    // The stream a device is currently running on, as the number a kernel
    // launch is given.  Generated code asks for this by name rather than going
    // through the stream object, because a launch is handed the number and
    // building a stream object only to take it apart again is work on the path
    // where a launch is timed.
    m.def("_cuda_getCurrentRawStream", [](int64_t device_index) -> int64_t {
#ifdef USE_CUDA
        return static_cast<int64_t>(
            tensorplay::cuda::getCurrentCUDAStream(device_index).id());
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
        p.warp_size = prop.warpSize;
        p.max_threads_per_multi_processor = prop.maxThreadsPerMultiProcessor;
        p.max_threads_per_block = prop.maxThreadsPerBlock;
        p.regs_per_multiprocessor = prop.regsPerMultiprocessor;
        p.shared_memory_per_block = prop.sharedMemPerBlock;
        p.shared_memory_per_block_optin = prop.sharedMemPerBlockOptin;
        p.shared_memory_per_multiprocessor = prop.sharedMemPerMultiprocessor;
        p.memory_bus_width = prop.memoryBusWidth;
        // Clock fields moved out of cudaDeviceProp in CUDA 12; read them
        // through the device attribute interface instead.
        int memory_clock_khz = 0;
        int clock_khz = 0;
        cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device);
        cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device);
        p.memory_clock_rate = memory_clock_khz * 1000;
        p.clock_rate = clock_khz * 1000;
        p.L2_cache_size = prop.l2CacheSize;
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

    cuda.def("get_rng_state", [](int device) {
#ifdef USE_CUDA
        tensorplay::cuda::CUDAGuard guard(device);
        return tensorplay::cuda::get_rng_state();
#else
        (void)device;
        throw std::runtime_error("CUDA is not available");
#endif
    }, "device"_a = -1);

    cuda.def("set_rng_state", [](const Tensor& state, int device) {
#ifdef USE_CUDA
        tensorplay::cuda::CUDAGuard guard(device);
        tensorplay::cuda::set_rng_state(state);
#else
        (void)state;
        (void)device;
        throw std::runtime_error("CUDA is not available");
#endif
    }, "state"_a, "device"_a = -1);

    cuda.def("current_seed", [](int device) {
#ifdef USE_CUDA
        tensorplay::cuda::CUDAGuard guard(device);
        return tensorplay::cuda::current_seed();
#else
        (void)device;
        throw std::runtime_error("CUDA is not available");
#endif
    }, "device"_a = -1);

    // Where the stream of random values was left off, and the ability to say
    // where it starts.  Both are needed to make a graph of random operations
    // reproducible: the same seed read at the same position gives the same
    // values, and a position is a seed together with how far the stream was
    // already read.  Setting the offset is what lets one operation's answer be
    // folded into the next one's position instead of both reading the same one.
    cuda.def("_get_rng_state_offset", [](int device) {
#ifdef USE_CUDA
        tensorplay::cuda::CUDAGuard guard(device);
        return tensorplay::cuda::current_offset();
#else
        (void)device;
        throw std::runtime_error("CUDA is not available");
#endif
    }, "device"_a = -1);

    cuda.def("_set_rng_state_offset", [](uint64_t offset, int device) {
#ifdef USE_CUDA
        tensorplay::cuda::CUDAGuard guard(device);
        tensorplay::cuda::set_offset(offset);
#else
        (void)offset;
        (void)device;
        throw std::runtime_error("CUDA is not available");
#endif
    }, "offset"_a, "device"_a = -1);

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

    py::module_ cpu = m.def_submodule("_cpu", "Host computation backend");
    cpu.def("_get_cpu_capability", []() {
        py::dict out;
#if defined(__x86_64__) || defined(_M_X64)
        out["architecture"] = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
        out["architecture"] = "arm64";
#elif defined(__powerpc64__) || defined(__PPC64__)
        out["architecture"] = "ppc64";
#elif defined(__s390x__)
        out["architecture"] = "s390x";
#elif defined(__riscv) && (__riscv_xlen == 64)
        out["architecture"] = "riscv64";
#else
        out["architecture"] = "unknown";
#endif
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
        // The probe below only accepts string literals, so each feature
        // is queried at its own call site.
        out["sse"] = __builtin_cpu_supports("sse") != 0;
        out["sse2"] = __builtin_cpu_supports("sse2") != 0;
        out["sse3"] = __builtin_cpu_supports("sse3") != 0;
        out["ssse3"] = __builtin_cpu_supports("ssse3") != 0;
        out["sse4_1"] = __builtin_cpu_supports("sse4.1") != 0;
        out["sse4_2"] = __builtin_cpu_supports("sse4.2") != 0;
        out["avx"] = __builtin_cpu_supports("avx") != 0;
        out["avx2"] = __builtin_cpu_supports("avx2") != 0;
        out["avx512_f"] = __builtin_cpu_supports("avx512f") != 0;
        out["avx512_cd"] = __builtin_cpu_supports("avx512cd") != 0;
        out["avx512_dq"] = __builtin_cpu_supports("avx512dq") != 0;
        out["avx512_bw"] = __builtin_cpu_supports("avx512bw") != 0;
        out["avx512_vl"] = __builtin_cpu_supports("avx512vl") != 0;
        out["avx512_vnni"] = __builtin_cpu_supports("avx512vnni") != 0;
        out["avx512_bf16"] = __builtin_cpu_supports("avx512bf16") != 0;
        out["amx_bf16"] = __builtin_cpu_supports("amx-bf16") != 0;
        out["amx_tile"] = __builtin_cpu_supports("amx-tile") != 0;
        out["amx_int8"] = __builtin_cpu_supports("amx-int8") != 0;
        out["fma3"] = __builtin_cpu_supports("fma") != 0;
        out["aes"] = __builtin_cpu_supports("aes") != 0;
        out["sha"] = __builtin_cpu_supports("sha") != 0;
        out["f16c"] = __builtin_cpu_supports("f16c") != 0;
        out["bmi"] = __builtin_cpu_supports("bmi") != 0;
        out["bmi2"] = __builtin_cpu_supports("bmi2") != 0;
        out["popcnt"] = __builtin_cpu_supports("popcnt") != 0;
#elif defined(__aarch64__)
        out["neon"] = true;
        const bool sve_bf16 = tp_cpu_has_arm_sve_bf16();
        out["sve"] = sve_bf16;
        out["sve_bf16"] = sve_bf16;
        const int sve_bits = tp_cpu_sve_vector_length_bits();
        if (sve_bits > 0) {
            out["sve_max_length"] = sve_bits;
        }
#endif
        return out;
    });
    cpu.def("_init_amx", []() {
#if defined(__linux__) && defined(__x86_64__) && !defined(__ANDROID__) && \
    (defined(__GNUC__) || defined(__clang__))
        if (__builtin_cpu_supports("amx-tile") == 0) {
            return false;
        }
        constexpr unsigned long kTileCfg = 17;
        constexpr unsigned long kTileData = 18;
        constexpr int kGetPerm = 0x1022;
        constexpr int kReqPerm = 0x1023;
        if (syscall(SYS_arch_prctl, kReqPerm, kTileData) != 0) {
            return false;
        }
        unsigned long bitmask = 0;
        if (syscall(SYS_arch_prctl, kGetPerm, &bitmask) != 0) {
            return false;
        }
        return (bitmask & ((1UL << kTileCfg) | (1UL << kTileData))) != 0;
#else
        return false;
#endif
    });

    m.def("_accelerator_getAccelerator", []() -> py::object {
#ifdef USE_CUDA
        return py::cast(Device(DeviceType::CUDA));
#else
        return py::none();
#endif
    });
    m.def("_accelerator_getDeviceIndex", []() {
#ifdef USE_CUDA
        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGetDevice failed: ") +
                cudaGetErrorString(err));
        }
        return device;
#else
        throw std::runtime_error("No accelerator device in this build");
#endif
    });
    m.def("_accelerator_setDeviceIndex", [](int64_t index) {
        if (index < 0) {
            return;
        }
#ifdef USE_CUDA
        cudaError_t err = cudaSetDevice(static_cast<int>(index));
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaSetDevice failed: ") +
                cudaGetErrorString(err));
        }
#else
        throw std::runtime_error("No accelerator device in this build");
#endif
    });
    m.def("_accelerator_exchangeDevice", [](int64_t index) {
#ifdef USE_CUDA
        int current = 0;
        cudaError_t err = cudaGetDevice(&current);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGetDevice failed: ") +
                cudaGetErrorString(err));
        }
        if (index >= 0 && static_cast<int>(index) != current) {
            err = cudaSetDevice(static_cast<int>(index));
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaSetDevice failed: ") +
                    cudaGetErrorString(err));
            }
        }
        return current;
#else
        (void)index;
        throw std::runtime_error("No accelerator device in this build");
#endif
    });
    m.def("_accelerator_maybeExchangeDevice", [](int64_t index) {
#ifdef USE_CUDA
        int current = 0;
        cudaError_t err = cudaGetDevice(&current);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGetDevice failed: ") +
                cudaGetErrorString(err));
        }
        if (index >= 0 && static_cast<int>(index) != current) {
            err = cudaSetDevice(static_cast<int>(index));
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaSetDevice failed: ") +
                    cudaGetErrorString(err));
            }
        }
        return current;
#else
        (void)index;
        return -1;
#endif
    });
    m.def(
        "_accelerator_synchronizeDevice",
        [](int64_t index) {
#ifdef USE_CUDA
            tensorplay::cuda::CUDAGuard guard(static_cast<int>(index));
            tensorplay::cuda::checkCuda(
                cudaDeviceSynchronize(), "cudaDeviceSynchronize");
#else
            (void)index;
#endif
        },
        "device_index"_a = -1);
    m.def(
        "_accelerator_getMemoryInfo",
        [](int64_t index) {
#ifdef USE_CUDA
            int device = static_cast<int>(index);
            if (device < 0) {
                cudaError_t err = cudaGetDevice(&device);
                if (err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("cudaGetDevice failed: ") +
                        cudaGetErrorString(err));
                }
            }
            tensorplay::cuda::CUDAGuard guard(device);
            size_t free = 0;
            size_t total = 0;
            tensorplay::cuda::checkCuda(
                cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
            return std::make_pair(free, total);
#else
            (void)index;
            throw std::runtime_error("No accelerator device in this build");
#endif
        },
        "device_index"_a = -1);
    m.def(
        "_accelerator_getDeviceCapability",
        [](int64_t index) {
            (void)index;
            py::dict out;
            py::set supported;
            supported.add(py::cast(DType::UInt8));
            supported.add(py::cast(DType::Int8));
            supported.add(py::cast(DType::Int16));
            supported.add(py::cast(DType::Int32));
            supported.add(py::cast(DType::Int64));
            supported.add(py::cast(DType::UInt16));
            supported.add(py::cast(DType::Bool));
            supported.add(py::cast(DType::Float16));
            supported.add(py::cast(DType::BFloat16));
            supported.add(py::cast(DType::Float32));
            supported.add(py::cast(DType::Float64));
            supported.add(py::cast(DType::ComplexFloat));
            supported.add(py::cast(DType::ComplexDouble));
            out["supported_dtypes"] = std::move(supported);
            return out;
        },
        "device_index"_a = -1);
}

#include "python_bindings.h"

#ifdef USE_CUDA
#include "CUDAGraph.h"
#endif

#include <stdexcept>

// Native surface consumed by tensorplay/compiler/cudagraphs.py and
// tensorplay/cuda/graphs.py.  The four required symbols mirror the contract
// documented in the CudaGraphManager docstring; the stream helpers exist so
// the Python orchestration can run warmup and capture on the same side
// stream, as ATen requires.

void init_cuda_graph(py::module_& m) {
#ifdef USE_CUDA
    namespace cg = tensorplay::cuda::graph;

    m.def("cuda_graph_begin_capture", &cg::beginCapture,
          "Start CUDA graph capture on a pooled side stream (becomes the "
          "calling thread's current stream until end_capture)");

    m.def("cuda_graph_end_capture", &cg::endCapture,
          py::call_guard<py::gil_scoped_release>(),
          "Stop capture and return an opaque graph handle");

    m.def("cuda_graph_instantiate", &cg::instantiate, "handle"_a,
          py::call_guard<py::gil_scoped_release>(),
          "Compile a captured graph into an executable; returns its handle");

    m.def("cuda_graph_launch", &cg::launch, "handle"_a,
          py::call_guard<py::gil_scoped_release>(),
          "Enqueue an executable graph on the current stream");

    m.def("cuda_graph_destroy", &cg::destroy, "handle"_a,
          py::call_guard<py::gil_scoped_release>(),
          "Destroy a graph handle and release its private memory pool");

    m.def("cuda_graph_capture_stream",
          [](int device) { return cg::captureStream(device); },
          "device"_a = -1,
          "The dedicated side stream capture runs on (reuse it for warmup)");

    m.def("cuda_stream_create",
          [](int device, int priority) {
              return tensorplay::cuda::getStreamFromPool(priority, device);
          },
          "device"_a = -1, "priority"_a = 0,
          "Create a side stream from the stream pool");

    m.def("cuda_stream_set_current",
          [](const tensorplay::cuda::CUDAStream& stream) {
              tensorplay::cuda::setCurrentCUDAStream(stream);
          },
          "stream"_a);

    m.def("cuda_stream_get_current",
          [](int device) {
              return tensorplay::cuda::getCurrentCUDAStream(device);
          },
          "device"_a = -1);

    m.def("cuda_is_capturing", &tensorplay::cuda::isCapturing,
          "True while a CUDA graph capture scope is open");
#else
    m.def("cuda_graph_begin_capture", []() -> void {
        throw std::runtime_error(
            "CUDA graphs require a TensorPlay build with CUDA support");
    });
    m.def("cuda_graph_end_capture", []() -> int64_t {
        throw std::runtime_error(
            "CUDA graphs require a TensorPlay build with CUDA support");
    });
    m.def("cuda_graph_instantiate", [](int64_t) -> int64_t {
        throw std::runtime_error(
            "CUDA graphs require a TensorPlay build with CUDA support");
    }, "handle"_a);
    m.def("cuda_graph_launch", [](int64_t) -> void {
        throw std::runtime_error(
            "CUDA graphs require a TensorPlay build with CUDA support");
    }, "handle"_a);
    m.def("cuda_graph_destroy", [](int64_t) -> void {
        throw std::runtime_error(
            "CUDA graphs require a TensorPlay build with CUDA support");
    }, "handle"_a);
    m.def("cuda_graph_capture_stream", [](int) -> int {
        throw std::runtime_error(
            "CUDA graphs require a TensorPlay build with CUDA support");
    }, "device"_a = -1);
#endif
}

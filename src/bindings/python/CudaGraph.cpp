#include "python_bindings.h"

#ifdef USE_CUDA
#include "CUDAGraph.h"
#include "CUDARuntime.h"
#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cuda_runtime.h>
#endif

#include <optional>
#include <string>
#include <vector>

// CUDAGraph class: capture_begin/capture_end manage the side stream, the
// graph-private allocator pool and graph-safe RNG; replay() launches the
// cached executable directly (no registry lookups).  stage_and_launch is the
// TensorPlay-only bulk replay entry: it stages every input with a raw
// device-to-device copy (dispatcher bypass) and launches in one call.

namespace {

#ifdef USE_CUDA

void stageAndLaunch(tensorplay::cuda::graph::CUDAGraph& graph,
                    std::vector<Tensor>& static_inputs,
                    const std::vector<Tensor>& inputs) {
    if (static_inputs.size() != inputs.size()) {
        TP_THROW(ValueError,
                 "graph expects " + std::to_string(static_inputs.size()) +
                     " staged inputs, got " + std::to_string(inputs.size()));
    }

    py::gil_scoped_release release;
    namespace cuda = tensorplay::cuda;
    const cuda::CUDAStream stream = cuda::getCurrentCUDAStream();
    for (size_t i = 0; i < static_inputs.size(); ++i) {
        Tensor& dst = static_inputs[i];
        const Tensor& src = inputs[i];
        const bool memcpyable =
            dst.dtype() == src.dtype() && dst.device() == src.device() &&
            dst.device().is_cuda() && dst.is_contiguous() &&
            src.is_contiguous() && dst.numel() == src.numel();
        if (memcpyable) {
            // Raw D2D copy on the launch stream: no dispatcher round trip,
            // no version bump, no autograd bookkeeping per staged input.
            // The source is recorded against the launch stream so dropping
            // it right after enqueue cannot hand its memory back early.
            cuda::recordStream(src.data_ptr(), stream);
            cuda::checkCuda(
                cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(),
                                dst.numel() * dst.itemsize(),
                                cudaMemcpyDeviceToDevice, stream.stream()),
                "graph input staging");
        } else {
            // Cross-device / non-contiguous / dtype-drifted inputs fall back
            // to the full copy semantics.
            tensorplay::tpx::ops::copy_(dst, src);
        }
    }
    graph.replay();
}

#endif // USE_CUDA

} // namespace

void init_cuda_graph(py::module_& m) {
#ifdef USE_CUDA
    namespace cg = tensorplay::cuda::graph;

    py::class_<cg::CUDAGraph, std::shared_ptr<cg::CUDAGraph>>(m, "CUDAGraph")
        .def(py::init<>())
        .def("capture_begin",
             [](cg::CUDAGraph& self, uint64_t pool,
                const std::string& capture_error_mode,
                std::optional<tensorplay::cuda::CUDAStream> stream) {
                 self.capture_begin(
                     pool, cg::captureModeFromName(capture_error_mode),
                     stream.value_or(tensorplay::cuda::CUDAStream::undefined()));
             },
             "pool"_a = 0, "capture_error_mode"_a = "global", "stream"_a = py::none(),
             "Start capture; allocations are routed into pool (0 = fresh "
             "private pool, share via _C.graph_pool_handle())")
        .def("capture_end", &cg::CUDAGraph::capture_end,
             py::call_guard<py::gil_scoped_release>(),
             "End capture and instantiate eagerly")
        .def("instantiate", &cg::CUDAGraph::instantiate,
             "No-op once instantiated; kept for late callers")
        .def("replay",
             [](cg::CUDAGraph& self,
                const std::optional<tensorplay::cuda::CUDAStream>& stream) {
                 py::gil_scoped_release release;
                 if (stream.has_value()) {
                     self.replay(*stream);
                 } else {
                     self.replay();
                 }
             },
             "stream"_a = py::none(),
             "Launch the cached executable on the current stream, or on an "
             "explicit stream for hot loops pinned to one")
        .def("stage_and_launch", &stageAndLaunch,
             "static_inputs"_a, "inputs"_a,
             "Stage every input onto its static buffer (raw async copies) "
             "and replay in one call - the low-overhead replay path")
        .def("reset", &cg::CUDAGraph::reset,
             "Destroy the executable and release the pool reference")
        .def("enable_debug_mode", &cg::CUDAGraph::enable_debug_mode,
             "Retain the captured template for debug_dump()")
        .def("debug_dump", &cg::CUDAGraph::debug_dump, "path"_a,
             "Write a DOT rendering of the graph to ``path``")
        .def("pool_id", &cg::CUDAGraph::pool_id,
             "Allocator pool id this graph captures against")
        .def_property_readonly("device", &cg::CUDAGraph::device)
        .def_property_readonly("has_graph_exec", &cg::CUDAGraph::has_graph_exec)
        // --- conditional nodes (CUDA >= 12.4) ---
        .def("begin_capture_to_if_node",
             [](cg::CUDAGraph& self, Tensor pred) {
                 self.begin_capture_to_if_node(pred);
             },
             "scalar_pred"_a,
             "Split the remaining capture into an if-node gated on the Bool "
             "device scalar; body ops run inside the node until "
             "end_capture_to_conditional_node()")
        .def("begin_capture_to_while_node",
             [](cg::CUDAGraph& self, Tensor pred) {
                 self.begin_capture_to_while_node(pred);
             },
             "scalar_pred"_a,
             "Like begin_capture_to_if_node but the body loops while the "
             "predicate stays true")
        .def("set_conditional_handle_for_current_node",
             [](cg::CUDAGraph& self, Tensor pred) {
                 self.set_conditional_handle_for_current_node(pred);
             },
             "scalar_pred"_a,
             "Refresh the predicate of the innermost open conditional node")
        .def("end_capture_to_conditional_node",
             &cg::CUDAGraph::end_capture_to_conditional_node,
             "Close the open conditional body and resume capturing on the "
             "parent stream");

    m.def("conditional_nodes_supported", []() {
        return tensorplay::cuda::graph::conditionalNodesSupported();
    }, "True when the CUDA runtime supports graph conditional nodes (>= 12.4)");

    m.def("graph_pool_handle", &tensorplay::cuda::graph_pool_handle,
          "Reserve a unique memory-pool id usable as CUDAGraph.capture_begin(pool=...)");

    m.def("cuda_is_capturing",
          []() { return tensorplay::cuda::graph::isCapturing(); },
          "True while any CUDA graph capture scope is open in this process");
    m.def("cuda_stream_is_capturing",
          []() {
              cudaStreamCaptureStatus status =
                  cudaStreamCaptureStatusNone;
              cudaError_t error = cudaStreamIsCapturing(
                  tensorplay::cuda::getCurrentCUDAStream().stream(), &status);
              if (error != cudaSuccess) {
                  (void)cudaGetLastError();
                  return false;
              }
              return status == cudaStreamCaptureStatusActive;
          },
          "True when this thread's current stream is participating in a "
          "capture status)");
#else
    // Non-CUDA builds expose no CUDA graph names; the Python layer
    // (tensorplay/cuda/graphs.py) raises descriptive errors on use.
#endif
}

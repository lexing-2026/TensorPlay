// Python bindings for the NCCL communicator context, mirroring the
// torch.distributed._C collective entry points used by distributed_c10d.py.

#include "python_bindings.h"

#include "NCCLContext.h"

#ifdef USE_CUDA
#include "CUDARuntime.h"
#endif

namespace py = pybind11;

void init_distributed(py::module_& m) {
    py::module_ dist = m.def_submodule("_distributed", "Distributed (NCCL) backend");

    using tensorplay::nccl::Comm;

    auto current_stream_ptr = [](const Tensor& t) -> void* {
#ifdef USE_CUDA
        int device_index = t.device().index();
        return reinterpret_cast<void*>(
            tensorplay::cuda::getCurrentCUDAStream(device_index).stream());
#else
        (void)t;
        TP_THROW(RuntimeError, "TensorPlay was compiled without CUDA support");
#endif
    };

    auto require_cuda = [](const Tensor& t) -> void* {
        if (!t.device().is_cuda()) {
            TP_THROW(RuntimeError,
                     "ProcessGroupNCCL only supports CUDA tensors; "
                     "no device index is specified");
        }
        return t.data_ptr();
    };

    dist.def("is_available", []() {
#ifdef USE_CUDA
        return tensorplay::nccl::available();
#else
        return false;
#endif
    });

    dist.def("version", []() -> std::string {
#ifdef USE_CUDA
        return tensorplay::nccl::version();
#else
        return "";
#endif
    });

    dist.def("get_unique_id", []() -> py::bytes {
#ifdef USE_CUDA
        uint8_t uid[tensorplay::nccl::kUniqueIdBytes];
        tensorplay::nccl::getUniqueId(uid);
        return py::bytes(reinterpret_cast<const char*>(uid),
                         tensorplay::nccl::kUniqueIdBytes);
#else
        TP_THROW(RuntimeError, "TensorPlay was compiled without CUDA support");
        return py::bytes("", 0);
#endif
    });

    dist.def("comm_init_rank", [](int rank, int world_size, py::bytes uid_bytes) -> uint64_t {
#ifdef USE_CUDA
        std::string raw = uid_bytes;
        if (raw.size() != tensorplay::nccl::kUniqueIdBytes) {
            TP_THROW(ValueError, "unique id must be 128 bytes");
        }
        Comm comm = tensorplay::nccl::commInitRank(
            rank, world_size, reinterpret_cast<const uint8_t*>(raw.data()));
        return reinterpret_cast<uint64_t>(comm);
#else
        (void)rank; (void)world_size; (void)uid_bytes;
        TP_THROW(RuntimeError, "TensorPlay was compiled without CUDA support");
        return 0;
#endif
    }, "rank"_a, "world_size"_a, "uid"_a);

    dist.def("comm_destroy", [](uint64_t handle) {
#ifdef USE_CUDA
        tensorplay::nccl::commDestroy(reinterpret_cast<Comm>(handle));
#else
        (void)handle;
#endif
    }, "comm"_a);

    dist.def("comm_abort", [](uint64_t handle) {
#ifdef USE_CUDA
        tensorplay::nccl::commAbort(reinterpret_cast<Comm>(handle));
#else
        (void)handle;
#endif
    }, "comm"_a);

    dist.def("comm_count", [](uint64_t handle) -> int {
#ifdef USE_CUDA
        return tensorplay::nccl::commCount(reinterpret_cast<Comm>(handle));
#else
        (void)handle;
        return 0;
#endif
    }, "comm"_a);

    dist.def("all_reduce", [&](Tensor& t, int op, uint64_t handle) {
#ifdef USE_CUDA
        void* ptr = require_cuda(t);
        tensorplay::nccl::allReduce(ptr, t.numel(), t.dtype(),
                                    static_cast<tensorplay::nccl::ReduceOp>(op),
                                    reinterpret_cast<Comm>(handle),
                                    current_stream_ptr(t));
#else
        (void)t; (void)op; (void)handle;
#endif
    }, "tensor"_a, "op"_a, "comm"_a);

    dist.def("broadcast", [&](Tensor& t, int root, uint64_t handle) {
#ifdef USE_CUDA
        void* ptr = require_cuda(t);
        tensorplay::nccl::broadcast(ptr, t.numel(), t.dtype(), root,
                                    reinterpret_cast<Comm>(handle),
                                    current_stream_ptr(t));
#else
        (void)t; (void)root; (void)handle;
#endif
    }, "tensor"_a, "root"_a, "comm"_a);

    dist.def("reduce", [&](Tensor& t, int op, int root, uint64_t handle) {
#ifdef USE_CUDA
        void* ptr = require_cuda(t);
        tensorplay::nccl::reduce(ptr, t.numel(), t.dtype(),
                                 static_cast<tensorplay::nccl::ReduceOp>(op), root,
                                 reinterpret_cast<Comm>(handle),
                                 current_stream_ptr(t));
#else
        (void)t; (void)op; (void)root; (void)handle;
#endif
    }, "tensor"_a, "op"_a, "root"_a, "comm"_a);

    dist.def("all_gather", [&](Tensor& recv, Tensor& send, uint64_t handle) {
#ifdef USE_CUDA
        void* send_ptr = require_cuda(send);
        void* recv_ptr = require_cuda(recv);
        tensorplay::nccl::allGather(send_ptr, recv_ptr, send.numel(), send.dtype(),
                                    reinterpret_cast<Comm>(handle),
                                    current_stream_ptr(send));
#else
        (void)recv; (void)send; (void)handle;
#endif
    }, "recv"_a, "send"_a, "comm"_a);

    dist.def("reduce_scatter", [&](Tensor& recv, Tensor& send, int op, uint64_t handle) {
#ifdef USE_CUDA
        void* send_ptr = require_cuda(send);
        void* recv_ptr = require_cuda(recv);
        tensorplay::nccl::reduceScatter(send_ptr, recv_ptr, recv.numel(), recv.dtype(),
                                        static_cast<tensorplay::nccl::ReduceOp>(op),
                                        reinterpret_cast<Comm>(handle),
                                        current_stream_ptr(recv));
#else
        (void)recv; (void)send; (void)op; (void)handle;
#endif
    }, "recv"_a, "send"_a, "op"_a, "comm"_a);

    dist.def("gather", [&](py::object recv, Tensor& send, int root, uint64_t handle) {
#ifdef USE_CUDA
        void* send_ptr = require_cuda(send);
        void* recv_ptr = nullptr;
        if (!recv.is_none()) {
            recv_ptr = require_cuda(recv.cast<Tensor&>());
        }
        tensorplay::nccl::gather(send_ptr, recv_ptr, send.numel(), send.dtype(),
                                 root, reinterpret_cast<Comm>(handle),
                                 current_stream_ptr(send));
#else
        (void)recv; (void)send; (void)root; (void)handle;
#endif
    }, "recv"_a, "send"_a, "root"_a, "comm"_a);

    dist.def("scatter", [&](Tensor& recv, py::object send, int root, uint64_t handle) {
#ifdef USE_CUDA
        void* recv_ptr = require_cuda(recv);
        void* send_ptr = nullptr;
        if (!send.is_none()) {
            send_ptr = require_cuda(send.cast<Tensor&>());
        }
        tensorplay::nccl::scatter(send_ptr, recv_ptr, recv.numel(), recv.dtype(),
                                  root, reinterpret_cast<Comm>(handle),
                                  current_stream_ptr(recv));
#else
        (void)recv; (void)send; (void)root; (void)handle;
#endif
    }, "recv"_a, "send"_a, "root"_a, "comm"_a);

    dist.def("send", [&](Tensor& t, int peer, uint64_t handle) {
#ifdef USE_CUDA
        void* ptr = require_cuda(t);
        tensorplay::nccl::send(ptr, t.numel(), t.dtype(), peer,
                               reinterpret_cast<Comm>(handle),
                               current_stream_ptr(t));
#else
        (void)t; (void)peer; (void)handle;
#endif
    }, "tensor"_a, "peer"_a, "comm"_a);

    dist.def("recv", [&](Tensor& t, int peer, uint64_t handle) {
#ifdef USE_CUDA
        void* ptr = require_cuda(t);
        tensorplay::nccl::recv(ptr, t.numel(), t.dtype(), peer,
                               reinterpret_cast<Comm>(handle),
                               current_stream_ptr(t));
#else
        (void)t; (void)peer; (void)handle;
#endif
    }, "tensor"_a, "peer"_a, "comm"_a);
}

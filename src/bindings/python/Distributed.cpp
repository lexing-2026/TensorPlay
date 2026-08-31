// Python bindings for the NCCL communicator context.

#include "python_bindings.h"

#include "NCCLContext.h"

#include <vector>

#ifdef USE_CUDA
#include "CUDARuntime.h"
#include "AccumulateGrad.h"
#include "AutogradMeta.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <memory>
#include <mutex>
#endif

namespace py = pybind11;

#ifdef USE_CUDA
namespace {

// fast path: find_unused_parameters=False, no comm hook).
//
// Post-accumulate grad hooks are registered as pure C++ callbacks on each
// parameter's AccumulateGrad node, so the backward hot path never crosses
// into Python and never takes the GIL (the Python hook bridge acquires the
// GIL per hook on the engine worker thread, which dominated small-model DDP
// overhead).  Copy-in happens eagerly in the hook (spreading copies across
// the bucket all-reduce is one fused multi-tensor copy on a dedicated comm
// stream, joined into the compute stream once the final bucket is reduced.
class DDPReducer : public std::enable_shared_from_this<DDPReducer> {
public:
    struct Bucket {
        int64_t index = 0;
        Tensor buffer;
        std::vector<Tensor> params;
        std::vector<Tensor> views;  // pre-sliced bucket views (param shapes)
        int64_t remaining = 0;
    };

    DDPReducer(std::vector<std::vector<Tensor>> bucket_params,
               std::vector<Tensor> bucket_buffers,
               uint64_t comm, int64_t world_size,
               bool gradient_as_bucket_view)
        : comm_(reinterpret_cast<tensorplay::nccl::Comm>(comm)),
          world_size_(world_size),
          gabv_(gradient_as_bucket_view) {
        if (bucket_params.size() != bucket_buffers.size()) {
            TP_THROW(ValueError,
                     "DDPReducer: bucket params and buffers must match");
        }
        device_index_ = bucket_buffers.empty()
            ? 0 : bucket_buffers[0].device().index();
        comm_stream_ = tensorplay::cuda::getStreamFromPool(0, device_index_);
        buckets_.reserve(bucket_params.size());
        for (size_t bi = 0; bi < bucket_params.size(); ++bi) {
            Bucket b;
            b.index = static_cast<int64_t>(bi);
            b.buffer = bucket_buffers[bi];
            b.params = std::move(bucket_params[bi]);
            int64_t off = 0;
            for (const auto& p : b.params) {
                const auto shape = p.shape();
                std::vector<int64_t> sizes(shape.begin(), shape.end());
                b.views.push_back(
                    tensorplay::tpx::ops::narrow(b.buffer, 0, off, p.numel())
                        .view(sizes));
                off += p.numel();
            }
            buckets_.push_back(std::move(b));
        }
    }

    // Second-phase init (needs shared_from_this): attach the C++ hooks.
    void install_hooks() {
        for (size_t bi = 0; bi < buckets_.size(); ++bi) {
            for (size_t idx = 0; idx < buckets_[bi].params.size(); ++idx) {
                const Tensor& param = buckets_[bi].params[idx];
                auto* meta =
                    tensorplay::tpx::impl::get_or_create_autograd_meta(param);
                if (meta == nullptr) continue;
                std::shared_ptr<tensorplay::tpx::Node> node =
                    meta->grad_accumulator();
                if (!node) {
                    node = std::make_shared<tensorplay::tpx::AccumulateGrad>(
                        param);
                    meta->set_grad_accumulator(node);
                }
                accum_nodes_.push_back(node);
                std::weak_ptr<DDPReducer> weak(shared_from_this());
                node->add_post_hook(
                    [weak, bi, idx](const tensorplay::tpx::variable_list&,
                                    tensorplay::tpx::variable_list&& outputs) {
                        if (auto self = weak.lock()) {
                            self->mark_ready(bi, idx);
                        }
                        return std::move(outputs);
                    });
            }
        }
    }

    // ensure_prior_reduction_finished).
    void prepare_for_iteration() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (require_finalize_) {
            TP_THROW(RuntimeError,
                     "Expected to have finished reduction in the prior "
                     "iteration before starting a new one. This error "
                     "indicates that your module has parameters that were "
                     "not used in producing loss. You can enable "
                     "find_unused_parameters=True in the "
                     "DistributedDataParallel constructor to work around "
                     "this error.");
        }
        next_bucket_ = 0;
        for (auto& b : buckets_) {
            b.remaining = static_cast<int64_t>(b.params.size());
        }
        require_finalize_ = true;
    }

    // no_sync support.
    void set_require_sync(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        require_sync_ = enabled;
    }

    // Join path: the shadow iteration issued no real backward, so drop any
    // outstanding finalize expectation without raising on the next forward.
    void abort_iteration() {
        std::lock_guard<std::mutex> lock(mutex_);
        require_finalize_ = false;
        next_bucket_ = 0;
    }

    // Engine worker thread (no GIL): one parameter's grad is ready.
    void mark_ready(size_t bi, size_t idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!require_sync_) return;
        Bucket& b = buckets_[bi];
        auto* meta =
            tensorplay::tpx::impl::get_autograd_meta(b.params[idx]);
        Tensor grad = (meta != nullptr) ? meta->grad() : Tensor();
        if (!grad.defined()) {
            // Leave the bucket incomplete; prepare_for_iteration raises on
            return;
        }
        const Tensor& view_ref = b.views[idx];
        if (gabv_ && grad.data_ptr() == view_ref.data_ptr() &&
            grad.numel() == view_ref.numel()) {
        } else {
            Tensor view = view_ref;  // copy_ is non-const
            view.copy_(grad);
            if (gabv_ && meta != nullptr) {
                meta->set_grad(view);
            }
        }
        if (--b.remaining == 0) {
            flush_ready_locked();
        }
    }

private:
    void flush_ready_locked() {
        while (next_bucket_ < static_cast<int64_t>(buckets_.size()) &&
               buckets_[next_bucket_].remaining == 0) {
            reduce_bucket_locked(buckets_[next_bucket_]);
            ++next_bucket_;
        }
        if (next_bucket_ == static_cast<int64_t>(buckets_.size())) {
            require_finalize_ = false;
        }
    }

    void reduce_bucket_locked(Bucket& b) {
        if (world_size_ <= 1) return;
        namespace nccl = tensorplay::nccl;
        nccl::ReduceOp op = nccl::ReduceOp::Avg;
        if (b.buffer.dtype() != tensorplay::DType::Float32) {
            // accumulating unscaled values at reduced precision.
            b.buffer.div_(tensorplay::Scalar(static_cast<int64_t>(world_size_)));
            op = nccl::ReduceOp::Sum;
        }
        // Hand the bucket to the comm stream (copy-in ran on the current
        // compute stream), all-reduce and copy the reduced grads back there,
        // then join the compute stream once the final bucket is reduced.
        tensorplay::cuda::CUDAStream cur =
            tensorplay::cuda::getCurrentCUDAStream(device_index_);
        order_ev_.record(cur);
        order_ev_.block(comm_stream_);
        {
            tensorplay::cuda::CUDAStreamGuard guard(comm_stream_);
            nccl::allReduce(b.buffer.data_ptr(),
                            static_cast<size_t>(b.buffer.numel()),
                            b.buffer.dtype(), op, comm_,
                            reinterpret_cast<void*>(comm_stream_.stream()));
            if (!gabv_) {
                std::vector<Tensor> dsts;
                std::vector<Tensor> srcs;
                dsts.reserve(b.params.size());
                srcs.reserve(b.params.size());
                for (size_t i = 0; i < b.params.size(); ++i) {
                    auto* meta = tensorplay::tpx::impl::get_autograd_meta(
                        b.params[i]);
                    Tensor g = (meta != nullptr) ? meta->grad() : Tensor();
                    if (g.defined()) {
                        dsts.push_back(std::move(g));
                        srcs.push_back(b.views[i]);
                    }
                }
                if (!dsts.empty()) {
                    tensorplay::tpx::ops::_foreach_copy_(dsts, srcs, false);
                }
            }
        }
        if (b.index == static_cast<int64_t>(buckets_.size()) - 1) {
            done_ev_.record(comm_stream_);
            done_ev_.block(cur);
        }
    }

    tensorplay::nccl::Comm comm_;
    int64_t world_size_;
    bool gabv_;
    int device_index_ = 0;
    tensorplay::cuda::CUDAStream comm_stream_ =
        tensorplay::cuda::CUDAStream::undefined();
    tensorplay::cuda::CUDAEvent order_ev_;
    tensorplay::cuda::CUDAEvent done_ev_;
    std::vector<Bucket> buckets_;
    std::vector<std::shared_ptr<tensorplay::tpx::Node>> accum_nodes_;
    std::mutex mutex_;
    int64_t next_bucket_ = 0;
    bool require_finalize_ = false;
    bool require_sync_ = true;
};

}  // namespace
#endif  // USE_CUDA

namespace tensorplay {
namespace distributed {
void init_gloo_bindings(py::module_& dist);
void init_mpi_bindings(py::module_& dist);
}  // namespace distributed
}  // namespace tensorplay

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

    auto dtype_item_size = [](const Tensor& t) -> size_t {
        switch (t.dtype()) {
            case DType::Int8:
            case DType::UInt8: return 1;
            case DType::Int16:
            case DType::UInt16:
            case DType::Float16:
            case DType::BFloat16: return 2;
            case DType::Int32:
            case DType::UInt32:
            case DType::Float32: return 4;
            case DType::Int64:
            case DType::UInt64:
            case DType::Float64: return 8;
            default:
                TP_THROW(RuntimeError,
                         "unsupported dtype for all_to_all_single");
        }
    };

    dist.def("all_to_all_single_equal_split", [&](Tensor& output, Tensor& input, uint64_t handle) {
#ifdef USE_CUDA
        void* send_ptr = require_cuda(input);
        void* recv_ptr = require_cuda(output);
        if (input.dtype() != output.dtype()) {
            TP_THROW(ValueError, "output tensor must have the same type as input tensor");
        }
        if (output.numel() != input.numel()) {
            TP_THROW(ValueError, "output tensor must have the same number of elements as input tensor");
        }
        tensorplay::nccl::allToAllSingleEqualSplit(
            send_ptr, recv_ptr, input.numel(), input.dtype(),
            reinterpret_cast<Comm>(handle), current_stream_ptr(input));
#else
        (void)output; (void)input; (void)handle;
#endif
    }, "recv"_a, "send"_a, "comm"_a);

    dist.def("all_to_all_single_unequal_split",
             [&](Tensor& output, Tensor& input,
                 std::vector<int64_t> output_split_sizes,
                 std::vector<int64_t> input_split_sizes, uint64_t handle) {
#ifdef USE_CUDA
        void* send_ptr = require_cuda(input);
        void* recv_ptr = require_cuda(output);
        if (input.dtype() != output.dtype()) {
            TP_THROW(ValueError, "output tensor must have the same type as input tensor");
        }
        size_t elem = dtype_item_size(input);
        std::vector<size_t> sendcounts, senddispls, recvcounts, recvdispls;
        size_t acc = 0;
        for (int64_t s : input_split_sizes) {
            sendcounts.push_back(static_cast<size_t>(s));
            senddispls.push_back(acc);
            acc += static_cast<size_t>(s);
        }
        if (acc != static_cast<size_t>(input.numel())) {
            TP_THROW(ValueError, "input_split_sizes sum must equal input numel");
        }
        acc = 0;
        for (int64_t s : output_split_sizes) {
            recvcounts.push_back(static_cast<size_t>(s));
            recvdispls.push_back(acc);
            acc += static_cast<size_t>(s);
        }
        if (acc != static_cast<size_t>(output.numel())) {
            TP_THROW(ValueError, "output_split_sizes sum must equal output numel");
        }
        tensorplay::nccl::allToAllSingleUnequalSplit(
            send_ptr, sendcounts.data(), senddispls.data(),
            recv_ptr, recvcounts.data(), recvdispls.data(),
            elem, input.dtype(), reinterpret_cast<Comm>(handle),
            current_stream_ptr(input));
#else
        (void)output; (void)input; (void)output_split_sizes;
        (void)input_split_sizes; (void)handle;
#endif
    }, "recv"_a, "send"_a, "output_split_sizes"_a, "input_split_sizes"_a, "comm"_a);

    dist.def("group_start", []() {
#ifdef USE_CUDA
        tensorplay::nccl::groupStart();
#endif
    });
    dist.def("group_end", []() {
#ifdef USE_CUDA
        tensorplay::nccl::groupEnd();
#endif
    });

#ifdef USE_CUDA
    py::class_<DDPReducer, std::shared_ptr<DDPReducer>>(dist, "DDPReducer")
        .def(py::init([](std::vector<std::vector<Tensor>> bucket_params,
                         std::vector<Tensor> bucket_buffers,
                         uint64_t comm, int64_t world_size,
                         bool gradient_as_bucket_view) {
                 auto reducer = std::make_shared<DDPReducer>(
                     std::move(bucket_params), std::move(bucket_buffers),
                     comm, world_size, gradient_as_bucket_view);
                 reducer->install_hooks();
                 return reducer;
             }),
             py::arg("bucket_params"), py::arg("bucket_buffers"),
             py::arg("comm"), py::arg("world_size"),
             py::arg("gradient_as_bucket_view"))
        .def("prepare_for_iteration", &DDPReducer::prepare_for_iteration)
        .def("set_require_sync", &DDPReducer::set_require_sync)
        .def("abort_iteration", &DDPReducer::abort_iteration);
#endif

    tensorplay::distributed::init_gloo_bindings(dist);
    tensorplay::distributed::init_mpi_bindings(dist);
}

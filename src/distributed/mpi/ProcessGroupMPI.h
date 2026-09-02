#pragma once

#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <Tensor.h>

#include "../gloo/ProcessGroupGloo.h"
#include "../Types.h"

#ifdef USE_P10D_MPI
#include <mpi.h>
#endif

namespace tensorplay {
namespace distributed {

using Tensor = tensorplay::Tensor;

// ProcessGroupMPI: process group over a Message Passing Interface runtime.
//
// All functions on this class are expected to be called in the same order
// across processes in the group. Collective calls are serialized onto a
// single worker thread, which requires the MPI implementation to provide at
// least MPI_THREAD_SERIALIZED support; only one such group may exist when
// the runtime does not offer MPI_THREAD_MULTIPLE.
//
// Collective entry points validate tensor count, shape, dtype, and device
// requirements before queueing work on the MPI worker.
#ifdef USE_P10D_MPI
class ProcessGroupMPI {
 public:
  class WorkMPI : public GlooWork {
   public:
    explicit WorkMPI(std::vector<Tensor> outputTensors, std::string opName)
        : GlooWork(std::move(opName)) {
      outputTensors_ = {std::move(outputTensors)};
    }

    void finishWorkMPI() {
      finish();
    }
    void finishWorkMPIError(std::exception_ptr eptr) {
      finishWithError(std::move(eptr));
    }
  };

  class AsyncWork : public GlooWork {
   public:
    AsyncWork(
        MPI_Request request,
        std::vector<Tensor> outputTensors,
        std::string opName,
        std::vector<Tensor> inputTensors = {})
        : GlooWork(std::move(opName)),
          outputTensors_(std::move(outputTensors)),
          inputTensors_(std::move(inputTensors)),
          request_(request) {
      std::memset(&status_, 0, sizeof(status_));
    }

    ~AsyncWork() override;

    bool is_completed() override;
    int source_rank() const override;
    bool wait(int64_t timeout_ms = -1) override;
    void abort() override;
    std::vector<Tensor> result() const override {
      return outputTensors_;
    }

   protected:
    std::exception_ptr populateException() const;

   private:
    const std::vector<Tensor> outputTensors_;
    const std::vector<Tensor> inputTensors_;
    MPI_Request request_;
    MPI_Status status_{};
  };

  // Constructor spawns the worker thread loop.
  ProcessGroupMPI(int rank, int size, MPI_Comm pgComm, bool ownsCommunicator);
  ~ProcessGroupMPI();

  // Abort the MPI program; must be called when an exception is detected.
  void abort();

  static std::shared_ptr<ProcessGroupMPI> createProcessGroupMPI(
      std::vector<int> ranks = {});

  std::shared_ptr<GlooWork> broadcast(
      std::vector<Tensor>& tensors,
      int rootRank,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> broadcast(
      std::vector<Tensor>& tensors,
      const BroadcastOptions& options) {
    if (options.rootTensor != 0) {
      throw std::invalid_argument(
          "MPI broadcast supports only root tensor index 0");
    }
    return broadcast(tensors, options.rootRank, options.timeout);
  }
  std::shared_ptr<GlooWork> allreduce(
      std::vector<Tensor>& tensors,
      ReduceOp reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> allreduce(
      std::vector<Tensor>& tensors,
      const AllreduceOptions& options) {
    return allreduce(tensors, options.reduceOp, options.timeout);
  }
  std::shared_ptr<GlooWork> allreduce_sparse(
      std::vector<Tensor>& tensors,
      const AllreduceOptions& options);
  std::shared_ptr<GlooWork> allreduce_coalesced(
      std::vector<Tensor>& tensors,
      ReduceOp reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> allreduce_coalesced(
      std::vector<Tensor>& tensors,
      const AllreduceCoalescedOptions& options) {
    return allreduce_coalesced(tensors, options.reduceOp, options.timeout);
  }
  std::shared_ptr<GlooWork> reduce(
      std::vector<Tensor>& tensors,
      int rootRank,
      ReduceOp reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> reduce(
      std::vector<Tensor>& tensors,
      const ReduceOptions& options) {
    if (options.rootTensor != 0) {
      throw std::invalid_argument(
          "MPI reduce supports only root tensor index 0");
    }
    return reduce(tensors, options.rootRank, options.reduceOp, options.timeout);
  }
  std::shared_ptr<GlooWork> allgather(
      std::vector<std::vector<Tensor>>& outputTensors,
      std::vector<Tensor>& inputTensors,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> allgather(
      std::vector<std::vector<Tensor>>& outputTensors,
      std::vector<Tensor>& inputTensors,
      const AllgatherOptions& options) {
    return allgather(outputTensors, inputTensors, options.timeout);
  }
  std::shared_ptr<GlooWork> all_gather_single(
      Tensor& output,
      Tensor& input,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> all_gather_single(
      Tensor& output,
      Tensor& input,
      const AllgatherOptions& options) {
    return all_gather_single(output, input, options.timeout);
  }
  std::shared_ptr<GlooWork> allgather_coalesced(
      std::vector<std::vector<Tensor>>& outputTensors,
      std::vector<Tensor>& inputTensors,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> allgather_coalesced(
      std::vector<std::vector<Tensor>>& outputTensors,
      std::vector<Tensor>& inputTensors,
      const AllgatherOptions& options) {
    return allgather_coalesced(outputTensors, inputTensors, options.timeout);
  }
  std::shared_ptr<GlooWork> all_gather_single_coalesced(
      std::vector<Tensor>& outputs,
      std::vector<Tensor>& inputs,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> all_gather_single_coalesced(
      std::vector<Tensor>& outputs,
      std::vector<Tensor>& inputs,
      const AllgatherOptions& options) {
    return all_gather_single_coalesced(outputs, inputs, options.timeout);
  }
  std::shared_ptr<GlooWork> all_gather_into_tensor(
      Tensor& output,
      Tensor& input,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> all_gather_into_tensor(
      Tensor& output,
      Tensor& input,
      const AllgatherOptions& options) {
    return all_gather_into_tensor(output, input, options.timeout);
  }
  std::shared_ptr<GlooWork> gather_single(
      Tensor& output,
      Tensor& input,
      int rootRank,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> gather_single(
      Tensor& output,
      Tensor& input,
      const GatherOptions& options) {
    return gather_single(output, input, options.rootRank, options.timeout);
  }
  std::shared_ptr<GlooWork> gather(
      std::vector<std::vector<Tensor>>& outputTensors,
      std::vector<Tensor>& inputTensors,
      int rootRank,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> gather(
      std::vector<std::vector<Tensor>>& outputTensors,
      std::vector<Tensor>& inputTensors,
      const GatherOptions& options) {
    return gather(outputTensors, inputTensors, options.rootRank, options.timeout);
  }
  std::shared_ptr<GlooWork> scatter(
      std::vector<Tensor>& outputTensors,
      std::vector<std::vector<Tensor>>& inputTensors,
      int rootRank,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> scatter(
      std::vector<Tensor>& outputTensors,
      std::vector<std::vector<Tensor>>& inputTensors,
      const ScatterOptions& options) {
    return scatter(outputTensors, inputTensors, options.rootRank, options.timeout);
  }
  std::shared_ptr<GlooWork> reduce_scatter(
      std::vector<Tensor>& outputTensors,
      std::vector<std::vector<Tensor>>& inputTensors,
      ReduceOp reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> reduce_scatter(
      std::vector<Tensor>& outputTensors,
      std::vector<std::vector<Tensor>>& inputTensors,
      const ReduceScatterOptions& options) {
    return reduce_scatter(
        outputTensors, inputTensors, options.reduceOp, options.timeout);
  }
  std::shared_ptr<GlooWork> reduce_scatter_tensor(
      Tensor& output,
      Tensor& input,
      ReduceOp reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> reduce_scatter_tensor(
      Tensor& output,
      Tensor& input,
      const ReduceScatterOptions& options) {
    return reduce_scatter_tensor(output, input, options.reduceOp, options.timeout);
  }
  std::shared_ptr<GlooWork> reduce_scatter_single(
      Tensor& output,
      Tensor& input,
      ReduceOp reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> reduce_scatter_single(
      Tensor& output,
      Tensor& input,
      const ReduceScatterOptions& options) {
    return reduce_scatter_single(output, input, options.reduceOp, options.timeout);
  }
  std::shared_ptr<GlooWork> reduce_scatter_single_coalesced(
      std::vector<Tensor>& outputs,
      std::vector<Tensor>& inputs,
      ReduceOp reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> reduce_scatter_single_coalesced(
      std::vector<Tensor>& outputs,
      std::vector<Tensor>& inputs,
      const ReduceScatterOptions& options) {
    return reduce_scatter_single_coalesced(
        outputs, inputs, options.reduceOp, options.timeout);
  }
  std::shared_ptr<GlooWork> all_to_all_single(
      Tensor& outputTensor,
      Tensor& inputTensor,
      std::vector<int64_t> outputSplitSizes,
      std::vector<int64_t> inputSplitSizes,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> all_to_all_single(
      Tensor& outputTensor,
      Tensor& inputTensor,
      std::vector<int64_t> outputSplitSizes,
      std::vector<int64_t> inputSplitSizes,
      const AllToAllOptions& options) {
    return all_to_all_single(
        outputTensor,
        inputTensor,
        std::move(outputSplitSizes),
        std::move(inputSplitSizes),
        options.timeout);
  }
  std::shared_ptr<GlooWork> alltoall(
      std::vector<Tensor>& outputTensors,
      std::vector<Tensor>& inputTensors,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> alltoall(
      std::vector<Tensor>& outputTensors,
      std::vector<Tensor>& inputTensors,
      const AllToAllOptions& options) {
    return alltoall(outputTensors, inputTensors, options.timeout);
  }
  std::shared_ptr<GlooWork> send(
      std::vector<Tensor>& tensors,
      int dstRank,
      int tag);
  std::shared_ptr<GlooWork> recv(
      std::vector<Tensor>& tensors,
      int srcRank,
      int tag);
  std::shared_ptr<GlooWork> recvAnysource(
      std::vector<Tensor>& tensors,
      int tag);
  std::shared_ptr<GlooWork> barrier(std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> barrier(const BarrierOptions& options) {
    return barrier(options.timeout);
  }

  int getRank() const {
    return rank_;
  }
  int getSize() const {
    return size_;
  }

 protected:
  struct WorkEntry {
    WorkEntry(
        std::vector<Tensor>* srcPtr,
        std::vector<Tensor>* dstPtr,
        std::function<void(WorkEntry&)> run)
        : dst(dstPtr ? *dstPtr : std::vector<Tensor>()),
          run(std::move(run)) {
      if (srcPtr) {
        src = *srcPtr;
      }
    }

    std::vector<Tensor> src;
    std::vector<Tensor> dst;
    int* srcRank = nullptr;
    std::function<void(WorkEntry&)> run;
  };

  using WorkType =
      std::tuple<std::unique_ptr<WorkEntry>, std::shared_ptr<WorkMPI>>;

  void runLoop();
  void destroy();
  std::shared_ptr<GlooWork> enqueue(
      std::unique_ptr<WorkEntry> entry,
      std::string opName);

  bool stop_{false};

  std::mutex pgMutex_;
  std::thread workerThread_;

  std::deque<WorkType> queue_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;

  static void initMPIOnce();
  static void mpiExit();

  static std::mutex pgGlobalMutex_;
  static int mpiThreadSupport_;

  MPI_Comm pgComm_;
  bool ownsCommunicator_{false};
  bool destroyed_{false};

  int rank_{-1};
  int size_{-1};
};
#else
// Keep the type available so bindings can report missing runtime support.
class ProcessGroupMPI {
 public:
  static std::shared_ptr<ProcessGroupMPI> createProcessGroupMPI(
      std::vector<int> ranks = {});
};
#endif

} // namespace distributed
} // namespace tensorplay

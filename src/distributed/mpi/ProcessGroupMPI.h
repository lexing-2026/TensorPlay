#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Tensor.h>

#include "../gloo/ProcessGroupGloo.h"

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
// Only single-tensor collectives are supported: the input tensor vector of
// every operation must have exactly one element.
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
        std::string opName)
        : GlooWork(std::move(opName)),
          outputTensors_(std::move(outputTensors)),
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
    void populateException();

   private:
    const std::vector<Tensor> outputTensors_;
    MPI_Request request_;
    MPI_Status status_{};
  };

  // Constructor spawns the worker thread loop.
  ProcessGroupMPI(int rank, int size, MPI_Comm pgComm);
  ~ProcessGroupMPI();

  // Abort the MPI program; must be called when an exception is detected.
  void abort();

  static std::shared_ptr<ProcessGroupMPI> createProcessGroupMPI(
      std::vector<int> ranks = {});

  std::shared_ptr<GlooWork> broadcast(
      std::vector<Tensor>& tensors,
      int rootRank,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> allreduce(
      std::vector<Tensor>& tensors,
      int reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> allreduce_coalesced(
      std::vector<Tensor>& tensors,
      int reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> reduce(
      std::vector<Tensor>& tensors,
      int rootRank,
      int reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> allgather(
      std::vector<std::vector<Tensor>>& outputTensors,
      std::vector<Tensor>& inputTensors,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> all_gather_into_tensor(
      Tensor& output,
      Tensor& input,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> gather(
      std::vector<std::vector<Tensor>>& outputTensors,
      std::vector<Tensor>& inputTensors,
      int rootRank,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> scatter(
      std::vector<Tensor>& outputTensors,
      std::vector<std::vector<Tensor>>& inputTensors,
      int rootRank,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> reduce_scatter(
      std::vector<Tensor>& outputTensors,
      std::vector<std::vector<Tensor>>& inputTensors,
      int reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> reduce_scatter_tensor(
      Tensor& output,
      Tensor& input,
      int reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> all_to_all_single(
      Tensor& outputTensor,
      Tensor& inputTensor,
      std::vector<int64_t> outputSplitSizes,
      std::vector<int64_t> inputSplitSizes,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> alltoall(
      std::vector<Tensor>& outputTensors,
      std::vector<Tensor>& inputTensors,
      std::chrono::milliseconds timeout);
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

  int rank_{-1};
  int size_{-1};
};
#else
// Placeholder keeping the Python-side surface importable on builds without
// an MPI runtime; construction reports the missing support.
class ProcessGroupMPI {
 public:
  static std::shared_ptr<ProcessGroupMPI> createProcessGroupMPI(
      std::vector<int> ranks = {});
};
#endif

} // namespace distributed
} // namespace tensorplay

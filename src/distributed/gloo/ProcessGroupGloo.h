#pragma once

#include <pybind11/pybind11.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gloo/context.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/device.h>
#include <gloo/transport/unbound_buffer.h>

#include <Tensor.h>

#include "store/Store.h"

namespace tensorplay {
namespace distributed {

using Tensor = tensorplay::Tensor;
using Store = distributed::Store;

// Reduction op codes mirror the Python-layer ReduceOp values.
enum class ReduceOpCode : int {
  SUM = 0,
  PRODUCT = 1,
  MAX = 2,
  MIN = 3,
  AVG = 4,
};

// Base of every asynchronous work item. Completion state is guarded by
// waitMutex_; wait() blocks until the worker thread marks completion or an
// error was captured.
class GlooWork {
 public:
  explicit GlooWork(std::string opName = "")
      : opName_(std::move(opName)) {}
  virtual ~GlooWork() = default;

  virtual bool wait(int64_t timeout_ms = -1);
  virtual bool is_completed();
  virtual void abort() {
    // Abort support is optional per work type.
  }
  virtual int source_rank() const {
    return -1;
  }
  virtual std::vector<Tensor> result() const {
    return outputTensors_.empty() ? std::vector<Tensor>{}
                                  : outputTensors_.front();
  }

  const std::string& op_name() const {
    return opName_;
  }
  uint64_t seq() const {
    return seq_;
  }

  // Completion plumbing used by the worker thread.
  void finish();
  void finishWithError(std::exception_ptr eptr);

 protected:
  std::string opName_;
  uint64_t seq_{0};
  std::vector<std::vector<Tensor>> outputTensors_;

  mutable std::mutex waitMutex_;
  std::condition_variable waitCV_;
  bool completed_{false};
  std::exception_ptr exception_{nullptr};
};

// Work executed on a backend worker thread. `run` is invoked without any
// locks held; completion is signaled by the executor.
class GlooAsyncWork : public GlooWork {
 public:
  GlooAsyncWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<std::vector<Tensor>> outputTensors,
      std::string opName,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooWork(std::move(opName)),
        context_(std::move(context)),
        timeout_(
            timeout == std::chrono::milliseconds(-1)
                ? context_->getTimeout()
                : timeout) {
    outputTensors_ = std::move(outputTensors);
    seq_ = seq;
  }

  virtual void run() = 0;

  void execute() {
    try {
      run();
    } catch (...) {
      finishWithError(std::current_exception());
      return;
    }
    finish();
  }

  std::chrono::milliseconds getTimeout() const {
    return timeout_;
  }

 protected:
  std::shared_ptr<gloo::Context> context_;
  std::chrono::milliseconds timeout_;
};

// Point-to-point completions are driven entirely by the device I/O thread;
// the work object only synchronizes on the underlying unbound buffer.
class GlooRecvWork : public GlooWork {
 public:
  GlooRecvWork(
      Tensor tensor,
      std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
      uint64_t seq,
      std::string opName = "recv")
      : GlooWork(std::move(opName)),
        tensor_(std::move(tensor)),
        buffer_(std::move(buffer)) {
    seq_ = seq;
    outputTensors_ = {{tensor_}};
  }

  int source_rank() const override;
  bool wait(int64_t timeout_ms = -1) override;
  void abort() override {
    buffer_->abortWaitRecv();
  }

 protected:
  Tensor tensor_;
  std::unique_ptr<::gloo::transport::UnboundBuffer> buffer_;
  int srcRank_{-1};
};

struct GlooOptions {
  // Timeout applied to every collective issued through the group.
  std::chrono::milliseconds timeout{std::chrono::minutes(30)};
  // One context per device; each context carries its own I/O threads.
  std::vector<std::shared_ptr<::gloo::transport::Device>> devices;
  // Worker threads draining the collective queue.
  int threads{2};
  // Global ranks participating in this group (identity mapping when empty).
  std::vector<int64_t> global_ranks_in_group;
  std::string group_name;
};

// ProcessGroupGloo: CPU process group over the gloo transport library.
//
// All functions on this class are expected to be called in the same order
// across processes in the group; the increasing sequence number doubles as
// the collective tag so concurrent operations match up correctly.
class ProcessGroupGloo {
 public:
  explicit ProcessGroupGloo(
      std::shared_ptr<Store> store,
      int rank,
      int size,
      GlooOptions options);
  ~ProcessGroupGloo();

  int getRank() const {
    return rank_;
  }
  int getSize() const {
    return size_;
  }
  const std::string& groupName() const {
    return options_.group_name;
  }

  void setTimeout(std::chrono::milliseconds timeout) {
    options_.timeout = timeout;
    for (auto& context : contexts_) {
      context->setTimeout(timeout);
    }
  }

  // Device helpers.
  static std::shared_ptr<::gloo::transport::Device> createDeviceForInterface(
      const std::string& interface,
      bool lazyInit = false);
  static std::shared_ptr<::gloo::transport::Device> createDeviceForHostname(
      const std::string& hostname,
      bool lazyInit = false);
  static std::shared_ptr<::gloo::transport::Device> createDefaultDevice(
      bool lazyInit = false);

  // Collectives. All operate in-place on the given tensors, matching the
  // conventions of the Python layer; root ranks are group ranks.
  std::shared_ptr<GlooWork> broadcast(
      std::vector<Tensor>& tensors,
      int rootRank,
      int rootTensor,
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
      int rootTensor,
      int reduceOp,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> allgather(
      std::vector<std::vector<Tensor>>& outputs,
      std::vector<Tensor>& inputs,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> all_gather_into_tensor(
      Tensor& output,
      Tensor& input,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> gather(
      std::vector<std::vector<Tensor>>& outputs,
      std::vector<Tensor>& inputs,
      int rootRank,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> scatter(
      std::vector<Tensor>& outputs,
      std::vector<std::vector<Tensor>>& inputs,
      int rootRank,
      std::chrono::milliseconds timeout);
  std::shared_ptr<GlooWork> reduce_scatter(
      std::vector<Tensor>& outputs,
      std::vector<std::vector<Tensor>>& inputs,
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
      std::vector<int64_t> outputCounts,
      std::vector<int64_t> inputCounts,
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
  void monitoredBarrier(std::chrono::milliseconds timeout, bool waitAllRanks);

  uint64_t getSequenceNumberForGroup() const {
    return seq_;
  }

 protected:
  uint32_t nextTag();
  std::shared_ptr<::gloo::Context> getContext(uint32_t tag);
  void checkInitialized() const;
  void connectContexts(
      int rank,
      int size,
      std::shared_ptr<Store> store);
  void runLoop(int workerIndex);
  void enqueue(std::shared_ptr<GlooAsyncWork> work);
  // Runs synchronously on the calling thread (used by monitoredBarrier).
  void runInline(GlooAsyncWork* work);
  // Splits dim 0 of ``tensor`` into ``size`` equal contiguous chunks.
  std::vector<Tensor> splitEven(const Tensor& tensor);

  std::shared_ptr<Store> store_;
  GlooOptions options_;

  bool initialized_{false};

  // Every context is one fully connected replica set; multiple contexts
  // spread collectives over independent I/O threads round-robin.
  std::vector<std::shared_ptr<::gloo::Context>> contexts_;
  std::vector<std::thread> threads_;
  bool stop_{false};

  uint32_t collectiveCounter_{0};

  std::deque<std::shared_ptr<GlooAsyncWork>> workQueue_;
  std::vector<std::shared_ptr<GlooAsyncWork>> workInProgress_;
  std::mutex workMutex_;
  std::condition_variable workProduceCV_;
  std::condition_variable workConsumeCV_;
  uint64_t seq_{0};

  int rank_{-1};
  int size_{-1};
};

} // namespace distributed
} // namespace tensorplay

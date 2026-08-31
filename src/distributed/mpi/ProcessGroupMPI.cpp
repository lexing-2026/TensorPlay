#include "ProcessGroupMPI.h"

#ifdef USE_P10D_MPI

#include <mpi.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>

namespace tensorplay {
namespace distributed {

namespace {

#define MPI_CHECK(cmd)                                                     \
  do {                                                                     \
    int mpiStatus = cmd;                                                   \
    if (mpiStatus != MPI_SUCCESS) {                                        \
      std::string err = "MPI error in " + std::string(__FILE__) + ":" +    \
          std::to_string(__LINE__) + " with error code " +                 \
          std::to_string(mpiStatus);                                       \
      TP_THROW(RuntimeError, err);                                         \
    }                                                                      \
  } while (0)

[[noreturn]] void invalidArgument(const std::string& msg) {
  TP_THROW(ValueError, msg);
}

[[noreturn]] void runtimeFailure(const std::string& msg) {
  TP_THROW(RuntimeError, msg);
}

// Op mapping (codes mirror the Python-layer ReduceOp values).
MPI_Op mpiOpOf(int op) {
  switch (op) {
    case 0: // SUM
      return MPI_SUM;
    case 1: // PRODUCT
      return MPI_PROD;
    case 2: // MAX
      return MPI_MAX;
    case 3: // MIN
      return MPI_MIN;
    default:
      runtimeFailure("Unhandled reduce op for the MPI backend");
  }
}

// Type mapping.
MPI_Datatype mpiDatatypeOf(::tensorplay::ScalarType type) {
  switch (type) {
    case ::tensorplay::ScalarType::UInt8:
      return MPI_UNSIGNED_CHAR;
    case ::tensorplay::ScalarType::Int8:
      return MPI_CHAR;
    case ::tensorplay::ScalarType::Int16:
      return MPI_SHORT;
    case ::tensorplay::ScalarType::Int32:
      return MPI_INT;
    case ::tensorplay::ScalarType::Int64:
      return MPI_LONG;
    case ::tensorplay::ScalarType::Float32:
      return MPI_FLOAT;
    case ::tensorplay::ScalarType::Float64:
      return MPI_DOUBLE;
#if defined(MPIX_C_FLOAT16)
    case ::tensorplay::ScalarType::Float16:
      return MPIX_C_FLOAT16;
#endif
    default:
      runtimeFailure(
          "Tensor dtype is not supported by the MPI backend");
  }
}

void checkSingleTensorHelper(const Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    runtimeFailure("input tensor has to be contiguous");
  }
  if (tensor.device().is_cuda()) {
    runtimeFailure(
        "CUDA tensor detected; the MPI backend in this build is CPU-only");
  }
}

void checkSingleTensor(const std::vector<Tensor>& tensors) {
  if (tensors.size() != 1) {
    runtimeFailure("MPI process group does not support multi-tensor ops");
  }
  checkSingleTensorHelper(tensors[0]);
}

void checkSameSizeAndType(
    const Tensor& t_in,
    const std::vector<Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    if ((tensor.numel() != t_in.numel()) ||
        (tensor.dtype() != t_in.dtype())) {
      runtimeFailure("Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensor);
  }
}

Tensor newLikeFlat(const std::vector<Tensor>& tensors) {
  int64_t numel = 0;
  for (const auto& tensor : tensors) {
    numel += tensor.numel();
  }
  return Tensor::empty({numel}, tensors[0].dtype(), tensors[0].device());
}

std::chrono::milliseconds timeoutOr(std::chrono::milliseconds t) {
  return t == std::chrono::milliseconds(-1) ? std::chrono::milliseconds(0) : t;
}

} // namespace

ProcessGroupMPI::AsyncWork::~AsyncWork() {
  if (request_ != MPI_REQUEST_NULL) {
    std::cerr
        << "Attempted destruction of AsyncWork before work has completed, "
        << "terminating the program." << '\n';
    std::terminate();
  }
}

bool ProcessGroupMPI::AsyncWork::is_completed() {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  int flag = 0;
  MPI_CHECK(MPI_Test(&request_, &flag, &status_));
  if (request_ != MPI_REQUEST_NULL) {
    return false;
  }
  if (status_.MPI_ERROR != MPI_SUCCESS) {
    populateException();
  }
  return true;
}

int ProcessGroupMPI::AsyncWork::source_rank() const {
  return status_.MPI_SOURCE;
}

bool ProcessGroupMPI::AsyncWork::wait(int64_t /* timeout_ms */) {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  MPI_CHECK(MPI_Wait(&request_, &status_));
  auto ok = (status_.MPI_ERROR == MPI_SUCCESS);
  if (!ok) {
    populateException();
    std::rethrow_exception(exception_);
  }
  return true;
}

void ProcessGroupMPI::AsyncWork::abort() {
  TP_THROW(RuntimeError, "ProcessGroupMPI::AsyncWork::abort not implemented.");
}

void ProcessGroupMPI::AsyncWork::populateException() {
  std::array<char, MPI_MAX_ERROR_STRING> buf{};
  int len = (int)buf.size();
  MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
  exception_ = std::make_exception_ptr(
      std::runtime_error(std::string(buf.data(), (size_t)len)));
}

// Static global states
int ProcessGroupMPI::mpiThreadSupport_ = 0;
std::mutex ProcessGroupMPI::pgGlobalMutex_;

void ProcessGroupMPI::mpiExit() {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  MPI_CHECK(MPI_Finalize());
}

void ProcessGroupMPI::initMPIOnce() {
  static bool init_mpi_flag = []() {
    int mpi_was_initialized = 0;
    MPI_CHECK(MPI_Initialized(&mpi_was_initialized));
    if (mpi_was_initialized == 0) {
      MPI_CHECK(MPI_Init_thread(
          nullptr, nullptr, MPI_THREAD_SERIALIZED, &mpiThreadSupport_));
      if (mpiThreadSupport_ < MPI_THREAD_SERIALIZED) {
        TP_THROW(
            RuntimeError,
            "Used MPI implementation doesn't have the minimum level of "
            "threading support: MPI_THREAD_SERIALIZED. This is required by "
            "the distributed package");
      }
      if (std::atexit(ProcessGroupMPI::mpiExit)) {
        TP_THROW(RuntimeError, "Fail to register the MPI exit handler");
      }
    }
    return true;
  }();
  (void)init_mpi_flag;
}

std::shared_ptr<ProcessGroupMPI> ProcessGroupMPI::createProcessGroupMPI(
    std::vector<int> ranks) {
  initMPIOnce();

  MPI_Comm groupComm = MPI_COMM_WORLD;
  int rank = -1;
  int size = -1;

  {
    std::lock_guard<std::mutex> globalLock(pgGlobalMutex_);

    if (!ranks.empty()) {
      MPI_Group worldGroup{};
      MPI_Group ranksGroup{};
      MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));
      MPI_CHECK(
          MPI_Group_incl(worldGroup, (int)ranks.size(), ranks.data(), &ranksGroup));
      // MPI_Comm_create can be flaky in certain cases; retry a few times.
      constexpr int kMaxNumRetries = 3;
      bool groupComm_updated = false;
      MPI_Barrier(MPI_COMM_WORLD);
      for (int i = 0; i < kMaxNumRetries; ++i) {
        if (MPI_Comm_create(MPI_COMM_WORLD, ranksGroup, &groupComm)) {
          groupComm_updated = true;
          break;
        }
      }
      MPI_CHECK(groupComm_updated);
      MPI_CHECK(MPI_Group_free(&worldGroup));
      MPI_CHECK(MPI_Group_free(&ranksGroup));
    }

    if (groupComm != MPI_COMM_NULL) {
      MPI_CHECK(MPI_Comm_rank(groupComm, &rank));
      MPI_CHECK(MPI_Comm_size(groupComm, &size));
      if (rank < 0 || size < 0) {
        TP_THROW(RuntimeError, "Failed to get the world_size / rank");
      }
    }
  }

  if (groupComm == MPI_COMM_NULL) {
    return nullptr;
  }

  return std::shared_ptr<ProcessGroupMPI>(new ProcessGroupMPI(rank, size, groupComm));
}

ProcessGroupMPI::ProcessGroupMPI(int rank, int size, MPI_Comm pgComm)
    : pgComm_(pgComm), rank_(rank), size_(size) {
  if (pgComm_ == MPI_COMM_NULL) {
    TP_THROW(RuntimeError, "pgComm_ must not be MPI_COMM_NULL");
  }
  workerThread_ = std::thread(&ProcessGroupMPI::runLoop, this);
}

ProcessGroupMPI::~ProcessGroupMPI() {
  destroy();
}

void ProcessGroupMPI::destroy() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  queueConsumeCV_.wait(lock, [&] { return queue_.empty(); });
  stop_ = true;
  lock.unlock();
  queueProduceCV_.notify_all();
  workerThread_.join();
}

void ProcessGroupMPI::abort() {
  destroy();
  MPI_Abort(pgComm_, EXIT_FAILURE);
}

void ProcessGroupMPI::runLoop() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  while (!stop_) {
    if (queue_.empty()) {
      queueProduceCV_.wait(lock);
      continue;
    }
    auto workTuple = std::move(queue_.front());
    queue_.pop_front();
    auto& workEntry = std::get<0>(workTuple);
    auto& work = std::get<1>(workTuple);
    lock.unlock();
    queueConsumeCV_.notify_one();
    try {
      workEntry->run(*workEntry);
      work->finishWorkMPI();
    } catch (...) {
      work->finishWorkMPIError(std::current_exception());
    }
    lock.lock();
  }
}

std::shared_ptr<GlooWork> ProcessGroupMPI::enqueue(
    std::unique_ptr<WorkEntry> entry,
    std::string opName) {
  auto work = std::make_shared<WorkMPI>(entry->dst, std::move(opName));
  std::unique_lock<std::mutex> lock(pgMutex_);
  queue_.emplace_back(std::move(entry), work);
  lock.unlock();
  queueProduceCV_.notify_one();
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupMPI::broadcast(
    std::vector<Tensor>& tensors,
    int rootRank,
    std::chrono::milliseconds timeout) {
  checkSingleTensor(tensors);
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc =
      [rootRank, this](WorkEntry& entry) {
        auto data = entry.src[0];
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Bcast(
            data.data_ptr(),
            (int)data.numel(),
            mpiDatatypeOf(data.dtype()),
            rootRank,
            pgComm_));
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:broadcast");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::allreduce(
    std::vector<Tensor>& tensors,
    int reduceOp,
    std::chrono::milliseconds timeout) {
  checkSingleTensor(tensors);
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc =
      [reduceOp, this](WorkEntry& entry) {
        auto data = entry.src[0];
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Allreduce(
            MPI_IN_PLACE,
            data.data_ptr(),
            (int)data.numel(),
            mpiDatatypeOf(data.dtype()),
            mpiOpOf(reduceOp),
            pgComm_));
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:all_reduce");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::allreduce_coalesced(
    std::vector<Tensor>& tensors,
    int reduceOp,
    std::chrono::milliseconds timeout) {
  (void)tensors;
  (void)reduceOp;
  (void)timeout;
  TP_THROW(RuntimeError, "allreduce_coalesced is currently not supported with MPI");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::reduce(
    std::vector<Tensor>& tensors,
    int rootRank,
    int reduceOp,
    std::chrono::milliseconds timeout) {
  checkSingleTensor(tensors);
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc =
      [rootRank, reduceOp, this](WorkEntry& entry) {
        auto data = entry.src[0];
        auto dataPtr = data.data_ptr();
        void* sendbuf = (rank_ == rootRank) ? MPI_IN_PLACE : dataPtr;
        void* recvbuf = (rank_ == rootRank) ? dataPtr : nullptr;
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Reduce(
            sendbuf,
            recvbuf,
            (int)data.numel(),
            mpiDatatypeOf(data.dtype()),
            mpiOpOf(reduceOp),
            rootRank,
            pgComm_));
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:reduce");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::allgather(
    std::vector<std::vector<Tensor>>& outputTensors,
    std::vector<Tensor>& inputTensors,
    std::chrono::milliseconds timeout) {
  checkSingleTensor(inputTensors);
  (void)timeout;
  if (outputTensors.size() != 1) {
    runtimeFailure("MPI process group only supports a single tensor op");
  }
  if ((int64_t)size_ != (int64_t)outputTensors[0].size()) {
    runtimeFailure("All gather: number of output tensors should equal world size");
  }
  checkSameSizeAndType(inputTensors[0], outputTensors[0]);

  std::function<void(WorkEntry&)> runFunc = [this](WorkEntry& entry) {
    auto data = entry.src[0];
    std::vector<Tensor> outputDataVec = entry.dst;
    auto flatOutputTensor = newLikeFlat(outputDataVec);
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Allgather(
        data.data_ptr(),
        (int)data.numel(),
        mpiDatatypeOf(data.dtype()),
        flatOutputTensor.data_ptr(),
        (int)data.numel(),
        mpiDatatypeOf(data.dtype()),
        pgComm_));
    for (size_t i = 0; i < outputDataVec.size(); ++i) {
      outputDataVec[i].copy_(
          flatOutputTensor.narrow(0, (int64_t)i, 1).reshape(
              outputDataVec[i].shape()));
    }
  };
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors[0], std::move(runFunc));
  return enqueue(std::move(entry), "mpi:all_gather");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::all_gather_into_tensor(
    Tensor& output,
    Tensor& input,
    std::chrono::milliseconds timeout) {
  if (output.numel() != input.numel() * (int64_t)size_) {
    runtimeFailure(
        "All gather: output tensor size must equal input tensor size times "
        "the world size");
  }
  checkSingleTensorHelper(input);
  checkSingleTensorHelper(output);
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc = [this](WorkEntry& entry) {
    auto dstdata = entry.dst[0];
    auto srcdata = entry.src[0];
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Allgather(
        srcdata.data_ptr(),
        (int)srcdata.numel(),
        mpiDatatypeOf(srcdata.dtype()),
        dstdata.data_ptr(),
        (int)srcdata.numel(),
        mpiDatatypeOf(dstdata.dtype()),
        pgComm_));
  };
  auto inputTensors = std::vector<Tensor>({input});
  auto outputTensors = std::vector<Tensor>({output});
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:_allgather_base");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::gather(
    std::vector<std::vector<Tensor>>& outputTensors,
    std::vector<Tensor>& inputTensors,
    int rootRank,
    std::chrono::milliseconds timeout) {
  checkSingleTensor(inputTensors);
  (void)timeout;

  if (rank_ != rootRank) {
    if (!outputTensors.empty()) {
      runtimeFailure("Gather: number of output tensors should be 0 for non-root");
    }
  } else {
    if (outputTensors.size() != 1) {
      runtimeFailure("Gather: multi-tensor collective is not supported");
    }
    if ((int64_t)size_ != (int64_t)outputTensors[0].size()) {
      runtimeFailure("Gather: number of output tensors should equal world size");
    }
    checkSameSizeAndType(inputTensors[0], outputTensors[0]);
  }

  std::function<void(WorkEntry&)> runFunc =
      [rootRank, this](WorkEntry& entry) {
        auto data = entry.src[0];
        void* recvbuf = nullptr;
        Tensor flatOutputTensor;

        std::vector<Tensor> dstdata = entry.dst;
        if (rank_ == rootRank) {
          flatOutputTensor = newLikeFlat(dstdata);
          recvbuf = flatOutputTensor.data_ptr();
        }

        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Gather(
            data.data_ptr(),
            (int)data.numel(),
            mpiDatatypeOf(data.dtype()),
            recvbuf,
            (int)data.numel(),
            mpiDatatypeOf(data.dtype()),
            rootRank,
            pgComm_));

        if (rank_ == rootRank) {
          std::vector<Tensor>& outputDataVec = entry.dst;
          for (size_t i = 0; i < outputDataVec.size(); ++i) {
            outputDataVec.at(i).copy_(
                flatOutputTensor.narrow(0, (int64_t)i, 1).reshape(
                    outputDataVec.at(i).shape()));
          }
        }
      };

  if (rank_ == rootRank) {
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors[0], std::move(runFunc));
    return enqueue(std::move(entry), "mpi:gather");
  }
  auto entry =
      std::make_unique<WorkEntry>(&inputTensors, nullptr, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:gather");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::scatter(
    std::vector<Tensor>& outputTensors,
    std::vector<std::vector<Tensor>>& inputTensors,
    int rootRank,
    std::chrono::milliseconds timeout) {
  checkSingleTensor(outputTensors);
  (void)timeout;

  if (rank_ != rootRank) {
    if (!inputTensors.empty()) {
      runtimeFailure("Scatter: number of input tensors should be 0 for non-root");
    }
  } else {
    if (inputTensors.size() != 1) {
      runtimeFailure("Scatter: multi-tensor collective is not supported");
    }
    if ((int64_t)size_ != (int64_t)inputTensors[0].size()) {
      runtimeFailure("Scatter: number of input tensors should equal world size");
    }
    checkSameSizeAndType(outputTensors[0], inputTensors[0]);
  }

  std::function<void(WorkEntry&)> runFunc =
      [rootRank, this](WorkEntry& entry) {
        auto data = entry.dst[0];
        void* sendbuf = nullptr;
        Tensor flatInputTensor;

        if (rank_ == rootRank) {
          std::vector<Tensor>& inputDataVec = entry.src;
          flatInputTensor = newLikeFlat(inputDataVec);
          sendbuf = flatInputTensor.data_ptr();
          for (size_t i = 0; i < inputDataVec.size(); ++i) {
            flatInputTensor.narrow(0, (int64_t)i, 1).reshape(
                inputDataVec.at(i).shape()).copy_(inputDataVec.at(i));
          }
        }

        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Scatter(
            sendbuf,
            (int)data.numel(),
            mpiDatatypeOf(data.dtype()),
            data.data_ptr(),
            (int)data.numel(),
            mpiDatatypeOf(data.dtype()),
            rootRank,
            pgComm_));
      };

  if (rank_ == rootRank) {
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors[0], &outputTensors, std::move(runFunc));
    return enqueue(std::move(entry), "mpi:scatter");
  }
  auto entry = std::make_unique<WorkEntry>(
      nullptr, &outputTensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:scatter");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::reduce_scatter(
    std::vector<Tensor>& outputTensors,
    std::vector<std::vector<Tensor>>& inputTensors,
    int reduceOp,
    std::chrono::milliseconds timeout) {
  checkSingleTensor(outputTensors);
  (void)timeout;
  if (inputTensors.size() != 1) {
    runtimeFailure("MPI process group only supports a single tensor op");
  }
  if ((int64_t)size_ != (int64_t)inputTensors[0].size()) {
    runtimeFailure(
        "Reduce scatter: number of input tensors should equal world size");
  }
  checkSameSizeAndType(outputTensors[0], inputTensors[0]);

  std::function<void(WorkEntry&)> runFunc =
      [reduceOp, this](WorkEntry& entry) {
        auto data = entry.dst[0];
        auto flatInputTensor = newLikeFlat(entry.src);
        for (size_t i = 0; i < entry.src.size(); ++i) {
          flatInputTensor.narrow(0, (int64_t)i, 1).reshape(
              entry.src[i].shape()).copy_(entry.src[i]);
        }
        int recvcount = (int)(flatInputTensor.numel() / (int64_t)size_);

        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Reduce_scatter_block(
            flatInputTensor.data_ptr(),
            data.data_ptr(),
            recvcount,
            mpiDatatypeOf(data.dtype()),
            mpiOpOf(reduceOp),
            pgComm_));
      };

  auto entry = std::make_unique<WorkEntry>(
      &inputTensors[0], &outputTensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:reduce_scatter");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::reduce_scatter_tensor(
    Tensor& output,
    Tensor& input,
    int reduceOp,
    std::chrono::milliseconds timeout) {
  if (output.numel() * (int64_t)size_ != input.numel()) {
    runtimeFailure(
        "Reduce scatter: input tensor size must equal output tensor size "
        "times the world size");
  }
  checkSingleTensorHelper(output);
  checkSingleTensorHelper(input);
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc =
      [reduceOp, this](WorkEntry& entry) {
        auto dstdata = entry.dst[0];
        auto srcdata = entry.src[0];
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Reduce_scatter_block(
            srcdata.data_ptr(),
            dstdata.data_ptr(),
            (int)dstdata.numel(),
            mpiDatatypeOf(srcdata.dtype()),
            mpiOpOf(reduceOp),
            pgComm_));
      };

  auto inputTensors = std::vector<Tensor>({input});
  auto outputTensors = std::vector<Tensor>({output});
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:_reduce_scatter_base");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::all_to_all_single(
    Tensor& outputTensor,
    Tensor& inputTensor,
    std::vector<int64_t> outputSplitSizes,
    std::vector<int64_t> inputSplitSizes,
    std::chrono::milliseconds timeout) {
  checkSingleTensorHelper(inputTensor);
  checkSingleTensorHelper(outputTensor);
  (void)timeout;

  if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
    if (!(outputTensor.numel() == inputTensor.numel() &&
          outputTensor.dtype() == inputTensor.dtype())) {
      runtimeFailure("Tensors are not equal in size or data type");
    }
    if (outputTensor.size(0) % size_ != 0) {
      runtimeFailure(
          "Tensor's dim 0 does not divide equally across group size");
    }

    std::function<void(WorkEntry&)> runFunc = [this](WorkEntry& entry) {
      auto srcdata = entry.src[0];
      auto dstdata = entry.dst[0];
      std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
      MPI_CHECK(MPI_Alltoall(
          srcdata.data_ptr(),
          (int)(srcdata.numel() / (int64_t)size_),
          mpiDatatypeOf(srcdata.dtype()),
          dstdata.data_ptr(),
          (int)(dstdata.numel() / (int64_t)size_),
          mpiDatatypeOf(dstdata.dtype()),
          pgComm_));
    };
    std::vector<Tensor> inputTensors = {inputTensor};
    std::vector<Tensor> outputTensors = {outputTensor};
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors, std::move(runFunc));
    return enqueue(std::move(entry), "mpi:all_to_all");
  }

  // Variable-size path (alltoallv).
  if (inputSplitSizes.size() != (size_t)size_ ||
      outputSplitSizes.size() != (size_t)size_) {
    runtimeFailure("Number of split sizes must equal group size");
  }
  int64_t inSum = 0;
  for (auto v : inputSplitSizes) inSum += v;
  if (inSum != inputTensor.size(0)) {
    runtimeFailure("Split sizes doesn't match total dim 0 size");
  }
  int64_t outSum = 0;
  for (auto v : outputSplitSizes) outSum += v;
  if (outSum != outputTensor.size(0)) {
    runtimeFailure("Split sizes doesn't match total dim 0 size");
  }

  std::function<void(WorkEntry&)> runFunc =
      [this, inputSplitSizes, outputSplitSizes](WorkEntry& entry) {
        auto srcdata = entry.src[0];
        auto dstdata = entry.dst[0];
        std::vector<int> send_lengths(size_);
        std::vector<int> recv_lengths(size_);
        std::vector<int> send_offsets(size_);
        std::vector<int> recv_offsets(size_);
        int64_t send_offset = 0;
        for (int i = 0; i < size_; ++i) {
          send_lengths[i] = (int)inputSplitSizes[i];
          send_offsets[i] = (int)send_offset;
          send_offset += inputSplitSizes[i];
        }
        int64_t recv_offset = 0;
        for (int i = 0; i < size_; ++i) {
          recv_lengths[i] = (int)outputSplitSizes[i];
          recv_offsets[i] = (int)recv_offset;
          recv_offset += outputSplitSizes[i];
        }
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Alltoallv(
            srcdata.data_ptr(),
            send_lengths.data(),
            send_offsets.data(),
            mpiDatatypeOf(srcdata.dtype()),
            dstdata.data_ptr(),
            recv_lengths.data(),
            recv_offsets.data(),
            mpiDatatypeOf(dstdata.dtype()),
            pgComm_));
      };
  std::vector<Tensor> inputTensors = {inputTensor};
  std::vector<Tensor> outputTensors = {outputTensor};
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:all_to_all");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::alltoall(
    std::vector<Tensor>& outputTensors,
    std::vector<Tensor>& inputTensors,
    std::chrono::milliseconds timeout) {
  (void)timeout;
  if ((int64_t)inputTensors.size() != size_) {
    runtimeFailure("Number of input tensors are not equal to group size");
  }
  if ((int64_t)outputTensors.size() != size_) {
    runtimeFailure("Number of output tensors are not equal to group size");
  }
  std::function<void(WorkEntry&)> runFunc = [this](WorkEntry& entry) {
    std::vector<int> send_lengths(size_);
    std::vector<int> recv_lengths(size_);
    std::vector<int> send_offsets(size_);
    std::vector<int> recv_offsets(size_);
    auto srcdata = entry.src;
    auto dstdata = entry.dst;
    int64_t src_len = 0;
    int64_t dst_len = 0;
    for (int i = 0; i < size_; ++i) {
      send_lengths[i] = (int)srcdata[i].numel();
      send_offsets[i] = (int)src_len;
      src_len += srcdata[i].numel();
      recv_lengths[i] = (int)dstdata[i].numel();
      recv_offsets[i] = (int)dst_len;
      dst_len += dstdata[i].numel();
    }
    Tensor srcFlatData = Tensor::empty(
        {src_len}, srcdata[0].dtype(), srcdata[0].device());
    Tensor dstFlatData = Tensor::empty(
        {dst_len}, dstdata[0].dtype(), dstdata[0].device());
    int64_t cursor = 0;
    for (int i = 0; i < size_; ++i) {
      srcFlatData.narrow(0, cursor, srcdata[i].numel())
          .reshape(srcdata[i].shape())
          .copy_(srcdata[i]);
      cursor += srcdata[i].numel();
    }
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Alltoallv(
        srcFlatData.data_ptr(),
        send_lengths.data(),
        send_offsets.data(),
        mpiDatatypeOf(srcdata[0].dtype()),
        dstFlatData.data_ptr(),
        recv_lengths.data(),
        recv_offsets.data(),
        mpiDatatypeOf(dstdata[0].dtype()),
        pgComm_));
    cursor = 0;
    for (int i = 0; i < size_; ++i) {
      dstdata[i].copy_(
          dstFlatData.narrow(0, cursor, dstdata[i].numel()).reshape(
              dstdata[i].shape()));
      cursor += dstdata[i].numel();
    }
  };
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:all_to_all");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::send(
    std::vector<Tensor>& tensors,
    int dstRank,
    int tag) {
  checkSingleTensor(tensors);
  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;
  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Isend(
        tensor.data_ptr(),
        (int)tensor.numel(),
        mpiDatatypeOf(tensor.dtype()),
        dstRank,
        tag,
        pgComm_,
        &request));
  }
  return std::make_shared<AsyncWork>(
      request, std::vector<Tensor>(), "mpi:send");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::recv(
    std::vector<Tensor>& tensors,
    int srcRank,
    int tag) {
  checkSingleTensor(tensors);
  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;
  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        (int)tensor.numel(),
        mpiDatatypeOf(tensor.dtype()),
        srcRank,
        tag,
        pgComm_,
        &request));
  }
  return std::make_shared<AsyncWork>(request, tensors, "mpi:recv");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::recvAnysource(
    std::vector<Tensor>& tensors,
    int tag) {
  checkSingleTensor(tensors);
  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;
  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        (int)tensor.numel(),
        mpiDatatypeOf(tensor.dtype()),
        MPI_ANY_SOURCE,
        tag,
        pgComm_,
        &request));
  }
  return std::make_shared<AsyncWork>(request, tensors, "mpi:recvAnySource");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::barrier(
    std::chrono::milliseconds timeout) {
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc = [this](WorkEntry&) {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Barrier(pgComm_));
  };
  auto entry = std::make_unique<WorkEntry>(nullptr, nullptr, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:barrier");
}

} // namespace distributed
} // namespace tensorplay

#else // !USE_P10D_MPI

#include "ProcessGroupMPI.h"

#include <vector>

namespace tensorplay {
namespace distributed {

std::shared_ptr<ProcessGroupMPI> ProcessGroupMPI::createProcessGroupMPI(
    std::vector<int> ranks) {
  (void)ranks;
  TP_THROW(
      RuntimeError,
      "Distributed package doesn't have MPI built in. MPI is only included "
      "if the package is built on a host that has MPI installed.");
}

} // namespace distributed
} // namespace tensorplay

#endif // USE_P10D_MPI

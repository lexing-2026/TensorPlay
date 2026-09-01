#include "ProcessGroupMPI.h"

#ifdef USE_P10D_MPI

#include <mpi.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
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

void checkRootRank(int rootRank, int size, const char* op) {
  if (rootRank < 0 || rootRank >= size) {
    invalidArgument(
        std::string(op) + ": root rank " + std::to_string(rootRank) +
        " is out of range for group of size " + std::to_string(size));
  }
}

void checkPeerRank(int peerRank, int size, const char* op) {
  if (peerRank < 0 || peerRank >= size) {
    invalidArgument(
        std::string(op) + ": peer rank " + std::to_string(peerRank) +
        " is out of range for group of size " + std::to_string(size));
  }
}

int mpiCount(int64_t value, const char* op) {
  if (value < 0 || value > std::numeric_limits<int>::max()) {
    runtimeFailure(
        std::string(op) + ": tensor element count exceeds the MPI limit");
  }
  return static_cast<int>(value);
}

// Op mapping for the distributed reduction operations.
MPI_Op mpiOpOf(ReduceOp op) {
  switch (op.op()) {
    case ReduceOp::SUM:
      return MPI_SUM;
    case ReduceOp::PRODUCT:
      return MPI_PROD;
    case ReduceOp::MIN:
      return MPI_MIN;
    case ReduceOp::MAX:
      return MPI_MAX;
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
#if defined(MPIX_C_BF16)
    case ::tensorplay::ScalarType::BFloat16:
      return MPIX_C_BF16;
#elif defined(MPIX_BFLOAT16)
    case ::tensorplay::ScalarType::BFloat16:
      return MPIX_BFLOAT16;
#endif
    default:
      runtimeFailure(
          "Tensor dtype is not supported by the MPI backend");
  }
}

void checkSingleTensorHelper(const Tensor& tensor) {
  if (!tensor.defined()) {
    runtimeFailure("input tensor is undefined");
  }
  if (!tensor.is_contiguous()) {
    runtimeFailure("input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    runtimeFailure("input tensor has to be dense");
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
  checkSingleTensorHelper(t_in);
  for (const auto& tensor : tensors) {
    if ((tensor.numel() != t_in.numel()) ||
        (tensor.dtype() != t_in.dtype())) {
      runtimeFailure("Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensor);
  }
}

void checkSameDtype(
    const Tensor& t_in,
    const std::vector<Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    if (tensor.dtype() != t_in.dtype()) {
      runtimeFailure("Tensors are not equal in data type");
    }
  }
}

void checkSplitSizes(
    const std::vector<int64_t>& splitSizes,
    const Tensor& tensor,
    int groupSize) {
  if (tensor.dim() == 0) {
    runtimeFailure("all_to_all_single requires tensors with a dimension 0");
  }
  for (const auto splitSize : splitSizes) {
    if (splitSize < 0) {
      runtimeFailure("Split sizes must be non-negative");
    }
  }
  if (splitSizes.empty()) {
    if (tensor.size(0) % groupSize != 0) {
      runtimeFailure(
          "Tensor's dim 0 does not divide equally across group size");
    }
    return;
  }
  if (splitSizes.size() != static_cast<size_t>(groupSize)) {
    runtimeFailure("Number of tensor split sizes not equal to group size");
  }
  const int64_t sum =
      std::accumulate(splitSizes.begin(), splitSizes.end(), int64_t{0});
  if (sum != tensor.size(0)) {
    runtimeFailure("Split sizes doesn't match total dim 0 size");
  }
}

void computeLengthsAndOffsets(
    const std::vector<int64_t>& splitSizes,
    const Tensor& tensor,
    int groupSize,
    std::vector<int>* lengths,
    std::vector<int>* offsets) {
  const int64_t dim0Size = tensor.size(0);
  const int64_t rowSize = dim0Size == 0 ? 1 : tensor.numel() / dim0Size;
  const bool equalSplits = splitSizes.empty();
  const int64_t equalSplitSize =
      equalSplits ? dim0Size / static_cast<int64_t>(groupSize) : 0;
  lengths->resize(static_cast<size_t>(groupSize));
  offsets->resize(static_cast<size_t>(groupSize));
  int64_t offset = 0;
  for (int i = 0; i < groupSize; ++i) {
    const int64_t splitSize =
        equalSplits ? equalSplitSize : splitSizes[static_cast<size_t>(i)];
    if (splitSize > 0 && rowSize > std::numeric_limits<int64_t>::max() / splitSize) {
      runtimeFailure("all_to_all_single split size overflow");
    }
    const int64_t length = rowSize * splitSize;
    (*lengths)[static_cast<size_t>(i)] = mpiCount(length, "MPI alltoallv");
    (*offsets)[static_cast<size_t>(i)] = mpiCount(offset, "MPI alltoallv");
    if (length > std::numeric_limits<int64_t>::max() - offset) {
      runtimeFailure("all_to_all_single offset overflow");
    }
    offset += length;
  }
}

void computeLengthsAndOffsets(
    const std::vector<Tensor>& tensors,
    std::vector<int>* lengths,
    std::vector<int>* offsets) {
  lengths->resize(tensors.size());
  offsets->resize(tensors.size());
  int64_t offset = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    (*lengths)[i] = mpiCount(tensors[i].numel(), "MPI alltoallv");
    (*offsets)[i] = mpiCount(offset, "MPI alltoallv");
    if (tensors[i].numel() > std::numeric_limits<int64_t>::max() - offset) {
      runtimeFailure("alltoall tensor offset overflow");
    }
    offset += tensors[i].numel();
  }
}

Tensor newLikeFlat(const std::vector<Tensor>& tensors) {
  if (tensors.empty()) {
    runtimeFailure("Received an empty tensor list");
  }
  std::vector<int64_t> sizes = tensors[0].shape();
  sizes.insert(sizes.begin(), static_cast<int64_t>(tensors.size()));
  return Tensor::empty(sizes, tensors[0].dtype(), tensors[0].device());
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
    finishWithError(exception_);
  } else {
    finish();
  }
  return true;
}

int ProcessGroupMPI::AsyncWork::source_rank() const {
  return status_.MPI_SOURCE;
}

bool ProcessGroupMPI::AsyncWork::wait(int64_t /* timeout_ms */) {
  if (request_ == MPI_REQUEST_NULL) {
    if (exception_ != nullptr) {
      std::rethrow_exception(exception_);
    }
    return true;
  }
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  MPI_CHECK(MPI_Wait(&request_, &status_));
  auto ok = (status_.MPI_ERROR == MPI_SUCCESS);
  if (!ok) {
    populateException();
    finishWithError(exception_);
    std::rethrow_exception(exception_);
  }
  finish();
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
        if (MPI_Comm_create(MPI_COMM_WORLD, ranksGroup, &groupComm) ==
            MPI_SUCCESS) {
          groupComm_updated = true;
          break;
        }
      }
      if (!groupComm_updated) {
        runtimeFailure("Failed to create the MPI process group");
      }
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
  if (size_ <= 0 || rank_ < 0 || rank_ >= size_) {
    TP_THROW(RuntimeError, "Invalid process-group rank or size");
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
  checkRootRank(rootRank, size_, "Broadcast");
  checkSingleTensor(tensors);
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc =
      [rootRank, this](WorkEntry& entry) {
        auto data = entry.src[0];
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Bcast(
            data.data_ptr(),
            mpiCount(data.numel(), "MPI broadcast"),
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
    ReduceOp reduceOp,
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
            mpiCount(data.numel(), "MPI allreduce"),
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
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  (void)tensors;
  (void)reduceOp;
  (void)timeout;
  TP_THROW(RuntimeError, "allreduce_coalesced is currently not supported with MPI");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::reduce(
    std::vector<Tensor>& tensors,
    int rootRank,
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  checkRootRank(rootRank, size_, "Reduce");
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
            mpiCount(data.numel(), "MPI reduce"),
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
            mpiCount(data.numel(), "MPI allgather"),
            mpiDatatypeOf(data.dtype()),
            flatOutputTensor.data_ptr(),
            mpiCount(data.numel(), "MPI allgather"),
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

std::shared_ptr<GlooWork> ProcessGroupMPI::all_gather_single(
    Tensor& output,
    Tensor& input,
    std::chrono::milliseconds timeout) {
  checkSingleTensorHelper(input);
  checkSingleTensorHelper(output);
  if (output.numel() != input.numel() * (int64_t)size_) {
    runtimeFailure(
        "All gather: output tensor size must equal input tensor size times "
        "the world size");
  }
  if (output.dtype() != input.dtype()) {
    runtimeFailure("Tensors are not equal in data type");
  }
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc = [this](WorkEntry& entry) {
    auto dstdata = entry.dst[0];
    auto srcdata = entry.src[0];
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Allgather(
        srcdata.data_ptr(),
        mpiCount(srcdata.numel(), "MPI allgather"),
        mpiDatatypeOf(srcdata.dtype()),
        dstdata.data_ptr(),
        mpiCount(srcdata.numel(), "MPI allgather"),
        mpiDatatypeOf(dstdata.dtype()),
        pgComm_));
  };
  auto inputTensors = std::vector<Tensor>({input});
  auto outputTensors = std::vector<Tensor>({output});
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:all_gather_single");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::allgather_coalesced(
    std::vector<std::vector<Tensor>>& outputTensors,
    std::vector<Tensor>& inputTensors,
    std::chrono::milliseconds timeout) {
  (void)outputTensors;
  (void)inputTensors;
  (void)timeout;
  TP_THROW(RuntimeError, "ProcessGroupMPI does not support allgather_coalesced");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::all_gather_into_tensor(
    Tensor& output,
    Tensor& input,
    std::chrono::milliseconds timeout) {
  checkSingleTensorHelper(input);
  checkSingleTensorHelper(output);
  if (output.numel() != input.numel() * (int64_t)size_) {
    runtimeFailure(
        "All gather: output tensor size must equal input tensor size times "
        "the world size");
  }
  if (output.dtype() != input.dtype()) {
    runtimeFailure("Tensors are not equal in data type");
  }
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc = [this](WorkEntry& entry) {
    auto dstdata = entry.dst[0];
    auto srcdata = entry.src[0];
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Allgather(
        srcdata.data_ptr(),
        mpiCount(srcdata.numel(), "MPI allgather"),
        mpiDatatypeOf(srcdata.dtype()),
        dstdata.data_ptr(),
        mpiCount(srcdata.numel(), "MPI allgather"),
        mpiDatatypeOf(dstdata.dtype()),
        pgComm_));
  };
  auto inputTensors = std::vector<Tensor>({input});
  auto outputTensors = std::vector<Tensor>({output});
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:all_gather_into_tensor");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::gather_single(
    Tensor& output,
    Tensor& input,
    int rootRank,
    std::chrono::milliseconds timeout) {
  checkRootRank(rootRank, size_, "Gather");
  checkSingleTensorHelper(input);
  if (getRank() == rootRank) {
    checkSingleTensorHelper(output);
    if (output.numel() != input.numel() * static_cast<int64_t>(size_)) {
      runtimeFailure(
          "Gather: output tensor size must equal input tensor size times "
          "the world size");
    }
    if (output.dtype() != input.dtype()) {
      runtimeFailure("Tensors are not equal in data type");
    }
  }
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc =
      [rootRank, this](WorkEntry& entry) {
        auto srcdata = entry.src[0];
        void* recvbuf = nullptr;
        if (rank_ == rootRank) {
          recvbuf = entry.dst[0].data_ptr();
        }
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Gather(
            srcdata.data_ptr(),
            mpiCount(srcdata.numel(), "MPI gather"),
            mpiDatatypeOf(srcdata.dtype()),
            recvbuf,
            mpiCount(srcdata.numel(), "MPI gather"),
            mpiDatatypeOf(srcdata.dtype()),
            rootRank,
            pgComm_));
      };
  auto inputTensors = std::vector<Tensor>({input});
  auto outputTensors = std::vector<Tensor>({output});
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:gather");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::gather(
    std::vector<std::vector<Tensor>>& outputTensors,
    std::vector<Tensor>& inputTensors,
    int rootRank,
    std::chrono::milliseconds timeout) {
  checkRootRank(rootRank, size_, "Gather");
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
            mpiCount(data.numel(), "MPI gather"),
            mpiDatatypeOf(data.dtype()),
            recvbuf,
            mpiCount(data.numel(), "MPI gather"),
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
  checkRootRank(rootRank, size_, "Scatter");
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
            mpiCount(data.numel(), "MPI scatter"),
            mpiDatatypeOf(data.dtype()),
            data.data_ptr(),
            mpiCount(data.numel(), "MPI scatter"),
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
    ReduceOp reduceOp,
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
        int recvcount = mpiCount(
            flatInputTensor.numel() / (int64_t)size_,
            "MPI reduce scatter");

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
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  checkSingleTensorHelper(output);
  checkSingleTensorHelper(input);
  if (output.numel() * (int64_t)size_ != input.numel()) {
    runtimeFailure(
        "Reduce scatter: input tensor size must equal output tensor size "
        "times the world size");
  }
  if (output.dtype() != input.dtype()) {
    runtimeFailure("Tensors are not equal in data type");
  }
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc =
      [reduceOp, this](WorkEntry& entry) {
        auto dstdata = entry.dst[0];
        auto srcdata = entry.src[0];
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Reduce_scatter_block(
            srcdata.data_ptr(),
            dstdata.data_ptr(),
            mpiCount(dstdata.numel(), "MPI reduce scatter"),
            mpiDatatypeOf(srcdata.dtype()),
            mpiOpOf(reduceOp),
            pgComm_));
      };

  auto inputTensors = std::vector<Tensor>({input});
  auto outputTensors = std::vector<Tensor>({output});
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:reduce_scatter_tensor");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::reduce_scatter_single(
    Tensor& output,
    Tensor& input,
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  if (output.numel() * (int64_t)size_ != input.numel()) {
    runtimeFailure(
        "Reduce scatter: input tensor size must equal output tensor size "
        "times the world size");
  }
  checkSingleTensorHelper(output);
  checkSingleTensorHelper(input);
  if (output.dtype() != input.dtype()) {
    runtimeFailure("Tensors are not equal in data type");
  }
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc =
      [reduceOp, this](WorkEntry& entry) {
        auto dstdata = entry.dst[0];
        auto srcdata = entry.src[0];
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Reduce_scatter_block(
            srcdata.data_ptr(),
            dstdata.data_ptr(),
            mpiCount(dstdata.numel(), "MPI reduce scatter"),
            mpiDatatypeOf(srcdata.dtype()),
            mpiOpOf(reduceOp),
            pgComm_));
      };
  auto inputTensors = std::vector<Tensor>({input});
  auto outputTensors = std::vector<Tensor>({output});
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:reduce_scatter_single");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::all_to_all_single(
    Tensor& outputTensor,
    Tensor& inputTensor,
    std::vector<int64_t> outputSplitSizes,
    std::vector<int64_t> inputSplitSizes,
    std::chrono::milliseconds timeout) {
  checkSingleTensorHelper(inputTensor);
  checkSingleTensorHelper(outputTensor);
  if (inputTensor.dtype() != outputTensor.dtype()) {
    runtimeFailure("Tensors are not equal in data type");
  }
  if (inputTensor.dim() == 0 || outputTensor.dim() == 0) {
    runtimeFailure("all_to_all_single requires tensors with a dimension 0");
  }
  (void)timeout;

  if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
    checkSplitSizes(outputSplitSizes, outputTensor, size_);
    if (outputTensor.numel() != inputTensor.numel()) {
      runtimeFailure("Tensors are not equal in size or data type");
    }

    std::function<void(WorkEntry&)> runFunc = [this](WorkEntry& entry) {
      auto srcdata = entry.src[0];
      auto dstdata = entry.dst[0];
      std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
      MPI_CHECK(MPI_Alltoall(
          srcdata.data_ptr(),
          mpiCount(srcdata.numel() / (int64_t)size_, "MPI alltoall"),
          mpiDatatypeOf(srcdata.dtype()),
          dstdata.data_ptr(),
          mpiCount(dstdata.numel() / (int64_t)size_, "MPI alltoall"),
          mpiDatatypeOf(dstdata.dtype()),
          pgComm_));
    };
    std::vector<Tensor> inputTensors = {inputTensor};
    std::vector<Tensor> outputTensors = {outputTensor};
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors, std::move(runFunc));
    return enqueue(std::move(entry), "mpi:all_to_all");
  }

  checkSplitSizes(inputSplitSizes, inputTensor, size_);
  checkSplitSizes(outputSplitSizes, outputTensor, size_);

  std::function<void(WorkEntry&)> runFunc =
      [this, inputSplitSizes, outputSplitSizes](WorkEntry& entry) {
        auto srcdata = entry.src[0];
        auto dstdata = entry.dst[0];
        std::vector<int> send_lengths(size_);
        std::vector<int> recv_lengths(size_);
        std::vector<int> send_offsets(size_);
        std::vector<int> recv_offsets(size_);
        computeLengthsAndOffsets(
            inputSplitSizes,
            srcdata,
            size_,
            &send_lengths,
            &send_offsets);
        computeLengthsAndOffsets(
            outputSplitSizes,
            dstdata,
            size_,
            &recv_lengths,
            &recv_offsets);
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
  for (const auto& tensor : inputTensors) {
    checkSingleTensorHelper(tensor);
  }
  for (const auto& tensor : outputTensors) {
    checkSingleTensorHelper(tensor);
  }
  checkSameDtype(inputTensors[0], inputTensors);
  checkSameDtype(inputTensors[0], outputTensors);
  std::function<void(WorkEntry&)> runFunc = [this](WorkEntry& entry) {
    std::vector<int> send_lengths(size_);
    std::vector<int> recv_lengths(size_);
    std::vector<int> send_offsets(size_);
    std::vector<int> recv_offsets(size_);
    auto srcdata = entry.src;
    auto dstdata = entry.dst;
    computeLengthsAndOffsets(srcdata, &send_lengths, &send_offsets);
    computeLengthsAndOffsets(dstdata, &recv_lengths, &recv_offsets);
    int64_t src_len = 0;
    int64_t dst_len = 0;
    for (int i = 0; i < size_; ++i) {
      src_len += srcdata[i].numel();
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
  checkPeerRank(dstRank, size_, "Send");
  if (tag < 0) {
    invalidArgument("Send: tag must be non-negative");
  }
  checkSingleTensor(tensors);
  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;
  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Isend(
        tensor.data_ptr(),
        mpiCount(tensor.numel(), "MPI send"),
        mpiDatatypeOf(tensor.dtype()),
        dstRank,
        tag,
        pgComm_,
        &request));
  }
  return std::make_shared<AsyncWork>(
      request, std::vector<Tensor>(), "mpi:send", tensors);
}

std::shared_ptr<GlooWork> ProcessGroupMPI::recv(
    std::vector<Tensor>& tensors,
    int srcRank,
    int tag) {
  checkPeerRank(srcRank, size_, "Receive");
  if (tag < 0) {
    invalidArgument("Receive: tag must be non-negative");
  }
  checkSingleTensor(tensors);
  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;
  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        mpiCount(tensor.numel(), "MPI receive"),
        mpiDatatypeOf(tensor.dtype()),
        srcRank,
        tag,
        pgComm_,
        &request));
  }
  return std::make_shared<AsyncWork>(request, tensors, "mpi:recv", tensors);
}

std::shared_ptr<GlooWork> ProcessGroupMPI::recvAnysource(
    std::vector<Tensor>& tensors,
    int tag) {
  if (tag < 0) {
    invalidArgument("Receive: tag must be non-negative");
  }
  checkSingleTensor(tensors);
  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;
  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        mpiCount(tensor.numel(), "MPI receive"),
        mpiDatatypeOf(tensor.dtype()),
        MPI_ANY_SOURCE,
        tag,
        pgComm_,
        &request));
  }
  return std::make_shared<AsyncWork>(
      request, tensors, "mpi:recvAnySource", tensors);
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

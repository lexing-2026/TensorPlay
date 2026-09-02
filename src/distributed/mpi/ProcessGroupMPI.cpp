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
    case ::tensorplay::ScalarType::ComplexFloat:
      return MPI_C_FLOAT_COMPLEX;
    case ::tensorplay::ScalarType::ComplexDouble:
      return MPI_C_DOUBLE_COMPLEX;
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
  int64_t sum = 0;
  for (const auto splitSize : splitSizes) {
    if (splitSize > std::numeric_limits<int64_t>::max() - sum) {
      runtimeFailure("all_to_all split sizes overflow the index range");
    }
    sum += splitSize;
  }
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
    if (tensors[i].numel() >
        std::numeric_limits<int64_t>::max() - offset) {
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

int64_t checkedProduct(int64_t left, int64_t right, const char* op) {
  if (left < 0 || right < 0) {
    runtimeFailure(
        std::string(op) + ": tensor metadata contains a negative size");
  }
  if (right != 0 && left > std::numeric_limits<int64_t>::max() / right) {
    runtimeFailure(
        std::string(op) + ": tensor metadata size overflows the index range");
  }
  return left * right;
}

bool productEquals(int64_t left, int64_t right, int64_t expected) {
  if (left < 0 || right < 0 || expected < 0) {
    return false;
  }
  if (right != 0 && left > std::numeric_limits<int64_t>::max() / right) {
    return false;
  }
  return left * right == expected;
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
  std::exception_ptr exception;
  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    if (request_ == MPI_REQUEST_NULL) {
      return true;
    }
    int flag = 0;
    MPI_CHECK(MPI_Test(&request_, &flag, &status_));
    if (request_ != MPI_REQUEST_NULL) {
      return false;
    }
    if (status_.MPI_ERROR != MPI_SUCCESS) {
      exception = populateException();
    }
  }
  if (exception != nullptr) {
    finishWithError(exception);
  } else {
    finish();
  }
  return true;
}

int ProcessGroupMPI::AsyncWork::source_rank() const {
  std::lock_guard<std::mutex> globalLock(pgGlobalMutex_);
  return status_.MPI_SOURCE;
}

bool ProcessGroupMPI::AsyncWork::wait(int64_t timeout_ms) {
  if (timeout_ms < 0) {
    std::exception_ptr exception;
    bool alreadyCompleted = false;
    {
      std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
      if (request_ == MPI_REQUEST_NULL) {
        alreadyCompleted = true;
      } else {
        MPI_CHECK(MPI_Wait(&request_, &status_));
        if (status_.MPI_ERROR != MPI_SUCCESS) {
          exception = populateException();
        }
      }
    }
    if (alreadyCompleted) {
      std::lock_guard<std::mutex> waitLock(waitMutex_);
      exception = exception_;
    }
    if (exception != nullptr) {
      finishWithError(exception);
      std::rethrow_exception(exception);
    }
    finish();
    return true;
  }

  const auto timeout = std::chrono::milliseconds(timeout_ms);
  const auto start = std::chrono::steady_clock::now();
  while (true) {
    std::exception_ptr exception;
    bool completed = false;
    {
      std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
      if (request_ == MPI_REQUEST_NULL) {
        completed = true;
      } else {
        int flag = 0;
        MPI_CHECK(MPI_Test(&request_, &flag, &status_));
        completed = request_ == MPI_REQUEST_NULL;
        if (completed && status_.MPI_ERROR != MPI_SUCCESS) {
          exception = populateException();
        }
      }
    }
    if (completed) {
      if (exception == nullptr) {
        std::lock_guard<std::mutex> waitLock(waitMutex_);
        exception = exception_;
      }
      if (exception != nullptr) {
        finishWithError(exception);
        std::rethrow_exception(exception);
      }
      finish();
      return true;
    }
    if (std::chrono::steady_clock::now() - start >= timeout) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void ProcessGroupMPI::AsyncWork::abort() {
  int cancelStatus = MPI_SUCCESS;
  int waitStatus = MPI_SUCCESS;
  std::exception_ptr exception;
  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    if (request_ == MPI_REQUEST_NULL) {
      return;
    }
    cancelStatus = MPI_Cancel(&request_);
    waitStatus = MPI_Wait(&request_, &status_);
    if (waitStatus != MPI_SUCCESS) {
      exception = std::make_exception_ptr(std::runtime_error(
        "MPI request wait failed while aborting asynchronous work"));
    } else if (cancelStatus != MPI_SUCCESS) {
      exception = std::make_exception_ptr(std::runtime_error(
        "MPI request cancellation failed while aborting asynchronous work"));
    } else {
      exception = std::make_exception_ptr(
        std::runtime_error("MPI asynchronous work was aborted"));
    }
    request_ = MPI_REQUEST_NULL;
  }
  finishWithError(exception);
}

std::exception_ptr ProcessGroupMPI::AsyncWork::populateException() const {
  std::array<char, MPI_MAX_ERROR_STRING> buf{};
  int len = (int)buf.size();
  MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
  return std::make_exception_ptr(
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
    } else {
      MPI_CHECK(MPI_Query_thread(&mpiThreadSupport_));
      if (mpiThreadSupport_ < MPI_THREAD_SERIALIZED) {
        TP_THROW(
            RuntimeError,
            "The active MPI runtime does not provide the minimum level of "
            "threading support required by the distributed package");
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
      if (ranks.size() >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
        runtimeFailure("MPI process-group rank list is too large");
      }
      MPI_Group worldGroup{};
      MPI_Group ranksGroup{};
      MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));
      MPI_CHECK(
          MPI_Group_incl(
              worldGroup,
              static_cast<int>(ranks.size()),
              ranks.data(),
              &ranksGroup));
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

  return std::shared_ptr<ProcessGroupMPI>(
      new ProcessGroupMPI(rank, size, groupComm, !ranks.empty()));
}

ProcessGroupMPI::ProcessGroupMPI(
    int rank,
    int size,
    MPI_Comm pgComm,
    bool ownsCommunicator)
    : pgComm_(pgComm),
      ownsCommunicator_(ownsCommunicator),
      rank_(rank),
      size_(size) {
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
  {
    std::unique_lock<std::mutex> lock(pgMutex_);
    if (destroyed_) {
      return;
    }
    queueConsumeCV_.wait(lock, [&] { return queue_.empty(); });
    stop_ = true;
    destroyed_ = true;
  }
  queueProduceCV_.notify_all();
  if (workerThread_.joinable()) {
    workerThread_.join();
  }

  MPI_Comm communicator = MPI_COMM_NULL;
  {
    std::lock_guard<std::mutex> globalLock(pgGlobalMutex_);
    if (ownsCommunicator_) {
      communicator = pgComm_;
      pgComm_ = MPI_COMM_NULL;
    }
  }
  if (communicator != MPI_COMM_NULL) {
    int finalized = 0;
    if (MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0) {
      const int status = MPI_Comm_free(&communicator);
      if (status != MPI_SUCCESS) {
        std::cerr << "Failed to release the MPI process-group communicator"
                  << '\n';
      }
    }
  }
}

void ProcessGroupMPI::abort() {
  destroy();
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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
  if (stop_ || destroyed_) {
    runtimeFailure("MPI process group has already been destroyed");
  }
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
  if (!tensors.empty() && tensors[0].is_sparse()) {
    AllreduceOptions options;
    options.reduceOp = reduceOp;
    options.timeout = timeout;
    return allreduce_sparse(tensors, options);
  }
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

std::shared_ptr<GlooWork> ProcessGroupMPI::allreduce_sparse(
    std::vector<Tensor>& tensors,
    const AllreduceOptions& options) {
  if (tensors.empty()) {
    runtimeFailure("MPI allreduce_sparse requires a non-empty tensor list");
  }
  if (options.reduceOp != ReduceOp::SUM) {
    runtimeFailure("MPI allreduce_sparse supports SUM only");
  }

  const auto& reference = tensors[0];
  if (!reference.defined()) {
    runtimeFailure("MPI allreduce_sparse received an undefined tensor");
  }
  if (!reference.device().is_cpu()) {
    runtimeFailure("MPI allreduce_sparse only supports CPU tensors");
  }
  if (!reference.is_sparse() || reference.is_sparse_csr()) {
    runtimeFailure(
        "MPI allreduce_sparse requires sparse coordinate tensors");
  }
  for (const auto& tensor : tensors) {
    if (!tensor.defined() || !tensor.device().is_cpu() ||
        !tensor.is_sparse() || tensor.is_sparse_csr()) {
      runtimeFailure(
          "MPI allreduce_sparse requires matching sparse coordinate tensors");
    }
    if (tensor.dtype() != reference.dtype() ||
        tensor.shape() != reference.shape() ||
        tensor.sparse_dim() != reference.sparse_dim() ||
        tensor.dense_dim() != reference.dense_dim()) {
      runtimeFailure(
          "MPI allreduce_sparse tensors must have matching dtype, shape, and "
          "sparse dimensions");
    }
  }

  std::function<void(WorkEntry&)> runFunc = [this](WorkEntry& entry) {
    Tensor input = entry.src[0];
    for (size_t index = 1; index < entry.src.size(); ++index) {
      input = Tensor::sparse_add(input, entry.src[index]);
    }
    if (!input.is_coalesced()) {
      input = input.coalesce();
    }

    struct Metadata {
      int64_t sparseDim{0};
      int64_t denseDim{0};
      int64_t nnz{0};
      std::vector<int64_t> sizes;
    };

    const int64_t sparseDim = input.sparse_dim();
    const int64_t denseDim = input.dense_dim();
    const int64_t nnz = input._values().size(0);
    const auto sizes = static_cast<std::vector<int64_t>>(input.shape());
    const int64_t metadataCount = checkedProduct(
        1, static_cast<int64_t>(sizes.size()) + 3, "MPI sparse metadata");
    std::vector<int64_t> metadataPayload;
    metadataPayload.reserve(static_cast<size_t>(metadataCount));
    metadataPayload.push_back(sparseDim);
    metadataPayload.push_back(denseDim);
    metadataPayload.push_back(nnz);
    metadataPayload.insert(
        metadataPayload.end(), sizes.begin(), sizes.end());

    const int metadataCountInt =
        mpiCount(metadataCount, "MPI sparse metadata");
    std::vector<int> metadataCounts(static_cast<size_t>(size_));
    std::vector<int> metadataOffsets(static_cast<size_t>(size_));
    {
      std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
      MPI_CHECK(MPI_Allgather(
          &metadataCountInt,
          1,
          MPI_INT,
          metadataCounts.data(),
          1,
          MPI_INT,
          pgComm_));
    }

    int64_t totalMetadata = 0;
    for (int rank = 0; rank < size_; ++rank) {
      metadataOffsets[static_cast<size_t>(rank)] =
          mpiCount(totalMetadata, "MPI sparse metadata");
      const int64_t count = metadataCounts[static_cast<size_t>(rank)];
      if (count < 0 ||
          count > std::numeric_limits<int64_t>::max() - totalMetadata) {
        runtimeFailure("MPI sparse metadata exceeds the index range");
      }
      totalMetadata += count;
    }
    std::vector<int64_t> gatheredMetadata(
        static_cast<size_t>(totalMetadata));
    {
      std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
      MPI_CHECK(MPI_Allgatherv(
          metadataPayload.data(),
          metadataCountInt,
          mpiDatatypeOf(::tensorplay::ScalarType::Int64),
          gatheredMetadata.data(),
          metadataCounts.data(),
          metadataOffsets.data(),
          mpiDatatypeOf(::tensorplay::ScalarType::Int64),
          pgComm_));
    }

    std::vector<Metadata> metadata;
    metadata.reserve(static_cast<size_t>(size_));
    size_t metadataOffset = 0;
    for (int rank = 0; rank < size_; ++rank) {
      const size_t payloadSize =
          static_cast<size_t>(metadataCounts[static_cast<size_t>(rank)]);
      if (payloadSize < 3 || payloadSize > gatheredMetadata.size() ||
          metadataOffset > gatheredMetadata.size() - payloadSize) {
        runtimeFailure("MPI sparse metadata payload is truncated");
      }
      const auto* payload = gatheredMetadata.data() + metadataOffset;
      Metadata item;
      item.sparseDim = payload[0];
      item.denseDim = payload[1];
      item.nnz = payload[2];
      if (item.sparseDim < 0 || item.denseDim < 0 || item.nnz < 0) {
        runtimeFailure("MPI sparse metadata contains a negative dimension");
      }
      if (item.sparseDim >
          std::numeric_limits<int64_t>::max() - item.denseDim) {
        runtimeFailure("MPI sparse metadata dimension count overflows");
      }
      const int64_t dimensionCount =
          item.sparseDim + item.denseDim;
      if (dimensionCount > std::numeric_limits<int64_t>::max() - 3 ||
          dimensionCount != static_cast<int64_t>(sizes.size()) ||
          payloadSize != static_cast<size_t>(dimensionCount + 3)) {
        runtimeFailure("MPI sparse metadata has an invalid dimension count");
      }
      item.sizes.assign(
          payload + 3, payload + 3 + static_cast<size_t>(dimensionCount));
      for (const auto size : item.sizes) {
        if (size < 0) {
          runtimeFailure("MPI sparse metadata contains a negative size");
        }
      }
      if (item.sparseDim != sparseDim || item.denseDim != denseDim ||
          item.sizes != sizes) {
        runtimeFailure("MPI sparse tensor dimensions do not match across ranks");
      }
      metadata.push_back(std::move(item));
      metadataOffset += payloadSize;
    }

    int64_t denseNumel = 1;
    for (int64_t dim = sparseDim;
         dim < static_cast<int64_t>(sizes.size());
         ++dim) {
      denseNumel = checkedProduct(denseNumel, sizes[static_cast<size_t>(dim)],
                                  "MPI sparse values");
    }

    std::vector<int> indexCounts(static_cast<size_t>(size_));
    std::vector<int> indexOffsets(static_cast<size_t>(size_));
    int64_t totalIndices = 0;
    for (int rank = 0; rank < size_; ++rank) {
      const int64_t count = checkedProduct(
          metadata[static_cast<size_t>(rank)].nnz,
          sparseDim,
          "MPI sparse indices");
      indexOffsets[static_cast<size_t>(rank)] =
          mpiCount(totalIndices, "MPI sparse indices");
      indexCounts[static_cast<size_t>(rank)] =
          mpiCount(count, "MPI sparse indices");
      if (count > std::numeric_limits<int64_t>::max() - totalIndices) {
        runtimeFailure("MPI sparse indices exceed the index range");
      }
      totalIndices += count;
    }
    Tensor localIndices = input._indices().contiguous();
    Tensor gatheredIndices = Tensor::empty(
        {totalIndices},
        ::tensorplay::DType::Int64,
        input.device());
    {
      std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
      MPI_CHECK(MPI_Allgatherv(
          localIndices.data_ptr<int64_t>(),
          mpiCount(localIndices.numel(), "MPI sparse indices"),
          mpiDatatypeOf(::tensorplay::ScalarType::Int64),
          gatheredIndices.data_ptr<int64_t>(),
          indexCounts.data(),
          indexOffsets.data(),
          mpiDatatypeOf(::tensorplay::ScalarType::Int64),
          pgComm_));
    }

    const bool complexValues = isComplexType(input.dtype());
    Tensor localValues = input._values().contiguous();
    Tensor transportValues = complexValues ? localValues.view_as_real()
                                           : localValues;
    std::vector<int> valueCounts(static_cast<size_t>(size_));
    std::vector<int> valueOffsets(static_cast<size_t>(size_));
    int64_t totalValues = 0;
    for (int rank = 0; rank < size_; ++rank) {
      const int64_t logicalCount = checkedProduct(
          metadata[static_cast<size_t>(rank)].nnz,
          denseNumel,
          "MPI sparse values");
      const int64_t wireCount = complexValues
          ? checkedProduct(logicalCount, 2, "MPI sparse values")
          : logicalCount;
      valueOffsets[static_cast<size_t>(rank)] =
          mpiCount(totalValues, "MPI sparse values");
      valueCounts[static_cast<size_t>(rank)] =
          mpiCount(wireCount, "MPI sparse values");
      if (wireCount > std::numeric_limits<int64_t>::max() - totalValues) {
        runtimeFailure("MPI sparse values exceed the index range");
      }
      totalValues += wireCount;
    }
    Tensor gatheredValues = Tensor::empty(
        {totalValues}, transportValues.dtype(), input.device());
    {
      std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
      MPI_CHECK(MPI_Allgatherv(
          transportValues.data_ptr(),
          mpiCount(transportValues.numel(), "MPI sparse values"),
          mpiDatatypeOf(transportValues.dtype()),
          gatheredValues.data_ptr(),
          valueCounts.data(),
          valueOffsets.data(),
          mpiDatatypeOf(transportValues.dtype()),
          pgComm_));
    }

    Tensor output;
    int64_t indexOffset = 0;
    int64_t valueOffset = 0;
    std::vector<int64_t> denseShape(
        sizes.begin() + static_cast<size_t>(sparseDim), sizes.end());
    for (int rank = 0; rank < size_; ++rank) {
      const auto& item = metadata[static_cast<size_t>(rank)];
      const int64_t indexCount = checkedProduct(
          item.nnz, sparseDim, "MPI sparse indices");
      const int64_t logicalValueCount = checkedProduct(
          item.nnz, denseNumel, "MPI sparse values");
      const int64_t wireValueCount = complexValues
          ? checkedProduct(logicalValueCount, 2, "MPI sparse values")
          : logicalValueCount;
      Tensor peerIndices = gatheredIndices.narrow(0, indexOffset, indexCount)
          .reshape({sparseDim, item.nnz});
      Tensor peerValues = gatheredValues.narrow(0, valueOffset, wireValueCount);
      std::vector<int64_t> valueShape{item.nnz};
      valueShape.insert(valueShape.end(), denseShape.begin(), denseShape.end());
      if (complexValues) {
        valueShape.push_back(2);
        peerValues = peerValues.reshape(valueShape).view_as_complex();
      } else {
        peerValues = peerValues.reshape(valueShape);
      }
      Tensor peer = Tensor::make_sparse_coo_tensor(
          peerIndices, peerValues, sizes, true);
      output = output.defined() ? Tensor::sparse_add(output, peer) : peer;
      indexOffset += indexCount;
      valueOffset += wireValueCount;
    }
    output = output.coalesce();
    for (auto& tensor : entry.src) {
      auto indices = output._indices().clone();
      auto values = output._values().clone();
      tensor.unsafeGetTensorImpl()->set_sparse_state(
          indices.unsafeGetTensorImpl(),
          values.unsafeGetTensorImpl(),
          sizes,
          output.is_coalesced());
    }
  };

  auto entry = std::make_unique<WorkEntry>(
      &tensors, &tensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:sparse_all_reduce");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::allreduce_coalesced(
    std::vector<Tensor>& tensors,
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  if (tensors.empty()) {
    runtimeFailure("MPI allreduce_coalesced requires a non-empty tensor list");
  }
  for (const auto& tensor : tensors) {
    checkSingleTensorHelper(tensor);
    if (tensor.dtype() != tensors[0].dtype() ||
        tensor.device() != tensors[0].device()) {
      runtimeFailure(
          "MPI allreduce_coalesced tensors must share dtype and device");
    }
  }
  (void)timeout;
  std::function<void(WorkEntry&)> runFunc =
      [reduceOp, this](WorkEntry& entry) {
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        for (auto& data : entry.src) {
          MPI_CHECK(MPI_Allreduce(
              MPI_IN_PLACE,
              data.data_ptr(),
              mpiCount(data.numel(), "MPI allreduce_coalesced"),
              mpiDatatypeOf(data.dtype()),
              mpiOpOf(reduceOp),
              pgComm_));
        }
      };
  auto entry = std::make_unique<WorkEntry>(
      &tensors, &tensors, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:all_reduce_coalesced");
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
  if (!productEquals(input.numel(), static_cast<int64_t>(size_), output.numel())) {
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
  if (inputTensors.empty()) {
    runtimeFailure(
        "MPI allgather_coalesced requires a non-empty input tensor list");
  }
  if (outputTensors.size() != static_cast<size_t>(size_)) {
    runtimeFailure(
        "MPI allgather_coalesced output list count must equal group size");
  }
  for (const auto& input : inputTensors) {
    checkSingleTensorHelper(input);
  }
  for (const auto& outputList : outputTensors) {
    if (outputList.size() != inputTensors.size()) {
      runtimeFailure(
          "MPI allgather_coalesced output lists must match input list size");
    }
    for (size_t index = 0; index < inputTensors.size(); ++index) {
      const auto& input = inputTensors[index];
      const auto& output = outputList[index];
      checkSingleTensorHelper(output);
      if (output.shape() != input.shape() ||
          output.dtype() != input.dtype() ||
          output.device() != input.device()) {
        runtimeFailure(
            "MPI allgather_coalesced output tensors do not match inputs");
      }
    }
  }

  std::vector<Tensor> flattenedOutputs;
  flattenedOutputs.reserve(outputTensors.size() * inputTensors.size());
  for (const auto& outputList : outputTensors) {
    flattenedOutputs.insert(
        flattenedOutputs.end(), outputList.begin(), outputList.end());
  }
  (void)timeout;
  const size_t tensorCount = inputTensors.size();
  std::function<void(WorkEntry&)> runFunc =
      [this, tensorCount](WorkEntry& entry) {
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        for (size_t index = 0; index < tensorCount; ++index) {
          std::vector<Tensor> outputs;
          outputs.reserve(static_cast<size_t>(size_));
          for (int rank = 0; rank < size_; ++rank) {
            outputs.push_back(
                entry.dst[static_cast<size_t>(rank) * tensorCount + index]);
          }
          auto flatOutput = newLikeFlat(outputs);
          const auto& input = entry.src[index];
          MPI_CHECK(MPI_Allgather(
              input.data_ptr(),
              mpiCount(input.numel(), "MPI allgather_coalesced"),
              mpiDatatypeOf(input.dtype()),
              flatOutput.data_ptr(),
              mpiCount(input.numel(), "MPI allgather_coalesced"),
              mpiDatatypeOf(input.dtype()),
              pgComm_));
          for (int rank = 0; rank < size_; ++rank) {
            auto& output = outputs[static_cast<size_t>(rank)];
            output.copy_(flatOutput
                             .narrow(0, static_cast<int64_t>(rank), 1)
                             .reshape(output.shape()));
          }
        }
      };
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &flattenedOutputs, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:all_gather_coalesced");
}

std::shared_ptr<GlooWork> ProcessGroupMPI::all_gather_single_coalesced(
    std::vector<Tensor>& outputs,
    std::vector<Tensor>& inputs,
    std::chrono::milliseconds timeout) {
  if (outputs.size() != inputs.size()) {
    runtimeFailure(
        "MPI all_gather_single_coalesced input/output tensor lists must "
        "have the same length");
  }
  if (inputs.empty()) {
    runtimeFailure(
        "MPI all_gather_single_coalesced requires a non-empty tensor list");
  }

  std::vector<std::vector<Tensor>> outputLists(
      static_cast<size_t>(size_));
  for (size_t index = 0; index < inputs.size(); ++index) {
    const auto& input = inputs[index];
    auto& output = outputs[index];
    checkSingleTensorHelper(input);
    checkSingleTensorHelper(output);
    if (input.dtype() != output.dtype() || input.device() != output.device()) {
      runtimeFailure(
          "MPI all_gather_single_coalesced input/output tensor types do not "
          "match");
    }

    const auto inputShape = static_cast<std::vector<int64_t>>(input.shape());
    const auto outputShape =
        static_cast<std::vector<int64_t>>(output.shape());
    auto expectedShape = inputShape;
    if (inputShape.empty()) {
      expectedShape = {static_cast<int64_t>(size_)};
    } else {
      if (inputShape[0] >
          std::numeric_limits<int64_t>::max() / static_cast<int64_t>(size_)) {
        runtimeFailure(
            "MPI all_gather_single_coalesced output shape overflows");
      }
      expectedShape[0] *= static_cast<int64_t>(size_);
    }
    if (outputShape != expectedShape) {
      runtimeFailure(
          "MPI all_gather_single_coalesced output shape is invalid");
    }

    if (inputShape.empty()) {
      for (int rank = 0; rank < size_; ++rank) {
        outputLists[static_cast<size_t>(rank)].push_back(
            output.narrow(0, rank, 1).reshape(inputShape));
      }
    } else {
      const int64_t chunk = inputShape[0];
      for (int rank = 0; rank < size_; ++rank) {
        outputLists[static_cast<size_t>(rank)].push_back(
            output.narrow(0, static_cast<int64_t>(rank) * chunk, chunk));
      }
    }
  }
  return allgather_coalesced(outputLists, inputs, timeout);
}

std::shared_ptr<GlooWork> ProcessGroupMPI::all_gather_into_tensor(
    Tensor& output,
    Tensor& input,
    std::chrono::milliseconds timeout) {
  checkSingleTensorHelper(input);
  checkSingleTensorHelper(output);
  if (!productEquals(input.numel(), static_cast<int64_t>(size_), output.numel())) {
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
    if (!productEquals(
            input.numel(), static_cast<int64_t>(size_), output.numel())) {
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
  if (!productEquals(
          output.numel(), static_cast<int64_t>(size_), input.numel())) {
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
  if (!productEquals(
          output.numel(), static_cast<int64_t>(size_), input.numel())) {
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

std::shared_ptr<GlooWork> ProcessGroupMPI::reduce_scatter_single_coalesced(
    std::vector<Tensor>& outputs,
    std::vector<Tensor>& inputs,
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  if (outputs.size() != inputs.size()) {
    runtimeFailure(
        "MPI reduce_scatter_single_coalesced input/output tensor lists must "
        "have the same length");
  }
  if (inputs.empty()) {
    runtimeFailure(
        "MPI reduce_scatter_single_coalesced requires a non-empty tensor list");
  }
  for (size_t index = 0; index < inputs.size(); ++index) {
    const auto& input = inputs[index];
    const auto& output = outputs[index];
    checkSingleTensorHelper(input);
    checkSingleTensorHelper(output);
    if (input.dtype() != output.dtype() || input.device() != output.device()) {
      runtimeFailure(
          "MPI reduce_scatter_single_coalesced input/output tensor types do "
          "not match");
    }
    if (!productEquals(
            output.numel(), static_cast<int64_t>(size_), input.numel())) {
      runtimeFailure(
          "MPI reduce_scatter_single_coalesced input size must equal output "
          "size times the world size");
    }
  }

  (void)timeout;
  const size_t tensorCount = inputs.size();
  std::function<void(WorkEntry&)> runFunc =
      [reduceOp, tensorCount, this](WorkEntry& entry) {
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        for (size_t index = 0; index < tensorCount; ++index) {
          const auto& input = entry.src[index];
          auto& output = entry.dst[index];
          MPI_CHECK(MPI_Reduce_scatter_block(
              input.data_ptr(),
              output.data_ptr(),
              mpiCount(output.numel(), "MPI reduce_scatter_single_coalesced"),
              mpiDatatypeOf(input.dtype()),
              mpiOpOf(reduceOp),
              pgComm_));
        }
      };
  auto entry = std::make_unique<WorkEntry>(
      &inputs, &outputs, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:reduce_scatter_single_coalesced");
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
    checkSplitSizes(inputSplitSizes, inputTensor, size_);
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
      if (srcdata[i].numel() >
          std::numeric_limits<int64_t>::max() - src_len) {
        runtimeFailure("all_to_all input size overflows the index range");
      }
      if (dstdata[i].numel() >
          std::numeric_limits<int64_t>::max() - dst_len) {
        runtimeFailure("all_to_all output size overflows the index range");
      }
      src_len += srcdata[i].numel();
      dst_len += dstdata[i].numel();
    }
    Tensor srcFlatData = Tensor::empty(
        {src_len}, srcdata[0].dtype(), srcdata[0].device());
    Tensor dstFlatData = Tensor::empty(
        {dst_len}, dstdata[0].dtype(), dstdata[0].device());
    int64_t cursor = 0;
    for (int i = 0; i < size_; ++i) {
      if (srcdata[i].numel() >
          std::numeric_limits<int64_t>::max() - cursor) {
        runtimeFailure("all_to_all input offset overflows the index range");
      }
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
      if (dstdata[i].numel() >
          std::numeric_limits<int64_t>::max() - cursor) {
        runtimeFailure("all_to_all output offset overflows the index range");
      }
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

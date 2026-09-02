#include "ProcessGroupGloo.h"

#include <gloo/allgather.h>
#include <gloo/allgatherv.h>
#include <gloo/allreduce.h>
#include <gloo/alltoall.h>
#include <gloo/alltoallv.h>
#include <gloo/barrier.h>
#include <gloo/broadcast.h>
#include <gloo/gather.h>
#include <gloo/math.h>
#include <gloo/reduce.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/scatter.h>
#include <gloo/transport/tcp/device.h>

#include <BFloat16.h>
#include <Half.h>

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <gloo/common/win.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <ifaddrs.h>
#endif


namespace tensorplay {
namespace distributed {

namespace py = pybind11;

namespace {

constexpr const char* kLoopbackAddress = "127.0.0.1";

[[noreturn]] void runtimeFailure(const std::string& msg);

std::chrono::milliseconds remainingBarrierTime(
    std::chrono::steady_clock::time_point start,
    std::chrono::milliseconds timeout,
    bool waitAllRanks) {
  if (waitAllRanks) {
    return timeout;
  }
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  const auto remaining = timeout - elapsed;
  return remaining.count() <= 0 ? std::chrono::milliseconds(-1) : remaining;
}

void checkBarrierTime(
    std::chrono::milliseconds timeout,
    std::chrono::milliseconds remaining,
    const std::vector<int>& processedRanks,
    int currentRank) {
  if (remaining.count() >= 0) {
    return;
  }
  std::string message =
      "Rank " + std::to_string(currentRank) +
      " timed out in monitoredBarrier after " +
      std::to_string(timeout.count()) + " ms.";
  if (processedRanks.empty()) {
    message += "\nNo ranks successfully processed in monitoredBarrier.";
  } else {
    message += "\nSuccessfully processed ranks: ";
    for (size_t i = 0; i < processedRanks.size(); ++i) {
      if (i != 0) {
        message += ", ";
      }
      message += std::to_string(processedRanks[i]);
    }
  }
  runtimeFailure(message);
}

// Adapts the project store onto the rendezvous interface the gloo transport
// expects. Extended batch APIs are not wired; gloo falls back to the plain
// per-key operations.
class GlooStoreAdapter : public ::gloo::rendezvous::Store {
 public:
  // The base class injects its own `Store` name into member-scope lookup;
  // this alias makes the project store the unambiguous `Store` here.
  using Store = tensorplay::distributed::Store;

  explicit GlooStoreAdapter(std::shared_ptr<Store> store)
      : store_(std::move(store)) {}

  void set(const std::string& key, const std::vector<char>& data) override {
    store_->set(
        key, std::vector<uint8_t>(data.begin(), data.end()));
  }

  std::vector<char> get(const std::string& key) override {
    auto value = store_->get(key);
    return std::vector<char>(value.begin(), value.end());
  }

  void wait(const std::vector<std::string>& keys) override {
    store_->wait(keys);
  }

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override {
    store_->wait(keys, timeout);
  }

  int64_t add(const std::string& key, int64_t value) override {
    return store_->add(key, value);
  }

 private:
  std::shared_ptr<Store> store_;
};

[[noreturn]] void invalidArgument(const std::string& msg) {
  TP_THROW(ValueError, msg);
}

[[noreturn]] void runtimeFailure(const std::string& msg) {
  TP_THROW(RuntimeError, msg);
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

int64_t checkedByteSize(const Tensor& tensor, const char* op) {
  const int64_t numel = tensor.numel();
  const int64_t itemsize = static_cast<int64_t>(tensor.itemsize());
  if (numel < 0 || itemsize < 0 ||
      (itemsize != 0 &&
       numel > std::numeric_limits<int64_t>::max() / itemsize)) {
    invalidArgument(std::string(op) + ": tensor byte size overflows");
  }
  return numel * itemsize;
}

void assertRootRank(int rootRank, int size, const char* op) {
  if (rootRank < 0 || rootRank >= size) {
    invalidArgument(
        std::string(op) + ": root rank " + std::to_string(rootRank) +
        " is out of range for group of size " + std::to_string(size));
  }
}

void assertRootTensor(int rootTensor, int64_t count, const char* op) {
  if (rootTensor < 0 || rootTensor >= count) {
    invalidArgument(
        std::string(op) + ": root tensor index out of range");
  }
}

void assertTypeAndSizesMatch(
    const std::vector<Tensor>& tensors,
    const char* op) {
  if (tensors.empty()) {
    invalidArgument(std::string(op) + ": requires a non-empty tensor list");
  }
  for (const auto& tensor : tensors) {
    if (!tensor.defined() ||
        tensor.dtype() != tensors[0].dtype() ||
        tensor.shape() != tensors[0].shape() ||
        tensor.device() != tensors[0].device() ||
        tensor.is_sparse() != tensors[0].is_sparse() ||
        tensor.is_sparse_csr() != tensors[0].is_sparse_csr()) {
      invalidArgument(
          std::string(op) +
          ": all tensors must have the same type, device, layout and sizes");
    }
  }
}

void assertTensorTypeAndSizesMatch(
    const Tensor& expected,
    const Tensor& actual,
    const char* op) {
  if (!expected.defined() || !actual.defined() ||
      expected.dtype() != actual.dtype() ||
      expected.shape() != actual.shape() ||
      expected.device() != actual.device() ||
      expected.is_sparse() != actual.is_sparse() ||
      expected.is_sparse_csr() != actual.is_sparse_csr()) {
    invalidArgument(
        std::string(op) +
        ": tensors must have the same type, device, layout and sizes");
  }
}

void assertDense(const Tensor& tensor, const char* op) {
  if (!tensor.defined()) {
    invalidArgument(std::string(op) + ": tensor is undefined");
  }
  if (tensor.is_sparse()) {
    invalidArgument(std::string(op) + ": only dense tensors are supported");
  }
}

void assertDense(const std::vector<Tensor>& tensors, const char* op) {
  if (tensors.empty()) {
    invalidArgument(std::string(op) + ": requires a non-empty tensor list");
  }
  for (const auto& tensor : tensors) {
    assertDense(tensor, op);
  }
}

void assertSingleElement(const std::vector<Tensor>& tensors, const char* op) {
  if (tensors.size() != 1) {
    invalidArgument(std::string(op) + ": requires a single tensor");
  }
}

Tensor flattenDenseTensors(const std::vector<Tensor>& tensors) {
  if (tensors.size() == 1 && tensors[0].is_contiguous()) {
    return tensors[0].view({tensors[0].numel()});
  }
  std::vector<int64_t> sizes;
  sizes.reserve(tensors.size());
  std::vector<Tensor> flat;
  flat.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    flat.push_back(tensor.contiguous().view({tensor.numel()}));
    sizes.push_back(tensor.numel());
  }
  if (flat.size() == 1) {
    return flat[0];
  }
  return Tensor::cat(flat, 0);
}

Tensor newLikeFlat(const std::vector<Tensor>& tensors) {
  // Shape [group size, t0.shape...]: the transport lays out one
  // contribution per rank along dim 0, so rows unflatten directly.
  std::vector<int64_t> sizes{(int64_t)tensors.size()};
  const auto t0shape = tensors[0].shape();
  sizes.insert(sizes.end(), t0shape.begin(), t0shape.end());
  return Tensor::empty(sizes, tensors[0].dtype(), tensors[0].device());
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
    if (tensor.dim() == 0 || tensor.size(0) % groupSize != 0) {
      runtimeFailure(
          "Tensor's dim 0 does not divide equally across group size");
    }
  } else {
    if (splitSizes.size() != static_cast<size_t>(groupSize)) {
      runtimeFailure(
          "Number of tensor split sizes not equal to group size");
    }
    int64_t sum = 0;
    for (const auto splitSize : splitSizes) {
      if (splitSize > std::numeric_limits<int64_t>::max() - sum) {
        runtimeFailure("all_to_all split sizes overflow the index range");
      }
      sum += splitSize;
    }
    if (sum != tensor.size(0)) {
      runtimeFailure(
          "Split sizes doesn't match total dim 0 size");
    }
  }
}

void computeLengthsAndOffsets(
    const std::vector<int64_t>& splitSizes,
    const Tensor& tensor,
    std::vector<int64_t>* lengths,
    std::vector<int64_t>* offsets) {
  const size_t groupSize = lengths->size();
  const int64_t dim0Size = tensor.size(0);
  const int64_t rowSize = dim0Size == 0 ? 1 : tensor.numel() / dim0Size;
  const bool equalSplits = splitSizes.empty();
  const int64_t equalSplitSize =
      equalSplits ? dim0Size / static_cast<int64_t>(groupSize) : 0;
  if (!equalSplits && splitSizes.size() != groupSize) {
    runtimeFailure("Number of tensor split sizes not equal to group size");
  }
  if (!equalSplits) {
    lengths->resize(groupSize);
  }
  offsets->resize(groupSize);
  int64_t offset = 0;
  for (size_t i = 0; i < groupSize; ++i) {
    const int64_t splitSize =
        equalSplits ? equalSplitSize : splitSizes[i];
    if (splitSize < 0 ||
        (rowSize != 0 &&
         splitSize > std::numeric_limits<int64_t>::max() / rowSize)) {
      runtimeFailure("all_to_all split size overflows the index range");
    }
    (*lengths)[i] = rowSize * splitSize;
    (*offsets)[i] = offset;
    if ((*lengths)[i] > std::numeric_limits<int64_t>::max() - offset) {
      runtimeFailure("all_to_all offset overflows the index range");
    }
    offset += (*lengths)[i];
  }
}

// Elementwise reduce functions. Standard C++ arithmetic types delegate to the
// transport library's implementations; 16-bit float types are reduced through
// float promotion to avoid relying on their conversion operators.
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
void reduceInto(ReduceOp op, void* c, const void* a, const void* b, size_t n) {
  switch (op.op()) {
    case ReduceOp::SUM:
    case ReduceOp::AVG:
      ::gloo::sum<T>(c, a, b, n);
      break;
    case ReduceOp::PRODUCT:
      ::gloo::product<T>(c, a, b, n);
      break;
    case ReduceOp::MIN:
      ::gloo::min<T>(c, a, b, n);
      break;
    case ReduceOp::MAX:
      ::gloo::max<T>(c, a, b, n);
      break;
    case ReduceOp::BAND:
      if constexpr (std::is_integral_v<T>) {
        auto* tc = static_cast<T*>(c);
        const auto* ta = static_cast<const T*>(a);
        const auto* tb = static_cast<const T*>(b);
        for (size_t i = 0; i < n; ++i) {
          tc[i] = ta[i] & tb[i];
        }
      } else {
        runtimeFailure(
            "Cannot use the bitwise AND reduction with a non-integral dtype");
      }
      break;
    case ReduceOp::BOR:
      if constexpr (std::is_integral_v<T>) {
        auto* tc = static_cast<T*>(c);
        const auto* ta = static_cast<const T*>(a);
        const auto* tb = static_cast<const T*>(b);
        for (size_t i = 0; i < n; ++i) {
          tc[i] = ta[i] | tb[i];
        }
      } else {
        runtimeFailure(
            "Cannot use the bitwise OR reduction with a non-integral dtype");
      }
      break;
    case ReduceOp::BXOR:
      if constexpr (std::is_integral_v<T>) {
        auto* tc = static_cast<T*>(c);
        const auto* ta = static_cast<const T*>(a);
        const auto* tb = static_cast<const T*>(b);
        for (size_t i = 0; i < n; ++i) {
          tc[i] = ta[i] ^ tb[i];
        }
      } else {
        runtimeFailure(
            "Cannot use the bitwise XOR reduction with a non-integral dtype");
      }
      break;
    case ReduceOp::PREMUL_SUM:
      runtimeFailure("Cannot use the pre-multiply sum reduction with Gloo");
      break;
    default:
      runtimeFailure("Unhandled reduce op for the gloo backend");
  }
}

template <typename T>
void halfOp(ReduceOp op, void* c, const void* a, const void* b, size_t n) {
  auto* tc = static_cast<T*>(c);
  auto* ta = static_cast<const T*>(a);
  auto* tb = static_cast<const T*>(b);
  for (size_t i = 0; i < n; ++i) {
    float x = static_cast<float>(ta[i]);
    float y = static_cast<float>(tb[i]);
    float r = x;
    switch (op.op()) {
      case ReduceOp::SUM:
      case ReduceOp::AVG:
        r = x + y;
        break;
      case ReduceOp::PRODUCT:
        r = x * y;
        break;
      case ReduceOp::MIN:
        r = x < y ? x : y;
        break;
      case ReduceOp::MAX:
        r = x > y ? x : y;
        break;
      case ReduceOp::BAND:
      case ReduceOp::BOR:
      case ReduceOp::BXOR:
        runtimeFailure(
            "Cannot use a bitwise reduction with a non-integral dtype");
        break;
      case ReduceOp::PREMUL_SUM:
        runtimeFailure(
            "Cannot use the pre-multiply sum reduction with Gloo");
        break;
      default:
        runtimeFailure("Unhandled reduce op for the gloo backend");
    }
    tc[i] = static_cast<T>(r);
  }
}

template <typename T>
gloo::AllreduceOptions::Func makeReduceFunction(ReduceOp op) {
  if constexpr (std::is_arithmetic_v<T>) {
    return [op](void* c, const void* a, const void* b, size_t n) {
      reduceInto<T>(op, c, a, b, n);
    };
  } else {
    return [op](void* c, const void* a, const void* b, size_t n) {
      halfOp<T>(op, c, a, b, n);
    };
  }
}

template <typename T, typename O>
void setReduceFn(O& opts, ReduceOp op) {
  opts.setReduceFunction(makeReduceFunction<T>(op));
}

// Dtype dispatch over the tensor library's scalar types.
#define TP_GLOO_GENERATE_ALL_TYPES(type, func, ...)      \
  switch (type) {                                        \
    case ::tensorplay::ScalarType::Float32:              \
      func<float>(__VA_ARGS__);                          \
      break;                                             \
    case ::tensorplay::ScalarType::Float64:              \
      func<double>(__VA_ARGS__);                         \
      break;                                             \
    case ::tensorplay::ScalarType::Float16:              \
      func<::tensorplay::Half>(__VA_ARGS__);             \
      break;                                             \
    case ::tensorplay::ScalarType::BFloat16:             \
      func<::tensorplay::BFloat16>(__VA_ARGS__);         \
      break;                                             \
    case ::tensorplay::ScalarType::Int8:                 \
      func<int8_t>(__VA_ARGS__);                         \
      break;                                             \
    case ::tensorplay::ScalarType::Int16:                \
      func<int16_t>(__VA_ARGS__);                        \
      break;                                             \
    case ::tensorplay::ScalarType::Int32:                \
      func<int32_t>(__VA_ARGS__);                        \
      break;                                             \
    case ::tensorplay::ScalarType::Int64:                \
      func<int64_t>(__VA_ARGS__);                        \
      break;                                             \
    case ::tensorplay::ScalarType::UInt8:                \
    case ::tensorplay::ScalarType::Bool:                 \
      func<uint8_t>(__VA_ARGS__);                        \
      break;                                             \
    case ::tensorplay::ScalarType::UInt16:               \
      func<uint16_t>(__VA_ARGS__);                       \
      break;                                             \
    case ::tensorplay::ScalarType::UInt32:               \
      func<uint32_t>(__VA_ARGS__);                       \
      break;                                             \
    case ::tensorplay::ScalarType::UInt64:               \
      func<uint64_t>(__VA_ARGS__);                       \
      break;                                             \
    default:                                             \
      runtimeFailure("Invalid scalar type");             \
  }

// Buffer registration helpers. The pointer is bound to the dispatched C++
// type so the transport records the correct element size; the count stays in
// units of tensor elements.
template <typename T>
void setInputs(gloo::AllreduceOptions& opts, std::vector<Tensor>& tensors) {
  std::vector<T*> ptrs;
  ptrs.reserve(tensors.size());
  for (auto& tensor : tensors) {
    ptrs.push_back(static_cast<T*>(tensor.data_ptr()));
  }
  opts.setInputs(ptrs, tensors[0].numel());
}

template <typename T>
void setInputs(gloo::ScatterOptions& opts, std::vector<Tensor>& tensors) {
  std::vector<T*> ptrs;
  ptrs.reserve(tensors.size());
  for (auto& tensor : tensors) {
    ptrs.push_back(static_cast<T*>(tensor.data_ptr()));
  }
  opts.setInputs(ptrs, tensors[0].numel());
}

template <typename T, typename O>
void setInput(O& opts, Tensor& tensor) {
  opts.setInput(static_cast<T*>(tensor.data_ptr()), tensor.numel());
}

template <typename T, typename O>
void setInput(O& opts, Tensor& tensor, std::vector<int64_t>& counts) {
  opts.setInput(static_cast<T*>(tensor.data_ptr()), counts);
}

template <typename T, typename O>
void setOutput(O& opts, Tensor& tensor) {
  opts.setOutput(static_cast<T*>(tensor.data_ptr()), tensor.numel());
}

template <typename T, typename O>
void setOutput(O& opts, Tensor& tensor, std::vector<int64_t>& counts) {
  opts.setOutput(static_cast<T*>(tensor.data_ptr()), counts);
}

template <typename T, typename O>
void setOutput(O& opts, Tensor& tensor, const std::vector<size_t>& counts) {
  opts.setOutput(static_cast<T*>(tensor.data_ptr()), counts);
}

template <typename T>
void setOutputs(
    gloo::AllreduceOptions& opts,
    std::vector<Tensor>& tensors,
    int64_t count) {
  std::vector<T*> ptrs;
  ptrs.reserve(tensors.size());
  for (auto& tensor : tensors) {
    ptrs.push_back(static_cast<T*>(tensor.data_ptr()));
  }
  opts.setOutputs(ptrs, count);
}

void assertNonEmptyDeviceCpu(const std::vector<Tensor>& tensors, const char* op) {
  if (tensors.empty()) {
    invalidArgument(std::string(op) + ": requires a non-empty tensor list");
  }
  for (const auto& tensor : tensors) {
    if (!tensor.defined()) {
      invalidArgument(std::string(op) + ": tensor is undefined");
    }
    if (!tensor.device().is_cpu()) {
      invalidArgument(
          std::string(op) +
          ": only CPU tensors are supported by the gloo backend");
    }
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Work plumbing
// ---------------------------------------------------------------------------

bool GlooWork::wait(int64_t timeout_ms) {
  std::unique_lock<std::mutex> lock(waitMutex_);
  if (timeout_ms < 0) {
    waitCV_.wait(lock, [&] { return completed_; });
  } else {
    if (!waitCV_.wait_for(
            lock,
            std::chrono::milliseconds(timeout_ms),
            [&] { return completed_; })) {
      return false;
    }
  }
  if (exception_ != nullptr) {
    std::rethrow_exception(exception_);
  }
  return true;
}

bool GlooWork::is_completed() {
  std::lock_guard<std::mutex> lock(waitMutex_);
  return completed_;
}

void GlooWork::abort() {
  TP_THROW(RuntimeError, "work abort is unavailable for this operation");
}

std::vector<Tensor> GlooWork::result() const {
  std::lock_guard<std::mutex> lock(waitMutex_);
  if (!completed_) {
    TP_THROW(
        RuntimeError,
        "work must be completed before its result can be read");
  }
  if (exception_ != nullptr) {
    std::rethrow_exception(exception_);
  }
  if (outputTensors_.size() > 1) {
    TP_THROW(
        RuntimeError,
        "work result does not support multiple tensor lists");
  }
  return outputTensors_.empty() ? std::vector<Tensor>{}
                                : outputTensors_.front();
}

void GlooWork::finish() {
  {
    std::lock_guard<std::mutex> lock(waitMutex_);
    completed_ = true;
  }
  waitCV_.notify_all();
}

void GlooWork::finishWithError(std::exception_ptr eptr) {
  {
    std::lock_guard<std::mutex> lock(waitMutex_);
    completed_ = true;
    exception_ = std::move(eptr);
  }
  waitCV_.notify_all();
}

namespace {

class GlooCompositeWork : public GlooWork {
 public:
  GlooCompositeWork(
      std::string opName,
      std::vector<std::vector<Tensor>> outputTensors,
      std::vector<std::shared_ptr<GlooWork>> children,
      std::function<void()> finalize)
      : GlooWork(std::move(opName)),
        children_(std::move(children)),
        finalize_(std::move(finalize)) {
    outputTensors_ = std::move(outputTensors);
  }

  bool is_completed() override {
    std::lock_guard<std::mutex> lock(compositeMutex_);
    if (baseCompleted()) {
      return true;
    }
    for (const auto& child : children_) {
      if (!child->is_completed()) {
        return false;
      }
    }
    try {
      for (const auto& child : children_) {
        if (!child->wait(0)) {
          return false;
        }
      }
      finalizeLocked();
    } catch (...) {
      failLocked(std::current_exception());
    }
    return true;
  }

  bool wait(int64_t timeout_ms = -1) override {
    std::lock_guard<std::mutex> lock(compositeMutex_);
    if (baseCompleted()) {
      GlooWork::wait(0);
      return true;
    }

    const auto start = std::chrono::steady_clock::now();
    try {
      for (const auto& child : children_) {
        const auto remaining = remainingTimeout(timeout_ms, start);
        if (!child->wait(remaining)) {
          return false;
        }
      }
      finalizeLocked();
    } catch (...) {
      failLocked(std::current_exception());
      throw;
    }
    return true;
  }

 private:
  bool baseCompleted() const {
    std::lock_guard<std::mutex> lock(waitMutex_);
    return completed_;
  }

  static int64_t remainingTimeout(
      int64_t timeout_ms,
      std::chrono::steady_clock::time_point start) {
    if (timeout_ms < 0) {
      return -1;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    if (elapsed >= timeout_ms) {
      return 0;
    }
    return timeout_ms - elapsed;
  }

  void finalizeLocked() {
    if (finalized_) {
      return;
    }
    finalized_ = true;
    finalize_();
    GlooWork::finish();
  }

  void failLocked(std::exception_ptr eptr) {
    if (!baseCompleted()) {
      finalized_ = true;
      finishWithError(std::move(eptr));
    }
  }

  std::vector<std::shared_ptr<GlooWork>> children_;
  std::function<void()> finalize_;
  std::mutex compositeMutex_;
  bool finalized_{false};
};

} // namespace

int GlooRecvWork::source_rank() const {
  std::lock_guard<std::mutex> lock(waitMutex_);
  return srcRank_;
}

bool GlooRecvWork::wait(int64_t timeout_ms) {
  {
    std::lock_guard<std::mutex> lock(waitMutex_);
    if (completed_) {
      if (exception_ != nullptr) {
        std::rethrow_exception(exception_);
      }
      return true;
    }
  }
  std::exception_ptr exception{nullptr};
  bool completed = false;
  int receivedRank = -1;
  try {
    if (timeout_ms < 0) {
      completed = buffer_->waitRecv(&receivedRank);
    } else {
      completed =
          buffer_->waitRecv(
              &receivedRank, std::chrono::milliseconds(timeout_ms));
    }
  } catch (...) {
    exception = std::current_exception();
  }
  if (exception != nullptr) {
    finishWithError(exception);
    std::rethrow_exception(exception);
  }
  if (completed) {
    {
      std::lock_guard<std::mutex> lock(waitMutex_);
      srcRank_ = receivedRank;
    }
    GlooWork::finish();
  }
  return completed;
}

void GlooRecvWork::abort() {
  buffer_->abortWaitRecv();
  finishWithError(std::make_exception_ptr(
      std::runtime_error("gloo receive work was aborted")));
}

// ---------------------------------------------------------------------------
// Device creation
// ---------------------------------------------------------------------------

namespace {

void socketInitialize() {
#ifdef _WIN32
  ::gloo::init_winsock();
#endif
}

bool doesHostnameResolveToUsableAddress(const std::string& hostname) {
  socketInitialize();
  struct addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* result = nullptr;
  auto rv = getaddrinfo(hostname.c_str(), nullptr, &hints, &result);
  if (rv != 0) {
    return false;
  }
  struct addrinfo* rp = nullptr;
  for (rp = result; rp != nullptr; rp = rp->ai_next) {
    auto fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd == -1) {
      continue;
    }
    rv = bind(fd, rp->ai_addr, rp->ai_addrlen);
#ifdef _WIN32
    closesocket(fd);
#else
    close(fd);
#endif
    if (rv == -1) {
      continue;
    }
    break;
  }
  freeaddrinfo(result);
  return rp != nullptr;
}

std::shared_ptr<::gloo::transport::Device> makeTcpDevice(
    const ::gloo::transport::tcp::attr& attr,
    bool lazyInit) {
  if (lazyInit) {
    return ::gloo::transport::tcp::CreateLazyDevice(attr);
  }
  return ::gloo::transport::tcp::CreateDevice(attr);
}

} // namespace

std::shared_ptr<::gloo::transport::Device>
ProcessGroupGloo::createDeviceForInterface(
    const std::string& interface_name,
    bool lazyInit) {
  ::gloo::transport::tcp::attr attr;
  attr.iface = interface_name;
  return makeTcpDevice(attr, lazyInit);
}

std::shared_ptr<::gloo::transport::Device>
ProcessGroupGloo::createDeviceForHostname(
    const std::string& hostname,
    bool lazyInit) {
  if (!doesHostnameResolveToUsableAddress(hostname)) {
    runtimeFailure("Cannot resolve " + hostname + " to a (local) address");
  }
  ::gloo::transport::tcp::attr attr;
  attr.hostname = hostname;
  return makeTcpDevice(attr, lazyInit);
}

std::shared_ptr<::gloo::transport::Device>
ProcessGroupGloo::createDefaultDevice(bool lazyInit) {
  socketInitialize();
  // An explicit interface selection takes precedence over hostname discovery.
  const char* ifname = std::getenv("GLOO_SOCKET_IFNAME");
  if (ifname != nullptr && std::strlen(ifname) > 1) {
    return createDeviceForInterface(ifname, lazyInit);
  }
  char hostname[256];
  auto rv = gethostname(hostname, sizeof(hostname));
  if (rv == 0 && doesHostnameResolveToUsableAddress(hostname)) {
    return createDeviceForHostname(hostname, lazyInit);
  }
  // Hostname did not resolve to a usable local address; bind loopback so
  // single-host jobs still work.
  return createDeviceForHostname(kLoopbackAddress, lazyInit);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

ProcessGroupGloo::ProcessGroupGloo(
    std::shared_ptr<Store> store,
    int rank,
    int size,
    GlooOptions options)
    : store_(std::move(store)),
      options_(std::move(options)),
      rank_(rank),
      size_(size) {
  if (!store_) {
    runtimeFailure("ProcessGroupGloo: store must not be null");
  }
  if (size_ <= 0 || rank_ < 0 || rank_ >= size_) {
    runtimeFailure("ProcessGroupGloo: rank and size are invalid");
  }
  if (options_.devices.empty()) {
    runtimeFailure("ProcessGroupGloo: no device(s) specified");
  }
  for (const auto& device : options_.devices) {
    if (!device) {
      runtimeFailure("ProcessGroupGloo: device must not be null");
    }
  }
  if (options_.threads <= 0) {
    runtimeFailure("ProcessGroupGloo: thread count must be positive");
  }
  if (!options_.global_ranks_in_group.empty() &&
      options_.global_ranks_in_group.size() != static_cast<size_t>(size_)) {
    runtimeFailure(
        "ProcessGroupGloo: global rank list must match the group size");
  }
  connectContexts(rank_, size_, store_);
  initialized_ = true;

  workInProgress_.resize(options_.threads);
  threads_.resize(options_.threads);
  for (size_t i = 0; i < threads_.size(); ++i) {
    threads_[i] = std::thread(&ProcessGroupGloo::runLoop, this, (int)i);
  }
}

ProcessGroupGloo::~ProcessGroupGloo() {
  std::unique_lock<std::mutex> lock(workMutex_);
  workConsumeCV_.wait(lock, [&] { return workQueue_.empty(); });
  stop_ = true;
  lock.unlock();
  workProduceCV_.notify_all();
  for (auto& thread : threads_) {
    thread.join();
  }
}

void ProcessGroupGloo::connectContexts(
    int rank,
    int size,
    std::shared_ptr<Store> store) {
  auto glooStore = std::make_shared<GlooStoreAdapter>(std::move(store));
  std::vector<std::shared_ptr<::gloo::Context>> contexts;
  contexts.reserve(options_.devices.size());
  for (size_t i = 0; i < options_.devices.size(); ++i) {
    auto context = std::make_shared<::gloo::rendezvous::Context>(rank, size);
    auto prefixedStore = std::make_shared<::gloo::rendezvous::PrefixStore>(
        std::to_string(i), glooStore);
    context->setTimeout(options_.timeout);
    try {
      context->connectFullMesh(prefixedStore, options_.devices[i]);
    } catch (const std::runtime_error& e) {
      runtimeFailure(
          std::string("Gloo connectFullMesh failed with ") + e.what());
    }
    contexts.push_back(std::move(context));
  }
  contexts_ = std::move(contexts);
}

uint32_t ProcessGroupGloo::nextTag() {
  checkInitialized();
  return collectiveCounter_++;
}

std::shared_ptr<::gloo::Context> ProcessGroupGloo::getContext(uint32_t tag) {
  checkInitialized();
  return contexts_[tag % contexts_.size()];
}

void ProcessGroupGloo::checkInitialized() const {
  if (!initialized_ || contexts_.empty()) {
    runtimeFailure("ProcessGroupGloo has not been initialized");
  }
}

void ProcessGroupGloo::runLoop(int workerIndex) {
  std::unique_lock<std::mutex> lock(workMutex_);
  while (!stop_) {
    if (workQueue_.empty()) {
      workProduceCV_.wait(lock);
      continue;
    }
    auto work = std::move(workQueue_.front());
    workQueue_.pop_front();
    workInProgress_[workerIndex] = work;
    lock.unlock();
    workConsumeCV_.notify_one();
    work->execute();
    lock.lock();
    workInProgress_[workerIndex].reset();
  }
}

void ProcessGroupGloo::enqueue(std::shared_ptr<GlooAsyncWork> work) {
  std::unique_lock<std::mutex> lock(workMutex_);
  workQueue_.push_back(std::move(work));
  lock.unlock();
  workProduceCV_.notify_one();
}

void ProcessGroupGloo::runInline(GlooAsyncWork* work) {
  work->execute();
}

// ---------------------------------------------------------------------------
// Work class bodies (op implementations)
// ---------------------------------------------------------------------------

namespace {

class AsyncBroadcastWork : public GlooAsyncWork {
 public:
  AsyncBroadcastWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<Tensor>& inputs,
      int rootRank,
      int rootTensor,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(
            std::move(context),
            {inputs},
            "broadcast",
            seq,
            timeout),
        inputs(inputs),
        rootRank(rootRank),
        rootTensor(rootTensor),
        tag(tag) {}

  std::vector<Tensor> inputs;
  const int rootRank;
  const int rootTensor;
  const uint32_t tag;

  void broadcastOne(Tensor tensor) {
    if (isComplexType(tensor.dtype())) {
      tensor = tensor.view_as_real();
    }
    const auto scalarType = tensor.dtype();
    gloo::BroadcastOptions opts(context_);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);
    gloo::broadcast(opts);
  }

  void run() override {
    broadcastOne(inputs[rootTensor]);
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (i == static_cast<size_t>(rootTensor)) {
        continue;
      }
      inputs[i].copy_(inputs[rootTensor]);
    }
  }
};

class AsyncAllreduceWork : public GlooAsyncWork {
 public:
  AsyncAllreduceWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(
            std::move(context),
            {inputs},
            "all_reduce",
            seq,
            timeout),
        inputs(inputs),
        reduceOp(reduceOp),
        tag(tag) {}

  std::vector<Tensor> inputs;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void allreduceOne(std::vector<Tensor>& tensors) {
    std::vector<Tensor> tensorViews;
    tensorViews.reserve(tensors.size());
    for (const auto& tensor : tensors) {
      Tensor view = tensor;
      if (isComplexType(tensor.dtype())) {
        if (!isComplexViewAsRealAllowed(reduceOp)) {
          runtimeFailure(
              "all_reduce does not support this reduction operation on "
              "complex tensors");
        }
        view = tensor.view_as_real();
      }
      tensorViews.push_back(std::move(view));
    }
    gloo::AllreduceOptions opts(context_);
    const auto scalarType = tensorViews[0].dtype();
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setReduceFn, opts, reduceOp);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());
    TP_GLOO_GENERATE_ALL_TYPES(
        scalarType,
        setOutputs,
        opts,
        tensorViews,
        tensorViews[0].numel());
    gloo::allreduce(opts);
    if (reduceOp == ReduceOp::AVG) {
      for (auto& tensor : tensors) {
        tensor /= (double)context_->size;
      }
    }
  }

  void run() override {
    allreduceOne(inputs);
  }
};

class AsyncSparseAllreduceWork : public GlooAsyncWork {
 public:
  AsyncSparseAllreduceWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(
            std::move(context),
            {inputs},
            "sparse_all_reduce",
            seq,
            timeout),
        inputs(inputs),
        tag(tag) {}

  std::vector<Tensor> inputs;
  const uint32_t tag;

  struct Metadata {
    int64_t sparseDim{0};
    int64_t denseDim{0};
    int64_t nnz{0};
    std::vector<int64_t> sizes;
  };

  static size_t checkedCount(int64_t left, int64_t right) {
    if (left < 0 || right < 0) {
      runtimeFailure("Sparse tensor metadata contains a negative size");
    }
    const auto lhs = static_cast<size_t>(left);
    const auto rhs = static_cast<size_t>(right);
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
      runtimeFailure("Sparse tensor metadata size overflows the transport count");
    }
    return lhs * rhs;
  }

  static int64_t checkedProduct(int64_t left, int64_t right) {
    if (left < 0 || right < 0) {
      runtimeFailure("Sparse tensor shape contains a negative size");
    }
    if (right != 0 && left > std::numeric_limits<int64_t>::max() / right) {
      runtimeFailure("Sparse tensor shape overflows the index range");
    }
    return left * right;
  }

  static size_t checkedScale(size_t value, size_t factor) {
    if (factor != 0 && value > std::numeric_limits<size_t>::max() / factor) {
      runtimeFailure("Sparse values exceed the transport size");
    }
    return value * factor;
  }

  std::vector<int64_t> metadataPayload(const Tensor& tensor) const {
    if (!tensor.is_sparse() || tensor.is_sparse_csr()) {
      runtimeFailure(
          "sparse allreduce requires sparse coordinate tensors");
    }
    const int64_t sparseDim = tensor.sparse_dim();
    const int64_t denseDim = tensor.dense_dim();
    const int64_t nnz = tensor._values().size(0);
    std::vector<int64_t> payload;
    payload.reserve(static_cast<size_t>(3 + sparseDim + denseDim));
    payload.push_back(sparseDim);
    payload.push_back(denseDim);
    payload.push_back(nnz);
    for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
      payload.push_back(tensor.size(dim));
    }
    return payload;
  }

  std::vector<Metadata> allgatherMetadata(const Tensor& tensor) {
    const auto payload = metadataPayload(tensor);
    Tensor localSize = Tensor::empty({1}, ::tensorplay::DType::Int64,
                                     tensor.device());
    localSize.data_ptr<int64_t>()[0] = static_cast<int64_t>(payload.size());
    Tensor sizes = Tensor::empty(
        {static_cast<int64_t>(context_->size)},
        ::tensorplay::DType::Int64,
        tensor.device());

    gloo::AllgatherOptions sizeOptions(context_);
    sizeOptions.setInput(localSize.data_ptr<int64_t>(), 1);
    sizeOptions.setOutput(
        sizes.data_ptr<int64_t>(), static_cast<size_t>(context_->size));
    sizeOptions.setTag(tag);
    sizeOptions.setTimeout(getTimeout());
    gloo::allgather(sizeOptions);

    std::vector<size_t> counts;
    counts.reserve(static_cast<size_t>(context_->size));
    size_t total = 0;
    for (int rank = 0; rank < context_->size; ++rank) {
      const int64_t count = sizes.data_ptr<int64_t>()[rank];
      if (count < 3) {
        runtimeFailure("Sparse tensor metadata payload is truncated");
      }
      const auto countSize = static_cast<size_t>(count);
      if (total > std::numeric_limits<size_t>::max() - countSize) {
        runtimeFailure("Sparse tensor metadata exceeds the transport size");
      }
      counts.push_back(countSize);
      total += countSize;
    }

    Tensor localPayload = Tensor::empty(
        {static_cast<int64_t>(payload.size())},
        ::tensorplay::DType::Int64,
        tensor.device());
    std::copy(payload.begin(), payload.end(), localPayload.data_ptr<int64_t>());
    Tensor gathered = Tensor::empty(
        {static_cast<int64_t>(total)},
        ::tensorplay::DType::Int64,
        tensor.device());

    gloo::AllgathervOptions payloadOptions(context_);
    payloadOptions.setInput(
        localPayload.data_ptr<int64_t>(), localPayload.numel());
    payloadOptions.setOutput(gathered.data_ptr<int64_t>(), counts);
    payloadOptions.setTag(tag);
    payloadOptions.setTimeout(getTimeout());
    gloo::allgatherv(payloadOptions);

    std::vector<Metadata> metadata;
    metadata.reserve(static_cast<size_t>(context_->size));
    size_t offset = 0;
    const auto* data = gathered.data_ptr<int64_t>();
    for (const auto count : counts) {
      const auto* entry = data + offset;
      Metadata item;
      item.sparseDim = entry[0];
      item.denseDim = entry[1];
      item.nnz = entry[2];
      if (item.sparseDim < 0 || item.denseDim < 0 || item.nnz < 0) {
        runtimeFailure("Sparse tensor metadata contains a negative dimension");
      }
      const auto dimensionCount = checkedCount(
          item.sparseDim + item.denseDim, 1);
      if (count != static_cast<size_t>(3) + dimensionCount) {
        runtimeFailure("Sparse tensor metadata has an invalid dimension count");
      }
      item.sizes.assign(entry + 3, entry + 3 + dimensionCount);
      for (const auto size : item.sizes) {
        if (size < 0) {
          runtimeFailure("Sparse tensor metadata contains a negative size");
        }
      }
      metadata.push_back(std::move(item));
      offset += count;
    }
    return metadata;
  }

  std::vector<Tensor> allgatherIndices(
      const Tensor& tensor,
      const std::vector<Metadata>& metadata) {
    const int64_t sparseDim = tensor.sparse_dim();
    std::vector<size_t> counts;
    counts.reserve(metadata.size());
    size_t total = 0;
    for (const auto& item : metadata) {
      const auto count = checkedCount(item.nnz, sparseDim);
      if (total > std::numeric_limits<size_t>::max() - count) {
        runtimeFailure("Sparse indices exceed the transport size");
      }
      counts.push_back(count);
      total += count;
    }

    Tensor input = tensor._indices().contiguous();
    Tensor output = Tensor::empty(
        {static_cast<int64_t>(total)},
        ::tensorplay::DType::Int64,
        tensor.device());
    gloo::AllgathervOptions options(context_);
    options.setInput(input.data_ptr<int64_t>(), input.numel());
    options.setOutput(output.data_ptr<int64_t>(), counts);
    options.setTag(tag);
    options.setTimeout(getTimeout());
    gloo::allgatherv(options);

    std::vector<Tensor> indices;
    indices.reserve(metadata.size());
    int64_t offset = 0;
    for (const auto& item : metadata) {
      const auto count = checkedCount(item.nnz, sparseDim);
      indices.push_back(output.narrow(
          0, offset, static_cast<int64_t>(count))
          .reshape({sparseDim, item.nnz}));
      offset += static_cast<int64_t>(count);
    }
    return indices;
  }

  std::vector<Tensor> allgatherValues(
      const Tensor& tensor,
      const std::vector<Metadata>& metadata) {
    const int64_t sparseDim = tensor.sparse_dim();
    int64_t denseNumel = 1;
    for (int64_t dim = sparseDim; dim < tensor.dim(); ++dim) {
      denseNumel = checkedProduct(denseNumel, tensor.size(dim));
    }

    const bool complexValues = isComplexType(tensor.dtype());
    std::vector<size_t> counts;
    counts.reserve(metadata.size());
    size_t total = 0;
    for (const auto& item : metadata) {
      const auto logicalCount = checkedCount(item.nnz, denseNumel);
      const auto count = complexValues
          ? checkedScale(logicalCount, 2)
          : logicalCount;
      if (total > std::numeric_limits<size_t>::max() - count) {
        runtimeFailure("Sparse values exceed the transport size");
      }
      counts.push_back(count);
      total += count;
    }

    Tensor input = tensor._values().contiguous();
    Tensor inputTransport = complexValues ? input.view_as_real() : input;
    Tensor output = Tensor::empty(
        {static_cast<int64_t>(total)},
        complexValues ? toRealValueType(tensor.dtype()) : tensor.dtype(),
        tensor.device());
    gloo::AllgathervOptions options(context_);
    const auto scalarType = inputTransport.dtype();
    TP_GLOO_GENERATE_ALL_TYPES(
        scalarType, setInput, options, inputTransport);
    TP_GLOO_GENERATE_ALL_TYPES(
        scalarType, setOutput, options, output, counts);
    options.setTag(tag);
    options.setTimeout(getTimeout());
    gloo::allgatherv(options);

    std::vector<Tensor> values;
    values.reserve(metadata.size());
    int64_t offset = 0;
    std::vector<int64_t> valueShape;
    for (int64_t dim = sparseDim; dim < tensor.dim(); ++dim) {
      valueShape.push_back(tensor.size(dim));
    }
    for (const auto& item : metadata) {
      std::vector<int64_t> shape{item.nnz};
      shape.insert(shape.end(), valueShape.begin(), valueShape.end());
      const auto logicalCount = checkedCount(item.nnz, denseNumel);
      const auto count = complexValues
          ? checkedScale(logicalCount, 2)
          : logicalCount;
      Tensor value = output.narrow(
          0, offset, static_cast<int64_t>(count));
      if (complexValues) {
        shape.push_back(2);
        value = value.reshape(shape).view_as_complex();
        shape.pop_back();
      } else {
        value = value.reshape(shape);
      }
      values.push_back(std::move(value));
      offset += static_cast<int64_t>(count);
    }
    return values;
  }

static void replaceSparseTensor(Tensor& destination, const Tensor& source) {
    auto indices = source._indices().clone();
    auto values = source._values().clone();
    destination.unsafeGetTensorImpl()->set_sparse_state(
        indices.unsafeGetTensorImpl(),
        values.unsafeGetTensorImpl(),
        static_cast<std::vector<int64_t>>(source.shape()),
        source.is_coalesced());
  }

  Tensor allreduceSparse(std::vector<Tensor>& tensors) {
    Tensor input = tensors[0];
    for (size_t i = 1; i < tensors.size(); ++i) {
      input = Tensor::sparse_add(input, tensors[i]);
    }
    input = input.coalesce();

    auto metadata = allgatherMetadata(input);
    const auto& expected = metadata[static_cast<size_t>(context_->rank)];
    for (const auto& item : metadata) {
      if (item.sparseDim != expected.sparseDim ||
          item.denseDim != expected.denseDim ||
          item.sizes != expected.sizes) {
        runtimeFailure("Sparse tensor dimensions do not match across ranks");
      }
    }

    auto indices = allgatherIndices(input, metadata);
    auto values = allgatherValues(input, metadata);
    Tensor output = Tensor::make_sparse_coo_tensor(
        indices[0], values[0], expected.sizes, true);
    for (int rank = 1; rank < context_->size; ++rank) {
      Tensor peer = Tensor::make_sparse_coo_tensor(
          indices[static_cast<size_t>(rank)],
          values[static_cast<size_t>(rank)],
          expected.sizes,
          true);
      output = Tensor::sparse_add(output, peer);
    }
    return output.coalesce();
  }

  void run() override {
    Tensor output = allreduceSparse(inputs);
    for (auto& tensor : inputs) {
      replaceSparseTensor(tensor, output);
    }
  }
};

class AsyncAllreduceCoalescedWork : public AsyncAllreduceWork {
 public:
  using AsyncAllreduceWork::AsyncAllreduceWork;

  void run() override {
    Tensor coalescedTensor = flattenDenseTensors(inputs);
    std::vector<Tensor> allreduceInput = {coalescedTensor};
    allreduceOne(allreduceInput);
    int64_t offset = 0;
    for (Tensor& tensor : inputs) {
      const int64_t tensorNumel = tensor.numel();
      const auto tensorShape = tensor.shape();
      tensor.copy_(
          coalescedTensor.slice(0, offset, offset + tensorNumel)
              .reshape(tensorShape));
      offset += tensorNumel;
    }
  }
};

class AsyncReduceWork : public GlooAsyncWork {
 public:
  AsyncReduceWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<Tensor>& inputs,
      int rootRank,
      int rootTensor,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(std::move(context), {inputs}, "reduce", seq, timeout),
        inputs(inputs),
        rootRank(rootRank),
        rootTensor(rootTensor),
        reduceOp(reduceOp),
        tag(tag) {}

  std::vector<Tensor> inputs;
  const int rootRank;
  const int rootTensor;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void reduceOne(std::vector<Tensor>& tensors) {
    Tensor tensor = tensors[0];
    if (isComplexType(tensor.dtype())) {
      if (!isComplexViewAsRealAllowed(reduceOp)) {
        runtimeFailure(
            "reduce does not support this reduction operation on complex "
            "tensors");
      }
      tensor = tensor.view_as_real();
    }
    gloo::ReduceOptions opts(context_);
    const auto scalarType = tensor.dtype();
    opts.setRoot(rootRank);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setReduceFn, opts, reduceOp);
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);
    gloo::reduce(opts);
    if (reduceOp == ReduceOp::AVG) {
      tensors[0] /= (double)context_->size;
    }
  }

  void run() override {
    reduceOne(inputs);
  }
};

class AsyncAllgatherWork : public GlooAsyncWork {
 public:
  AsyncAllgatherWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<std::vector<Tensor>>& outputs,
      std::vector<Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(
            std::move(context),
            outputs,
            "all_gather",
            seq,
            timeout),
        outputs(outputs),
        inputs(inputs),
        tag(tag) {}

  std::vector<std::vector<Tensor>> outputs;
  std::vector<Tensor> inputs;
  const uint32_t tag;

  void allgatherOne(
      std::vector<std::vector<Tensor>>& outputs,
      std::vector<Tensor>& inputs) {
    const auto scalarType = inputs[0].dtype();
    gloo::AllgatherOptions opts(context_);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());

    Tensor flatInputTensor = flattenDenseTensors(inputs);
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);

    Tensor flatOutputTensor = newLikeFlat(outputs[0]);
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    gloo::allgather(opts);

    for (auto& outputgroup : outputs) {
      for (size_t j = 0; j < outputgroup.size(); ++j) {
        outputgroup[j].copy_(
            flatOutputTensor.narrow(0, (int64_t)j, 1).reshape(
                outputgroup[j].shape()));
      }
    }
  }

  void run() override {
    allgatherOne(outputs, inputs);
  }
};

class AsyncAllgatherCoalescedWork : public GlooAsyncWork {
 public:
  AsyncAllgatherCoalescedWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<std::vector<Tensor>>& output_lists,
      std::vector<Tensor>& input_list,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(
            std::move(context),
            output_lists,
            "all_gather",
            seq,
            timeout),
        output_lists(output_lists),
        input_list(input_list),
        tag(tag) {}

  std::vector<std::vector<Tensor>> output_lists;
  std::vector<Tensor> input_list;
  const uint32_t tag;

  void run() override {
    const auto scalarType = input_list[0].dtype();
    gloo::AllgatherOptions opts(context_);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());

    Tensor flatInputTensor = flattenDenseTensors(input_list);
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);

    int64_t output_numel = 0;
    for (const auto& t : output_lists[0]) {
      output_numel += t.numel();
    }
    output_numel *= (int64_t)output_lists.size();
    Tensor flatOutputTensor = Tensor::empty(
        {output_numel}, output_lists[0][0].dtype(), output_lists[0][0].device());
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    gloo::allgather(opts);

    int64_t current_element = 0;
    for (auto& output_list : output_lists) {
      for (auto& output_tensor : output_list) {
        output_tensor.copy_(
            flatOutputTensor.narrow(0, current_element, output_tensor.numel())
                .reshape(output_tensor.shape()));
        current_element += output_tensor.numel();
      }
    }
  }
};

class AsyncGatherSingleWork : public GlooAsyncWork {
 public:
  AsyncGatherSingleWork(
      std::shared_ptr<gloo::Context> context,
      Tensor& output,
      Tensor& input,
      int root,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(
            std::move(context),
            {{output}},
            "gather",
            seq,
            timeout),
        output(output),
        input(input),
        root(root),
        tag(tag) {}

  Tensor output;
  Tensor input;
  const int root;
  const uint32_t tag;

  void run() override {
    const auto scalarType = input.dtype();
    gloo::GatherOptions opts(context_);
    opts.setRoot(root);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setInput, opts, input);
    if (context_->rank == root) {
      TP_GLOO_GENERATE_ALL_TYPES(scalarType, setOutput, opts, output);
    }
    gloo::gather(opts);
  }
};

class AsyncGatherWork : public GlooAsyncWork {
 public:
  AsyncGatherWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<std::vector<Tensor>>& outputs,
      std::vector<Tensor>& inputs,
      int root,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(std::move(context), outputs, "gather", seq, timeout),
        outputs(outputs),
        inputs(inputs),
        root(root),
        tag(tag) {}

  std::vector<std::vector<Tensor>> outputs;
  std::vector<Tensor> inputs;
  const int root;
  const uint32_t tag;

  void gatherOne(
      std::vector<std::vector<Tensor>>& outputs,
      std::vector<Tensor>& inputs) {
    const auto scalarType = inputs[0].dtype();
    gloo::GatherOptions opts(context_);
    opts.setRoot(root);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());

    Tensor flatOutputTensor;
    if (context_->rank == root) {
      flatOutputTensor = newLikeFlat(outputs[0]);
      TP_GLOO_GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    }

    Tensor flatInputTensor = flattenDenseTensors(inputs);
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);
    gloo::gather(opts);

    if (context_->rank == root) {
      for (size_t i = 0; i < outputs[0].size(); ++i) {
        outputs[0][i].copy_(
            flatOutputTensor.narrow(0, (int64_t)i, 1).reshape(
                outputs[0][i].shape()));
      }
    }
  }

  void run() override {
    gatherOne(outputs, inputs);
  }
};

class AsyncScatterWork : public GlooAsyncWork {
 public:
  AsyncScatterWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<Tensor>& outputs,
      std::vector<std::vector<Tensor>>& inputs,
      int root,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(
            std::move(context),
            {outputs},
            "scatter",
            seq,
            timeout),
        outputs(outputs),
        inputs(inputs),
        root(root),
        tag(tag) {}

  std::vector<Tensor> outputs;
  std::vector<std::vector<Tensor>> inputs;
  const int root;
  const uint32_t tag;

  void scatterOne(
      std::vector<Tensor>& outputs,
      std::vector<std::vector<Tensor>>& inputs) {
    const auto scalarType = outputs[0].dtype();
    gloo::ScatterOptions opts(context_);
    opts.setRoot(root);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());

    if (context_->rank == root) {
      TP_GLOO_GENERATE_ALL_TYPES(scalarType, setInputs, opts, inputs[0]);
    }
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputs[0]);
    gloo::scatter(opts);
  }

  void run() override {
    scatterOne(outputs, inputs);
  }
};

class AsyncAlltoallWork : public GlooAsyncWork {
 public:
  AsyncAlltoallWork(
      std::shared_ptr<gloo::Context> context,
      Tensor& outputTensor,
      Tensor& inputTensor,
      std::vector<int64_t> outputCounts,
      std::vector<int64_t> inputCounts,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(
            std::move(context),
            {{outputTensor}},
            "all_to_all",
            seq,
            timeout),
        outputTensor(outputTensor),
        inputTensor(inputTensor),
        outputCounts(std::move(outputCounts)),
        inputCounts(std::move(inputCounts)),
        tag(tag) {}

  Tensor outputTensor;
  Tensor inputTensor;
  std::vector<int64_t> outputCounts;
  std::vector<int64_t> inputCounts;
  const uint32_t tag;

  void alltoallOne(Tensor& outputTensor, Tensor& inputTensor) {
    const auto scalarType = outputTensor.dtype();
    if (outputCounts.empty() && inputCounts.empty()) {
      gloo::AlltoallOptions opts(context_);
      opts.setTag(tag);
      opts.setTimeout(getTimeout());
      TP_GLOO_GENERATE_ALL_TYPES(scalarType, setInput, opts, inputTensor);
      TP_GLOO_GENERATE_ALL_TYPES(scalarType, setOutput, opts, outputTensor);
      gloo::alltoall(opts);
    } else {
      std::vector<int64_t> sendCounts(context_->size);
      std::vector<int64_t> recvCounts(context_->size);
      std::vector<int64_t> sendOffsets(context_->size);
      std::vector<int64_t> recvOffsets(context_->size);
      computeLengthsAndOffsets(
          inputCounts, inputTensor, &sendCounts, &sendOffsets);
      computeLengthsAndOffsets(
          outputCounts, outputTensor, &recvCounts, &recvOffsets);
      gloo::AlltoallvOptions opts(context_);
      opts.setTag(tag);
      opts.setTimeout(getTimeout());
      TP_GLOO_GENERATE_ALL_TYPES(
          scalarType, setInput, opts, inputTensor, sendCounts);
      TP_GLOO_GENERATE_ALL_TYPES(
          scalarType, setOutput, opts, outputTensor, recvCounts);
      gloo::alltoallv(opts);
    }
  }

  void run() override {
    alltoallOne(outputTensor, inputTensor);
  }
};

class AsyncAlltoallListWork : public GlooAsyncWork {
 public:
  AsyncAlltoallListWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<Tensor>& outputTensors,
      std::vector<Tensor>& inputTensors,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(
            std::move(context),
            {outputTensors},
            "all_to_all",
            seq,
            timeout),
        outputTensors(outputTensors),
        inputTensors(inputTensors),
        tag(tag) {}

  std::vector<Tensor> outputTensors;
  std::vector<Tensor> inputTensors;
  const uint32_t tag;

  void alltoallListOne(
      std::vector<Tensor>& outputTensors,
      std::vector<Tensor>& inputTensors) {
    const auto scalarType = inputTensors[0].dtype();
    gloo::AlltoallOptions opts(context_);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());

    Tensor flatInputTensor = flattenDenseTensors(inputTensors);
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);

    Tensor flatOutputTensor = newLikeFlat(outputTensors);
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);

    gloo::alltoall(opts);

    for (size_t i = 0; i < outputTensors.size(); ++i) {
      outputTensors[i].copy_(
          flatOutputTensor.narrow(0, (int64_t)i, 1).reshape(
              outputTensors[i].shape()));
    }
  }

  void run() override {
    alltoallListOne(outputTensors, inputTensors);
  }
};

class AsyncBarrierWork : public GlooAsyncWork {
 public:
  AsyncBarrierWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<std::weak_ptr<GlooAsyncWork>> priorWork,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(std::move(context), {}, "barrier", seq, timeout),
        priorWork(std::move(priorWork)),
        tag(tag) {}

  std::vector<std::weak_ptr<GlooAsyncWork>> priorWork;
  const uint32_t tag;

  void run() override {
    for (auto& weakWork : priorWork) {
      auto work = weakWork.lock();
      if (work) {
        work->wait();
      }
    }
    gloo::BarrierOptions opts(context_);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());
    gloo::barrier(opts);
  }
};

}  // namespace

GlooSendWork::GlooSendWork(
    Tensor tensor,
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
    uint64_t seq,
    std::string opName)
    : GlooWork(std::move(opName)),
      tensor_(std::move(tensor)),
      buffer_(std::move(buffer)) {
  seq_ = seq;
}

bool GlooSendWork::wait(int64_t timeout_ms) {
  {
    std::lock_guard<std::mutex> lock(waitMutex_);
    if (completed_) {
      if (exception_ != nullptr) {
        std::rethrow_exception(exception_);
      }
      return true;
    }
  }

  try {
    const bool completed = timeout_ms < 0
        ? buffer_->waitSend()
        : buffer_->waitSend(std::chrono::milliseconds(timeout_ms));
    if (completed) {
      finish();
    }
    return completed;
  } catch (...) {
    auto exception = std::current_exception();
    finishWithError(exception);
    std::rethrow_exception(exception);
  }
}

void GlooSendWork::abort() {
  buffer_->abortWaitSend();
  finishWithError(std::make_exception_ptr(
      std::runtime_error("gloo send work was aborted")));
}

// ---------------------------------------------------------------------------
// Collective entry points
// ---------------------------------------------------------------------------

std::shared_ptr<GlooWork> ProcessGroupGloo::broadcast(
    std::vector<Tensor>& inputs,
    int rootRank,
    int rootTensor,
    std::chrono::milliseconds timeout) {
  assertRootRank(rootRank, size_, "ProcessGroupGloo::broadcast");
  assertRootTensor(rootTensor, (int64_t)inputs.size(), "ProcessGroupGloo::broadcast");
  assertNonEmptyDeviceCpu(inputs, "ProcessGroupGloo::broadcast");
  assertDense(inputs, "ProcessGroupGloo::broadcast");
  assertTypeAndSizesMatch(inputs, "ProcessGroupGloo::broadcast");

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  auto work = std::make_shared<AsyncBroadcastWork>(
      std::move(context), inputs, rootRank, rootTensor, tag, seq_, timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::allreduce(
    std::vector<Tensor>& inputs,
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  assertNonEmptyDeviceCpu(inputs, "ProcessGroupGloo::allreduce");
  assertTypeAndSizesMatch(inputs, "ProcessGroupGloo::allreduce");
  if (inputs[0].is_sparse()) {
    if (inputs[0].is_sparse_csr()) {
      invalidArgument(
          "ProcessGroupGloo::allreduce: sparse CSR tensors are not supported");
    }
    if (reduceOp != ReduceOp::SUM) {
      invalidArgument(
          "ProcessGroupGloo::allreduce: sparse tensors support SUM only");
    }
    for (const auto& tensor : inputs) {
      if (tensor.is_sparse_csr()) {
        invalidArgument(
            "ProcessGroupGloo::allreduce: sparse CSR tensors are not supported");
      }
    }
  }

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work;
  if (inputs[0].is_sparse()) {
    work = std::make_shared<AsyncSparseAllreduceWork>(
        std::move(context), inputs, tag, seq_, timeout);
  } else {
    work = std::make_shared<AsyncAllreduceWork>(
        std::move(context), inputs, reduceOp, tag, seq_, timeout);
  }
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::allreduce_coalesced(
    std::vector<Tensor>& tensors,
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  assertNonEmptyDeviceCpu(tensors, "ProcessGroupGloo::allreduce_coalesced");
  assertDense(tensors, "ProcessGroupGloo::allreduce_coalesced");
  for (const auto& t : tensors) {
    if (t.is_sparse() || t.dtype() != tensors[0].dtype() ||
        t.device() != tensors[0].device()) {
      invalidArgument(
          "ProcessGroupGloo::allreduce_coalesced: tensors must share type "
          "and device and must be dense");
    }
  }
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work =
      std::make_shared<AsyncAllreduceCoalescedWork>(
          std::move(context), tensors, reduceOp, tag, seq_, timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::reduce(
    std::vector<Tensor>& inputs,
    int rootRank,
    int rootTensor,
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  assertRootRank(rootRank, size_, "ProcessGroupGloo::reduce");
  assertRootTensor(rootTensor, (int64_t)inputs.size(), "ProcessGroupGloo::reduce");
  assertSingleElement(inputs, "ProcessGroupGloo::reduce");
  assertNonEmptyDeviceCpu(inputs, "ProcessGroupGloo::reduce");
  assertDense(inputs, "ProcessGroupGloo::reduce");

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work = std::make_shared<AsyncReduceWork>(
      std::move(context),
      inputs,
      rootRank,
      rootTensor,
      reduceOp,
      tag,
      seq_,
      timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::allgather(
    std::vector<std::vector<Tensor>>& outputs,
    std::vector<Tensor>& inputs,
    std::chrono::milliseconds timeout) {
  if (inputs.empty()) {
    invalidArgument("ProcessGroupGloo::allgather: requires non-empty inputs");
  }
  if (inputs.size() != outputs.size()) {
    invalidArgument(
        "ProcessGroupGloo::allgather: input/output lists must have the same "
        "length");
  }
  for (const auto& outputList : outputs) {
    if ((int64_t)outputList.size() != (int64_t)inputs.size() * size_) {
      invalidArgument(
          "ProcessGroupGloo::allgather: invalid output tensor list length");
    }
  }
  assertNonEmptyDeviceCpu(inputs, "ProcessGroupGloo::allgather");
  assertDense(inputs, "ProcessGroupGloo::allgather");
  assertTypeAndSizesMatch(inputs, "ProcessGroupGloo::allgather");
  if (outputs.empty()) {
    invalidArgument(
        "ProcessGroupGloo::allgather: output lists must not be empty");
  }
  for (const auto& outputList : outputs) {
    assertNonEmptyDeviceCpu(outputList, "ProcessGroupGloo::allgather");
    assertDense(outputList, "ProcessGroupGloo::allgather");
    for (const auto& output : outputList) {
      assertTensorTypeAndSizesMatch(
          inputs[0], output, "ProcessGroupGloo::allgather");
    }
  }

  std::vector<Tensor> inputViews;
  inputViews.reserve(inputs.size());
  for (const auto& input : inputs) {
    inputViews.push_back(
        isComplexType(input.dtype()) ? input.view_as_real() : input);
  }
  std::vector<std::vector<Tensor>> outputViews;
  outputViews.reserve(outputs.size());
  for (const auto& outputList : outputs) {
    auto& views = outputViews.emplace_back();
    views.reserve(outputList.size());
    for (const auto& output : outputList) {
      views.push_back(
          isComplexType(output.dtype()) ? output.view_as_real() : output);
    }
  }

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work = std::make_shared<AsyncAllgatherWork>(
      std::move(context), outputViews, inputViews, tag, seq_, timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::all_gather_single(
    Tensor& output,
    Tensor& input,
    std::chrono::milliseconds timeout) {
  assertNonEmptyDeviceCpu({output}, "ProcessGroupGloo::all_gather_single");
  assertNonEmptyDeviceCpu({input}, "ProcessGroupGloo::all_gather_single");
  assertDense(output, "ProcessGroupGloo::all_gather_single");
  assertDense(input, "ProcessGroupGloo::all_gather_single");
  if (output.dtype() != input.dtype() || output.device() != input.device()) {
    invalidArgument(
        "ProcessGroupGloo::all_gather_single: input and output tensors "
        "must have the same type and device");
  }

  const auto inputView =
      isComplexType(input.dtype()) ? input.view_as_real() : input;
  const auto outputView =
      isComplexType(output.dtype()) ? output.view_as_real() : output;
  const auto inputShape = static_cast<std::vector<int64_t>>(input.shape());
  const auto outputShape = static_cast<std::vector<int64_t>>(output.shape());
  const auto inputViewShape =
      static_cast<std::vector<int64_t>>(inputView.shape());
  auto concatenatedShape = inputShape;
  if (concatenatedShape.empty()) {
    concatenatedShape.push_back(static_cast<int64_t>(size_));
  } else {
    if (concatenatedShape[0] >
        std::numeric_limits<int64_t>::max() / size_) {
      invalidArgument(
          "ProcessGroupGloo::all_gather_single: output shape overflows");
    }
    concatenatedShape[0] *= static_cast<int64_t>(size_);
  }
  auto stackedShape = inputShape;
  stackedShape.insert(stackedShape.begin(), static_cast<int64_t>(size_));

  std::vector<Tensor> outputChunks;
  if (outputShape == concatenatedShape) {
    if (inputShape.empty()) {
      outputChunks.reserve(static_cast<size_t>(size_));
      for (int rank = 0; rank < size_; ++rank) {
        outputChunks.push_back(
            outputView.narrow(0, rank, 1).reshape(inputViewShape));
      }
    } else {
      outputChunks = outputView.chunk(size_, 0);
    }
  } else if (outputShape == stackedShape) {
    outputChunks.reserve(static_cast<size_t>(size_));
    for (int rank = 0; rank < size_; ++rank) {
      outputChunks.push_back(outputView.select(0, rank));
    }
  } else {
    invalidArgument(
        "ProcessGroupGloo::all_gather_single: output tensor shape must be "
        "either concatenated or stacked along dim 0");
  }
  if (outputChunks.size() != static_cast<size_t>(size_)) {
    invalidArgument(
        "ProcessGroupGloo::all_gather_single: output tensor must provide "
        "one chunk per group rank");
  }
  std::vector<std::vector<Tensor>> outputs = {outputChunks};
  std::vector<Tensor> inputs = {inputView};
  return allgather(outputs, inputs, timeout);
}

std::shared_ptr<GlooWork> ProcessGroupGloo::allgather_coalesced(
    std::vector<std::vector<Tensor>>& outputLists,
    std::vector<Tensor>& inputList,
    std::chrono::milliseconds timeout) {
  if (inputList.empty()) {
    invalidArgument(
        "ProcessGroupGloo::allgather_coalesced: requires non-empty input "
        "tensor list");
  }
  if (outputLists.size() != static_cast<size_t>(size_)) {
    invalidArgument(
        "ProcessGroupGloo::allgather_coalesced: output lists should be equal "
        "to world size");
  }
  for (const auto& outputList : outputLists) {
    if (outputList.size() != inputList.size()) {
      invalidArgument(
          "ProcessGroupGloo::allgather_coalesced: invalid output tensor list "
          "length");
    }
  }
  assertNonEmptyDeviceCpu(inputList, "ProcessGroupGloo::allgather_coalesced");
  assertDense(inputList, "ProcessGroupGloo::allgather_coalesced");
  for (size_t i = 1; i < inputList.size(); ++i) {
    if (inputList[i].dtype() != inputList[0].dtype() ||
        inputList[i].device() != inputList[0].device()) {
      invalidArgument(
          "ProcessGroupGloo::allgather_coalesced: input tensors must share "
          "dtype and device");
    }
  }
  for (const auto& outputList : outputLists) {
    assertNonEmptyDeviceCpu(outputList, "ProcessGroupGloo::allgather_coalesced");
    assertDense(outputList, "ProcessGroupGloo::allgather_coalesced");
    for (size_t i = 0; i < outputList.size(); ++i) {
      if (outputList[i].shape() != inputList[i].shape()) {
        invalidArgument(
            "ProcessGroupGloo::allgather_coalesced: output tensor sizes do "
            "not match input tensor sizes");
      }
      if (outputList[i].dtype() != inputList[i].dtype() ||
          outputList[i].device() != inputList[i].device()) {
        invalidArgument(
            "ProcessGroupGloo::allgather_coalesced: output tensor types do "
            "not match input tensor types");
      }
    }
  }

  std::vector<Tensor> inputViews;
  inputViews.reserve(inputList.size());
  for (const auto& input : inputList) {
    inputViews.push_back(
        isComplexType(input.dtype()) ? input.view_as_real() : input);
  }
  std::vector<std::vector<Tensor>> outputViews;
  outputViews.reserve(outputLists.size());
  for (const auto& outputList : outputLists) {
    auto& views = outputViews.emplace_back();
    views.reserve(outputList.size());
    for (const auto& output : outputList) {
      views.push_back(
          isComplexType(output.dtype()) ? output.view_as_real() : output);
    }
  }

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work =
      std::make_shared<AsyncAllgatherCoalescedWork>(
          std::move(context), outputViews, inputViews, tag, seq_, timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::all_gather_single_coalesced(
    std::vector<Tensor>& outputs,
    std::vector<Tensor>& inputs,
    std::chrono::milliseconds timeout) {
  if (outputs.size() != inputs.size()) {
    invalidArgument(
        "ProcessGroupGloo::all_gather_single_coalesced: input/output tensor "
        "lists must have the same length");
  }
  if (inputs.empty()) {
    invalidArgument(
        "ProcessGroupGloo::all_gather_single_coalesced: requires a "
        "non-empty tensor list");
  }

  std::vector<Tensor> inputViews;
  inputViews.reserve(inputs.size());
  std::vector<std::vector<Tensor>> outputLists(size_);
  for (size_t index = 0; index < inputs.size(); ++index) {
    auto& input = inputs[index];
    auto& output = outputs[index];
    assertNonEmptyDeviceCpu(
        {input}, "ProcessGroupGloo::all_gather_single_coalesced");
    assertNonEmptyDeviceCpu(
        {output}, "ProcessGroupGloo::all_gather_single_coalesced");
    assertDense(input, "ProcessGroupGloo::all_gather_single_coalesced");
    assertDense(output, "ProcessGroupGloo::all_gather_single_coalesced");
    if (input.dtype() != output.dtype() || input.device() != output.device()) {
      invalidArgument(
          "ProcessGroupGloo::all_gather_single_coalesced: input/output "
          "tensor types do not match");
    }

    const auto inputShape = static_cast<std::vector<int64_t>>(input.shape());
    const auto outputShape = static_cast<std::vector<int64_t>>(output.shape());
    Tensor inputView =
        isComplexType(input.dtype()) ? input.view_as_real() : input;
    Tensor outputView =
        isComplexType(output.dtype()) ? output.view_as_real() : output;
    inputViews.push_back(inputView);
    const auto inputViewShape =
        static_cast<std::vector<int64_t>>(inputView.shape());
    const auto outputViewShape =
        static_cast<std::vector<int64_t>>(outputView.shape());
    auto expectedShape = inputShape;
    if (inputShape.empty()) {
        expectedShape = {static_cast<int64_t>(size_)};
    } else {
      if (inputShape[0] >
          std::numeric_limits<int64_t>::max() /
              static_cast<int64_t>(size_)) {
        invalidArgument(
            "ProcessGroupGloo::all_gather_single_coalesced: output shape "
            "overflows");
      }
      expectedShape[0] *= static_cast<int64_t>(size_);
    }
    auto stackedViewShape = inputViewShape;
    stackedViewShape.insert(
        stackedViewShape.begin(), static_cast<int64_t>(size_));
    const bool outputIsStacked = outputViewShape == stackedViewShape;
    if (outputShape != expectedShape && !outputIsStacked) {
      invalidArgument(
          "ProcessGroupGloo::all_gather_single_coalesced: output tensor "
          "shape is invalid");
    }

    if (outputIsStacked) {
      for (int rank = 0; rank < size_; ++rank) {
        outputLists[static_cast<size_t>(rank)].push_back(
            outputView.select(0, rank));
      }
    } else if (inputShape.empty()) {
      for (int rank = 0; rank < size_; ++rank) {
        outputLists[static_cast<size_t>(rank)].push_back(
            outputView.narrow(0, rank, 1).reshape(
                inputViewShape));
      }
    } else {
      const int64_t chunk = inputShape[0];
      for (int rank = 0; rank < size_; ++rank) {
        outputLists[static_cast<size_t>(rank)].push_back(
            outputView.narrow(
                0, static_cast<int64_t>(rank) * chunk, chunk));
      }
    }
  }
  return allgather_coalesced(outputLists, inputViews, timeout);
}

std::shared_ptr<GlooWork> ProcessGroupGloo::all_gather_into_tensor(
    Tensor& output,
    Tensor& input,
    std::chrono::milliseconds timeout) {
  return all_gather_single(output, input, timeout);
}

std::shared_ptr<GlooWork> ProcessGroupGloo::gather_single(
    Tensor& output,
    Tensor& input,
    int rootRank,
    std::chrono::milliseconds timeout) {
  assertRootRank(rootRank, size_, "ProcessGroupGloo::gather_single");
  assertNonEmptyDeviceCpu({input}, "ProcessGroupGloo::gather_single");
  if (!input.is_contiguous()) {
    invalidArgument(
        "ProcessGroupGloo::gather_single: input must be contiguous");
  }
  if (getRank() == rootRank) {
    if (!output.defined() || !output.is_contiguous()) {
      invalidArgument(
          "ProcessGroupGloo::gather_single: output must be a contiguous "
          "tensor on the destination rank");
    }
    if (!productEquals(
            input.numel(), static_cast<int64_t>(size_), output.numel())) {
      invalidArgument(
          "ProcessGroupGloo::gather_single: output size must equal input "
          "size times world size");
    }
    if (output.dtype() != input.dtype() || output.device() != input.device()) {
      invalidArgument(
          "ProcessGroupGloo::gather_single: input/output tensor types do not "
          "match");
    }
  }

  Tensor inputView =
      isComplexType(input.dtype()) ? input.view_as_real() : input;
  Tensor outputView =
      output.defined() && isComplexType(output.dtype())
      ? output.view_as_real()
      : output;

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work = std::make_shared<AsyncGatherSingleWork>(
      std::move(context), outputView, inputView, rootRank, tag, seq_, timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::gather(
    std::vector<std::vector<Tensor>>& outputs,
    std::vector<Tensor>& inputs,
    int rootRank,
    std::chrono::milliseconds timeout) {
  assertRootRank(rootRank, size_, "ProcessGroupGloo::gather");
  assertSingleElement(inputs, "ProcessGroupGloo::gather");
  assertNonEmptyDeviceCpu(inputs, "ProcessGroupGloo::gather");
  assertDense(inputs, "ProcessGroupGloo::gather");
  if (getRank() == rootRank) {
    if ((int64_t)outputs.size() != 1 ||
        (int64_t)outputs[0].size() != size_) {
      invalidArgument(
          "ProcessGroupGloo::gather: root expects one tensor per rank");
    }
    assertNonEmptyDeviceCpu(outputs[0], "ProcessGroupGloo::gather");
    assertDense(outputs[0], "ProcessGroupGloo::gather");
    for (const auto& output : outputs[0]) {
      assertTensorTypeAndSizesMatch(
          inputs[0], output, "ProcessGroupGloo::gather");
    }
  } else if (!outputs.empty()) {
    invalidArgument(
        "ProcessGroupGloo::gather: non-root output list must be empty");
  }

  std::vector<Tensor> inputViews;
  inputViews.reserve(inputs.size());
  for (const auto& input : inputs) {
    inputViews.push_back(
        isComplexType(input.dtype()) ? input.view_as_real() : input);
  }
  std::vector<std::vector<Tensor>> outputViews;
  outputViews.reserve(outputs.size());
  for (const auto& outputList : outputs) {
    auto& views = outputViews.emplace_back();
    views.reserve(outputList.size());
    for (const auto& output : outputList) {
      views.push_back(
          isComplexType(output.dtype()) ? output.view_as_real() : output);
    }
  }

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work = std::make_shared<AsyncGatherWork>(
      std::move(context), outputViews, inputViews, rootRank, tag, seq_, timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::scatter(
    std::vector<Tensor>& outputs,
    std::vector<std::vector<Tensor>>& inputs,
    int rootRank,
    std::chrono::milliseconds timeout) {
  assertRootRank(rootRank, size_, "ProcessGroupGloo::scatter");
  if (outputs.size() != 1) {
    invalidArgument("ProcessGroupGloo::scatter: requires a single output");
  }
  assertNonEmptyDeviceCpu(outputs, "ProcessGroupGloo::scatter");
  assertDense(outputs, "ProcessGroupGloo::scatter");
  if (getRank() == rootRank) {
    if (inputs.size() != 1 || (int64_t)inputs[0].size() != size_) {
      invalidArgument(
          "ProcessGroupGloo::scatter: root expects one input tensor per rank");
    }
    assertTypeAndSizesMatch(inputs[0], "ProcessGroupGloo::scatter");
    for (const auto& input : inputs[0]) {
      assertTensorTypeAndSizesMatch(
          outputs[0], input, "ProcessGroupGloo::scatter");
    }
  } else if (!inputs.empty()) {
    invalidArgument(
        "ProcessGroupGloo::scatter: non-root input list must be empty");
  }

  std::vector<Tensor> outputViews;
  outputViews.reserve(outputs.size());
  for (const auto& output : outputs) {
    outputViews.push_back(
        isComplexType(output.dtype()) ? output.view_as_real() : output);
  }
  std::vector<std::vector<Tensor>> inputViews;
  inputViews.reserve(inputs.size());
  for (const auto& inputList : inputs) {
    auto& views = inputViews.emplace_back();
    views.reserve(inputList.size());
    for (const auto& input : inputList) {
      views.push_back(
          isComplexType(input.dtype()) ? input.view_as_real() : input);
    }
  }

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work = std::make_shared<AsyncScatterWork>(
      std::move(context), outputViews, inputViews, rootRank, tag, seq_, timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::reduce_scatter(
    std::vector<Tensor>& outputs,
    std::vector<std::vector<Tensor>>& inputs,
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  const auto rank = getRank();
  const auto worldSize = getSize();
  if (outputs.size() != 1) {
    invalidArgument("ProcessGroupGloo::reduce_scatter: 1 output only");
  }
  if (inputs.size() != 1 || (int64_t)inputs[0].size() != worldSize) {
    invalidArgument(
        "ProcessGroupGloo::reduce_scatter: input list length must equal "
        "world size");
  }
  assertNonEmptyDeviceCpu(outputs, "ProcessGroupGloo::reduce_scatter");
  assertDense(outputs, "ProcessGroupGloo::reduce_scatter");
  assertNonEmptyDeviceCpu(inputs[0], "ProcessGroupGloo::reduce_scatter");
  assertDense(inputs[0], "ProcessGroupGloo::reduce_scatter");
  assertTypeAndSizesMatch(inputs[0], "ProcessGroupGloo::reduce_scatter");
  assertTensorTypeAndSizesMatch(
      outputs[0], inputs[0][0], "ProcessGroupGloo::reduce_scatter");

  std::vector<Tensor> buffers;
  for (size_t i = 0; i < (size_t)worldSize; ++i) {
    if ((int)i == rank) {
      outputs[0].copy_(inputs[0][i]);
      buffers.push_back(outputs[0]);
    } else {
      buffers.push_back(inputs[0][i].clone());
    }
  }
  std::vector<std::shared_ptr<GlooWork>> works;
  for (auto& buffer : buffers) {
    std::vector<Tensor> inp = {buffer};
    works.push_back(allreduce(inp, reduceOp, timeout));
  }
  return std::make_shared<GlooCompositeWork>(
      "reduce_scatter",
      std::vector<std::vector<Tensor>>{outputs},
      std::move(works),
      [] {});
}

std::shared_ptr<GlooWork> ProcessGroupGloo::reduce_scatter_tensor(
    Tensor& output,
    Tensor& input,
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  const auto worldSize = getSize();
  assertNonEmptyDeviceCpu({output}, "ProcessGroupGloo::reduce_scatter_tensor");
  assertNonEmptyDeviceCpu({input}, "ProcessGroupGloo::reduce_scatter_tensor");
  assertDense(output, "ProcessGroupGloo::reduce_scatter_tensor");
  assertDense(input, "ProcessGroupGloo::reduce_scatter_tensor");
  if (output.dtype() != input.dtype() || output.device() != input.device()) {
    invalidArgument(
        "ProcessGroupGloo::reduce_scatter_tensor: input/output tensors must "
        "have the same type and device");
  }
  if (output.dim() == 0 || input.dim() == 0 ||
      !productEquals(output.size(0), worldSize, input.size(0))) {
    invalidArgument(
        "ProcessGroupGloo::reduce_scatter_tensor: dim 0 of input must equal "
        "output dim 0 times world size");
  }
  if (output.shape().size() != input.shape().size()) {
    invalidArgument(
        "ProcessGroupGloo::reduce_scatter_tensor: input/output dimensions "
        "do not match");
  }
  for (size_t dim = 1; dim < output.shape().size(); ++dim) {
    if (output.shape()[dim] != input.shape()[dim]) {
      invalidArgument(
          "ProcessGroupGloo::reduce_scatter_tensor: input/output sizes do "
          "not match");
    }
  }
  Tensor inputClone = input.clone();
  std::vector<Tensor> inp = {inputClone};
  auto work = allreduce(inp, reduceOp, timeout);
  const auto rank = getRank();
  auto finalize = [output, inputClone, rank, worldSize]() mutable {
    const int64_t chunk = inputClone.numel() / worldSize;
    output.copy_(
        inputClone.narrow(0, static_cast<int64_t>(rank) * chunk, chunk)
            .reshape(output.shape()));
  };
  return std::make_shared<GlooCompositeWork>(
      "reduce_scatter_tensor",
      std::vector<std::vector<Tensor>>{{output}},
      std::vector<std::shared_ptr<GlooWork>>{std::move(work)},
      std::move(finalize));
}

std::shared_ptr<GlooWork> ProcessGroupGloo::reduce_scatter_single(
    Tensor& output,
    Tensor& input,
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  std::vector<Tensor> outputs = {output};
  std::vector<Tensor> inputs = {input};
  return reduce_scatter_single_coalesced(
      outputs, inputs, reduceOp, timeout);
}

std::shared_ptr<GlooWork> ProcessGroupGloo::reduce_scatter_single_coalesced(
    std::vector<Tensor>& outputs,
    std::vector<Tensor>& inputs,
    ReduceOp reduceOp,
    std::chrono::milliseconds timeout) {
  if (outputs.size() != inputs.size()) {
    invalidArgument(
        "ProcessGroupGloo::reduce_scatter_single_coalesced: input/output "
        "tensor lists must have the same length");
  }
  if (!inputs.empty()) {
    assertNonEmptyDeviceCpu(
        inputs, "ProcessGroupGloo::reduce_scatter_single_coalesced");
  }
  if (!outputs.empty()) {
    assertNonEmptyDeviceCpu(
        outputs, "ProcessGroupGloo::reduce_scatter_single_coalesced");
  }
  if (inputs.empty()) {
    invalidArgument(
        "ProcessGroupGloo::reduce_scatter_single_coalesced: requires a "
        "non-empty tensor list");
  }

  const auto rank = getRank();
  const auto worldSize = getSize();
  std::vector<Tensor> buffers;
  buffers.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (outputs[i].dtype() != inputs[i].dtype() ||
        outputs[i].device() != inputs[i].device()) {
      invalidArgument(
          "ProcessGroupGloo::reduce_scatter_single_coalesced: input/output "
          "tensor types do not match");
    }
    if (!productEquals(
            outputs[i].numel(), static_cast<int64_t>(worldSize),
            inputs[i].numel())) {
      invalidArgument(
          "ProcessGroupGloo::reduce_scatter_single_coalesced: input size "
          "must equal output size times world size");
    }
    buffers.push_back(inputs[i].clone());
  }

  std::vector<std::shared_ptr<GlooWork>> works;
  works.reserve(buffers.size());
  for (auto& buffer : buffers) {
    std::vector<Tensor> input = {buffer};
    works.push_back(allreduce(input, reduceOp, timeout));
  }

  auto finalize = [outputs, buffers, rank, worldSize]() mutable {
    for (size_t i = 0; i < buffers.size(); ++i) {
      const auto flatBuffer = buffers[i].reshape({buffers[i].numel()});
      const auto chunk = flatBuffer.numel() / worldSize;
      outputs[i].copy_(
          flatBuffer
              .narrow(0, static_cast<int64_t>(rank) * chunk, chunk)
              .reshape(outputs[i].shape()));
    }
  };
  return std::make_shared<GlooCompositeWork>(
      "reduce_scatter",
      std::vector<std::vector<Tensor>>{outputs},
      std::move(works),
      std::move(finalize));
}

std::shared_ptr<GlooWork> ProcessGroupGloo::all_to_all_single(
    Tensor& outputTensor,
    Tensor& inputTensor,
    std::vector<int64_t> outputCounts,
    std::vector<int64_t> inputCounts,
    std::chrono::milliseconds timeout) {
  assertNonEmptyDeviceCpu({outputTensor}, "ProcessGroupGloo::all_to_all_single");
  assertNonEmptyDeviceCpu({inputTensor}, "ProcessGroupGloo::all_to_all_single");
  assertDense(outputTensor, "ProcessGroupGloo::all_to_all_single");
  assertDense(inputTensor, "ProcessGroupGloo::all_to_all_single");
  if (outputTensor.device() != inputTensor.device()) {
    invalidArgument(
        "ProcessGroupGloo::all_to_all_single: output and input must be on "
        "the same device");
  }
  if (outputTensor.dtype() != inputTensor.dtype()) {
    invalidArgument(
        "ProcessGroupGloo::all_to_all_single: output and input must have "
        "the same dtype");
  }
  if (!inputTensor.is_contiguous() || !outputTensor.is_contiguous()) {
    invalidArgument(
        "ProcessGroupGloo::all_to_all_single: input and output tensors "
        "must be contiguous");
  }
  if (outputCounts.empty() != inputCounts.empty()) {
    invalidArgument(
        "ProcessGroupGloo::all_to_all_single: input and output split sizes "
        "must be specified together");
  }
  checkSplitSizes(inputCounts, inputTensor, getSize());
  checkSplitSizes(outputCounts, outputTensor, getSize());

  Tensor inputView =
      isComplexType(inputTensor.dtype()) ? inputTensor.view_as_real()
                                         : inputTensor;
  Tensor outputView =
      isComplexType(outputTensor.dtype()) ? outputTensor.view_as_real()
                                          : outputTensor;

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work = std::make_shared<AsyncAlltoallWork>(
      std::move(context),
      outputView,
      inputView,
      std::move(outputCounts),
      std::move(inputCounts),
      tag,
      seq_,
      timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::alltoall(
    std::vector<Tensor>& outputTensors,
    std::vector<Tensor>& inputTensors,
    std::chrono::milliseconds timeout) {
  if ((int64_t)outputTensors.size() != size_ ||
      (int64_t)inputTensors.size() != size_) {
    invalidArgument(
        "ProcessGroupGloo::alltoall: tensor list length must equal world "
        "size");
  }
  assertNonEmptyDeviceCpu(inputTensors, "ProcessGroupGloo::alltoall");
  assertNonEmptyDeviceCpu(outputTensors, "ProcessGroupGloo::alltoall");
  assertDense(inputTensors, "ProcessGroupGloo::alltoall");
  assertDense(outputTensors, "ProcessGroupGloo::alltoall");
  assertTypeAndSizesMatch(inputTensors, "ProcessGroupGloo::alltoall");
  assertTypeAndSizesMatch(outputTensors, "ProcessGroupGloo::alltoall");
  for (size_t i = 0; i < inputTensors.size(); ++i) {
    assertTensorTypeAndSizesMatch(
        inputTensors[i], outputTensors[i], "ProcessGroupGloo::alltoall");
  }

  std::vector<Tensor> inputViews;
  inputViews.reserve(inputTensors.size());
  for (const auto& input : inputTensors) {
    inputViews.push_back(
        isComplexType(input.dtype()) ? input.view_as_real() : input);
  }
  std::vector<Tensor> outputViews;
  outputViews.reserve(outputTensors.size());
  for (const auto& output : outputTensors) {
    outputViews.push_back(
        isComplexType(output.dtype()) ? output.view_as_real() : output);
  }

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work =
      std::make_shared<AsyncAlltoallListWork>(
          std::move(context), outputViews, inputViews, tag, seq_, timeout);
  enqueue(work);
  return work;
}

namespace {

Tensor& checkSingleTensor(std::vector<Tensor>& tensors, const char* op) {
  if (tensors.size() != 1) {
    invalidArgument(std::string(op) + " takes a single tensor");
  }
  auto& tensor = tensors[0];
  if (!tensor.defined()) {
    invalidArgument(std::string(op) + ": tensor is undefined");
  }
  if (tensor.is_sparse()) {
    invalidArgument(std::string(op) + ": only dense tensors are supported");
  }
  if (!tensor.device().is_cpu()) {
    invalidArgument(std::string(op) + ": only CPU tensors are supported");
  }
  if (!tensor.is_contiguous()) {
    invalidArgument(std::string(op) + ": input tensor has to be contiguous");
  }
  return tensor;
}

uint32_t checkTag(int32_t tag, const char* op) {
  if (tag < 0) {
    invalidArgument(std::string(op) + ": tag must be nonnegative");
  }
  return static_cast<uint32_t>(tag);
}

} // namespace

std::shared_ptr<GlooWork> ProcessGroupGloo::send(
    std::vector<Tensor>& tensors,
    int dstRank,
    int tag) {
  assertRootRank(dstRank, size_, "ProcessGroupGloo::send");
  auto& tensor = checkSingleTensor(tensors, "ProcessGroupGloo::send");
  auto utag = checkTag(tag, "ProcessGroupGloo::send");
  void* ptr = tensor.data_ptr();
  auto size = checkedByteSize(tensor, "ProcessGroupGloo::send");

  auto context = getContext(utag);
  auto buf = context->createUnboundBuffer(ptr, size);
  buf->send(dstRank, utag);
  ++seq_;
  return std::make_shared<GlooSendWork>(
      tensor, std::move(buf), seq_, "send");
}

std::shared_ptr<GlooWork> ProcessGroupGloo::recv(
    std::vector<Tensor>& tensors,
    int srcRank,
    int tag) {
  assertRootRank(srcRank, size_, "ProcessGroupGloo::recv");
  auto& tensor = checkSingleTensor(tensors, "ProcessGroupGloo::recv");
  auto utag = checkTag(tag, "ProcessGroupGloo::recv");
  void* ptr = tensor.data_ptr();
  auto size = checkedByteSize(tensor, "ProcessGroupGloo::recv");

  auto context = getContext(utag);
  auto buf = context->createUnboundBuffer(ptr, size);
  buf->recv(srcRank, utag);
  ++seq_;
  return std::make_shared<GlooRecvWork>(tensor, std::move(buf), seq_, "recv");
}

std::shared_ptr<GlooWork> ProcessGroupGloo::recvAnysource(
    std::vector<Tensor>& tensors,
    int tag) {
  auto& tensor = checkSingleTensor(tensors, "ProcessGroupGloo::recvAnysource");
  auto utag = checkTag(tag, "ProcessGroupGloo::recvAnysource");
  void* ptr = tensor.data_ptr();
  auto size = checkedByteSize(tensor, "ProcessGroupGloo::recvAnysource");

  auto context = getContext(utag);
  auto buf = context->createUnboundBuffer(ptr, size);
  std::vector<int> srcRanks;
  srcRanks.reserve(size_);
  for (int i = 0; i < size_; ++i) {
    srcRanks.push_back(i);
  }
  buf->recv(srcRanks, utag);
  ++seq_;
  return std::make_shared<GlooRecvWork>(
      tensor, std::move(buf), seq_, "recvAnysource");
}

std::shared_ptr<GlooWork> ProcessGroupGloo::barrier(
    std::chrono::milliseconds timeout) {
  std::vector<std::weak_ptr<GlooAsyncWork>> priorWork;
  {
    std::unique_lock<std::mutex> lock(workMutex_);
    priorWork.insert(priorWork.end(), workInProgress_.begin(), workInProgress_.end());
    priorWork.insert(priorWork.end(), workQueue_.begin(), workQueue_.end());
  }
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  auto work = std::make_shared<AsyncBarrierWork>(
      std::move(context), std::move(priorWork), tag, seq_, timeout);
  enqueue(work);
  return work;
}

void ProcessGroupGloo::monitoredBarrier(
    std::chrono::milliseconds timeout,
    bool waitAllRanks) {
  const auto barrierTimeout =
      timeout == kUnsetTimeout ? options_.timeout : timeout;
  auto t1 = nextTag();
  auto t2 = nextTag();
  std::vector<Tensor> commTensor = {
      Tensor::full({1}, (double)getRank(), ::tensorplay::ScalarType::Int64)};
  auto rank = getRank();
  if (rank != 0) {
    auto sendWork = send(commTensor, 0, (int)t1);
    auto recvWork = recv(commTensor, 0, (int)t2);
    try {
      sendWork->wait();
      recvWork->wait();
    } catch (const std::exception& e) {
      runtimeFailure(
          "Rank " + std::to_string(rank) +
          " successfully reached monitoredBarrier, but received errors while "
          "waiting for send/recv from rank 0: " + e.what());
    }
    return;
  }
  auto startTime = std::chrono::steady_clock::now();
  auto worldSize = getSize();
  std::map<int, std::shared_ptr<GlooWork>> recvWorkMap;
  std::map<int, std::shared_ptr<GlooWork>> sendWorkMap;
  for (int dstRank = 1; dstRank < worldSize; ++dstRank) {
    recvWorkMap.emplace(dstRank, recv(commTensor, dstRank, (int)t1));
  }

  auto waitLoop = [&](const std::map<int, std::shared_ptr<GlooWork>>& works) {
    std::vector<int> processedRanks;
    for (auto& entry : works) {
      bool rankResponded = false;
      try {
        const auto remaining = remainingBarrierTime(
            startTime, barrierTimeout, waitAllRanks);
        if (!waitAllRanks) {
          checkBarrierTime(
              barrierTimeout, remaining, processedRanks, rank);
        }
        if (!entry.second->wait(remaining.count())) {
          runtimeFailure(
              "Rank " + std::to_string(entry.first) +
              " failed to pass monitoredBarrier in " +
              std::to_string(barrierTimeout.count()) + " ms");
        }
        rankResponded = true;
      } catch (const std::exception& e) {
        const std::string error =
            "[Rank 0]: Rank " + std::to_string(entry.first) +
            " failed to pass monitoredBarrier in " +
            std::to_string(barrierTimeout.count()) + " ms: " + e.what();
        if (waitAllRanks) {
          std::cerr << error << std::endl;
        } else {
          runtimeFailure(error);
        }
      }
      if (rankResponded) {
        processedRanks.push_back(entry.first);
      }
    }
    if (waitAllRanks &&
        processedRanks.size() != static_cast<size_t>(size_ - 1)) {
      std::string failed;
      for (int i = 1; i < size_; ++i) {
        if (std::find(
                processedRanks.begin(), processedRanks.end(), i) ==
            processedRanks.end()) {
          failed += (failed.empty() ? "" : ", ") + std::to_string(i);
        }
      }
      runtimeFailure(
          "[Rank 0]: Ranks " + failed + " failed to pass monitoredBarrier in " +
          std::to_string(barrierTimeout.count()) + " ms");
    }
  };

  waitLoop(recvWorkMap);
  for (int dstRank = 1; dstRank < worldSize; ++dstRank) {
    sendWorkMap.emplace(dstRank, send(commTensor, dstRank, (int)t2));
  }
  waitLoop(sendWorkMap);
}

std::vector<Tensor> ProcessGroupGloo::splitEven(const Tensor& tensor) {
  const auto worldSize = (size_t)getSize();
  auto sizes = tensor.shape();
  if (sizes.empty() || sizes[0] % (int64_t)worldSize != 0) {
    runtimeFailure(
        "Tensor's dim 0 does not divide equally across group size");
  }
  int64_t chunk = sizes[0] / (int64_t)worldSize;
  std::vector<Tensor> out;
  out.reserve(worldSize);
  for (size_t i = 0; i < worldSize; ++i) {
    out.push_back(tensor.narrow(0, (int64_t)i * chunk, chunk));
  }
  return out;
}

} // namespace distributed
} // namespace tensorplay

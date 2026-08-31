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
#include <map>
#include <numeric>
#include <stdexcept>

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


#include <algorithm>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>

namespace tensorplay {
namespace distributed {

namespace py = pybind11;

namespace {

constexpr const char* kLoopbackAddress = "127.0.0.1";

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
  for (const auto& tensor : tensors) {
    if (tensor.dtype() != tensors[0].dtype() ||
        tensor.shape() != tensors[0].shape()) {
      invalidArgument(
          std::string(op) + ": all tensors must have the same type and sizes");
    }
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
    int64_t sum = std::accumulate(splitSizes.begin(), splitSizes.end(), 0ll);
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
  if (splitSizes.empty()) {
    const auto split = tensor.numel() / tensor.size(0);
    const auto length = tensor.numel() / split;
    lengths->resize(split);
    std::fill(lengths->begin(), lengths->end(), length);
  } else {
    lengths->resize(splitSizes.size());
    for (size_t i = 0; i < splitSizes.size(); ++i) {
      int64_t length = splitSizes[i];
      for (int64_t d = 1; d < tensor.dim(); ++d) {
        length *= tensor.size(d);
      }
      (*lengths)[i] = length;
    }
  }
  offsets->resize(lengths->size());
  int64_t offset = 0;
  for (size_t i = 0; i < lengths->size(); ++i) {
    (*offsets)[i] = offset;
    offset += (*lengths)[i];
  }
}

void computeLengthsAndOffsets(
    const std::vector<Tensor>& tensors,
    std::vector<int64_t>* lengths,
    std::vector<int64_t>* offsets) {
  lengths->resize(tensors.size());
  int64_t offset = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    (*lengths)[i] = tensors[i].numel();
    (*offsets)[i] = offset;
    offset += (*lengths)[i];
  }
}

// Elementwise reduce functions. Standard C++ arithmetic types delegate to the
// transport library's implementations; 16-bit float types are reduced through
// float promotion to avoid relying on their conversion operators.
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
void reduceInto(int op, void* c, const void* a, const void* b, size_t n) {
  switch (op) {
    case 0: // SUM
    case 4: // AVG
      ::gloo::sum<T>(c, a, b, n);
      break;
    case 1: // PRODUCT
      ::gloo::product<T>(c, a, b, n);
      break;
    case 2: // MAX
      ::gloo::max<T>(c, a, b, n);
      break;
    case 3: // MIN
      ::gloo::min<T>(c, a, b, n);
      break;
    default:
      runtimeFailure("Unhandled reduce op for the gloo backend");
  }
}

template <typename T>
void halfOp(int op, void* c, const void* a, const void* b, size_t n) {
  auto* tc = static_cast<T*>(c);
  auto* ta = static_cast<const T*>(a);
  auto* tb = static_cast<const T*>(b);
  for (size_t i = 0; i < n; ++i) {
    float x = static_cast<float>(ta[i]);
    float y = static_cast<float>(tb[i]);
    float r = x;
    switch (op) {
      case 0:
      case 4:
        r = x + y;
        break;
      case 1:
        r = x * y;
        break;
      case 2:
        r = x > y ? x : y;
        break;
      case 3:
        r = x < y ? x : y;
        break;
      default:
        runtimeFailure("Unhandled reduce op for the gloo backend");
    }
    tc[i] = static_cast<T>(r);
  }
}

template <typename T>
gloo::AllreduceOptions::Func makeReduceFunction(int op) {
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
void setReduceFn(O& opts, int op) {
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
  if (tensors[0].device().is_cuda()) {
    invalidArgument(
        std::string(op) +
        ": CUDA tensors are not supported by the gloo backend in this build");
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

int GlooRecvWork::source_rank() const {
  std::lock_guard<std::mutex> lock(waitMutex_);
  return srcRank_;
}

bool GlooRecvWork::wait(int64_t timeout_ms) {
  std::exception_ptr exception{nullptr};
  bool completed = false;
  try {
    if (timeout_ms < 0) {
      completed = buffer_->waitRecv(&srcRank_);
    } else {
      completed =
          buffer_->waitRecv(&srcRank_, std::chrono::milliseconds(timeout_ms));
    }
  } catch (...) {
    exception = std::current_exception();
  }
  if (exception != nullptr) {
    std::lock_guard<std::mutex> lock(waitMutex_);
    completed_ = true;
    exception_ = exception;
    std::rethrow_exception(exception_);
  }
  GlooWork::finish();
  return completed;
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
  if (rv < 0) {
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
  // An explicit interface selection wins, mirroring the ecosystem-wide
  // GLOO_SOCKET_IFNAME escape hatch.
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
  if (options_.devices.empty()) {
    runtimeFailure("ProcessGroupGloo: no device(s) specified");
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
    if (tensor.dtype() == ::tensorplay::ScalarType::ComplexFloat ||
        tensor.dtype() == ::tensorplay::ScalarType::ComplexDouble) {
      tensor = tensor.view_as_real().contiguous();
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
      int reduceOp,
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
  const int reduceOp;
  const uint32_t tag;

  void allreduceOne(std::vector<Tensor>& tensors) {
    Tensor tensor = tensors[0];
    if (tensor.dtype() == ::tensorplay::ScalarType::ComplexFloat ||
        tensor.dtype() == ::tensorplay::ScalarType::ComplexDouble) {
      tensor = tensor.view_as_real().contiguous();
    }
    gloo::AllreduceOptions opts(context_);
    const auto scalarType = tensor.dtype();
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setReduceFn, opts, reduceOp);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());
    TP_GLOO_GENERATE_ALL_TYPES(
        scalarType, setOutputs, opts, tensors, tensor.numel());
    gloo::allreduce(opts);
    if (reduceOp == 4) { // AVG = SUM / size
      tensors[0] /= (double)context_->size;
    }
  }

  void run() override {
    allreduceOne(inputs);
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
      int reduceOp,
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
  const int reduceOp;
  const uint32_t tag;

  void reduceOne(std::vector<Tensor>& tensors) {
    Tensor tensor = tensors[0];
    if (tensor.dtype() == ::tensorplay::ScalarType::ComplexFloat ||
        tensor.dtype() == ::tensorplay::ScalarType::ComplexDouble) {
      tensor = tensor.view_as_real().contiguous();
    }
    gloo::ReduceOptions opts(context_);
    const auto scalarType = tensor.dtype();
    opts.setRoot(rootRank);
    opts.setTag(tag);
    opts.setTimeout(getTimeout());
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setReduceFn, opts, reduceOp);
    TP_GLOO_GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);
    gloo::reduce(opts);
    if (reduceOp == 4) {
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

// Runs on a worker thread so the send buffer outlives a dropped handle: the
// wait happens where the buffer lives, never on the caller's thread. The
// tensor is held for the same reason -- its storage backs the raw pointer
// the transport reads from.
class AsyncSendWork : public GlooAsyncWork {
 public:
  AsyncSendWork(
      std::shared_ptr<gloo::Context> context,
      Tensor tensor,
      std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
      int dstRank,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : GlooAsyncWork(std::move(context), {}, "send", seq, timeout),
        tensor_(std::move(tensor)),
        buffer_(std::move(buffer)),
        dstRank(dstRank),
        tag(tag) {}

  Tensor tensor_;
  std::unique_ptr<::gloo::transport::UnboundBuffer> buffer_;
  const int dstRank;
  const uint32_t tag;

  void run() override {
    buffer_->send(dstRank, tag);
    buffer_->waitSend(getTimeout());
  }
};

} // namespace

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
    int reduceOp,
    std::chrono::milliseconds timeout) {
  assertNonEmptyDeviceCpu(inputs, "ProcessGroupGloo::allreduce");
  assertTypeAndSizesMatch(inputs, "ProcessGroupGloo::allreduce");

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work =
      std::make_shared<AsyncAllreduceWork>(
          std::move(context), inputs, reduceOp, tag, seq_, timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::allreduce_coalesced(
    std::vector<Tensor>& tensors,
    int reduceOp,
    std::chrono::milliseconds timeout) {
  assertNonEmptyDeviceCpu(tensors, "ProcessGroupGloo::allreduce_coalesced");
  for (const auto& t : tensors) {
    if (t.dtype() != tensors[0].dtype() || t.device() != tensors[0].device()) {
      invalidArgument(
          "ProcessGroupGloo::allreduce_coalesced: tensors must share type "
          "and device");
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
    int reduceOp,
    std::chrono::milliseconds timeout) {
  assertRootRank(rootRank, size_, "ProcessGroupGloo::reduce");
  assertRootTensor(rootTensor, (int64_t)inputs.size(), "ProcessGroupGloo::reduce");
  assertSingleElement(inputs, "ProcessGroupGloo::reduce");
  assertNonEmptyDeviceCpu(inputs, "ProcessGroupGloo::reduce");

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
  for (const auto i : outputs) {
    if ((int64_t)i.size() != (int64_t)inputs.size() * size_) {
      invalidArgument(
          "ProcessGroupGloo::allgather: invalid output tensor list length");
    }
  }
  assertNonEmptyDeviceCpu(inputs, "ProcessGroupGloo::allgather");
  assertTypeAndSizesMatch(inputs, "ProcessGroupGloo::allgather");

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work = std::make_shared<AsyncAllgatherWork>(
      std::move(context), outputs, inputs, tag, seq_, timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::all_gather_into_tensor(
    Tensor& output,
    Tensor& input,
    std::chrono::milliseconds timeout) {
  auto tensor_list = splitEven(output);
  std::vector<std::vector<Tensor>> outputs = {tensor_list};
  std::vector<Tensor> inputs = {input};
  return allgather(outputs, inputs, timeout);
}

std::shared_ptr<GlooWork> ProcessGroupGloo::gather(
    std::vector<std::vector<Tensor>>& outputs,
    std::vector<Tensor>& inputs,
    int rootRank,
    std::chrono::milliseconds timeout) {
  assertRootRank(rootRank, size_, "ProcessGroupGloo::gather");
  assertSingleElement(inputs, "ProcessGroupGloo::gather");
  assertNonEmptyDeviceCpu(inputs, "ProcessGroupGloo::gather");
  if (getRank() == rootRank) {
    if ((int64_t)outputs.size() != 1 ||
        (int64_t)outputs[0].size() != size_) {
      invalidArgument(
          "ProcessGroupGloo::gather: root expects one tensor per rank");
    }
  } else if (!outputs.empty()) {
    invalidArgument(
        "ProcessGroupGloo::gather: non-root output list must be empty");
  }

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work = std::make_shared<AsyncGatherWork>(
      std::move(context), outputs, inputs, rootRank, tag, seq_, timeout);
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
  if (getRank() == rootRank) {
    if (inputs.size() != 1 || (int64_t)inputs[0].size() != size_) {
      invalidArgument(
          "ProcessGroupGloo::scatter: root expects one input tensor per rank");
    }
    assertTypeAndSizesMatch(inputs[0], "ProcessGroupGloo::scatter");
  } else if (!inputs.empty()) {
    invalidArgument(
        "ProcessGroupGloo::scatter: non-root input list must be empty");
  }

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work = std::make_shared<AsyncScatterWork>(
      std::move(context), outputs, inputs, rootRank, tag, seq_, timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::reduce_scatter(
    std::vector<Tensor>& outputs,
    std::vector<std::vector<Tensor>>& inputs,
    int reduceOp,
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
  // The lambda-based completion keeps the trailing wait off the queue: the
  // allreduces above were already enqueued in order.
  class ReduceScatterFinish : public GlooWork {
   public:
    ReduceScatterFinish(
        std::vector<Tensor> outputs,
        std::vector<std::shared_ptr<GlooWork>> works,
        int rank,
        int worldSize)
        : GlooWork("reduce_scatter"),
          outputs_(std::move(outputs)),
          works_(std::move(works)),
          rank_(rank),
          worldSize_(worldSize) {
      outputTensors_ = {outputs_};
    }

    bool wait(int64_t timeout_ms) override {
      for (auto& work : works_) {
        work->wait(timeout_ms);
      }
      // The output slot aliases this rank's own buffer, which the enqueued
      // allreduce already reduced in place.
      GlooWork::finish();
      return true;
    }

   protected:
    std::vector<Tensor> outputs_;
    std::vector<std::shared_ptr<GlooWork>> works_;
    int rank_;
    int worldSize_;
  };
  return std::make_shared<ReduceScatterFinish>(
      outputs, std::move(works), rank, worldSize);
}

std::shared_ptr<GlooWork> ProcessGroupGloo::reduce_scatter_tensor(
    Tensor& output,
    Tensor& input,
    int reduceOp,
    std::chrono::milliseconds timeout) {
  const auto worldSize = getSize();
  if (output.dim() == 0 || input.dim() == 0 ||
      output.size(0) * worldSize != input.size(0)) {
    invalidArgument(
        "ProcessGroupGloo::reduce_scatter_tensor: dim 0 of input must equal "
        "output dim 0 times world size");
  }
  Tensor inputClone = input.clone();
  std::vector<Tensor> inp = {inputClone};
  auto work = allreduce(inp, reduceOp, timeout);
  class ReduceScatterTensorFinish : public GlooWork {
   public:
    ReduceScatterTensorFinish(
        Tensor output,
        Tensor buffer,
        std::shared_ptr<GlooWork> work,
        int rank,
        int worldSize)
        : GlooWork("reduce_scatter_tensor"),
          output_(std::move(output)),
          buffer_(std::move(buffer)),
          work_(std::move(work)),
          rank_(rank),
          worldSize_(worldSize) {
      outputTensors_ = {{output_}};
    }

    bool wait(int64_t timeout_ms) override {
      work_->wait(timeout_ms);
      int64_t chunk = buffer_.numel() / worldSize_;
      output_.copy_(
          buffer_.narrow(0, (int64_t)rank_ * chunk, chunk).reshape(
              output_.shape()));
      GlooWork::finish();
      return true;
    }

   protected:
    Tensor output_;
    Tensor buffer_;
    std::shared_ptr<GlooWork> work_;
    int rank_;
    int worldSize_;
  };
  return std::make_shared<ReduceScatterTensorFinish>(
      output, inputClone, std::move(work), getRank(), worldSize);
}

std::shared_ptr<GlooWork> ProcessGroupGloo::all_to_all_single(
    Tensor& outputTensor,
    Tensor& inputTensor,
    std::vector<int64_t> outputCounts,
    std::vector<int64_t> inputCounts,
    std::chrono::milliseconds timeout) {
  if (outputTensor.device() != inputTensor.device()) {
    invalidArgument(
        "ProcessGroupGloo::all_to_all_single: output and input must be on "
        "the same device");
  }
  assertNonEmptyDeviceCpu({outputTensor}, "ProcessGroupGloo::all_to_all_single");
  checkSplitSizes(inputCounts, inputTensor, getSize());
  checkSplitSizes(outputCounts, outputTensor, getSize());

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work = std::make_shared<AsyncAlltoallWork>(
      std::move(context),
      outputTensor,
      inputTensor,
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
  assertTypeAndSizesMatch(inputTensors, "ProcessGroupGloo::alltoall");
  assertTypeAndSizesMatch(outputTensors, "ProcessGroupGloo::alltoall");

  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  std::shared_ptr<GlooAsyncWork> work =
      std::make_shared<AsyncAlltoallListWork>(
          std::move(context), outputTensors, inputTensors, tag, seq_, timeout);
  enqueue(work);
  return work;
}

namespace {

Tensor& checkSingleTensor(std::vector<Tensor>& tensors, const char* op) {
  if (tensors.size() != 1) {
    invalidArgument(std::string(op) + " takes a single tensor");
  }
  auto& tensor = tensors[0];
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
  auto& tensor = checkSingleTensor(tensors, "ProcessGroupGloo::send");
  auto utag = checkTag(tag, "ProcessGroupGloo::send");
  void* ptr = tensor.data_ptr();
  auto size = tensor.numel() * (int64_t)tensor.itemsize();

  auto context = getContext(utag);
  auto buf = context->createUnboundBuffer(ptr, size);
  ++seq_;
  auto work = std::make_shared<AsyncSendWork>(
      std::move(context),
      tensor,
      std::move(buf),
      dstRank,
      utag,
      seq_,
      options_.timeout);
  enqueue(work);
  return work;
}

std::shared_ptr<GlooWork> ProcessGroupGloo::recv(
    std::vector<Tensor>& tensors,
    int srcRank,
    int tag) {
  auto& tensor = checkSingleTensor(tensors, "ProcessGroupGloo::recv");
  auto utag = checkTag(tag, "ProcessGroupGloo::recv");
  void* ptr = tensor.data_ptr();
  auto size = tensor.numel() * (int64_t)tensor.itemsize();

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
  auto size = tensor.numel() * (int64_t)tensor.itemsize();

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
        entry.second->wait(timeout.count());
        rankResponded = true;
      } catch (const std::exception& e) {
        const std::string error =
            "[Rank 0]: Rank " + std::to_string(entry.first) +
            " failed to pass monitoredBarrier in " +
            std::to_string(timeout.count()) + " ms: " + e.what();
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
          std::to_string(timeout.count()) + " ms");
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
    out.push_back(
        tensor.narrow(0, (int64_t)i * chunk, chunk).contiguous());
  }
  return out;
}

} // namespace distributed
} // namespace tensorplay

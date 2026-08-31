#include "tensorpipe_backend.h"

#include "tensorpipe_utils.h"

#include <tensorpipe/tensorpipe.h>

#ifdef USE_CUDA
#include <tensorpipe/tensorpipe_cuda.h>
#endif

#include <stdexcept>
#include <utility>

namespace tensorplay::distributed::rpc {
namespace {

std::unique_ptr<TransportRegistration> make_uv_transport() {
    auto context = tensorpipe::transport::uv::create();
    return std::make_unique<TransportRegistration>(TransportRegistration{
        std::move(context), kUvTransportPriority, guess_address()});
}

#if TENSORPIPE_HAS_SHM_TRANSPORT
std::unique_ptr<TransportRegistration> make_shm_transport() {
    auto context = tensorpipe::transport::shm::create();
    return std::make_unique<TransportRegistration>(TransportRegistration{
        std::move(context), kShmTransportPriority, {}});
}
#endif

#if TENSORPIPE_HAS_IBV_TRANSPORT
std::unique_ptr<TransportRegistration> make_ibv_transport() {
    auto context = tensorpipe::transport::ibv::create();
    return std::make_unique<TransportRegistration>(TransportRegistration{
        std::move(context), kIbvTransportPriority, guess_address()});
}
#endif

std::unique_ptr<ChannelRegistration> make_basic_channel() {
    auto context = tensorpipe::channel::basic::create();
    return std::make_unique<ChannelRegistration>(ChannelRegistration{
        std::move(context), kBasicChannelPriority});
}

#if TENSORPIPE_HAS_CMA_CHANNEL
std::unique_ptr<ChannelRegistration> make_cma_channel() {
    auto context = tensorpipe::channel::cma::create();
    return std::make_unique<ChannelRegistration>(ChannelRegistration{
        std::move(context), kCmaChannelPriority});
}
#endif

constexpr size_t kNumUvThreads = 16;

std::unique_ptr<ChannelRegistration> make_multiplexed_uv_channel() {
    std::vector<std::shared_ptr<tensorpipe::transport::Context>> contexts;
    contexts.reserve(kNumUvThreads);
    std::vector<std::shared_ptr<tensorpipe::transport::Listener>> listeners;
    listeners.reserve(kNumUvThreads);
    const std::string address = guess_address();
    for (size_t lane = 0; lane < kNumUvThreads; ++lane) {
        auto context = tensorpipe::transport::uv::create();
        if (!context || !context->isViable()) {
            return std::make_unique<ChannelRegistration>(
                ChannelRegistration{});
        }
        contexts.push_back(std::move(context));
        listeners.push_back(contexts.back()->listen(address));
        if (!listeners.back()) {
            return std::make_unique<ChannelRegistration>(
                ChannelRegistration{});
        }
    }
    auto context = tensorpipe::channel::mpt::create(
        std::move(contexts), std::move(listeners));
    return std::make_unique<ChannelRegistration>(ChannelRegistration{
        std::move(context), kMultiplexedUvChannelPriority});
}

#ifdef USE_CUDA
std::unique_ptr<ChannelRegistration> make_cuda_basic_channel() {
    auto context = tensorpipe::channel::cuda_basic::create(
        tensorpipe::channel::basic::create());
    return std::make_unique<ChannelRegistration>(ChannelRegistration{
        std::move(context), kCudaBasicChannelPriority});
}

std::unique_ptr<ChannelRegistration> make_cuda_xth_channel() {
    auto context = tensorpipe::channel::cuda_xth::create();
    return std::make_unique<ChannelRegistration>(ChannelRegistration{
        std::move(context), kCudaXthChannelPriority});
}

#if TENSORPIPE_HAS_CUDA_IPC_CHANNEL
std::unique_ptr<ChannelRegistration> make_cuda_ipc_channel() {
    auto context = tensorpipe::channel::cuda_ipc::create();
    return std::make_unique<ChannelRegistration>(ChannelRegistration{
        std::move(context), kCudaIpcChannelPriority});
}
#endif

#if TENSORPIPE_HAS_CUDA_GDR_CHANNEL
std::unique_ptr<ChannelRegistration> make_cuda_gdr_channel() {
    auto context = tensorpipe::channel::cuda_gdr::create();
    return std::make_unique<ChannelRegistration>(ChannelRegistration{
        std::move(context), kCudaGdrChannelPriority});
}
#endif
#endif

struct BuiltinBackendRegistrations final {
    BuiltinBackendRegistrations() {
        TensorPipeTransportRegistry::instance().register_creator(
            "uv", make_uv_transport);
#if TENSORPIPE_HAS_SHM_TRANSPORT
        TensorPipeTransportRegistry::instance().register_creator(
            "shm", make_shm_transport);
#endif
#if TENSORPIPE_HAS_IBV_TRANSPORT
        TensorPipeTransportRegistry::instance().register_creator(
            "ibv", make_ibv_transport);
#endif
        TensorPipeChannelRegistry::instance().register_creator(
            "basic", make_basic_channel);
#if TENSORPIPE_HAS_CMA_CHANNEL
        TensorPipeChannelRegistry::instance().register_creator(
            "cma", make_cma_channel);
#endif
        TensorPipeChannelRegistry::instance().register_creator(
            "mpt_uv", make_multiplexed_uv_channel);
#ifdef USE_CUDA
        TensorPipeChannelRegistry::instance().register_creator(
            "cuda_basic", make_cuda_basic_channel);
        TensorPipeChannelRegistry::instance().register_creator(
            "cuda_xth", make_cuda_xth_channel);
#if TENSORPIPE_HAS_CUDA_IPC_CHANNEL
        TensorPipeChannelRegistry::instance().register_creator(
            "cuda_ipc", make_cuda_ipc_channel);
#endif
#if TENSORPIPE_HAS_CUDA_GDR_CHANNEL
        TensorPipeChannelRegistry::instance().register_creator(
            "cuda_gdr", make_cuda_gdr_channel);
#endif
#endif
    }
};

const BuiltinBackendRegistrations builtin_backend_registrations;

}  // namespace

TensorPipeTransportRegistry& TensorPipeTransportRegistry::instance() {
    static TensorPipeTransportRegistry registry;
    return registry;
}

void TensorPipeTransportRegistry::register_creator(
    std::string name,
    Creator creator) {
    if (name.empty() || !creator) {
        throw std::invalid_argument("backend registration is invalid");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto [iterator, inserted] =
        creators_.emplace(std::move(name), std::move(creator));
    if (!inserted) {
        throw std::invalid_argument(
            "backend registration already exists: " + iterator->first);
    }
}

bool TensorPipeTransportRegistry::has(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return creators_.find(name) != creators_.end();
}

std::vector<std::string> TensorPipeTransportRegistry::keys() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    result.reserve(creators_.size());
    for (const auto& entry : creators_) {
        result.push_back(entry.first);
    }
    return result;
}

std::unique_ptr<TransportRegistration> TensorPipeTransportRegistry::create(
    const std::string& name) const {
    Creator creator;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto iterator = creators_.find(name);
        if (iterator == creators_.end()) {
            throw std::invalid_argument(
                "backend registration does not exist: " + name);
        }
        creator = iterator->second;
    }
    return creator();
}

TensorPipeChannelRegistry& TensorPipeChannelRegistry::instance() {
    static TensorPipeChannelRegistry registry;
    return registry;
}

void TensorPipeChannelRegistry::register_creator(
    std::string name,
    Creator creator) {
    if (name.empty() || !creator) {
        throw std::invalid_argument("backend registration is invalid");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto [iterator, inserted] =
        creators_.emplace(std::move(name), std::move(creator));
    if (!inserted) {
        throw std::invalid_argument(
            "backend registration already exists: " + iterator->first);
    }
}

bool TensorPipeChannelRegistry::has(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return creators_.find(name) != creators_.end();
}

std::vector<std::string> TensorPipeChannelRegistry::keys() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> result;
    result.reserve(creators_.size());
    for (const auto& entry : creators_) {
        result.push_back(entry.first);
    }
    return result;
}

std::unique_ptr<ChannelRegistration> TensorPipeChannelRegistry::create(
    const std::string& name) const {
    Creator creator;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto iterator = creators_.find(name);
        if (iterator == creators_.end()) {
            throw std::invalid_argument(
                "backend registration does not exist: " + name);
        }
        creator = iterator->second;
    }
    return creator();
}

}  // namespace tensorplay::distributed::rpc

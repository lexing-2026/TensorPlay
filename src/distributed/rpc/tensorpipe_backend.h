#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tensorpipe {
namespace transport {
class Context;
}
namespace channel {
class Context;
}
}

namespace tensorplay::distributed::rpc {

constexpr int64_t kShmTransportPriority = 200;
constexpr int64_t kIbvTransportPriority = 100;
constexpr int64_t kUvTransportPriority = 0;

constexpr int64_t kCmaChannelPriority = 1200;
constexpr int64_t kMultiplexedUvChannelPriority = 1100;
constexpr int64_t kBasicChannelPriority = 1000;
constexpr int64_t kCudaIpcChannelPriority = 300;
constexpr int64_t kCudaGdrChannelPriority = 200;
constexpr int64_t kCudaXthChannelPriority = 400;
constexpr int64_t kCudaBasicChannelPriority = 0;

struct TransportRegistration final {
    std::shared_ptr<tensorpipe::transport::Context> transport;
    int64_t priority = 0;
    std::string address;
};

struct ChannelRegistration final {
    std::shared_ptr<tensorpipe::channel::Context> channel;
    int64_t priority = 0;
};

class TensorPipeTransportRegistry final {
public:
    using Creator = std::function<std::unique_ptr<TransportRegistration>()>;

    static TensorPipeTransportRegistry& instance();

    void register_creator(std::string name, Creator creator);
    bool has(const std::string& name) const;
    std::vector<std::string> keys() const;
    std::unique_ptr<TransportRegistration> create(
        const std::string& name) const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, Creator> creators_;
};

class TensorPipeChannelRegistry final {
public:
    using Creator = std::function<std::unique_ptr<ChannelRegistration>()>;

    static TensorPipeChannelRegistry& instance();

    void register_creator(std::string name, Creator creator);
    bool has(const std::string& name) const;
    std::vector<std::string> keys() const;
    std::unique_ptr<ChannelRegistration> create(
        const std::string& name) const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, Creator> creators_;
};

}  // namespace tensorplay::distributed::rpc

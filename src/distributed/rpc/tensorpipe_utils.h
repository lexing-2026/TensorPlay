#pragma once

#include "message.h"
#include "Storage.h"

#include <tensorpipe/tensorpipe.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorplay::distributed::rpc {

struct Endpoint final {
    std::string host;
    uint16_t port = 0;
};

Endpoint parse_endpoint(const std::string& value, uint16_t default_port = 29500);
std::string guess_address();

using DeviceMap = std::unordered_map<std::string, std::string>;

struct TensorPipeWriteState final {
    tensorpipe::Message message;
    std::unique_ptr<MessageType> type;
    std::unique_ptr<int64_t> id;
    std::string payload;
    std::string tensor_metadata;
    std::vector<tensorplay::Storage> storages;
};

struct TensorPipeReadState final {
    std::unique_ptr<MessageType> type;
    std::unique_ptr<int64_t> id;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> tensor_metadata;
    std::vector<tensorplay::Storage> storages;
};

struct TensorPipeReadAllocation final {
    tensorpipe::Allocation allocation;
    std::shared_ptr<TensorPipeReadState> state;
};

std::shared_ptr<TensorPipeWriteState> make_tensorpipe_message(
    const Message& message,
    const DeviceMap& device_map);
TensorPipeReadAllocation allocate_tensorpipe_message(
    const tensorpipe::Descriptor& descriptor);
MessagePtr decode_tensorpipe_message(
    const tensorpipe::Descriptor& descriptor,
    const TensorPipeReadState& state);

}  // namespace tensorplay::distributed::rpc

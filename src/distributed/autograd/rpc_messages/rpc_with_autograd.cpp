#include "rpc_with_autograd.h"

#include "rpc/python_functions.h"

#include <pybind11/stl.h>

#include <stdexcept>
#include <utility>

namespace tensorplay::distributed::autograd {

namespace {

pybind11::dict to_python_device_map(const rpc::DeviceMap& device_map) {
    pybind11::dict result;
    for (const auto& entry : device_map) {
        result[pybind11::str(entry.first)] = pybind11::str(entry.second);
    }
    return result;
}

rpc::DeviceMap from_python_device_map(const pybind11::handle& value) {
    const auto mapping = value.cast<pybind11::dict>();
    rpc::DeviceMap result;
    for (const auto& item : mapping) {
        result.emplace(
            item.first.cast<std::string>(), item.second.cast<std::string>());
    }
    return result;
}

}  // namespace

RpcWithAutograd::RpcWithAutograd(
    rpc::worker_id_t from_worker,
    rpc::MessageType message_type,
    AutogradMetadata metadata,
    rpc::MessagePtr wrapped_message,
    rpc::DeviceMap device_map)
    : from_worker_(from_worker),
      message_type_(message_type),
      metadata_(metadata),
      wrapped_message_(std::move(wrapped_message)),
      device_map_(std::move(device_map)) {
    if (!wrapped_message_ || !metadata_.valid() ||
        (message_type_ != rpc::MessageType::FORWARD_AUTOGRAD_REQ &&
         message_type_ != rpc::MessageType::FORWARD_AUTOGRAD_RESP)) {
        throw std::invalid_argument("invalid distributed autograd RPC wrapper");
    }
}

RpcWithAutograd RpcWithAutograd::from_message(const rpc::Message& message) {
    if (message.type() != rpc::MessageType::FORWARD_AUTOGRAD_REQ &&
        message.type() != rpc::MessageType::FORWARD_AUTOGRAD_RESP) {
        throw std::invalid_argument(
            "message is not a distributed autograd forward message");
    }
    pybind11::gil_scoped_acquire gil;
    const auto envelope = rpc::deserialize_python_object(rpc::SerializedPyObj(
        std::string(message.payload().begin(), message.payload().end()), {}));
    const auto values = envelope.cast<pybind11::tuple>();
    if (values.size() != 6) {
        throw std::runtime_error(
            "distributed autograd RPC envelope is malformed");
    }
    const auto wrapped_type = static_cast<rpc::MessageType>(
        values[0].cast<uint16_t>());
    AutogradMetadata metadata{
        values[1].cast<int64_t>(), values[2].cast<int64_t>()};
    if (!metadata.valid()) {
        throw std::runtime_error(
            "distributed autograd RPC envelope has invalid metadata");
    }
    const auto from_worker = values[3].cast<rpc::worker_id_t>();
    const auto device_map = from_python_device_map(values[4]);
    const auto wrapped_payload = values[5].cast<std::string>();
    auto wrapped = std::make_shared<rpc::Message>(
        std::vector<uint8_t>(wrapped_payload.begin(), wrapped_payload.end()),
        std::vector<pybind11::object>(
            message.tensors().begin(), message.tensors().end()),
        wrapped_type,
        message.id());
    return RpcWithAutograd(
        from_worker,
        message.type(),
        metadata,
        std::move(wrapped),
        device_map);
}

rpc::MessagePtr RpcWithAutograd::to_message() && {
    if (!wrapped_message_) {
        throw std::runtime_error(
            "distributed autograd RPC wrapper has no wrapped message");
    }
    pybind11::gil_scoped_acquire gil;
    const auto envelope = rpc::serialize_python_object(pybind11::make_tuple(
        static_cast<uint16_t>(wrapped_message_->type()),
        metadata_.context_id,
        metadata_.message_id,
        from_worker_,
        to_python_device_map(device_map_),
        pybind11::bytes(
            reinterpret_cast<const char*>(wrapped_message_->payload().data()),
            wrapped_message_->payload().size())));
    if (!envelope.tensors_.empty()) {
        throw std::runtime_error(
            "distributed autograd RPC envelope contains nested tensors");
    }
    const auto wrapped_id = wrapped_message_->id();
    auto tensors = std::move(*wrapped_message_).move_tensors();
    return std::make_shared<rpc::Message>(
        std::vector<uint8_t>(envelope.payload_.begin(), envelope.payload_.end()),
        std::move(tensors),
        message_type_,
        wrapped_id);
}

rpc::MessageType RpcWithAutograd::message_type() const noexcept {
    return message_type_;
}

rpc::MessageType RpcWithAutograd::wrapped_message_type() const noexcept {
    return wrapped_message_ ? wrapped_message_->type() : rpc::MessageType::UNKNOWN;
}

rpc::worker_id_t RpcWithAutograd::from_worker() const noexcept {
    return from_worker_;
}

const AutogradMetadata& RpcWithAutograd::metadata() const noexcept {
    return metadata_;
}

const rpc::DeviceMap& RpcWithAutograd::device_map() const noexcept {
    return device_map_;
}

const rpc::MessagePtr& RpcWithAutograd::wrapped_message() const noexcept {
    return wrapped_message_;
}

}  // namespace tensorplay::distributed::autograd

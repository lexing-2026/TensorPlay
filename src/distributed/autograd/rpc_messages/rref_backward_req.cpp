#include "rref_backward_req.h"

#include "rpc/python_functions.h"

#include <pybind11/stl.h>

#include <stdexcept>
#include <utility>

namespace tensorplay::distributed::autograd {

RRefBackwardReq::RRefBackwardReq(
    rpc::RRefId rref_id,
    int64_t context_id,
    bool retain_graph)
    : rref_id_(std::move(rref_id)),
      context_id_(context_id),
      retain_graph_(retain_graph) {}

rpc::MessagePtr RRefBackwardReq::to_message() const {
    pybind11::gil_scoped_acquire gil;
    const auto object = rpc::serialize_python_object(pybind11::make_tuple(
        rref_id_.to_python(), context_id_, retain_graph_));
    return std::make_shared<rpc::Message>(
        std::vector<uint8_t>(object.payload_.begin(), object.payload_.end()),
        std::move(object.tensors_),
        rpc::MessageType::RREF_BACKWARD_REQ);
}

RRefBackwardReq RRefBackwardReq::from_message(const rpc::Message& message) {
    if (message.type() != rpc::MessageType::RREF_BACKWARD_REQ) {
        throw std::runtime_error("RRef backward request type is invalid");
    }
    pybind11::gil_scoped_acquire gil;
    const auto object = rpc::deserialize_python_object(rpc::SerializedPyObj(
        std::string(message.payload().begin(), message.payload().end()),
        std::vector<pybind11::object>(
            message.tensors().begin(), message.tensors().end())));
    const auto values = object.cast<pybind11::tuple>();
    if (values.size() != 3) {
        throw std::runtime_error("RRef backward request is malformed");
    }
    return RRefBackwardReq(
        rpc::GloballyUniqueId::from_python(values[0]),
        values[1].cast<int64_t>(),
        values[2].cast<bool>());
}

const rpc::RRefId& RRefBackwardReq::rref_id() const noexcept {
    return rref_id_;
}

int64_t RRefBackwardReq::context_id() const noexcept {
    return context_id_;
}

bool RRefBackwardReq::retain_graph() const noexcept {
    return retain_graph_;
}

}  // namespace tensorplay::distributed::autograd

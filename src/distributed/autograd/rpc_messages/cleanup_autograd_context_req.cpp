#include "cleanup_autograd_context_req.h"

#include "rpc/python_functions.h"

#include <stdexcept>

namespace tensorplay::distributed::autograd {

CleanupAutogradContextReq::CleanupAutogradContextReq(int64_t context_id)
    : context_id_(context_id) {}

rpc::MessagePtr CleanupAutogradContextReq::to_message() const {
    pybind11::gil_scoped_acquire gil;
    auto object = rpc::serialize_python_object(pybind11::int_(context_id_));
    return std::make_shared<rpc::Message>(
        std::vector<uint8_t>(object.payload_.begin(), object.payload_.end()),
        std::move(object.tensors_),
        rpc::MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ);
}

CleanupAutogradContextReq CleanupAutogradContextReq::from_message(
    const rpc::Message& message) {
    if (message.type() != rpc::MessageType::CLEANUP_AUTOGRAD_CONTEXT_REQ) {
        throw std::runtime_error(
            "distributed autograd cleanup request type is invalid");
    }
    pybind11::gil_scoped_acquire gil;
    auto object = rpc::deserialize_python_object(rpc::SerializedPyObj(
        std::string(message.payload().begin(), message.payload().end()),
        std::vector<pybind11::object>(
            message.tensors().begin(), message.tensors().end())));
    return CleanupAutogradContextReq(object.cast<int64_t>());
}

int64_t CleanupAutogradContextReq::context_id() const noexcept {
    return context_id_;
}

}  // namespace tensorplay::distributed::autograd

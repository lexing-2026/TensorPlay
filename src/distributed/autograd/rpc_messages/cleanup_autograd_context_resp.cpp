#include "cleanup_autograd_context_resp.h"

#include <pybind11/pybind11.h>

#include <stdexcept>

namespace tensorplay::distributed::autograd {

rpc::MessagePtr CleanupAutogradContextResp::to_message() const {
    return std::make_shared<rpc::Message>(
        std::vector<uint8_t>(),
        std::vector<pybind11::object>(),
        rpc::MessageType::CLEANUP_AUTOGRAD_CONTEXT_RESP);
}

CleanupAutogradContextResp CleanupAutogradContextResp::from_message(
    const rpc::Message& message) {
    if (message.type() != rpc::MessageType::CLEANUP_AUTOGRAD_CONTEXT_RESP) {
        throw std::runtime_error(
            "distributed autograd cleanup response type is invalid");
    }
    return {};
}

}  // namespace tensorplay::distributed::autograd

#include "rref_backward_resp.h"

#include <pybind11/pybind11.h>

#include <stdexcept>

namespace tensorplay::distributed::autograd {

rpc::MessagePtr RRefBackwardResp::to_message() const {
    return std::make_shared<rpc::Message>(
        std::vector<uint8_t>(),
        std::vector<pybind11::object>(),
        rpc::MessageType::RREF_BACKWARD_RESP);
}

RRefBackwardResp RRefBackwardResp::from_message(const rpc::Message& message) {
    if (message.type() != rpc::MessageType::RREF_BACKWARD_RESP) {
        throw std::runtime_error("RRef backward response type is invalid");
    }
    return {};
}

}  // namespace tensorplay::distributed::autograd

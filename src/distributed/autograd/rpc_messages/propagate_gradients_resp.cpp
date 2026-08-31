#include "propagate_gradients_resp.h"

#include <pybind11/pybind11.h>

#include <stdexcept>

namespace tensorplay::distributed::autograd {

rpc::MessagePtr PropagateGradientsResp::to_message() const {
    return std::make_shared<rpc::Message>(
        std::vector<uint8_t>(),
        std::vector<pybind11::object>(),
        rpc::MessageType::BACKWARD_AUTOGRAD_RESP);
}

PropagateGradientsResp PropagateGradientsResp::from_message(
    const rpc::Message& message) {
    if (message.type() != rpc::MessageType::BACKWARD_AUTOGRAD_RESP) {
        throw std::runtime_error(
            "distributed autograd backward response type is invalid");
    }
    return {};
}

}  // namespace tensorplay::distributed::autograd

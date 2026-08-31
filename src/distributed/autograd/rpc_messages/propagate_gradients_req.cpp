#include "propagate_gradients_req.h"

#include "rpc/python_functions.h"

#include <pybind11/stl.h>

#include <stdexcept>

namespace tensorplay::distributed::autograd {

PropagateGradientsReq::PropagateGradientsReq(
    AutogradMetadata metadata,
    std::vector<tensorplay::Tensor> gradients,
    bool retain_graph)
    : metadata_(metadata),
      gradients_(std::move(gradients)),
      retain_graph_(retain_graph) {}

rpc::MessagePtr PropagateGradientsReq::to_message() const {
    pybind11::gil_scoped_acquire gil;
    pybind11::list gradients;
    for (const auto& gradient : gradients_) {
        gradients.append(gradient);
    }
    auto object = rpc::serialize_python_object(pybind11::make_tuple(
        gradients,
        metadata_.context_id,
        metadata_.message_id,
        retain_graph_));
    return std::make_shared<rpc::Message>(
        std::vector<uint8_t>(object.payload_.begin(), object.payload_.end()),
        std::move(object.tensors_),
        rpc::MessageType::BACKWARD_AUTOGRAD_REQ);
}

PropagateGradientsReq PropagateGradientsReq::from_message(
    const rpc::Message& message) {
    if (message.type() != rpc::MessageType::BACKWARD_AUTOGRAD_REQ) {
        throw std::runtime_error(
            "distributed autograd backward request type is invalid");
    }
    pybind11::gil_scoped_acquire gil;
    auto object = rpc::deserialize_python_object(rpc::SerializedPyObj(
        std::string(message.payload().begin(), message.payload().end()),
        std::vector<pybind11::object>(
            message.tensors().begin(), message.tensors().end())));
    const auto values = object.cast<pybind11::tuple>();
    if (values.size() != 4) {
        throw std::runtime_error(
            "distributed autograd backward request is malformed");
    }
    return PropagateGradientsReq(
        AutogradMetadata{
            values[1].cast<int64_t>(),
            values[2].cast<int64_t>()},
        values[0].cast<std::vector<tensorplay::Tensor>>(),
        values[3].cast<bool>());
}

const AutogradMetadata& PropagateGradientsReq::metadata() const noexcept {
    return metadata_;
}

const std::vector<tensorplay::Tensor>&
PropagateGradientsReq::gradients() const noexcept {
    return gradients_;
}

bool PropagateGradientsReq::retain_graph() const noexcept {
    return retain_graph_;
}

}  // namespace tensorplay::distributed::autograd

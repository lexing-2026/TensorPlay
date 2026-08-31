#pragma once

#include "context/container.h"
#include "functions/recvrpc_backward.h"
#include "functions/sendrpc_backward.h"
#include "rpc/tensorpipe_utils.h"
#include "rpc_messages/autograd_metadata.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace tensorplay::distributed::autograd {

void add_send_rpc_backward(
    const ContextPtr& context,
    const AutogradMetadata& metadata,
    std::vector<pybind11::object>& tensors);

ContextPtr add_recv_rpc_backward(
    const AutogradMetadata& metadata,
    std::vector<pybind11::object>& tensors,
    rpc::worker_id_t from_worker,
    rpc::DeviceMap device_map);

}  // namespace tensorplay::distributed::autograd

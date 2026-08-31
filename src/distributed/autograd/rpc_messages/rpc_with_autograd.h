#pragma once

#include "autograd_metadata.h"
#include "rpc/message.h"
#include "rpc/tensorpipe_utils.h"

#include <cstdint>
#include <memory>

namespace tensorplay::distributed::autograd {

class RpcWithAutograd final {
public:
    RpcWithAutograd(
        rpc::worker_id_t from_worker,
        rpc::MessageType message_type,
        AutogradMetadata metadata,
        rpc::MessagePtr wrapped_message,
        rpc::DeviceMap device_map = {});

    static RpcWithAutograd from_message(const rpc::Message& message);

    rpc::MessagePtr to_message() &&;

    rpc::MessageType message_type() const noexcept;
    rpc::MessageType wrapped_message_type() const noexcept;
    rpc::worker_id_t from_worker() const noexcept;
    const AutogradMetadata& metadata() const noexcept;
    const rpc::DeviceMap& device_map() const noexcept;
    const rpc::MessagePtr& wrapped_message() const noexcept;

private:
    rpc::worker_id_t from_worker_ = 0;
    rpc::MessageType message_type_ = rpc::MessageType::UNKNOWN;
    AutogradMetadata metadata_;
    rpc::MessagePtr wrapped_message_;
    rpc::DeviceMap device_map_;
};

}  // namespace tensorplay::distributed::autograd

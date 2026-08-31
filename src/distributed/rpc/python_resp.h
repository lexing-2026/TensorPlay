#pragma once

#include "rpc_command_base.h"
#include "types.h"

namespace tensorplay::distributed::rpc {

class PythonResp final : public RpcCommandBase {
public:
    explicit PythonResp(SerializedPyObj object);
    MessagePtr to_message_impl() && override;
    static std::unique_ptr<PythonResp> from_message(const Message& message);
    const SerializedPyObj& serialized_object() const noexcept;

private:
    SerializedPyObj object_;
};

}  // namespace tensorplay::distributed::rpc

#pragma once

#include "rpc_command_base.h"
#include "types.h"

namespace tensorplay::distributed::rpc {

class PythonCall final : public RpcCommandBase {
public:
    PythonCall(SerializedPyObj object, bool async_execution);
    MessagePtr to_message_impl() && override;
    static std::unique_ptr<PythonCall> from_message(const Message& message);
    const SerializedPyObj& serialized_object() const noexcept;
    bool is_async_execution() const noexcept;

private:
    SerializedPyObj object_;
    bool async_execution_ = false;
};

}  // namespace tensorplay::distributed::rpc

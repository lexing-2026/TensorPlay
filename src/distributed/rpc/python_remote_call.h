#pragma once

#include "python_call.h"
#include "rref_impl.h"

namespace tensorplay::distributed::rpc {

class PythonRemoteCall final : public RpcCommandBase {
public:
    PythonRemoteCall(SerializedPyObj object, RRefId rref_id);
    MessagePtr to_message_impl() && override;
    static std::unique_ptr<PythonRemoteCall> from_message(const Message& message);
    const SerializedPyObj& serialized_object() const noexcept;
    const RRefId& rref_id() const noexcept;

private:
    SerializedPyObj object_;
    RRefId rref_id_;
};

}  // namespace tensorplay::distributed::rpc

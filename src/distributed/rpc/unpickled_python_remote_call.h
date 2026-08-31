#pragma once

#include "unpickled_python_call.h"

namespace tensorplay::distributed::rpc {

class UnpickledPythonRemoteCall final : public UnpickledPythonCall {
public:
    UnpickledPythonRemoteCall(SerializedPyObj object, RRefId rref_id);
    const RRefId& rref_id() const noexcept;

private:
    RRefId rref_id_;
};

}  // namespace tensorplay::distributed::rpc

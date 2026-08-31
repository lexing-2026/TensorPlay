#include "unpickled_python_remote_call.h"

namespace tensorplay::distributed::rpc {

UnpickledPythonRemoteCall::UnpickledPythonRemoteCall(
    SerializedPyObj object,
    RRefId rref_id)
    : UnpickledPythonCall(std::move(object)), rref_id_(rref_id) {}

const RRefId& UnpickledPythonRemoteCall::rref_id() const noexcept {
    return rref_id_;
}

}  // namespace tensorplay::distributed::rpc

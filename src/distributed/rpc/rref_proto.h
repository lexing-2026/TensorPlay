#pragma once

#include "rpc_command_base.h"
#include "types.h"

#include <memory>
#include <utility>
#include <vector>

namespace tensorplay::distributed::rpc {

class RRefMessageBase : public RpcCommandBase {
public:
    RRefMessageBase(RRefId rref_id, MessageType type);
    const RRefId& rref_id() const noexcept;
    MessageType type() const noexcept;

protected:
    RRefId rref_id_;
    MessageType type_;
};

class ForkMessageBase : public RRefMessageBase {
public:
    ForkMessageBase(RRefId rref_id, ForkId fork_id, MessageType type);
    const ForkId& fork_id() const noexcept;
    MessagePtr to_message_impl() && override;

protected:
    ForkId fork_id_;
};

class PythonRRefFetchCall final : public RRefMessageBase {
public:
    PythonRRefFetchCall(worker_id_t from_worker, RRefId rref_id);
    worker_id_t from_worker() const noexcept;
    MessagePtr to_message_impl() && override;

private:
    worker_id_t from_worker_;
};

class RRefFetchRet final : public RpcCommandBase {
public:
    RRefFetchRet(std::vector<py::object> values, MessageType type);
    MessagePtr to_message_impl() && override;
    const std::vector<py::object>& values() const noexcept;

private:
    std::vector<py::object> values_;
    MessageType type_;
};

class RRefUserDelete final : public ForkMessageBase {
public:
    RRefUserDelete(RRefId rref_id, ForkId fork_id);
};

class RemoteRet final : public ForkMessageBase {
public:
    RemoteRet(RRefId rref_id, ForkId fork_id);
};

class RRefChildAccept final : public RpcCommandBase {
public:
    explicit RRefChildAccept(ForkId fork_id);
    const ForkId& fork_id() const noexcept;
    MessagePtr to_message_impl() && override;

private:
    ForkId fork_id_;
};

class RRefForkRequest final : public ForkMessageBase {
public:
    RRefForkRequest(RRefId rref_id, ForkId fork_id);
};

class RRefAck final : public RpcCommandBase {
public:
    MessagePtr to_message_impl() && override;
};

}  // namespace tensorplay::distributed::rpc

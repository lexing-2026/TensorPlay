#include "rref_proto.h"

#include <cstring>

namespace tensorplay::distributed::rpc {
namespace {

std::vector<uint8_t> id_payload(const GloballyUniqueId& id) {
    std::vector<uint8_t> payload(sizeof(id.created_on) + sizeof(id.local_id));
    std::memcpy(payload.data(), &id.created_on, sizeof(id.created_on));
    std::memcpy(payload.data() + sizeof(id.created_on), &id.local_id, sizeof(id.local_id));
    return payload;
}

std::vector<uint8_t> fork_payload(const GloballyUniqueId& rref, const GloballyUniqueId& fork) {
    auto payload = id_payload(rref);
    auto child = id_payload(fork);
    payload.insert(payload.end(), child.begin(), child.end());
    return payload;
}

MessagePtr make_message(
    std::vector<uint8_t> payload,
    MessageType type) {
    return std::make_shared<Message>(std::move(payload), std::vector<py::object>(), type);
}

}  // namespace

RRefMessageBase::RRefMessageBase(RRefId rref_id, MessageType type)
    : rref_id_(rref_id), type_(type) {}

const RRefId& RRefMessageBase::rref_id() const noexcept {
    return rref_id_;
}

MessageType RRefMessageBase::type() const noexcept {
    return type_;
}

ForkMessageBase::ForkMessageBase(RRefId rref_id, ForkId fork_id, MessageType type)
    : RRefMessageBase(rref_id, type), fork_id_(fork_id) {}

const ForkId& ForkMessageBase::fork_id() const noexcept {
    return fork_id_;
}

MessagePtr ForkMessageBase::to_message_impl() && {
    auto message = make_message(fork_payload(rref_id_, fork_id_), type_);
    message->set_id(rref_id_.local_id);
    return message;
}

PythonRRefFetchCall::PythonRRefFetchCall(worker_id_t from_worker, RRefId rref_id)
    : RRefMessageBase(rref_id, MessageType::PYTHON_RREF_FETCH_CALL),
      from_worker_(from_worker) {}

worker_id_t PythonRRefFetchCall::from_worker() const noexcept {
    return from_worker_;
}

MessagePtr PythonRRefFetchCall::to_message_impl() && {
    auto payload = id_payload(rref_id_);
    payload.push_back(static_cast<uint8_t>(from_worker_ & 0xff));
    return make_message(std::move(payload), type_);
}

RRefFetchRet::RRefFetchRet(std::vector<py::object> values, MessageType type)
    : values_(std::move(values)), type_(type) {}

MessagePtr RRefFetchRet::to_message_impl() && {
    return std::make_shared<Message>(std::vector<uint8_t>(), std::move(values_), type_);
}

const std::vector<py::object>& RRefFetchRet::values() const noexcept {
    return values_;
}

RRefUserDelete::RRefUserDelete(RRefId rref_id, ForkId fork_id)
    : ForkMessageBase(std::move(rref_id), std::move(fork_id), MessageType::RREF_USER_DELETE) {}

RemoteRet::RemoteRet(RRefId rref_id, ForkId fork_id)
    : ForkMessageBase(std::move(rref_id), std::move(fork_id), MessageType::REMOTE_RET) {}

RRefChildAccept::RRefChildAccept(ForkId fork_id) : fork_id_(std::move(fork_id)) {}

const ForkId& RRefChildAccept::fork_id() const noexcept {
    return fork_id_;
}

MessagePtr RRefChildAccept::to_message_impl() && {
    return make_message(id_payload(fork_id_), MessageType::RREF_CHILD_ACCEPT);
}

RRefForkRequest::RRefForkRequest(RRefId rref_id, ForkId fork_id)
    : ForkMessageBase(std::move(rref_id), std::move(fork_id), MessageType::RREF_FORK_REQUEST) {}

MessagePtr RRefAck::to_message_impl() && {
    return make_message({}, MessageType::RREF_ACK);
}

}  // namespace tensorplay::distributed::rpc

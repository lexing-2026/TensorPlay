#include "py_rref.h"

#include <stdexcept>

namespace tensorplay::distributed::rpc {

PyRRef::PyRRef(std::shared_ptr<RpcRRef> impl) : impl_(std::move(impl)) {
    if (!impl_) {
        throw std::invalid_argument("RRef implementation cannot be empty");
    }
}

std::string PyRRef::owner() const {
    return impl_->owner().name;
}

bool PyRRef::is_owner() const {
    return impl_->is_owner();
}

bool PyRRef::confirmed_by_owner() const {
    return impl_->confirmed_by_owner();
}

py::object PyRRef::to_here(double timeout) const {
    return impl_->to_here(timeout);
}

py::object PyRRef::local_value() const {
    return impl_->local_value();
}

void PyRRef::backward(int64_t context_id, bool retain_graph) const {
    impl_->backward(context_id, retain_graph);
}

std::shared_ptr<PyRRef> PyRRef::fork() const {
    return std::make_shared<PyRRef>(impl_->fork());
}

py::tuple PyRRef::rref_id() const {
    return impl_->rref_id().to_python();
}

py::tuple PyRRef::fork_id() const {
    return impl_->fork_id().to_python();
}

std::shared_ptr<RpcRRef> PyRRef::native() const noexcept {
    return impl_;
}

}  // namespace tensorplay::distributed::rpc

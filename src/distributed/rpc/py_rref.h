#pragma once

#include "rref_impl.h"

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>

namespace tensorplay::distributed::rpc {

namespace py = pybind11;

class PyRRef final {
public:
    explicit PyRRef(std::shared_ptr<RpcRRef> impl);

    std::string owner() const;
    bool is_owner() const;
    bool confirmed_by_owner() const;
    py::object to_here(double timeout = -1.0) const;
    py::object local_value() const;
    void backward(int64_t context_id = -1, bool retain_graph = false) const;
    std::shared_ptr<PyRRef> fork() const;
    py::tuple rref_id() const;
    py::tuple fork_id() const;
    std::shared_ptr<RpcRRef> native() const noexcept;

private:
    std::shared_ptr<RpcRRef> impl_;
};

}  // namespace tensorplay::distributed::rpc

#pragma once

#include "python_functions.h"

namespace tensorplay::distributed::rpc {

class UnpickledPythonCall {
public:
    explicit UnpickledPythonCall(SerializedPyObj object);
    py::object callable() const;
    py::tuple args() const;
    py::dict kwargs() const;
    bool is_async_execution() const noexcept;
    void set_async_execution(bool value) noexcept;

private:
    py::object callable_;
    py::tuple args_;
    py::dict kwargs_;
    bool async_execution_ = false;
};

}  // namespace tensorplay::distributed::rpc

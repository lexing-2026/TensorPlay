#pragma once

#include <cstdint>

namespace tensorplay::distributed::autograd {

struct AutogradMetadata final {
    int64_t context_id = -1;
    int64_t message_id = -1;

    bool valid() const noexcept {
        return context_id >= 0 && message_id >= 0;
    }
};

}  // namespace tensorplay::distributed::autograd

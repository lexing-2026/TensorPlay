#pragma once

#include <optional>
#include <vector>
#include "LocalDispatchKeySet.h"

namespace tensorplay { class Tensor; }

namespace tensorplay::transform {

P10_API void check_no_batched_argument(const Tensor& tensor);
P10_API void check_no_batched_argument(const std::optional<Tensor>& tensor);
template <typename T>
void check_no_batched_argument(const T&) {}
template <typename T>
void check_no_batched_argument(const std::vector<T>& tensors) {
    for (const auto& tensor : tensors) check_no_batched_argument(tensor);
}

enum class Randomness : uint8_t { Error = 0, Same = 1, Different = 2 };
enum class Kind : uint8_t { Vmap = 0, Grad = 1, Jvp = 2, Functionalize = 3 };

struct P10_API Layer {
    Kind kind = Kind::Vmap;
    int64_t level = -1;
    int64_t batch_size = 0;
    Randomness randomness = Randomness::Error;
    bool previous_grad_mode = true;
    bool previous_forward_grad_mode = true;
    bool add_back_views = false;
    std::optional<impl::LocalDispatchKeySet> saved_dispatch_keys;
};

class P10_API DynamicLayerFrontGuard {
public:
    DynamicLayerFrontGuard();
    ~DynamicLayerFrontGuard();
    DynamicLayerFrontGuard(const DynamicLayerFrontGuard&) = delete;
    DynamicLayerFrontGuard& operator=(const DynamicLayerFrontGuard&) = delete;
};

class P10_API DynamicLayerBackGuard {
public:
    DynamicLayerBackGuard();
    ~DynamicLayerBackGuard();
    bool at_base() const;
    DynamicLayerBackGuard(const DynamicLayerBackGuard&) = delete;
    DynamicLayerBackGuard& operator=(const DynamicLayerBackGuard&) = delete;
private:
    impl::LocalDispatchKeySet saved_keys_;
    Layer layer_;
};

} // namespace tensorplay::transform

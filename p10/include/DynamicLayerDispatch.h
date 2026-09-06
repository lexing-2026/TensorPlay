#pragma once

#include <optional>
#include "LocalDispatchKeySet.h"

namespace tensorplay::transform {

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
    DynamicLayerBackGuard(const DynamicLayerBackGuard&) = delete;
    DynamicLayerBackGuard& operator=(const DynamicLayerBackGuard&) = delete;
private:
    impl::LocalDispatchKeySet saved_keys_;
    Layer layer_;
};

} // namespace tensorplay::transform

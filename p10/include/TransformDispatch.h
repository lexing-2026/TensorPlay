#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "Macros.h"
#include "Tensor.h"

namespace tensorplay {
namespace transform {

enum class Randomness : uint8_t {
    Error = 0,
    Same = 1,
    Different = 2,
};

enum class Kind : uint8_t {
    Vmap = 0,
    Grad = 1,
    Jvp = 2,
    Functionalize = 3,
};

struct P10_API Layer {
    Kind kind = Kind::Vmap;
    int64_t level = -1;
    int64_t batch_size = 0;
    Randomness randomness = Randomness::Error;
    bool previous_grad_mode = true;
    bool previous_forward_grad_mode = true;
    bool add_back_views = false;
};

class P10_API DisableTransformsGuard {
public:
    DisableTransformsGuard();
    ~DisableTransformsGuard();
    DisableTransformsGuard(const DisableTransformsGuard&) = delete;
    DisableTransformsGuard& operator=(const DisableTransformsGuard&) = delete;

private:
    bool active_ = true;
};

P10_API int64_t push_vmap(int64_t batch_size, Randomness randomness);
P10_API Layer pop_layer();
P10_API std::optional<Layer> current_layer();
P10_API std::vector<Layer> layer_stack();
P10_API void clear_layers();
P10_API bool are_transforms_active();
P10_API DispatchKey dispatch_key_for_random(DispatchKey backend);

P10_API Tensor make_batched(const Tensor& value, int64_t dim, int64_t level);
P10_API std::tuple<Tensor, std::optional<int64_t>> unwrap_at_level(
    const Tensor& value, int64_t level);
P10_API Tensor unwrap_all(const Tensor& value);
P10_API bool is_batched_at_level(const Tensor& value, int64_t level);

P10_API int64_t actual_dim(const Tensor& value, int64_t public_dim,
                           bool wrap_dim = true);
P10_API Tensor move_batch_dim(const Tensor& value, int64_t dim);

} // namespace transform
} // namespace tensorplay

#pragma once

#include "Tensor.h"
#include "Macros.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace tensorplay {
namespace tpx {

// Forward-mode AD gradients are stored per level so that independent forward
// passes can run concurrently and be scoped by a simple integer handle (the
// level's index).  A ForwardGrad object holds the tangent(s) of one tensor,
// one slot per level; it registers itself with the level so the level's
// destructor can erase every tangent belonging to it.
//
// The maximum expected nesting depth is small (second-order gradients are
// the common ceiling); the sizes below are just a hint for the reserved
// capacity in the tangents map's small-size optimization.
#define TP_EXPECTED_MAX_LEVEL 2

struct ForwardGrad;

class TENSORPLAY_API ForwardADLevel {
public:
    ForwardADLevel(uint64_t idx) : idx_(idx) {}
    ~ForwardADLevel();

    static uint64_t get_next_idx();
    static void release_idx(uint64_t idx);
    static std::shared_ptr<ForwardADLevel> get_by_idx(uint64_t idx);
    // Returns nullptr when the level has already been released.
    static std::shared_ptr<ForwardADLevel> try_get_by_idx(uint64_t idx);
    static bool has_any_level();

    void erase(const std::shared_ptr<ForwardGrad>& grad);
    void insert(std::shared_ptr<ForwardGrad> grad);

private:
    std::unordered_set<std::shared_ptr<ForwardGrad>> grads_;
    std::mutex mutex_;
    uint64_t idx_;
};

class TENSORPLAY_API ForwardGrad : public std::enable_shared_from_this<ForwardGrad> {
public:
    ForwardGrad() = default;

    // Unregisters this object from every level it is still registered with.
    // Must be called from the owning object's destructor (AutogradMeta):
    // at that point no other thread may touch this object, and the level
    // destructors may call reset() concurrently.
    void clear();

    void set_value(Tensor value, uint64_t level);
    // ``update_level`` is false only when a level's destructor is erasing us.
    void reset(uint64_t level, bool update_level = true);

    const Tensor& value(uint64_t level) const;
    bool contains(uint64_t level);
    bool empty() const;

    static const Tensor& undef_grad();

private:
    std::unordered_map<uint64_t, Tensor> content_;
    mutable std::mutex mutex_;
};

// Thread-local switch gating forward-gradient reads.  Transformations that
// run a function under forward AD enable it only for the duration of the
// traced call so unrelated code never observes tangents.
class TENSORPLAY_API FwGradMode {
public:
    static bool is_enabled();
    static void set_enabled(bool enabled);

private:
    FwGradMode() = delete;
};

} // namespace tpx
} // namespace tensorplay

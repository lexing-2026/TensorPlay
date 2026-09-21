#include "ForwardGrad.h"

#include "Exception.h"
#include <atomic>

namespace tensorplay {
namespace tpx {

// See the discussion in ForwardGrad.h for why these live at process scope
// rather than in thread-local storage: level handles are plain integers and
// must stay valid across threads.
namespace {
std::mutex all_forward_levels_mutex_;
std::vector<std::shared_ptr<ForwardADLevel>> all_forward_levels_;
// Fast path for checking whether any forward AD level exists without
// grabbing all_forward_levels_mutex_.
std::atomic<uint64_t> active_forward_levels_{0};

const static Tensor singleton_undefined_tensor;
} // namespace

uint64_t ForwardADLevel::get_next_idx() {
  std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
  const uint64_t next_idx = all_forward_levels_.size();
  TP_CHECK(next_idx == 0,
           "Nested forward mode AD is not supported at the moment");
  all_forward_levels_.push_back(std::make_shared<ForwardADLevel>(next_idx));
  active_forward_levels_.store(all_forward_levels_.size(),
                               std::memory_order_release);
  return next_idx;
}

void ForwardADLevel::release_idx(uint64_t idx) {
  std::unique_lock<std::mutex> lock(all_forward_levels_mutex_);
  TP_CHECK(idx + 1 == all_forward_levels_.size(),
           "Exiting a forward AD level that is not the last one that was "
           "created is not supported. Levels must be released in the reverse "
           "order they were created.");
  TP_CHECK(!all_forward_levels_.empty(),
           "Trying to exit a forward AD level but no level is active");
  // Keep the level alive until we have released the lock; its destructor
  // erases every tangent registered with it.
  auto lvl = std::move(all_forward_levels_.back());
  all_forward_levels_.pop_back();
  active_forward_levels_.store(all_forward_levels_.size(),
                               std::memory_order_release);
  lock.unlock();
}

bool ForwardADLevel::has_any_level() {
  return active_forward_levels_.load(std::memory_order_acquire) != 0;
}

std::shared_ptr<ForwardADLevel> ForwardADLevel::get_by_idx(uint64_t idx) {
  std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
  TP_CHECK(idx < all_forward_levels_.size(),
           "Trying to access a forward AD level with an invalid index. "
           "This index was either not created or is already deleted.");
  return all_forward_levels_[idx];
}

std::shared_ptr<ForwardADLevel> ForwardADLevel::try_get_by_idx(uint64_t idx) {
  std::lock_guard<std::mutex> lock(all_forward_levels_mutex_);
  if (idx < all_forward_levels_.size()) {
    return all_forward_levels_[idx];
  }
  return nullptr;
}

ForwardADLevel::~ForwardADLevel() {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = grads_.begin();
  while (it != grads_.end()) {
    // This locks the ForwardGrad's own mutex; it is the only call chain that
    // reaches into another class's method while holding this one's lock, and
    // the ForwardGrad may already be half-destroyed (its owning AutogradMeta
    // is being torn down), so reset() must not re-enter the level.
    (*it)->reset(idx_, /* update_level */ false);
    it = grads_.erase(it);
  }
}

void ForwardADLevel::erase(const std::shared_ptr<ForwardGrad>& grad) {
  std::lock_guard<std::mutex> lock(mutex_);
  grads_.erase(grad);
}

void ForwardADLevel::insert(std::shared_ptr<ForwardGrad> grad) {
  std::lock_guard<std::mutex> lock(mutex_);
  grads_.insert(std::move(grad));
}

void ForwardGrad::clear() {
  std::vector<uint64_t> levels_idx;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    levels_idx.reserve(content_.size());
    for (const auto& c : content_) {
      levels_idx.push_back(c.first);
    }
  }
  // Owning reference keeps each level alive until we unregister (another
  // thread may be releasing it concurrently).
  for (const uint64_t l_idx : levels_idx) {
    auto level = ForwardADLevel::try_get_by_idx(l_idx);
    if (level) {
      level->erase(shared_from_this());
    }
  }
}

void ForwardGrad::set_value(Tensor value, uint64_t level) {
  // Owning reference ensures the level is not destroyed mid-update.
  auto forward_level = ForwardADLevel::get_by_idx(level);
  forward_level->insert(shared_from_this());

  std::lock_guard<std::mutex> lock(mutex_);
  content_.try_emplace(level, std::move(value));
}

void ForwardGrad::reset(uint64_t level, bool update_level) {
  if (update_level) {
    ForwardADLevel::get_by_idx(level)->erase(shared_from_this());
  }
  std::unique_lock<std::mutex> lock(mutex_);
  const auto it = content_.find(level);
  TP_CHECK(it != content_.end(), "Resetting a non-existent level.");
  // Keep the Tensor alive until the lock is released: this can be called
  // from a level destructor while another thread still holds a reference.
  auto t = std::move(it->second);
  content_.erase(it);
  lock.unlock();
}

const Tensor& ForwardGrad::value(uint64_t level) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto& it = content_.find(level);
  return it == content_.end() ? singleton_undefined_tensor : it->second;
}

bool ForwardGrad::contains(uint64_t level) {
  std::lock_guard<std::mutex> lock(mutex_);
  return content_.count(level) > 0;
}

bool ForwardGrad::empty() const {
  return content_.empty();
}

namespace {
// One thread-local slot shared by both accessors (a single translation unit,
// so no dll-interface concern).
bool& fw_grad_mode_slot() {
  static thread_local bool enabled = true;
  return enabled;
}
} // namespace

bool FwGradMode::is_enabled() {
  return fw_grad_mode_slot();
}

void FwGradMode::set_enabled(bool enabled) {
  fw_grad_mode_slot() = enabled;
}

const Tensor& ForwardGrad::undef_grad() {
  return singleton_undefined_tensor;
}

} // namespace tpx
} // namespace tensorplay
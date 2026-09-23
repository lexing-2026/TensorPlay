#pragma once

// Intrusive reference counting.
//
// The count lives inside the managed object itself, so a handle copy is a
// single relaxed atomic increment and a handle is one machine word instead
// of two.  Strong and weak counts share one 64-bit atomic word (strong in
// the low 32 bits, weak in the high 32 bits) so both can be observed and
// updated in one atomic operation.
//
// Invariant: strong > 0  =>  weak > 0, because every live strong reference
// contributes one to the weak count.  When the strong count reaches zero
// the object calls release_resources() and leaves only its (possibly empty)
// weak-count bookkeeping alive; the object itself is destroyed when the
// weak count also reaches zero.
//
// Objects must be heap-allocated (use make_intrusive).  Adopting a raw
// pointer whose counts are still zero is rejected in debug builds, which
// catches stack-allocated or already-owned objects being double-counted.

#include <atomic>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace tensorplay {

template <typename T>
class intrusive_ptr;

template <typename T>
class weak_intrusive_ptr;

class IntrusivePtrTarget {
public:
    IntrusivePtrTarget(const IntrusivePtrTarget&) noexcept : IntrusivePtrTarget() {}
    IntrusivePtrTarget& operator=(const IntrusivePtrTarget&) noexcept { return *this; }
    IntrusivePtrTarget(IntrusivePtrTarget&&) noexcept : IntrusivePtrTarget() {}
    IntrusivePtrTarget& operator=(IntrusivePtrTarget&&) noexcept { return *this; }

protected:
    constexpr IntrusivePtrTarget() noexcept : combined_(0) {}

    // The count reaching zero is what triggers destruction, so the base is
    // never destroyed through a pointer that still carries live references.
    virtual ~IntrusivePtrTarget() {
        assert(strong_count() == 0 &&
               "destroyed while intrusive_ptr references remain");
    }

    // Hook for releasing expensive resources when the strong count reaches
    // zero while weak references still observe the object.  The object body
    // is not destroyed at that point, so members must be left in a state
    // where the eventual destructor is still safe.
    virtual void release_resources() {}

    uint32_t strong_count(std::memory_order order = std::memory_order_relaxed) const {
        return static_cast<uint32_t>(combined_.load(order) & kStrongMask);
    }

    uint32_t weak_count(std::memory_order order = std::memory_order_relaxed) const {
        return static_cast<uint32_t>((combined_.load(order) & kWeakMask) >> 32);
    }

private:
    static constexpr uint64_t kStrongOne = 1;
    static constexpr uint64_t kWeakOne = uint64_t(1) << 32;
    static constexpr uint64_t kStrongMask = 0xFFFFFFFF;
    static constexpr uint64_t kWeakMask = kStrongMask << 32;
    // strong == 1 and weak == 1: the releasing thread holds the only strong
    // reference and no weak observer exists, so the object can be deleted
    // without a decrement round trip.
    static constexpr uint64_t kSoleOwner = kStrongOne | kWeakOne;

    template <typename T>
    friend class intrusive_ptr;

    template <typename T>
    friend class weak_intrusive_ptr;

    template <typename T, typename... Args>
    friend intrusive_ptr<T> make_intrusive(Args&&... args);

    // Increment the strong count.  Only the ordering against the final
    // decrement matters, so a relaxed increment is sufficient.
    void strong_retain_() noexcept {
        combined_.fetch_add(kStrongOne, std::memory_order_relaxed);
    }

    // Decrement the strong count; destroy the object when it was the last
    // one.  Must be heap-allocated and end with no members touched after
    // the final delete.
    void strong_release_() noexcept {
        if (combined_.load(std::memory_order_acquire) == kSoleOwner) {
            // Sole owner and no weak observers: nothing can observe this
            // destruction, so skip the decrement and delete directly.
            combined_.store(0, std::memory_order_relaxed);
            delete this;
            return;
        }
        uint64_t after = combined_.fetch_sub(kStrongOne, std::memory_order_acq_rel);
        if ((after & kStrongMask) == kStrongOne) {
            // This was the last strong reference.  If weak observers remain,
            // hand the object over to them after releasing heavy resources;
            // otherwise destroy it here.
            if ((after >> 32) == kStrongOne) {
                delete this;
            } else {
                release_resources();
                if (combined_.fetch_sub(kWeakOne, std::memory_order_acq_rel) ==
                    kWeakOne) {
                    delete this;
                }
            }
        }
    }

    // Raise the strong count from a weak reference.  Fails (and leaves the
    // counts untouched) if the strong count has already reached zero; the
    // CAS loop guards the strong > 0 window against a racing release.
    bool weak_try_retain_() noexcept {
        uint64_t cur = combined_.load(std::memory_order_relaxed);
        for (;;) {
            if ((cur & kStrongMask) == 0) {
                return false;
            }
            if (combined_.compare_exchange_weak(
                    cur, cur + kStrongOne,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                return true;
            }
        }
    }

    void weak_retain_() noexcept {
        combined_.fetch_add(kWeakOne, std::memory_order_relaxed);
    }

    void weak_release_() noexcept {
        if (combined_.fetch_sub(kWeakOne, std::memory_order_acq_rel) == kWeakOne) {
            delete this;
        }
    }

    mutable std::atomic<uint64_t> combined_;
};

template <typename T>
class weak_intrusive_ptr;

template <typename T>
class intrusive_ptr {
public:
    using element_type = T;

    intrusive_ptr() noexcept : target_(nullptr) {}
    intrusive_ptr(std::nullptr_t) noexcept : target_(nullptr) {}

    intrusive_ptr(const intrusive_ptr& rhs) noexcept : target_(rhs.target_) {
        retain();
    }

    intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
        rhs.target_ = nullptr;
    }

    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    intrusive_ptr(const intrusive_ptr<U>& rhs) noexcept
        : target_(rhs.target_) {
        retain();
    }

    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    intrusive_ptr(intrusive_ptr<U>&& rhs) noexcept : target_(rhs.target_) {
        rhs.target_ = nullptr;
    }

    ~intrusive_ptr() noexcept {
        release();
    }

    intrusive_ptr& operator=(const intrusive_ptr& rhs) noexcept {
        intrusive_ptr tmp(rhs);
        swap(tmp);
        return *this;
    }

    intrusive_ptr& operator=(intrusive_ptr&& rhs) noexcept {
        intrusive_ptr tmp(std::move(rhs));
        swap(tmp);
        return *this;
    }

    explicit operator bool() const noexcept { return target_ != nullptr; }

    T* get() const noexcept { return target_; }
    T& operator*() const noexcept { return *target_; }
    T* operator->() const noexcept { return target_; }

    bool defined() const noexcept { return target_ != nullptr; }

    void reset() noexcept {
        release();
        target_ = nullptr;
    }

    void swap(intrusive_ptr& rhs) noexcept { std::swap(target_, rhs.target_); }

    uint32_t use_count() const noexcept {
        return target_ ? target_->strong_count() : 0;
    }

    bool unique() const noexcept { return use_count() == 1; }

    // Identity comparisons; two handles alias the same object when their
    // raw pointers match.
    friend bool operator==(const intrusive_ptr& lhs, const intrusive_ptr& rhs) noexcept {
        return lhs.target_ == rhs.target_;
    }
    friend bool operator!=(const intrusive_ptr& lhs, const intrusive_ptr& rhs) noexcept {
        return lhs.target_ != rhs.target_;
    }
    friend bool operator==(const intrusive_ptr& lhs, std::nullptr_t) noexcept {
        return lhs.target_ == nullptr;
    }
    friend bool operator!=(const intrusive_ptr& lhs, std::nullptr_t) noexcept {
        return lhs.target_ != nullptr;
    }
    friend bool operator==(std::nullptr_t, const intrusive_ptr& rhs) noexcept {
        return rhs.target_ == nullptr;
    }
    friend bool operator!=(std::nullptr_t, const intrusive_ptr& rhs) noexcept {
        return rhs.target_ != nullptr;
    }
    friend bool operator<(const intrusive_ptr& lhs, const intrusive_ptr& rhs) noexcept {
        return lhs.target_ < rhs.target_;
    }

private:
    template <typename U>
    friend class intrusive_ptr;
    friend class weak_intrusive_ptr<T>;

    template <typename U, typename... Args>
    friend intrusive_ptr<U> make_intrusive(Args&&... args);

    enum class adopt_ref_t { adopt };
    static constexpr adopt_ref_t adopt_ref = adopt_ref_t::adopt;

    // Take over an already-retained reference (count is >= 1).
    intrusive_ptr(T* target, adopt_ref_t) noexcept : target_(target) {}

    // Take ownership of a freshly constructed object whose counts are zero.
    explicit intrusive_ptr(T* target) noexcept : target_(target) {
        assert(target->combined_.load(std::memory_order_relaxed) == 0 &&
               "adopting an object that is already owned or not heap allocated");
        target->combined_.store(IntrusivePtrTarget::kSoleOwner,
                                std::memory_order_relaxed);
    }

    void retain() noexcept {
        if (target_) {
            target_->strong_retain_();
        }
    }

    void release() noexcept {
        if (target_) {
            T* target = target_;
            target_ = nullptr;
            target->strong_release_();
        }
    }

    T* target_;
};

template <typename T>
class weak_intrusive_ptr {
public:
    using element_type = T;

    weak_intrusive_ptr() noexcept : target_(nullptr) {}
    weak_intrusive_ptr(std::nullptr_t) noexcept : target_(nullptr) {}

    weak_intrusive_ptr(const intrusive_ptr<T>& rhs) noexcept
        : target_(rhs.target_) {
        retain();
    }

    weak_intrusive_ptr(const weak_intrusive_ptr& rhs) noexcept
        : target_(rhs.target_) {
        retain();
    }

    weak_intrusive_ptr(weak_intrusive_ptr&& rhs) noexcept
        : target_(rhs.target_) {
        rhs.target_ = nullptr;
    }

    ~weak_intrusive_ptr() noexcept {
        release();
    }

    weak_intrusive_ptr& operator=(const weak_intrusive_ptr& rhs) noexcept {
        weak_intrusive_ptr tmp(rhs);
        swap(tmp);
        return *this;
    }

    weak_intrusive_ptr& operator=(weak_intrusive_ptr&& rhs) noexcept {
        weak_intrusive_ptr tmp(std::move(rhs));
        swap(tmp);
        return *this;
    }

    void reset() noexcept {
        release();
        target_ = nullptr;
    }

    void swap(weak_intrusive_ptr& rhs) noexcept {
        std::swap(target_, rhs.target_);
    }

    // True when the strong count has reached zero; the object body may
    // still be kept alive by other weak references.
    bool expired() const noexcept {
        return !target_ || target_->strong_count() == 0;
    }

    uint32_t use_count() const noexcept {
        return target_ ? target_->strong_count() : 0;
    }

    // Promote to a strong reference, or return null if the object is gone.
    intrusive_ptr<T> lock() const {
        if (target_ && target_->weak_try_retain_()) {
            return intrusive_ptr<T>(target_, intrusive_ptr<T>::adopt_ref);
        }
        return intrusive_ptr<T>();
    }

private:
    explicit weak_intrusive_ptr(T* target) noexcept : target_(target) {}

    void retain() noexcept {
        if (target_) {
            target_->weak_retain_();
        }
    }

    void release() noexcept {
        if (target_) {
            T* target = target_;
            target_ = nullptr;
            target->weak_release_();
        }
    }

    T* target_;
};

// Allocate a new T on the heap and wrap it with a single owning handle.
template <typename T, typename... Args>
intrusive_ptr<T> make_intrusive(Args&&... args) {
    return intrusive_ptr<T>(new T(std::forward<Args>(args)...));
}

} // namespace tensorplay

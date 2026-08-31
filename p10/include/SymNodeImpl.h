#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "Exception.h"
#include "Macros.h"

namespace tensorplay {

class SymNodeImpl;

class P10_API SymNode {
public:
    SymNode() noexcept = default;

    SymNode(const SymNode& other) noexcept;
    SymNode(SymNode&& other) noexcept;
    SymNode& operator=(const SymNode& other) noexcept;
    SymNode& operator=(SymNode&& other) noexcept;
    ~SymNode();

    static SymNode reclaim(SymNodeImpl* ptr) noexcept;
    static SymNode reclaim_copy(SymNodeImpl* ptr) noexcept;

    SymNodeImpl* get() const noexcept { return ptr_; }
    SymNodeImpl* operator->() const noexcept { return ptr_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    SymNodeImpl* release() && noexcept;

private:
    explicit SymNode(SymNodeImpl* ptr, bool add_ref) noexcept;
    SymNodeImpl* ptr_ = nullptr;
};

enum class SymNodeValueType : uint8_t {
    Integer,
    Boolean,
    Floating,
};

class P10_API SymNodeImpl {
public:
    SymNodeImpl() noexcept = default;
    SymNodeImpl(const SymNodeImpl&) = delete;
    SymNodeImpl& operator=(const SymNodeImpl&) = delete;
    virtual ~SymNodeImpl() = default;

    bool is_int() const { return value_type() == SymNodeValueType::Integer; }
    bool is_bool() const { return value_type() == SymNodeValueType::Boolean; }
    bool is_float() const { return value_type() == SymNodeValueType::Floating; }
    virtual SymNodeValueType value_type() const = 0;
    virtual bool is_nested_int() const { return false; }

    virtual SymNode add(const SymNode& other);
    virtual SymNode sub(const SymNode& other);
    virtual SymNode mul(const SymNode& other);
    virtual SymNode truediv(const SymNode& other);
    virtual SymNode float_truediv(const SymNode& other) {
        return truediv(other);
    }
    virtual SymNode int_truediv(const SymNode& other) {
        return truediv(other);
    }
    virtual SymNode pow(const SymNode& other);
    virtual SymNode float_pow(const SymNode& other) {
        return pow(other);
    }
    virtual SymNode pow_by_natural(const SymNode& other) {
        return pow(other);
    }
    virtual SymNode floordiv(const SymNode& other);
    virtual SymNode int_floordiv(const SymNode& other) {
        return floordiv(other);
    }
    virtual SymNode mod(const SymNode& other);

    virtual SymNode eq(const SymNode& other);
    virtual SymNode ne(const SymNode& other);
    virtual SymNode gt(const SymNode& other);
    virtual SymNode lt(const SymNode& other);
    virtual SymNode le(const SymNode& other);
    virtual SymNode ge(const SymNode& other);

    virtual SymNode ceil();
    virtual SymNode floor();
    virtual SymNode neg();
    virtual SymNode sym_min(const SymNode& other);
    virtual SymNode sym_max(const SymNode& other);
    virtual SymNode sym_or(const SymNode& other);
    virtual SymNode sym_and(const SymNode& other);
    virtual SymNode sym_not();
    virtual SymNode sym_ite(const SymNode& then_value,
                            const SymNode& else_value);

    virtual SymNode is_contiguous(const std::vector<SymNode>& sizes,
                                  const std::vector<SymNode>& strides);
    virtual SymNode is_channels_last_contiguous_2d(
        const std::vector<SymNode>& sizes,
        const std::vector<SymNode>& strides);
    virtual SymNode is_channels_last_contiguous_3d(
        const std::vector<SymNode>& sizes,
        const std::vector<SymNode>& strides);
    virtual SymNode is_channels_last_strides_2d(
        const std::vector<SymNode>& sizes,
        const std::vector<SymNode>& strides);
    virtual SymNode is_channels_last_strides_3d(
        const std::vector<SymNode>& sizes,
        const std::vector<SymNode>& strides);
    virtual SymNode is_non_overlapping_and_dense(
        const std::vector<SymNode>& sizes,
        const std::vector<SymNode>& strides);

    virtual SymNode clone();
    virtual SymNode sym_float();
    virtual SymNode wrap_int(int64_t value);
    virtual SymNode wrap_float(double value);
    virtual SymNode wrap_bool(bool value);

    virtual int64_t guard_int(const char* file, int64_t line);
    virtual bool guard_bool(const char* file, int64_t line);
    virtual double guard_float(const char* file, int64_t line);
    virtual bool guard_size_oblivious(const char* file, int64_t line);
    virtual bool guard_or_false(const char* file, int64_t line);
    virtual bool statically_known_true(const char* file, int64_t line);
    virtual bool guard_or_true(const char* file, int64_t line);
    virtual bool expect_true(const char* file, int64_t line);

    virtual int64_t int_();
    virtual bool bool_();
    virtual double float_();
    virtual bool has_hint();
    virtual std::string str();
    virtual std::string graph_repr();

    virtual std::optional<int64_t> nested_int();
    virtual std::optional<int64_t> nested_int_coeff();
    virtual std::optional<int64_t> constant_int();
    virtual std::optional<bool> constant_bool();
    virtual std::optional<double> constant_float();
    virtual std::optional<int64_t> maybe_as_int();
    virtual std::optional<double> maybe_as_float();
    virtual bool is_constant();
    virtual bool is_symbolic();

    void incref() const noexcept {
        refs_.fetch_add(1, std::memory_order_relaxed);
    }

    void decref() const noexcept {
        if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }

private:
    mutable std::atomic<uint32_t> refs_{1};
};

P10_API std::ostream& operator<<(std::ostream& os, const SymNode& node);

P10_API SymNode make_symbolic_int(
    std::string name, std::optional<int64_t> hint = std::nullopt);
P10_API SymNode make_symbolic_bool(
    std::string name, std::optional<bool> hint = std::nullopt);
P10_API SymNode make_symbolic_float(
    std::string name, std::optional<double> hint = std::nullopt);
P10_API SymNode make_constant_int(int64_t value);
P10_API SymNode make_constant_bool(bool value);
P10_API SymNode make_constant_float(double value);

} // namespace tensorplay

inline tensorplay::SymNode::SymNode(
    tensorplay::SymNodeImpl* ptr, bool add_ref) noexcept : ptr_(ptr) {
    if (ptr_ != nullptr && add_ref) {
        ptr_->incref();
    }
}

inline tensorplay::SymNode::SymNode(
    const tensorplay::SymNode& other) noexcept : ptr_(other.ptr_) {
    if (ptr_ != nullptr) {
        ptr_->incref();
    }
}

inline tensorplay::SymNode::SymNode(tensorplay::SymNode&& other) noexcept
    : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
}

inline tensorplay::SymNode& tensorplay::SymNode::operator=(
    const tensorplay::SymNode& other) noexcept {
    if (this != &other) {
        if (ptr_ != nullptr) {
            ptr_->decref();
        }
        ptr_ = other.ptr_;
        if (ptr_ != nullptr) {
            ptr_->incref();
        }
    }
    return *this;
}

inline tensorplay::SymNode& tensorplay::SymNode::operator=(
    tensorplay::SymNode&& other) noexcept {
    if (this != &other) {
        if (ptr_ != nullptr) {
            ptr_->decref();
        }
        ptr_ = other.ptr_;
        other.ptr_ = nullptr;
    }
    return *this;
}

inline tensorplay::SymNode::~SymNode() {
    if (ptr_ != nullptr) {
        ptr_->decref();
    }
}

inline tensorplay::SymNode tensorplay::SymNode::reclaim(
    tensorplay::SymNodeImpl* ptr) noexcept {
    return SymNode(ptr, false);
}

inline tensorplay::SymNode tensorplay::SymNode::reclaim_copy(
    tensorplay::SymNodeImpl* ptr) noexcept {
    return SymNode(ptr, true);
}

inline tensorplay::SymNodeImpl* tensorplay::SymNode::release() && noexcept {
    auto* result = ptr_;
    ptr_ = nullptr;
    return result;
}

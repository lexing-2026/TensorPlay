#include "SymNodeImpl.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>
#include <variant>

namespace tensorplay {

namespace {

enum class ExprOp : uint8_t {
    Symbol,
    Add,
    Sub,
    Mul,
    TrueDiv,
    FloorDiv,
    Mod,
    Pow,
    Eq,
    Ne,
    Gt,
    Lt,
    Le,
    Ge,
    Ceil,
    Floor,
    Neg,
    Min,
    Max,
    Or,
    And,
    Not,
    Ite,
    ToFloat,
    Contiguous,
    ChannelsLastContiguous2d,
    ChannelsLastContiguous3d,
    ChannelsLastStrides2d,
    ChannelsLastStrides3d,
    NonOverlappingAndDense,
};

using Value = std::variant<int64_t, bool, double>;

const char* op_name(ExprOp op) {
    switch (op) {
        case ExprOp::Add: return "+";
        case ExprOp::Sub: return "-";
        case ExprOp::Mul: return "*";
        case ExprOp::TrueDiv: return "/";
        case ExprOp::FloorDiv: return "//";
        case ExprOp::Mod: return "%";
        case ExprOp::Pow: return "**";
        case ExprOp::Eq: return "==";
        case ExprOp::Ne: return "!=";
        case ExprOp::Gt: return ">";
        case ExprOp::Lt: return "<";
        case ExprOp::Le: return "<=";
        case ExprOp::Ge: return ">=";
        case ExprOp::Min: return "min";
        case ExprOp::Max: return "max";
        case ExprOp::Or: return "|";
        case ExprOp::And: return "&";
        case ExprOp::Ite: return "ite";
        case ExprOp::ToFloat: return "float";
        default: return "?";
    }
}

std::string value_string(const Value& value) {
    return std::visit(
        [](const auto& item) {
            using T = std::decay_t<decltype(item)>;
            if constexpr (std::is_same_v<T, bool>) {
                return std::string(item ? "true" : "false");
            } else {
                std::ostringstream out;
                out << item;
                return out.str();
            }
        },
        value);
}

std::optional<int64_t> exact_int(const SymNode& node) {
    return node ? node->constant_int() : std::nullopt;
}

std::optional<bool> exact_bool(const SymNode& node) {
    return node ? node->constant_bool() : std::nullopt;
}

std::optional<double> exact_float(const SymNode& node) {
    return node ? node->constant_float() : std::nullopt;
}

template <typename T>
std::optional<T> hinted_value(const SymNode& node);

template <>
std::optional<int64_t> hinted_value<int64_t>(const SymNode& node) {
    if (!node || !node->is_int() || !node->has_hint()) return std::nullopt;
    try {
        return node->guard_int("symbolic", 0);
    } catch (const Exception&) {
        return std::nullopt;
    }
}

template <>
std::optional<bool> hinted_value<bool>(const SymNode& node) {
    if (!node || !node->is_bool() || !node->has_hint()) return std::nullopt;
    try {
        return node->guard_bool("symbolic", 0);
    } catch (const Exception&) {
        return std::nullopt;
    }
}

template <>
std::optional<double> hinted_value<double>(const SymNode& node) {
    if (!node || !node->is_float() || !node->has_hint()) return std::nullopt;
    try {
        return node->guard_float("symbolic", 0);
    } catch (const Exception&) {
        return std::nullopt;
    }
}

int64_t floor_divide(int64_t left, int64_t right) {
    TP_CHECK_VALUE(right != 0, "symbolic integer division by zero");
    TP_CHECK_VALUE(!(left == std::numeric_limits<int64_t>::min() && right == -1),
                   "symbolic integer division overflow");
    int64_t quotient = left / right;
    const int64_t remainder = left % right;
    if (remainder != 0 && ((remainder < 0) != (right < 0))) {
        --quotient;
    }
    return quotient;
}

int64_t checked_add(int64_t left, int64_t right) {
    const __int128 result = static_cast<__int128>(left) + right;
    TP_CHECK_VALUE(result >= std::numeric_limits<int64_t>::min() &&
                       result <= std::numeric_limits<int64_t>::max(),
                   "symbolic integer addition overflow");
    return static_cast<int64_t>(result);
}

int64_t checked_sub(int64_t left, int64_t right) {
    const __int128 result = static_cast<__int128>(left) - right;
    TP_CHECK_VALUE(result >= std::numeric_limits<int64_t>::min() &&
                       result <= std::numeric_limits<int64_t>::max(),
                   "symbolic integer subtraction overflow");
    return static_cast<int64_t>(result);
}

int64_t checked_mul(int64_t left, int64_t right) {
    const __int128 result = static_cast<__int128>(left) * right;
    TP_CHECK_VALUE(result >= std::numeric_limits<int64_t>::min() &&
                       result <= std::numeric_limits<int64_t>::max(),
                   "symbolic integer multiplication overflow");
    return static_cast<int64_t>(result);
}

int64_t checked_neg(int64_t value) {
    TP_CHECK_VALUE(value != std::numeric_limits<int64_t>::min(),
                   "symbolic integer negation overflow");
    return -value;
}

int64_t checked_pow(int64_t base, int64_t exponent) {
    TP_CHECK_VALUE(exponent >= 0,
                   "symbolic integer exponent must be non-negative");
    int64_t result = 1;
    uint64_t remaining = static_cast<uint64_t>(exponent);
    while (remaining != 0) {
        if (remaining & 1U) result = checked_mul(result, base);
        remaining >>= 1U;
        if (remaining != 0) base = checked_mul(base, base);
    }
    return result;
}

std::optional<Value> value_for(const SymNode& node, bool exact) {
    if (!node) return std::nullopt;
    if (node->is_int()) {
        auto value = exact ? exact_int(node) : hinted_value<int64_t>(node);
        if (value) return Value(*value);
    } else if (node->is_bool()) {
        auto value = exact ? exact_bool(node) : hinted_value<bool>(node);
        if (value) return Value(*value);
    } else if (node->is_float()) {
        auto value = exact ? exact_float(node) : hinted_value<double>(node);
        if (value) return Value(*value);
    }
    return std::nullopt;
}

std::optional<Value> apply_scalar(ExprOp op,
                                  SymNodeValueType result_type,
                                  const std::vector<SymNode>& inputs,
                                  bool exact) {
    if (op == ExprOp::Symbol) return std::nullopt;
    std::vector<Value> values;
    values.reserve(inputs.size());
    for (const auto& input : inputs) {
        auto value = value_for(input, exact);
        if (!value) return std::nullopt;
        values.push_back(*value);
    }

    auto int_value = [&](size_t index) {
        return std::get<int64_t>(values[index]);
    };
    auto bool_value = [&](size_t index) {
        return std::get<bool>(values[index]);
    };
    auto float_value = [&](size_t index) {
        return std::get<double>(values[index]);
    };

    if (result_type == SymNodeValueType::Integer) {
        if ((op == ExprOp::Neg || op == ExprOp::Ceil ||
             op == ExprOp::Floor) && inputs.size() != 1) {
            return std::nullopt;
        }
        if (op != ExprOp::Neg && op != ExprOp::Ceil &&
            op != ExprOp::Floor && inputs.size() != 2) {
            return std::nullopt;
        }
        const int64_t left = int_value(0);
        if (op == ExprOp::Neg) return Value(checked_neg(left));
        if (op == ExprOp::Ceil || op == ExprOp::Floor) return Value(left);
        if (op == ExprOp::ToFloat) return std::nullopt;
        const int64_t right = int_value(1);
        switch (op) {
            case ExprOp::Add: return Value(checked_add(left, right));
            case ExprOp::Sub: return Value(checked_sub(left, right));
            case ExprOp::Mul: return Value(checked_mul(left, right));
            case ExprOp::FloorDiv: return Value(floor_divide(left, right));
            case ExprOp::TrueDiv: return Value(floor_divide(left, right));
            case ExprOp::Mod:
                TP_CHECK_VALUE(right != 0, "symbolic integer remainder by zero");
                TP_CHECK_VALUE(!(left == std::numeric_limits<int64_t>::min() &&
                                 right == -1),
                               "symbolic integer remainder overflow");
                return Value(left % right);
            case ExprOp::Pow: return Value(checked_pow(left, right));
            case ExprOp::Min: return Value(std::min(left, right));
            case ExprOp::Max: return Value(std::max(left, right));
            default: break;
        }
    } else if (result_type == SymNodeValueType::Floating) {
        if ((op == ExprOp::ToFloat || op == ExprOp::Neg ||
             op == ExprOp::Ceil || op == ExprOp::Floor) && inputs.size() != 1) {
            return std::nullopt;
        }
        if (op != ExprOp::ToFloat && op != ExprOp::Neg &&
            op != ExprOp::Ceil && op != ExprOp::Floor && inputs.size() != 2) {
            return std::nullopt;
        }
        const double left = op == ExprOp::ToFloat
            ? (values[0].index() == 0
                   ? static_cast<double>(std::get<int64_t>(values[0]))
                   : float_value(0))
            : float_value(0);
        if (op == ExprOp::Neg) return Value(-left);
        if (op == ExprOp::Ceil) return Value(std::ceil(left));
        if (op == ExprOp::Floor) return Value(std::floor(left));
        if (op == ExprOp::ToFloat) return Value(left);
        const double right = float_value(1);
        switch (op) {
            case ExprOp::Add: return Value(left + right);
            case ExprOp::Sub: return Value(left - right);
            case ExprOp::Mul: return Value(left * right);
            case ExprOp::TrueDiv: return Value(left / right);
            case ExprOp::Pow: return Value(std::pow(left, right));
            case ExprOp::Min: return Value(std::min(left, right));
            case ExprOp::Max: return Value(std::max(left, right));
            default: break;
        }
    } else if (result_type == SymNodeValueType::Boolean &&
               (op == ExprOp::Not || op == ExprOp::And || op == ExprOp::Or)) {
        if ((op == ExprOp::Not && inputs.size() != 1) ||
            (op != ExprOp::Not && inputs.size() != 2)) {
            return std::nullopt;
        }
        const bool left = bool_value(0);
        if (op == ExprOp::Not) return Value(!left);
        const bool right = bool_value(1);
        if (op == ExprOp::And) return Value(left && right);
        if (op == ExprOp::Or) return Value(left || right);
    }

    if (result_type == SymNodeValueType::Boolean) {
        const bool result = [&]() {
            if (inputs.size() < 2) return false;
            const Value& left = values[0];
            const Value& right = values[1];
            if (left.index() == 0 && right.index() == 0) {
                const auto a = std::get<int64_t>(left);
                const auto b = std::get<int64_t>(right);
                switch (op) {
                    case ExprOp::Eq: return a == b;
                    case ExprOp::Ne: return a != b;
                    case ExprOp::Gt: return a > b;
                    case ExprOp::Lt: return a < b;
                    case ExprOp::Le: return a <= b;
                    case ExprOp::Ge: return a >= b;
                    default: return false;
                }
            }
            if (left.index() == 1 && right.index() == 1) {
                const auto a = std::get<bool>(left);
                const auto b = std::get<bool>(right);
                switch (op) {
                    case ExprOp::Eq: return a == b;
                    case ExprOp::Ne: return a != b;
                    default: return false;
                }
            }
            const auto a = std::get<double>(left);
            const auto b = std::get<double>(right);
            switch (op) {
                case ExprOp::Eq: return a == b;
                case ExprOp::Ne: return a != b;
                case ExprOp::Gt: return a > b;
                case ExprOp::Lt: return a < b;
                case ExprOp::Le: return a <= b;
                case ExprOp::Ge: return a >= b;
                default: return false;
            }
        }();
        return Value(result);
    }
    return std::nullopt;
}

bool layout_contiguous(const std::vector<int64_t>& sizes,
                       const std::vector<int64_t>& strides) {
    if (sizes.size() != strides.size()) return false;
    int64_t expected = 1;
    for (size_t i = sizes.size(); i > 0; --i) {
        const size_t dim = i - 1;
        if (sizes[dim] == 0) return true;
        if (sizes[dim] != 1) {
            if (strides[dim] != expected) return false;
            expected = checked_mul(expected, sizes[dim]);
        }
    }
    return true;
}

bool layout_channels_last(const std::vector<int64_t>& sizes,
                          const std::vector<int64_t>& strides,
                          size_t expected_rank) {
    if (sizes.size() != expected_rank || strides.size() != expected_rank) {
        return false;
    }
    const std::vector<size_t> order = expected_rank == 4
        ? std::vector<size_t>{1, 3, 2, 0}
        : std::vector<size_t>{1, 4, 3, 2, 0};
    int64_t expected = 1;
    for (const size_t dim : order) {
        if (sizes[dim] != 1) {
            if (strides[dim] != expected) return false;
            expected = checked_mul(expected, sizes[dim]);
        }
    }
    return true;
}

bool layout_channels_last_strides(const std::vector<int64_t>& sizes,
                                  const std::vector<int64_t>& strides,
                                  size_t expected_rank) {
    if (sizes.size() != expected_rank || strides.size() != expected_rank) {
        return false;
    }
    if (strides[1] == 0) return false;
    const std::vector<size_t> order = expected_rank == 4
        ? std::vector<size_t>{1, 3, 2, 0}
        : std::vector<size_t>{1, 4, 3, 2, 0};
    int64_t minimum = 0;
    for (const size_t dim : order) {
        if (sizes[dim] == 0 || strides[dim] < minimum) return false;
        if (dim == 0 && minimum == strides[1]) return false;
        minimum = checked_mul(
            strides[dim], std::max<int64_t>(sizes[dim], 1));
    }
    return true;
}

bool layout_dense(const std::vector<int64_t>& sizes,
                  const std::vector<int64_t>& strides) {
    if (sizes.size() != strides.size()) return false;
    if (sizes.size() == 1) {
        return sizes[0] < 2 || strides[0] == 1;
    }
    std::vector<size_t> order(sizes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
        if (sizes[left] < 2) return false;
        if (sizes[right] < 2) return true;
        return strides[left] < strides[right];
    });
    int64_t expected = 1;
    for (size_t dim : order) {
        if (sizes[dim] < 2) return true;
        if (strides[dim] != expected) return false;
        expected = checked_mul(expected, sizes[dim]);
    }
    return true;
}

std::optional<bool> layout_hint(ExprOp op,
                                const std::vector<SymNode>& inputs,
                                size_t split) {
    std::vector<int64_t> sizes;
    std::vector<int64_t> strides;
    sizes.reserve(split);
    strides.reserve(inputs.size() - split);
    for (size_t i = 0; i < split; ++i) {
        auto value = hinted_value<int64_t>(inputs[i]);
        if (!value) return std::nullopt;
        sizes.push_back(*value);
    }
    for (size_t i = split; i < inputs.size(); ++i) {
        auto value = hinted_value<int64_t>(inputs[i]);
        if (!value) return std::nullopt;
        strides.push_back(*value);
    }
    switch (op) {
        case ExprOp::Contiguous:
            return layout_contiguous(sizes, strides);
        case ExprOp::ChannelsLastContiguous2d:
            return layout_channels_last(sizes, strides, 4);
        case ExprOp::ChannelsLastContiguous3d:
            return layout_channels_last(sizes, strides, 5);
        case ExprOp::ChannelsLastStrides2d:
            return layout_channels_last_strides(sizes, strides, 4);
        case ExprOp::ChannelsLastStrides3d:
            return layout_channels_last_strides(sizes, strides, 5);
        case ExprOp::NonOverlappingAndDense:
            return layout_dense(sizes, strides);
        default:
            return std::nullopt;
    }
}

SymNode layout_owner(const std::vector<SymNode>& sizes,
                     const std::vector<SymNode>& strides) {
    for (const auto& value : sizes) {
        if (value) return value;
    }
    for (const auto& value : strides) {
        if (value) return value;
    }
    return make_constant_int(0);
}

SymNode symbolic_contiguous(const std::vector<SymNode>& sizes,
                            const std::vector<SymNode>& strides,
                            const std::vector<size_t>& order) {
    if (sizes.size() != strides.size() || order.size() != sizes.size()) {
        return make_constant_bool(false);
    }
    const SymNode owner = layout_owner(sizes, strides);
    const SymNode one = owner->wrap_int(1);
    const SymNode zero = owner->wrap_int(0);
    SymNode condition = owner->wrap_bool(true);
    SymNode empty = owner->wrap_bool(false);
    SymNode expected = one;
    for (const size_t dim : order) {
        const SymNode size_is_one = sizes[dim]->eq(one);
        const SymNode stride_is_expected = strides[dim]->eq(expected);
        condition = condition->sym_and(
            size_is_one->sym_or(stride_is_expected));
        empty = empty->sym_or(sizes[dim]->eq(zero));
        expected = expected->mul(sizes[dim]);
    }
    return condition->sym_or(empty);
}

SymNode symbolic_channels_last_strides(
    const std::vector<SymNode>& sizes,
    const std::vector<SymNode>& strides,
    const std::vector<size_t>& order) {
    if (sizes.size() != strides.size() || order.size() != sizes.size() ||
        sizes.size() < 2) {
        return make_constant_bool(false);
    }
    const SymNode owner = layout_owner(sizes, strides);
    const SymNode one = owner->wrap_int(1);
    const SymNode zero = owner->wrap_int(0);
    SymNode condition = strides[1]->ne(zero);
    SymNode minimum = zero;
    for (const size_t dim : order) {
        condition = condition->sym_and(sizes[dim]->ne(zero));
        condition = condition->sym_and(strides[dim]->ge(minimum));
        if (dim == 0) {
            condition = condition->sym_and(strides[dim]->ne(strides[1]));
        }
        minimum = strides[dim]->mul(sizes[dim]->sym_max(one));
    }
    return condition;
}

SymNode symbolic_non_overlapping_and_dense_indicator(
    const std::vector<SymNode>& sizes,
    const std::vector<SymNode>& strides);

class ExpressionSymNode final : public SymNodeImpl {
public:
    explicit ExpressionSymNode(Value value)
        : type_(std::holds_alternative<int64_t>(value)
                    ? SymNodeValueType::Integer
                    : std::holds_alternative<bool>(value)
                          ? SymNodeValueType::Boolean
                          : SymNodeValueType::Floating),
          op_(ExprOp::Symbol),
          value_(std::move(value)) {}

    ExpressionSymNode(SymNodeValueType type,
                      ExprOp op,
                      std::vector<SymNode> inputs,
                      std::string name,
                      size_t layout_split,
                      std::optional<Value> hint)
        : type_(type),
          op_(op),
          inputs_(std::move(inputs)),
          name_(std::move(name)),
          layout_split_(layout_split),
          hint_(std::move(hint)) {}

    SymNodeValueType value_type() const override { return type_; }

    SymNode add(const SymNode& other) override {
        return binary(ExprOp::Add, other, type_);
    }
    SymNode sub(const SymNode& other) override {
        return binary(ExprOp::Sub, other, type_);
    }
    SymNode mul(const SymNode& other) override {
        return binary(ExprOp::Mul, other, type_);
    }
    SymNode truediv(const SymNode& other) override {
        return binary(ExprOp::TrueDiv, other, type_);
    }
    SymNode float_truediv(const SymNode& other) override {
        return binary(ExprOp::TrueDiv, other, type_);
    }
    SymNode int_truediv(const SymNode& other) override {
        return binary(ExprOp::FloorDiv, other, type_);
    }
    SymNode pow(const SymNode& other) override {
        return binary(ExprOp::Pow, other, type_);
    }
    SymNode float_pow(const SymNode& other) override {
        return binary(ExprOp::Pow, other, type_);
    }
    SymNode pow_by_natural(const SymNode& other) override {
        return binary(ExprOp::Pow, other, type_);
    }
    SymNode floordiv(const SymNode& other) override {
        return binary(ExprOp::FloorDiv, other, type_);
    }
    SymNode int_floordiv(const SymNode& other) override {
        return binary(ExprOp::FloorDiv, other, type_);
    }
    SymNode mod(const SymNode& other) override {
        return binary(ExprOp::Mod, other, type_);
    }

    SymNode eq(const SymNode& other) override {
        return binary(ExprOp::Eq, other, SymNodeValueType::Boolean);
    }
    SymNode ne(const SymNode& other) override {
        return binary(ExprOp::Ne, other, SymNodeValueType::Boolean);
    }
    SymNode gt(const SymNode& other) override {
        return binary(ExprOp::Gt, other, SymNodeValueType::Boolean);
    }
    SymNode lt(const SymNode& other) override {
        return binary(ExprOp::Lt, other, SymNodeValueType::Boolean);
    }
    SymNode le(const SymNode& other) override {
        return binary(ExprOp::Le, other, SymNodeValueType::Boolean);
    }
    SymNode ge(const SymNode& other) override {
        return binary(ExprOp::Ge, other, SymNodeValueType::Boolean);
    }

    SymNode ceil() override { return unary(ExprOp::Ceil, type_); }
    SymNode floor() override { return unary(ExprOp::Floor, type_); }
    SymNode neg() override { return unary(ExprOp::Neg, type_); }
    SymNode sym_min(const SymNode& other) override {
        return binary(ExprOp::Min, other, type_);
    }
    SymNode sym_max(const SymNode& other) override {
        return binary(ExprOp::Max, other, type_);
    }
    SymNode sym_or(const SymNode& other) override {
        return binary(ExprOp::Or, other, SymNodeValueType::Boolean);
    }
    SymNode sym_and(const SymNode& other) override {
        return binary(ExprOp::And, other, SymNodeValueType::Boolean);
    }
    SymNode sym_not() override { return unary(ExprOp::Not, SymNodeValueType::Boolean); }
    SymNode sym_ite(const SymNode& then_value,
                   const SymNode& else_value) override {
        TP_CHECK_TYPE(type_ == SymNodeValueType::Boolean,
                      "symbolic condition must be boolean");
        TP_CHECK_TYPE(then_value && else_value,
                      "symbolic branches must be defined");
        TP_CHECK_TYPE(then_value->value_type() == else_value->value_type(),
                      "symbolic branches must have the same type");
        if (auto condition = constant_bool()) {
            return *condition ? then_value : else_value;
        }
        std::vector<SymNode> inputs = {SymNode::reclaim_copy(this), then_value,
                                       else_value};
        auto type = then_value->value_type();
        auto exact_then = value_for(then_value, true);
        auto exact_else = value_for(else_value, true);
        if (exact_then && exact_else) {
            return make_constant_value(type, *exact_then);
        }
        return SymNode::reclaim(new ExpressionSymNode(
            type, ExprOp::Ite, std::move(inputs), {}, 0, std::nullopt));
    }

    SymNode is_contiguous(const std::vector<SymNode>& sizes,
                          const std::vector<SymNode>& strides) override {
        return layout(ExprOp::Contiguous, sizes, strides);
    }
    SymNode is_channels_last_contiguous_2d(
        const std::vector<SymNode>& sizes,
        const std::vector<SymNode>& strides) override {
        return layout(ExprOp::ChannelsLastContiguous2d, sizes, strides);
    }
    SymNode is_channels_last_contiguous_3d(
        const std::vector<SymNode>& sizes,
        const std::vector<SymNode>& strides) override {
        return layout(ExprOp::ChannelsLastContiguous3d, sizes, strides);
    }
    SymNode is_channels_last_strides_2d(
        const std::vector<SymNode>& sizes,
        const std::vector<SymNode>& strides) override {
        return layout(ExprOp::ChannelsLastStrides2d, sizes, strides);
    }
    SymNode is_channels_last_strides_3d(
        const std::vector<SymNode>& sizes,
        const std::vector<SymNode>& strides) override {
        return layout(ExprOp::ChannelsLastStrides3d, sizes, strides);
    }
    SymNode is_non_overlapping_and_dense(
        const std::vector<SymNode>& sizes,
        const std::vector<SymNode>& strides) override {
        return layout(ExprOp::NonOverlappingAndDense, sizes, strides);
    }

    SymNode clone() override {
        if (value_) return make_constant_value(type_, *value_);
        return SymNode::reclaim(new ExpressionSymNode(
            type_, op_, inputs_, name_, layout_split_, hint_));
    }

    SymNode sym_float() override {
        if (type_ == SymNodeValueType::Floating) {
            return SymNode::reclaim_copy(this);
        }
        TP_CHECK_TYPE(type_ == SymNodeValueType::Integer,
                      "only symbolic integers can convert to symbolic floats");
        if (auto value = constant_int()) {
            return make_constant_float(static_cast<double>(*value));
        }
        return unary(ExprOp::ToFloat, SymNodeValueType::Floating);
    }

    SymNode wrap_int(int64_t value) override { return make_constant_int(value); }
    SymNode wrap_float(double value) override { return make_constant_float(value); }
    SymNode wrap_bool(bool value) override { return make_constant_bool(value); }

    int64_t guard_int(const char*, int64_t) override {
        TP_CHECK_TYPE(type_ == SymNodeValueType::Integer,
                      "symbolic node is not an integer");
        if (auto value = hinted_int()) return *value;
        TP_THROW(RuntimeError, "symbolic integer has no concrete value");
    }
    bool guard_bool(const char*, int64_t) override {
        TP_CHECK_TYPE(type_ == SymNodeValueType::Boolean,
                      "symbolic node is not a boolean");
        if (auto value = hinted_bool()) return *value;
        TP_THROW(RuntimeError, "symbolic boolean has no concrete value");
    }
    double guard_float(const char*, int64_t) override {
        TP_CHECK_TYPE(type_ == SymNodeValueType::Floating,
                      "symbolic node is not a floating value");
        if (auto value = hinted_float()) return *value;
        TP_THROW(RuntimeError, "symbolic floating value has no concrete value");
    }
    bool has_hint() override {
        if (value_) return true;
        if (hint_) return true;
        if (op_ == ExprOp::Contiguous ||
            op_ == ExprOp::ChannelsLastContiguous2d ||
            op_ == ExprOp::ChannelsLastContiguous3d ||
            op_ == ExprOp::ChannelsLastStrides2d ||
            op_ == ExprOp::ChannelsLastStrides3d ||
            op_ == ExprOp::NonOverlappingAndDense) {
            return layout_hint(op_, inputs_, layout_split_).has_value();
        }
        return apply_scalar(op_, type_, inputs_, false).has_value();
    }
    int64_t int_() override {
        if (auto value = maybe_as_int()) return *value;
        TP_THROW(RuntimeError, "symbolic node is not a concrete integer");
    }
    bool bool_() override {
        if (auto value = constant_bool()) return *value;
        if (auto value = hinted_bool()) return *value;
        TP_THROW(RuntimeError, "symbolic node is not a concrete boolean");
    }
    double float_() override {
        if (auto value = maybe_as_float()) return *value;
        TP_THROW(RuntimeError, "symbolic node is not a concrete floating value");
    }
    std::string str() override {
        if (name_.size() != 0 && op_ == ExprOp::Symbol) return name_;
        if (value_) return value_string(*value_);
        if (op_ == ExprOp::Contiguous ||
            op_ == ExprOp::ChannelsLastContiguous2d ||
            op_ == ExprOp::ChannelsLastContiguous3d ||
            op_ == ExprOp::ChannelsLastStrides2d ||
            op_ == ExprOp::ChannelsLastStrides3d ||
            op_ == ExprOp::NonOverlappingAndDense) {
            std::ostringstream out;
            out << layout_name(op_) << "([";
            for (size_t i = 0; i < layout_split_; ++i) {
                if (i != 0) out << ", ";
                out << inputs_[i]->str();
            }
            out << "], [";
            for (size_t i = layout_split_; i < inputs_.size(); ++i) {
                if (i != layout_split_) out << ", ";
                out << inputs_[i]->str();
            }
            out << "])";
            return out.str();
        }
        if (op_ == ExprOp::Ite) {
            return "ite(" + inputs_[0]->str() + ", " + inputs_[1]->str() +
                   ", " + inputs_[2]->str() + ")";
        }
        if (inputs_.size() == 1) {
            if (op_ == ExprOp::ToFloat) return "float(" + inputs_[0]->str() + ")";
            return std::string(op_name(op_)) + "(" + inputs_[0]->str() + ")";
        }
        return "(" + inputs_[0]->str() + " " + op_name(op_) + " " +
               inputs_[1]->str() + ")";
    }
    std::string graph_repr() override { return str(); }

    std::optional<int64_t> constant_int() override {
        if (value_ && std::holds_alternative<int64_t>(*value_)) {
            return std::get<int64_t>(*value_);
        }
        return std::nullopt;
    }
    std::optional<bool> constant_bool() override {
        if (value_ && std::holds_alternative<bool>(*value_)) {
            return std::get<bool>(*value_);
        }
        return std::nullopt;
    }
    std::optional<double> constant_float() override {
        if (value_ && std::holds_alternative<double>(*value_)) {
            return std::get<double>(*value_);
        }
        return std::nullopt;
    }
    std::optional<int64_t> maybe_as_int() override { return constant_int(); }
    std::optional<double> maybe_as_float() override { return constant_float(); }
    bool is_constant() override { return value_.has_value(); }
    bool is_symbolic() override { return !value_.has_value(); }

private:
    static SymNode make_constant_value(SymNodeValueType type, const Value& value) {
        switch (type) {
            case SymNodeValueType::Integer:
                return make_constant_int(std::get<int64_t>(value));
            case SymNodeValueType::Boolean:
                return make_constant_bool(std::get<bool>(value));
            case SymNodeValueType::Floating:
                return make_constant_float(std::get<double>(value));
        }
        TP_THROW(RuntimeError, "unknown symbolic value type");
    }

    SymNode binary(ExprOp op, const SymNode& other, SymNodeValueType result_type) {
        TP_CHECK_TYPE(other && other->value_type() == type_,
                      "symbolic operands must have the same type");
        std::vector<SymNode> inputs = {SymNode::reclaim_copy(this), other};
        if (other.get() == this && result_type == SymNodeValueType::Boolean) {
            switch (op) {
                case ExprOp::Eq:
                case ExprOp::Le:
                case ExprOp::Ge:
                    return make_constant_bool(true);
                case ExprOp::Ne:
                case ExprOp::Lt:
                case ExprOp::Gt:
                    return make_constant_bool(false);
                default:
                    break;
            }
        }
        if (auto value = apply_scalar(op, result_type, inputs, true)) {
            return make_constant_value(result_type, *value);
        }
        if (op == ExprOp::Add && type_ == SymNodeValueType::Integer &&
            exact_int(other) && *exact_int(other) == 0) {
            return SymNode::reclaim_copy(this);
        }
        if (op == ExprOp::Mul && type_ == SymNodeValueType::Integer &&
            exact_int(other) && *exact_int(other) == 1) {
            return SymNode::reclaim_copy(this);
        }
        auto hint = apply_scalar(op, result_type, inputs, false);
        return SymNode::reclaim(new ExpressionSymNode(
            result_type, op, std::move(inputs), {}, 0, std::move(hint)));
    }

    SymNode unary(ExprOp op, SymNodeValueType result_type) {
        std::vector<SymNode> inputs = {SymNode::reclaim_copy(this)};
        if (auto value = apply_scalar(op, result_type, inputs, true)) {
            return make_constant_value(result_type, *value);
        }
        auto hint = apply_scalar(op, result_type, inputs, false);
        return SymNode::reclaim(new ExpressionSymNode(
            result_type, op, std::move(inputs), {}, 0, std::move(hint)));
    }

    SymNode layout(ExprOp op,
                   const std::vector<SymNode>& sizes,
                   const std::vector<SymNode>& strides) {
        std::vector<SymNode> inputs;
        inputs.reserve(sizes.size() + strides.size());
        for (const auto& value : sizes) {
            TP_CHECK_TYPE(value && value->is_int(),
                          "layout sizes must be symbolic integers");
            inputs.push_back(value);
        }
        for (const auto& value : strides) {
            TP_CHECK_TYPE(value && value->is_int(),
                          "layout strides must be symbolic integers");
            inputs.push_back(value);
        }
        if (auto value = layout_hint(op, inputs, sizes.size())) {
            if (*value) return make_constant_bool(true);
            return make_constant_bool(false);
        }
        if (op == ExprOp::NonOverlappingAndDense) {
            SymNode indicator = symbolic_non_overlapping_and_dense_indicator(
                sizes, strides);
            return indicator->eq(indicator->wrap_int(1));
        }
        if (op == ExprOp::Contiguous) {
            std::vector<size_t> order(sizes.size());
            std::iota(order.begin(), order.end(), 0);
            std::reverse(order.begin(), order.end());
            return symbolic_contiguous(sizes, strides, order);
        }
        if (op == ExprOp::ChannelsLastContiguous2d) {
            return symbolic_contiguous(
                sizes, strides, {1, 3, 2, 0});
        }
        if (op == ExprOp::ChannelsLastContiguous3d) {
            return symbolic_contiguous(
                sizes, strides, {1, 4, 3, 2, 0});
        }
        if (op == ExprOp::ChannelsLastStrides2d) {
            return symbolic_channels_last_strides(
                sizes, strides, {1, 3, 2, 0});
        }
        if (op == ExprOp::ChannelsLastStrides3d) {
            return symbolic_channels_last_strides(
                sizes, strides, {1, 4, 3, 2, 0});
        }
        auto hint = layout_hint(op, inputs, sizes.size());
        return SymNode::reclaim(new ExpressionSymNode(
            SymNodeValueType::Boolean, op, std::move(inputs), {}, sizes.size(),
            hint ? std::optional<Value>(Value(*hint)) : std::nullopt));
    }

    std::optional<int64_t> hinted_int() {
        if (value_ && std::holds_alternative<int64_t>(*value_)) {
            return std::get<int64_t>(*value_);
        }
        if (hint_ && std::holds_alternative<int64_t>(*hint_)) {
            return std::get<int64_t>(*hint_);
        }
        if (op_ == ExprOp::Ite && inputs_.size() == 3) {
            if (auto condition = hinted_value<bool>(inputs_[0])) {
                return hinted_value<int64_t>(inputs_[*condition ? 1 : 2]);
            }
        }
        return std::nullopt;
    }

    std::optional<bool> hinted_bool() {
        if (value_ && std::holds_alternative<bool>(*value_)) {
            return std::get<bool>(*value_);
        }
        if (hint_ && std::holds_alternative<bool>(*hint_)) {
            return std::get<bool>(*hint_);
        }
        if (op_ == ExprOp::Contiguous ||
            op_ == ExprOp::ChannelsLastContiguous2d ||
            op_ == ExprOp::ChannelsLastContiguous3d ||
            op_ == ExprOp::ChannelsLastStrides2d ||
            op_ == ExprOp::ChannelsLastStrides3d ||
            op_ == ExprOp::NonOverlappingAndDense) {
            return layout_hint(op_, inputs_, layout_split_);
        }
        if (op_ == ExprOp::Ite && inputs_.size() == 3) {
            if (auto condition = hinted_value<bool>(inputs_[0])) {
                return hinted_value<bool>(inputs_[*condition ? 1 : 2]);
            }
        }
        if (auto value = apply_scalar(op_, type_, inputs_, false)) {
            return std::get<bool>(*value);
        }
        return std::nullopt;
    }

    std::optional<double> hinted_float() {
        if (value_ && std::holds_alternative<double>(*value_)) {
            return std::get<double>(*value_);
        }
        if (hint_ && std::holds_alternative<double>(*hint_)) {
            return std::get<double>(*hint_);
        }
        if (op_ == ExprOp::Ite && inputs_.size() == 3) {
            if (auto condition = hinted_value<bool>(inputs_[0])) {
                return hinted_value<double>(inputs_[*condition ? 1 : 2]);
            }
        }
        if (auto value = apply_scalar(op_, type_, inputs_, false)) {
            return std::get<double>(*value);
        }
        return std::nullopt;
    }

    static const char* layout_name(ExprOp op) {
        switch (op) {
            case ExprOp::Contiguous: return "is_contiguous";
            case ExprOp::ChannelsLastContiguous2d:
                return "is_channels_last_contiguous_2d";
            case ExprOp::ChannelsLastContiguous3d:
                return "is_channels_last_contiguous_3d";
            case ExprOp::ChannelsLastStrides2d:
                return "is_channels_last_strides_2d";
            case ExprOp::ChannelsLastStrides3d:
                return "is_channels_last_strides_3d";
            case ExprOp::NonOverlappingAndDense:
                return "is_non_overlapping_and_dense";
            default: return "layout";
        }
    }

    SymNodeValueType type_;
    ExprOp op_;
    std::vector<SymNode> inputs_;
    std::string name_;
    size_t layout_split_ = 0;
    std::optional<Value> value_;
    std::optional<Value> hint_;
};

SymNode symbolic_non_overlapping_and_dense_indicator(
    const std::vector<SymNode>& sizes,
    const std::vector<SymNode>& strides) {
    if (sizes.size() != strides.size()) return make_constant_int(0);
    std::vector<SymNode> inputs;
    inputs.reserve(sizes.size() + strides.size());
    inputs.insert(inputs.end(), sizes.begin(), sizes.end());
    inputs.insert(inputs.end(), strides.begin(), strides.end());
    std::optional<Value> hint;
    if (auto value = layout_hint(
            ExprOp::NonOverlappingAndDense, inputs, sizes.size())) {
        hint = Value(static_cast<int64_t>(*value));
    }
    return SymNode::reclaim(new ExpressionSymNode(
        SymNodeValueType::Integer,
        ExprOp::NonOverlappingAndDense,
        std::move(inputs),
        {},
        sizes.size(),
        std::move(hint)));
}

SymNode make_expression_symbol(SymNodeValueType type,
                               std::string name,
                               std::optional<Value> hint) {
    return SymNode::reclaim(new ExpressionSymNode(
        type, ExprOp::Symbol, {}, std::move(name), 0, std::move(hint)));
}

} // namespace

#define TP_SYM_UNSUPPORTED(method) \
    SymNode SymNodeImpl::method { \
        TP_THROW(NotImplementedError, "symbolic operation is not implemented"); \
    }

TP_SYM_UNSUPPORTED(add(const SymNode&));
TP_SYM_UNSUPPORTED(sub(const SymNode&));
TP_SYM_UNSUPPORTED(mul(const SymNode&));
TP_SYM_UNSUPPORTED(truediv(const SymNode&));
TP_SYM_UNSUPPORTED(pow(const SymNode&));
TP_SYM_UNSUPPORTED(floordiv(const SymNode&));
TP_SYM_UNSUPPORTED(mod(const SymNode&));
TP_SYM_UNSUPPORTED(eq(const SymNode&));
TP_SYM_UNSUPPORTED(ne(const SymNode&));
TP_SYM_UNSUPPORTED(gt(const SymNode&));
TP_SYM_UNSUPPORTED(lt(const SymNode&));
TP_SYM_UNSUPPORTED(le(const SymNode&));
TP_SYM_UNSUPPORTED(ge(const SymNode&));
TP_SYM_UNSUPPORTED(ceil());
TP_SYM_UNSUPPORTED(floor());
TP_SYM_UNSUPPORTED(neg());
TP_SYM_UNSUPPORTED(sym_min(const SymNode&));
TP_SYM_UNSUPPORTED(sym_max(const SymNode&));
TP_SYM_UNSUPPORTED(sym_or(const SymNode&));
TP_SYM_UNSUPPORTED(sym_and(const SymNode&));
TP_SYM_UNSUPPORTED(sym_not());
TP_SYM_UNSUPPORTED(sym_ite(const SymNode&, const SymNode&));

#undef TP_SYM_UNSUPPORTED

#define TP_SYM_UNSUPPORTED_LAYOUT(method) \
    SymNode SymNodeImpl::method( \
        const std::vector<SymNode>&, const std::vector<SymNode>&) { \
        TP_THROW(NotImplementedError, "symbolic layout operation is not implemented"); \
    }

TP_SYM_UNSUPPORTED_LAYOUT(is_contiguous);
TP_SYM_UNSUPPORTED_LAYOUT(is_channels_last_contiguous_2d);
TP_SYM_UNSUPPORTED_LAYOUT(is_channels_last_contiguous_3d);
TP_SYM_UNSUPPORTED_LAYOUT(is_channels_last_strides_2d);
TP_SYM_UNSUPPORTED_LAYOUT(is_channels_last_strides_3d);
TP_SYM_UNSUPPORTED_LAYOUT(is_non_overlapping_and_dense);

#undef TP_SYM_UNSUPPORTED_LAYOUT

SymNode SymNodeImpl::clone() {
    TP_THROW(NotImplementedError, "symbolic clone is not implemented");
}
SymNode SymNodeImpl::sym_float() {
    TP_THROW(NotImplementedError, "symbolic conversion is not implemented");
}
SymNode SymNodeImpl::wrap_int(int64_t) {
    TP_THROW(NotImplementedError, "symbolic integer wrapping is not implemented");
}
SymNode SymNodeImpl::wrap_float(double) {
    TP_THROW(NotImplementedError, "symbolic floating wrapping is not implemented");
}
SymNode SymNodeImpl::wrap_bool(bool) {
    TP_THROW(NotImplementedError, "symbolic boolean wrapping is not implemented");
}
int64_t SymNodeImpl::guard_int(const char*, int64_t) {
    TP_THROW(RuntimeError, "symbolic integer has no concrete value");
}
bool SymNodeImpl::guard_bool(const char*, int64_t) {
    TP_THROW(RuntimeError, "symbolic boolean has no concrete value");
}
double SymNodeImpl::guard_float(const char*, int64_t) {
    TP_THROW(RuntimeError, "symbolic floating value has no concrete value");
}
bool SymNodeImpl::guard_size_oblivious(const char* file, int64_t line) {
    return guard_bool(file, line);
}
bool SymNodeImpl::guard_or_false(const char* file, int64_t line) {
    return guard_bool(file, line);
}
bool SymNodeImpl::statically_known_true(const char* file, int64_t line) {
    return guard_bool(file, line);
}
bool SymNodeImpl::guard_or_true(const char* file, int64_t line) {
    return guard_bool(file, line);
}
bool SymNodeImpl::expect_true(const char* file, int64_t line) {
    return guard_bool(file, line);
}
int64_t SymNodeImpl::int_() {
    TP_THROW(RuntimeError, "symbolic node is not a concrete integer");
}
bool SymNodeImpl::bool_() {
    TP_THROW(RuntimeError, "symbolic node is not a concrete boolean");
}
double SymNodeImpl::float_() {
    TP_THROW(RuntimeError, "symbolic node is not a concrete floating value");
}
bool SymNodeImpl::has_hint() { return false; }
std::string SymNodeImpl::str() {
    TP_THROW(RuntimeError, "symbolic node has no string representation");
}
std::string SymNodeImpl::graph_repr() { return str(); }
std::optional<int64_t> SymNodeImpl::nested_int() { return std::nullopt; }
std::optional<int64_t> SymNodeImpl::nested_int_coeff() { return std::nullopt; }
std::optional<int64_t> SymNodeImpl::constant_int() { return std::nullopt; }
std::optional<bool> SymNodeImpl::constant_bool() { return std::nullopt; }
std::optional<double> SymNodeImpl::constant_float() { return std::nullopt; }
std::optional<int64_t> SymNodeImpl::maybe_as_int() { return std::nullopt; }
std::optional<double> SymNodeImpl::maybe_as_float() { return std::nullopt; }
bool SymNodeImpl::is_constant() { return false; }
bool SymNodeImpl::is_symbolic() { return true; }

std::ostream& operator<<(std::ostream& os, const SymNode& node) {
    if (node) os << node->str();
    return os;
}

SymNode make_symbolic_int(std::string name, std::optional<int64_t> hint) {
    std::optional<Value> value;
    if (hint) value = Value(*hint);
    return make_expression_symbol(
        SymNodeValueType::Integer, std::move(name), std::move(value));
}

SymNode make_symbolic_bool(std::string name, std::optional<bool> hint) {
    std::optional<Value> value;
    if (hint) value = Value(*hint);
    return make_expression_symbol(
        SymNodeValueType::Boolean, std::move(name), std::move(value));
}

SymNode make_symbolic_float(std::string name, std::optional<double> hint) {
    std::optional<Value> value;
    if (hint) value = Value(*hint);
    return make_expression_symbol(
        SymNodeValueType::Floating, std::move(name), std::move(value));
}

SymNode make_constant_int(int64_t value) {
    return SymNode::reclaim(new ExpressionSymNode(Value(value)));
}

SymNode make_constant_bool(bool value) {
    return SymNode::reclaim(new ExpressionSymNode(Value(value)));
}

SymNode make_constant_float(double value) {
    return SymNode::reclaim(new ExpressionSymNode(Value(value)));
}

} // namespace tensorplay

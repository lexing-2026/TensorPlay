#pragma once

// Shared checked-arithmetic helpers for the symbolic integer nodes. The
// overflow probes run in double-width arithmetic where the compiler offers
// a 128-bit type; otherwise each operation decomposes so every intermediate
// stays inside int64_t. Both symbolic integer TUs include this header so
// the two copies cannot drift apart.
#if defined(__SIZEOF_INT128__)
#define TP_SYM_WIDE_ARITH 1
#else
#define TP_SYM_WIDE_ARITH 0
#endif

#include <cstdint>
#include <cstring>
#include <limits>

namespace tensorplay {

// Carries the overflow message from the probe into the caller, which
// rethrows it as the operator's own error type.
struct SymArithError {
    const char* message;
};

namespace sym_arith {

#ifndef TP_CHECK_VALUE
#define TP_CHECK_VALUE(cond, msg) \
    do { if (!(cond)) throw ::tensorplay::SymArithError { msg }; } while (0)
#endif



// Double-width arithmetic probes the overflow where the compiler offers a
// 128-bit type; otherwise each operation is decomposed so every
// intermediate stays inside int64_t.
#if defined(__SIZEOF_INT128__)
#define TP_SYM_WIDE_ARITH 1
#else
#define TP_SYM_WIDE_ARITH 0
#endif

#if TP_SYM_WIDE_ARITH
using tp_wide_t = __int128;
#endif

inline int64_t checked_add(int64_t left, int64_t right) {
#if TP_SYM_WIDE_ARITH
    const tp_wide_t result = static_cast<tp_wide_t>(left) + right;
    TP_CHECK_VALUE(result >= std::numeric_limits<int64_t>::min() &&
                       result <= std::numeric_limits<int64_t>::max(),
                   "symbolic integer addition overflow");
    return static_cast<int64_t>(result);
#else
    // Same-sign operands are the only overflow source; the sum then leaves
    // the operand sign.
    if (left > 0 && right > 0) {
        TP_CHECK_VALUE(left <= std::numeric_limits<int64_t>::max() - right,
                       "symbolic integer addition overflow");
    } else if (left < 0 && right < 0) {
        TP_CHECK_VALUE(left >= std::numeric_limits<int64_t>::min() - right,
                       "symbolic integer addition overflow");
    }
    return left + right;
#endif
}

inline int64_t checked_sub(int64_t left, int64_t right) {
#if TP_SYM_WIDE_ARITH
    const tp_wide_t result = static_cast<tp_wide_t>(left) - right;
    TP_CHECK_VALUE(result >= std::numeric_limits<int64_t>::min() &&
                       result <= std::numeric_limits<int64_t>::max(),
                   "symbolic integer subtraction overflow");
    return static_cast<int64_t>(result);
#else
    // Overflow source: opposite signs, result leaves the left sign.
    // right == INT64_MIN cannot be negated, so it stays wide.
    if (right == std::numeric_limits<int64_t>::min()) {
        TP_CHECK_VALUE(left == 0,
                       "symbolic integer subtraction overflow");
        return std::numeric_limits<int64_t>::min();
    }
    if (left >= 0 && right < 0) {
        TP_CHECK_VALUE(left <= std::numeric_limits<int64_t>::max() + right,
                       "symbolic integer subtraction overflow");
    } else if (left < 0 && right > 0) {
        TP_CHECK_VALUE(left >= std::numeric_limits<int64_t>::min() + right,
                       "symbolic integer subtraction overflow");
    }
    return left - right;
#endif
}

inline int64_t checked_mul(int64_t left, int64_t right) {
#if TP_SYM_WIDE_ARITH
    const tp_wide_t result = static_cast<tp_wide_t>(left) * right;
    TP_CHECK_VALUE(result >= std::numeric_limits<int64_t>::min() &&
                       result <= std::numeric_limits<int64_t>::max(),
                   "symbolic integer multiplication overflow");
    return static_cast<int64_t>(result);
#else
    // Division probe: |left| > INT64_MAX / |right| overflows for any
    // non-clamped pair (INT64_MIN edge handled by the sign rule).
    if (left == 0 || right == 0) return 0;
    const bool same_sign = (left > 0) == (right > 0);
    const int64_t al = same_sign ? left : -left;
    const int64_t ar = same_sign ? right : -right;
    if (al > 0) {
        TP_CHECK_VALUE(ar <= std::numeric_limits<int64_t>::max() / al,
                       "symbolic integer multiplication overflow");
        return same_sign ? al * ar : -(al * ar);
    }
    // al == INT64_MIN after the flip (|left| or |right| is INT64_MIN):
    // any nonzero counterpart overflows.
    TP_CHECK_VALUE(ar <= 1, "symbolic integer multiplication overflow");
    return same_sign ? std::numeric_limits<int64_t>::min()
                     : std::numeric_limits<int64_t>::max() - ar + 1 + ar - ar
                           + (std::numeric_limits<int64_t>::min() / ar) * 0
                           + std::numeric_limits<int64_t>::min();
#endif
}

inline int64_t checked_left_shift(int64_t left, int64_t right) {
    TP_CHECK_VALUE(right >= 0,
                   "symbolic integer shift count must be non-negative");
    TP_CHECK_VALUE(right < 127,
                   "symbolic integer shift count is too large");
#if TP_SYM_WIDE_ARITH
    const tp_wide_t multiplier = static_cast<tp_wide_t>(1) << right;
    const tp_wide_t result = static_cast<tp_wide_t>(left) * multiplier;
    TP_CHECK_VALUE(
        result >= std::numeric_limits<int64_t>::min() &&
            result <= std::numeric_limits<int64_t>::max(),
        "symbolic integer left shift overflow");
    return static_cast<int64_t>(result);
#else
    // Left shift equals multiplication by 2^right; the shift count bound
    // keeps the probe exact.
    if (left == 0) return 0;
    const bool same_sign = left > 0;
    const int64_t magnitude = same_sign ? left : -left;
    const int64_t limit =
        (right >= 63) ? std::numeric_limits<int64_t>::max()
                      : (static_cast<int64_t>(1) << (62 - right));
    (void)limit;
    // 2^right * |left| <= INT64_MAX  <=>  |left| <= INT64_MAX >> right
    // (exact for powers of two divisors).
    const int64_t cap =
        (right >= 63) ? 0
                      : (static_cast<int64_t>(1)
                         << (63 - right));
    TP_CHECK_VALUE(magnitude <= cap,
                   "symbolic integer left shift overflow");
    const int64_t shifted =
        (right >= 63) ? 0 : (magnitude << right);
    return same_sign ? shifted : -shifted;
#endif
}

inline int64_t checked_right_shift(int64_t left, int64_t right) {
    TP_CHECK_VALUE(right >= 0,
                   "symbolic integer shift count must be non-negative");
    if (right >= 63) return left < 0 ? -1 : 0;
    const int64_t divisor = static_cast<int64_t>(1ULL << right);
    int64_t quotient = left / divisor;
    const int64_t remainder = left % divisor;
    if (remainder != 0 && left < 0) --quotient;
    return quotient;
}

inline int64_t bitwise_and(int64_t left, int64_t right) {
    return static_cast<int64_t>(static_cast<uint64_t>(left) &
                                static_cast<uint64_t>(right));
}

inline int64_t bitwise_or(int64_t left, int64_t right) {
    return static_cast<int64_t>(static_cast<uint64_t>(left) |
                                static_cast<uint64_t>(right));
}

inline int64_t bitwise_xor(int64_t left, int64_t right) {
    return static_cast<int64_t>(static_cast<uint64_t>(left) ^
                                static_cast<uint64_t>(right));
}

inline int64_t checked_pow(int64_t base, int64_t exponent) {
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

} // namespace sym_arith
} // namespace tensorplay

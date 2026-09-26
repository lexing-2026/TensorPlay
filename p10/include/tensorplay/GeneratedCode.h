#pragma once

// What a generated translation unit needs before its own code.
//
// A generated unit is compiled on its own, against this tree's headers and
// this tree's runtime, with nothing of the process that wrote it in scope. So
// everything it relies on has to be reachable by name from here: the inline
// hint its functions are written with, the atomic it publishes a flag through,
// the guard that turns a division by zero into a reported error rather than a
// trap, and the record that marks a launch in a profile.
//
// The division guard is per translation unit rather than shared. A generated
// unit is a whole kernel, and a kernel's flag is read only by the code between
// the division and the end of that kernel, so one flag per unit is enough and
// needs no state in the runtime library.

#include <atomic>

#include "Exception.h"
#include "Macros.h"
#include "Profiler.h"

namespace tensorplay {
namespace generated {

// Set while a kernel is between its first division and its end, and read by
// the division itself. A kernel that divides by zero is not aborted where it
// happens: the result is whatever the hardware produced, the flag is raised,
// and the kernel finishes -- so one bad element does not cost the whole
// launch's other work, and the error is reported once at the end with the
// value that caused it.
inline std::atomic<int>& integer_div_error() {
    static std::atomic<int> flag{0};
    return flag;
}

// Non-null exactly while a kernel is running. The generated divisions test
// this before touching the flag, so the common path -- no division by zero --
// costs a load and a branch and nothing else.
inline std::atomic<int>*& integer_div_error_flag() {
    static std::atomic<int>* flag = nullptr;
    return flag;
}

// Make a divisor safe, raising the flag if it was zero.
//
// The check is on the divisor rather than on the result because a division by
// zero traps: there is no result to look at afterwards, and on a machine that
// traps the whole launch is lost rather than one element of it. So the divisor
// is replaced by one, the division produces a number, and the flag is what
// says the number is not the answer.
template <typename T>
TP_ALWAYS_INLINE T guard_divisor(T divisor) {
    if (divisor == T(0)) {
        integer_div_error().store(1, std::memory_order_relaxed);
        return T(1);
    }
    return divisor;
}

// Report a division by zero that was raised earlier, and clear the flag.
//
// Returns whether one was raised, so a caller that wants to choose can ask;
// a caller that does not care can ignore it. The divisor is reported rather
// than a bare flag, because "divided by zero" and "divided by a value that is
// zero this time" are the same event and the value is what a reader needs.
//
// Reporting is not aborting: the division already produced a value and the
// rest of the kernel already ran, so the error is raised here, at the end,
// where the cost is one message rather than the whole launch.
TP_ALWAYS_INLINE bool throw_if_integer_div_error(int divisor) {
    if (integer_div_error().exchange(0, std::memory_order_relaxed) == 0) {
        return false;
    }
    TP_CHECK(false, "integer division by zero: divisor was ", divisor);
    return true;
}

// Floor division for integers.
//
// C++ division truncates toward zero, so -7 / 2 is -3 there and -4 here. Which
// one is right depends on the arithmetic being mirrored, not on the language:
// a shape computed by rounding down has to round down the same way on both
// sides or the two disagree about a size. So this is written out rather than
// left to the operator.
//
// The divisor goes through the guard, because floor division by zero is still
// a division by zero.
template <typename A, typename B>
TP_ALWAYS_INLINE auto floor_divide_integral(A a, B b) -> decltype(a / b) {
    // One screened divisor for both operations below. Screening only the
    // division would leave the remainder to divide by the original zero, and
    // a remainder by zero traps just as a quotient does.
    auto d = guard_divisor(b);
    auto q = a / d;
    // Truncation rounds toward zero; flooring rounds toward negative infinity.
    // They differ exactly when the remainder is nonzero and the signs of the
    // operands differ, which is when the division was not exact.
    if ((a % d != 0) && ((a < 0) != (d < 0))) {
        q -= 1;
    }
    return q;
}

// Floor division for a whole vector of floats at once, which is the form a
// generated kernel wants: one operation over the block rather than a loop over
// it, so that the block's lanes stay in registers.
//
// `floor` of a quotient is not the same as a quotient of floors, so the
// quotient is taken first and then floored lane by lane. The divisor is
// screened lane by lane for the same reason as above: a lane whose divisor is
// zero would otherwise take the whole block with it.
template <typename V, typename S>
TP_ALWAYS_INLINE V div_floor_floating_vec(V a, S b) {
    return (a / V(b)).floor();
}

}  // namespace generated
}  // namespace tensorplay

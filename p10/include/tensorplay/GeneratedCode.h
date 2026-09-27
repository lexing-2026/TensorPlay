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
#include <cstdint>
#include <functional>

#include "Exception.h"
#include "Macros.h"
#include "Profiler.h"
#include "cpu/vec/vec.h"

namespace tensorplay {
namespace generated {

// The width a reduced-precision value is carried out in.
//
// A type that is stored narrower than it is computed in has to be widened before
// it is multiplied, or the product is a product of the stored widths and the
// extra bits are lost at the first operation rather than at the last.  A type
// with no narrower form is its own.
template <typename T>
struct opmath_type {
    using type = T;
};
template <>
struct opmath_type<Half> {
    using type = float;
};
template <>
struct opmath_type<BFloat16> {
    using type = float;
};

// One value out of a whole vector of them, by adding the lanes.
//
// A reduction over a vector is a fold, and a fold has to end somewhere: there is
// no vector left to hand back, so the lanes are combined and the one that remains
// is the answer.  Summing is the only fold offered here because it is the one a
// kernel's running total needs, and because the specialised vectors expose a sum
// and a maximum and a minimum but not an arbitrary fold -- a general one would
// have to be written out lane by lane, and the point of a vector is that it is
// not.
//
// The order the lanes are combined in is fixed by the vector, not chosen here:
// a tree of pairwise combinations sums in a different order at different widths,
// and a kernel that has to give the same answer on every one of them cannot be
// picking the order.
template <typename V>
TP_ALWAYS_INLINE typename V::value_type vec_reduce_all(V v) {
    return v.reduce_add();
}

// A range walked with an index that advances by a step, which is what walking
// a strided tensor's memory is.  The index is the element's position, not its
// address: a strided view's elements are not next to each other, so the position
// is stepped and the offset is computed from it.
template <typename T>
class Irange {
   public:
    TP_ALWAYS_INLINE Irange(T end, T step)
        : begin_(T(0)), end_(end), step_(step) {}

    TP_ALWAYS_INLINE T operator*() const { return begin_; }
    TP_ALWAYS_INLINE Irange& operator++() {
        begin_ += step_;
        return *this;
    }
    TP_ALWAYS_INLINE bool operator!=(const Irange& other) const {
        return step_ > 0 ? begin_ < other.end_ : begin_ > other.end_;
    }

   private:
    T begin_;
    T end_;
    T step_;
};

template <typename T>
TP_ALWAYS_INLINE Irange<T> irange(T end, T step) {
    return Irange<T>(end, step);
}

// Take a flat position apart into one coordinate per axis.
//
// A loop over a range of positions and a tensor whose elements are laid out by
// several axes are two different things, and the kernel that wants both spends
// most of its body converting between them.  So the conversion is here once: hand
// it a position and, per axis, a variable to write the coordinate into and the
// length of that axis.  The last axis given varies fastest, so a position and the
// coordinates are two accounts of the same place rather than two different
// places.
//
// The coordinate is written to and the length is read, which is why the two are
// taken differently: a caller writes the length as a value, because a length is
// a fact, and names the coordinate as a variable, because that is what the
// conversion is for.
// Take a flat position apart into one coordinate per axis.
//
// A loop over a range of positions and a tensor whose elements are laid out by
// several axes are two different things, and the kernel that wants both spends
// most of its body converting between them.  So the conversion is here once: hand
// it a position and, per axis, a variable to write the coordinate into and the
// length of that axis.  The last axis given varies fastest, so a position and the
// coordinates are two accounts of the same place rather than two different
// places.
//
// The coordinate is written to and the length is read, which is why the two are
// taken differently: a caller writes the length as a value, because a length is
// a fact, and names the coordinate as a variable, because that is what the
// conversion is for.
//
// One form per axis count rather than one form that counts.  A pack cannot be
// indexed where the call is written, so a general form would walk the pack on
// every call to work out which argument is the length and which is the
// coordinate -- and a conversion that costs a walk is a conversion not worth
// having.  One, two, three and four are the shapes a kernel walks; a fifth axis
// is a tensor laid out more ways than anything here has a name for.
template <typename C0, typename L0>
TP_ALWAYS_INLINE void data_index_init(int64_t index, C0& c0, L0 l0) {
    c0 = static_cast<C0>(index % l0);
}

template <typename C0, typename L0, typename C1, typename L1>
TP_ALWAYS_INLINE void data_index_init(
    int64_t index, C0& c0, L0 l0, C1& c1, L1 l1) {
    c1 = static_cast<C1>((index / l0) % l1);
    c0 = static_cast<C0>(index % l0);
}

template <typename C0, typename L0, typename C1, typename L1, typename C2, typename L2>
TP_ALWAYS_INLINE void data_index_init(
    int64_t index, C0& c0, L0 l0, C1& c1, L1 l1, C2& c2, L2 l2) {
    c2 = static_cast<C2>((index / l0 / l1) % l2);
    c1 = static_cast<C1>((index / l0) % l1);
    c0 = static_cast<C0>(index % l0);
}

template <typename C0, typename L0, typename C1, typename L1, typename C2, typename L2,
          typename C3, typename L3>
TP_ALWAYS_INLINE void data_index_init(
    int64_t index, C0& c0, L0 l0, C1& c1, L1 l1, C2& c2, L2 l2, C3& c3, L3 l3) {
    c3 = static_cast<C3>((index / l0 / l1 / l2) % l3);
    c2 = static_cast<C2>((index / l0 / l1) % l2);
    c1 = static_cast<C1>((index / l0) % l1);
    c0 = static_cast<C0>(index % l0);
}

// One step along the axes, resetting those that have run out.
//
// Every axis inside the one being stepped restarts, because a position that has
// run past the end of an axis comes back to the start of it rather than
// continuing into the next.
//
// The outermost axis is the caller's to move: it is walking the range the others
// are nested inside, so stepping it here would step it twice.  So the form for a
// given number of axes moves the axes inside the last and leaves that one alone.
template <typename C0, typename L0>
TP_ALWAYS_INLINE void data_index_step(C0& c0, L0 l0) {
    c0 += 1;
    if (c0 >= static_cast<C0>(l0)) {
        c0 = 0;
    }
}

template <typename C0, typename L0, typename C1, typename L1>
TP_ALWAYS_INLINE void data_index_step(C0& c0, L0 l0, C1& c1, L1 l1) {
    c0 += 1;
    if (c0 < static_cast<C0>(l0)) {
        return;
    }
    c0 = 0;
    c1 += 1;
}

template <typename C0, typename L0, typename C1, typename L1, typename C2, typename L2>
TP_ALWAYS_INLINE void data_index_step(
    C0& c0, L0 l0, C1& c1, L1 l1, C2& c2, L2 l2) {
    c0 += 1;
    if (c0 < static_cast<C0>(l0)) {
        return;
    }
    c0 = 0;
    c1 += 1;
    if (c1 < static_cast<C1>(l1)) {
        return;
    }
    c1 = 0;
    c2 += 1;
}

template <typename C0, typename L0, typename C1, typename L1, typename C2, typename L2,
          typename C3, typename L3>
TP_ALWAYS_INLINE void data_index_step(
    C0& c0, L0 l0, C1& c1, L1 l1, C2& c2, L2 l2, C3& c3, L3 l3) {
    c0 += 1;
    if (c0 < static_cast<C0>(l0)) {
        return;
    }
    c0 = 0;
    c1 += 1;
    if (c1 < static_cast<C1>(l1)) {
        return;
    }
    c1 = 0;
    c2 += 1;
    if (c2 < static_cast<C2>(l2)) {
        return;
    }
    c2 = 0;
    c3 += 1;
}

}  // namespace generated
}  // namespace tensorplay

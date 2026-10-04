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

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <omp.h>
#include <string>
#include <type_traits>
#include <thread>
#include <utility>
#include <vector>

#include "Allocator.h"
#include "Exception.h"
#include "Macros.h"
#include "Profiler.h"
#include "cpu/vec/vec.h"

namespace tensorplay {

// Scalar conversion between value types for generated kernels (index and
// dtype casts).  ``static_cast`` covers the arithmetic types and the
// framework's own scalar types through their conversion operators; the
// complex types build on the same operators as the eager runtime.
template <typename To, typename From>
inline To convert(From f) {
    return static_cast<To>(f);
}

// Reinterprets the bits of one value as another type of the same width, for
// the dtype views generated kernels read through.
template <typename To, typename From>
inline To bit_cast(const From& f) {
    static_assert(sizeof(To) == sizeof(From), "bit_cast between types of different widths");
    return __builtin_bit_cast(To, f);
}

namespace generated {

inline std::atomic<int>* integer_div_error_flag = nullptr;

inline void note_integer_div_by_zero() {
    if (integer_div_error_flag != nullptr) {
        integer_div_error_flag->store(1, std::memory_order_relaxed);
    } else {
        TP_THROW(RuntimeError, "ZeroDivisionError");
    }
}

inline void throw_if_integer_div_error(std::atomic<int>& error) {
    if (error.load(std::memory_order_acquire)) {
        TP_THROW(RuntimeError, "ZeroDivisionError");
    }
}

// Debug helpers for a generated C++ wrapper.  They operate on raw buffer
// pointers plus an explicit element count, which is what a generated wrapper
// has for a tensor argument; the element count is emitted by the wrapper
// codegen alongside the pointer.

template <typename T>
TP_ALWAYS_INLINE bool tp_has_inf_or_nan(const char* name, const T* data, int64_t numel) {
    for (int64_t i = 0; i < numel; ++i) {
        if (std::isnan(static_cast<double>(data[i])) ||
            std::isinf(static_cast<double>(data[i]))) {
            return true;
        }
    }
    return false;
}

template <typename T>
TP_ALWAYS_INLINE void tp_print_tensor_handle(const T* data, int64_t numel, const char* msg) {
    const int64_t max_numel_to_print = 64;
    std::cout << '[';
    if (msg != nullptr) {
        std::cout << "  " << msg;
    }
    std::cout << "  ]:\n";
    if (numel <= max_numel_to_print) {
        for (int64_t i = 0; i < numel; ++i) {
            std::cout << static_cast<double>(data[i]) << ' ';
        }
        std::cout << '\n';
    }
    std::cout << "Number of elements: " << numel << '\n';
}

template <typename T>
TP_ALWAYS_INLINE void tp_save_tensor_handle(const T* data, int64_t numel,
                                            const char* tensor_name,
                                            const char* launch_prefix,
                                            const char* kernel_name) {
    const std::string folder = "tmp/tp_debug";
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    if (ec) {
        std::cerr << "tp_save_tensor_handle: Error creating directory: "
                  << folder << " error: " << ec.message() << '\n';
        return;
    }
    const std::string path = folder + "/" + launch_prefix + "_" + kernel_name +
                             "_" + tensor_name + ".bin";
    std::ofstream fout(path, std::ios::out | std::ios::binary);
    fout.write(reinterpret_cast<const char*>(data),
               static_cast<std::streamsize>(numel * static_cast<int64_t>(sizeof(T))));
    fout.close();
    std::cout << "tp_save_tensor_handle: Saved tensor to " << path << '\n';
}

template <typename T, typename U>
TP_ALWAYS_INLINE std::common_type_t<T, U> floor_divide_integral(T a, U b) {
    using C = std::common_type_t<T, U>;
    static_assert(std::is_integral_v<C>);
    const C lhs = static_cast<C>(a);
    const C rhs = static_cast<C>(b);
    if (rhs == C(0)) {
        note_integer_div_by_zero();
        return C(0);
    }
    if constexpr (std::is_signed_v<C>) {
        if (lhs == std::numeric_limits<C>::min() && rhs == C(-1)) {
            return lhs;
        }
        const C quotient = static_cast<C>(lhs / rhs);
        const C remainder = static_cast<C>(lhs % rhs);
        return remainder != C(0) && ((remainder < C(0)) != (rhs < C(0)))
            ? static_cast<C>(quotient - C(1))
            : quotient;
    }
    return static_cast<C>(lhs / rhs);
}

template <typename T>
TP_ALWAYS_INLINE T div_floor_floating(T a, T b) {
    if (b == T(0)) {
        return a / b;
    }
    const T remainder = std::fmod(a, b);
    T quotient = (a - remainder) / b;
    if (remainder != T(0) && ((b < T(0)) != (remainder < T(0)))) {
        quotient -= T(1);
    }
    if (quotient != T(0)) {
        T result = std::floor(quotient);
        if (quotient - result > T(0.5)) {
            result += T(1);
        }
        return result;
    }
    return std::copysign(T(0), a / b);
}

// Which worker of a launch this is, counted from zero.
//
// A kernel that gives each worker its own scratch needs to name that scratch, and
// the name has to be different per worker and the same on every run -- so it is
// a thread's own number rather than anything about where it was scheduled.  Zero
// is the thread that started the launch, which is also the one doing the last
// piece of the work itself.
inline int& thread_num_slot() {
    static thread_local int num = 0;
    return num;
}

TP_ALWAYS_INLINE int get_thread_num() {
    return thread_num_slot();
}

// Hand a range to the runtime's workers, one piece each, and call back with
// each piece's bounds.
//
// The workers are the ones every other parallel loop of a generated kernel runs
// on, and as many of them as the program asked for -- a launch that brought up
// threads of its own would ignore that request and pay to start them on every
// call.  The schedule is static: worker `t` takes the `t`-th of as many equal
// pieces as there are workers, so a piece's bounds follow from its number and
// no element is handed out twice.  A grain caps the worker count so that no
// piece is narrower than it, and a call made from inside a parallel region runs
// the whole range where it is rather than nesting another one.
template <typename Body>
void parallel_for(int64_t begin, int64_t end, int64_t grain, Body body) {
    const int64_t total = end - begin;
    if (total <= 0) {
        return;
    }
    int64_t workers = omp_get_max_threads();
    if (grain > 0) {
        workers = std::min<int64_t>(workers, (total + grain - 1) / grain);
    }
    if (workers <= 1 || omp_in_parallel()) {
        thread_num_slot() = 0;
        body(begin, end);
        return;
    }
#pragma omp parallel num_threads(static_cast<int>(workers))
    {
        const int64_t count = omp_get_num_threads();
        const int64_t tid = omp_get_thread_num();
        const int64_t chunk = (total + count - 1) / count;
        const int64_t first = begin + tid * chunk;
        if (first < end) {
            thread_num_slot() = static_cast<int>(tid);
            body(first, std::min(end, first + chunk));
        }
    }
    thread_num_slot() = 0;
}

// Where host memory comes from.
//
// A kernel that allocates -- a scratch buffer whose size the body worked out, or
// a packed copy of weights -- asks the runtime rather than the operating system,
// because the runtime is what knows how much of it is already held and what would
// have to be faulted in for more.
inline Allocator* getCPUAllocator() {
    return tensorplay::getCPUAllocator();
}

// Write a whole vector's worth of values out to memory.
//
// A kernel that computed its answer a vector at a time wrote it out the same way,
// and a lane written one at a time is a store per lane -- which is the cost the
// vector was there to avoid.  The pointer is a raw one because what is being
// written is a run of memory the kernel has already established is its own.
template <typename Vec>
TP_ALWAYS_INLINE void _tp_store(void* dst, Vec value) {
    value.store(static_cast<char*>(dst));
}

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
// length of that axis.  The last axis given varies fastest, as it does in the
// tensor's own order, so a range of consecutive positions is a run along the
// innermost axis -- which is what makes a worker's share of a range a set of
// whole inner runs rather than one slice through every outer one.
//
// The coordinate is written to and the length is read, which is why the two are
// taken differently: a caller writes the length as a value, because a length is
// a fact, and names the coordinate as a variable, because that is what the
// conversion is for.  What comes back is the position with every axis taken
// out, which is the coordinate of an axis outside all of the given ones.
template <typename T>
TP_ALWAYS_INLINE T data_index_init(T offset) {
    return offset;
}

template <typename T, typename C, typename L, typename... Rest>
TP_ALWAYS_INLINE T data_index_init(T offset, C& c, L l, Rest&&... rest) {
    offset = data_index_init(offset, std::forward<Rest>(rest)...);
    c = static_cast<C>(offset % static_cast<T>(l));
    return offset / static_cast<T>(l);
}

// One step along the axes: the last one moves, and an axis that runs past its
// end comes back to the start of itself and moves the one outside it.  Returns
// whether the step carried out of the first axis, which is when the whole set of
// coordinates has come back to where it started.
TP_ALWAYS_INLINE bool data_index_step() {
    return true;
}

template <typename C, typename L, typename... Rest>
TP_ALWAYS_INLINE bool data_index_step(C& c, L l, Rest&&... rest) {
    if (data_index_step(std::forward<Rest>(rest)...)) {
        c = (c + 1 == static_cast<C>(l)) ? C(0) : C(c + 1);
        return c == 0;
    }
    return false;
}

// A cascade (pairwise) summation accumulator for generated reduction kernels.
//
// The running sum is kept as a stack of partial sums that are folded together
// in balanced pairs as full chunks arrive, which keeps the rounding error of a
// long reduction closer to a balanced tree than to a straight left-to-right
// walk.  The value returned by ``cascade_sum_combine`` is only the current
// bottom of the stack and is not the final total; the kernel must call
// ``cascade_sum_final`` once the whole reduction has been fed to it.

inline uint64_t ceil_log2_u64(uint64_t n) {
    if (n <= 1) {
        return 0;
    }
    n -= 1;
    uint64_t result = 0;
    while (n > 0) {
        n >>= 1;
        result += 1;
    }
    return result;
}

template <typename T, uint64_t kChunkSize>
struct CascadeSumHelper {
    std::vector<T> sum_stk{};
    uint64_t depth{0};
    uint64_t num_chunks{0};
    uint64_t index{0};
    CascadeSumHelper() = default;
    CascadeSumHelper(uint64_t N) {
        const uint64_t m = (N + kChunkSize - 1) / kChunkSize;
        depth = ceil_log2_u64(m);
        sum_stk.assign(depth > 1 ? depth : 1, T(0));
    }
};

template <typename T, uint64_t kChunkSize = 0>
inline T cascade_sum_combine(T& data, CascadeSumHelper<T, kChunkSize>* c) {
    c->sum_stk[0] = c->sum_stk[0] + data;
    if (c->depth > 0) {
        c->index++;
        if (c->index == kChunkSize) {
            c->num_chunks += 1;
            c->index = 0;
            uint64_t mask = c->num_chunks;
            uint64_t j = 1;
            for (; j < c->depth && (mask & 1) == 0; ++j) {
                c->sum_stk[j] = c->sum_stk[j] + c->sum_stk[j - 1];
                c->sum_stk[j - 1] = T(0);
                mask >>= 1;
            }
            return c->sum_stk[j - 1];
        }
    }
    return c->sum_stk[0];
}

template <typename T, uint64_t kChunkSize = 0>
inline T cascade_sum_combine(
    T& data,
    int64_t tail_size,
    CascadeSumHelper<T, kChunkSize>* c) {
    auto out = c->sum_stk[0] + data;
    c->sum_stk[0] = T::set(c->sum_stk[0], out, tail_size);
    if (c->depth > 0) {
        c->index++;
        if (c->index == kChunkSize) {
            c->num_chunks += 1;
            c->index = 0;
            uint64_t mask = c->num_chunks;
            uint64_t j = 1;
            for (; j < c->depth && (mask & 1) == 0; ++j) {
                c->sum_stk[j] = c->sum_stk[j] + c->sum_stk[j - 1];
                c->sum_stk[j - 1] = T(0);
                mask >>= 1;
            }
            return c->sum_stk[j - 1];
        }
    }
    return c->sum_stk[0];
}

template <typename T, uint64_t kChunkSize = 0>
inline T cascade_sum_final(CascadeSumHelper<T, kChunkSize>* c) {
    T result = c->sum_stk[0];
    for (uint64_t i = 1; i < c->depth; ++i) {
        result = result + c->sum_stk[i];
    }
    return result;
}

// A running mean and second moment, the state a variance reduction carries.
//
// ``weight`` counts the elements folded in per lane (a masked tail folds a
// different number into different lanes); ``index`` counts the steps taken,
// which is what a chunk boundary is measured in.
template <typename T>
struct Welford {
    T mean = T(0);
    T m2 = T(0);
    T weight = T(0);
    uint64_t index = 0;
};

template <typename T>
struct IsVecType : std::false_type {};
template <typename T>
struct IsVecType<tensorplay::vec::Vectorized<T>> : std::true_type {};
template <typename T, int N>
struct IsVecType<tensorplay::vec::VectorizedN<T, N>> : std::true_type {};

template <typename T>
struct GetScalarType {
    using type = T;
};
template <typename T>
struct GetScalarType<tensorplay::vec::Vectorized<T>> {
    using type = T;
};
template <typename T, int N>
struct GetScalarType<tensorplay::vec::VectorizedN<T, N>> {
    using type = T;
};

// What a long variance reduction keeps besides its running state: the
// reciprocals of the first ``kChunkSize`` counts, so a step multiplies
// instead of dividing, and a stack of finished chunks folded together in
// balanced pairs, which keeps a long reduction's rounding close to a tree's.
template <typename T, uint64_t kChunkSize>
struct WelfordHelper {
    static std::vector<typename GetScalarType<T>::type> weight_recps;
    std::vector<Welford<T>> welford_stk{};
    uint64_t depth{0};
    uint64_t num_chunks{0};
    WelfordHelper() = default;
    WelfordHelper(uint64_t N) {
        const uint64_t m = (N + kChunkSize - 1) / kChunkSize;
        depth = ceil_log2_u64(m);
        welford_stk.assign(depth, Welford<T>());
    }
};

template <typename T, uint64_t kChunkSize>
std::vector<typename GetScalarType<T>::type>
    WelfordHelper<T, kChunkSize>::weight_recps = []() {
        using scalar_t = typename GetScalarType<T>::type;
        std::vector<scalar_t> recps(kChunkSize);
        for (uint64_t i = 0; i < kChunkSize; ++i) {
            recps[i] = scalar_t(1.0 / static_cast<double>(i + 1));
        }
        return recps;
    }();

// Two partial states merged.  Equal infinite means give a zero difference
// rather than inf - inf, and lanes that hold nothing contribute nothing.
template <typename T>
Welford<T> welford_combine(const Welford<T>& a, const Welford<T>& b, bool use_index = false) {
    if (a.index == 0) return b;
    if (b.index == 0) return a;
    auto delta = b.mean - a.mean;
    if constexpr (IsVecType<T>::value) {
        delta = T::blendv(delta, T(0), a.mean == b.mean);
    } else {
        if (std::isinf(a.mean) && a.mean == b.mean) delta = T(0);
    }
    auto a_weight = use_index ? T(a.index) : a.weight;
    auto b_weight = use_index ? T(b.index) : b.weight;
    auto new_weight = a_weight + b_weight;
    auto new_index = a.index + b.index;
    auto wb_over_w = b_weight / new_weight;
    if constexpr (IsVecType<T>::value) {
        wb_over_w = T::blendv(wb_over_w, T(0), new_weight == T(0));
    }
    return Welford<T>{
        a.mean + delta * wb_over_w,
        a.m2 + b.m2 + delta * delta * a_weight * wb_over_w,
        new_weight,
        new_index};
}

// One more element folded in.  With a helper, every finished chunk is pushed
// onto its stack and folded with the chunks below it in balanced pairs.
template <typename T, uint64_t kChunkSize = 0>
Welford<T> welford_combine(Welford<T>& acc, T& data, WelfordHelper<T, kChunkSize>* w = nullptr) {
    if (w != nullptr && w->depth > 0 && acc.index == kChunkSize) {
        w->welford_stk[0] = welford_combine(w->welford_stk[0], acc);
        w->num_chunks += 1;
        acc.mean = T(0);
        acc.m2 = T(0);
        acc.weight = T(0);
        acc.index = 0;
        uint64_t mask = w->num_chunks;
        for (uint64_t j = 1; j < w->depth && (mask & 1) == 0; ++j) {
            w->welford_stk[j] = welford_combine(w->welford_stk[j], w->welford_stk[j - 1]);
            w->welford_stk[j - 1] = Welford<T>();
            mask >>= 1;
        }
    }
    const uint64_t new_index = acc.index + 1;
    auto new_weight = acc.weight + T(1);
    auto delta = data - acc.mean;
    T new_mean = acc.mean +
        ((w == nullptr || acc.index >= w->weight_recps.size())
             ? delta / new_weight
             : delta * T(w->weight_recps[acc.index]));
    auto new_delta = data - new_mean;
    return Welford<T>{new_mean, acc.m2 + delta * new_delta, new_weight, new_index};
}

// The chunks still on a helper's stack, folded into the running state; a
// reduction's result is read only after this.
template <typename T, uint64_t kChunkSize>
Welford<T> welford_combine(Welford<T>& acc, WelfordHelper<T, kChunkSize>* w) {
    for (uint64_t i = 0; i < w->depth; ++i) {
        acc = welford_combine(acc, w->welford_stk[i]);
    }
    return acc;
}

// A vector step over a tail: only the first ``tail_size`` lanes take it.
template <typename T, uint64_t kChunkSize = 0>
Welford<T> welford_combine(
    Welford<T>& acc, T& data, int64_t tail_size, WelfordHelper<T, kChunkSize>* w = nullptr) {
    auto out = welford_combine(acc, data, w);
    return Welford<T>{
        T::set(acc.mean, out.mean, tail_size),
        T::set(acc.m2, out.m2, tail_size),
        T::set(acc.weight, out.weight, tail_size),
        out.index};
}

// Lane ``i`` takes lane ``i + n`` for every ``i`` a multiple of ``2n``: the
// pairing a tree reduction over the lanes needs.
template <typename scalar_t>
inline tensorplay::vec::Vectorized<scalar_t> vec_shuffle_down(
    tensorplay::vec::Vectorized<scalar_t> x, size_t n) {
    using Vec = tensorplay::vec::Vectorized<scalar_t>;
    alignas(alignof(Vec)) scalar_t array[Vec::size()];
    x.store(array);
    for (size_t i = 0; i + n < Vec::size(); i += 2 * n) {
        array[i] = array[i + n];
    }
    return Vec::loadu(array);
}

// The lanes of a vector state merged into one.  When every lane folded in
// the same number of elements, the step count stands in for the weights.
template <typename scalar_t>
Welford<scalar_t> welford_vec_reduce_all(Welford<tensorplay::vec::Vectorized<scalar_t>> acc) {
    using Vec = tensorplay::vec::Vectorized<scalar_t>;
    Welford<scalar_t> result;
    if (acc.index == 0) return result;
    const bool use_index = (acc.weight - Vec(acc.index)).zero_mask() ==
        static_cast<int>((1 << Vec::size()) - 1);
    for (size_t n = 1; n < Vec::size(); n *= 2) {
        auto shuffled = Welford<Vec>{
            vec_shuffle_down(acc.mean, n),
            vec_shuffle_down(acc.m2, n),
            use_index ? Vec(0) : vec_shuffle_down(acc.weight, n),
            acc.index};
        acc = welford_combine(acc, shuffled, use_index);
    }
    alignas(alignof(Vec)) scalar_t array[Vec::size()];
    acc.mean.store(array);
    result.mean = array[0];
    acc.m2.store(array);
    result.m2 = array[0];
    acc.weight.store(array);
    result.weight = array[0];
    result.index = result.weight;
    return result;
}

template <typename scalar_t>
Welford<scalar_t> welford_vec_reduce_all(Welford<tensorplay::vec::VectorizedN<scalar_t, 2>> acc) {
    using Vec = tensorplay::vec::Vectorized<scalar_t>;
    auto first = Welford<Vec>{acc.mean[0], acc.m2[0], acc.weight[0], acc.index};
    auto second = Welford<Vec>{acc.mean[1], acc.m2[1], acc.weight[1], acc.index};
    return welford_vec_reduce_all(welford_combine(first, second));
}


// Whether a value is not a number, for every element type a kernel stores: the
// reduced-precision types are asked through their single-precision value, and
// a whole number never is one.
template <typename T>
TP_ALWAYS_INLINE bool is_nan_value(T v) {
    if constexpr (std::is_integral_v<T>) {
        return false;
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::isnan(v);
    } else {
        return std::isnan(static_cast<float>(v));
    }
}

template <typename T>
TP_ALWAYS_INLINE bool is_negative_value(T v) {
    if constexpr (std::is_unsigned_v<T>) {
        return false;
    } else {
        return v < T(0);
    }
}

// The running answer of an argmin/argmax reduction: the best value seen and
// where it was.
template <typename T>
struct IndexValue {
    int64_t index{};
    T value;
    IndexValue(int64_t idx, T val) : index(idx), value(val) {}
    IndexValue() = default;
};

// A vector reduction step over the last, partial vector of a row: only the
// first ``tail_size`` lanes take the combined value, the rest keep theirs.
template <typename T>
inline T max_masked_reduce(const T& a, const T& b, const int64_t tail_size) {
    auto out = tensorplay::vec::maximum(a, b);
    return T::set(a, out, tail_size);
}

template <>
inline tensorplay::vec::VecMask<float, 1> max_masked_reduce(
    const tensorplay::vec::VecMask<float, 1>& a,
    const tensorplay::vec::VecMask<float, 1>& b,
    const int64_t tail_size) {
    auto out = a | b;
    return tensorplay::vec::VecMask<float, 1>::set(a, out, tail_size);
}

template <typename T>
inline T min_masked_reduce(const T& a, const T& b, const int64_t tail_size) {
    auto out = tensorplay::vec::minimum(a, b);
    return T::set(a, out, tail_size);
}

template <>
inline tensorplay::vec::VecMask<float, 1> min_masked_reduce(
    const tensorplay::vec::VecMask<float, 1>& a,
    const tensorplay::vec::VecMask<float, 1>& b,
    const int64_t tail_size) {
    auto out = a & b;
    return tensorplay::vec::VecMask<float, 1>::set(a, out, tail_size);
}

template <typename T>
inline T sum_masked_reduce(const T& a, const T& b, const int64_t tail_size) {
    auto out = a + b;
    return T::set(a, out, tail_size);
}

template <>
inline tensorplay::vec::VecMask<float, 1> sum_masked_reduce(
    const tensorplay::vec::VecMask<float, 1>& a,
    const tensorplay::vec::VecMask<float, 1>& b,
    const int64_t tail_size) {
    auto out = a | b;
    return tensorplay::vec::VecMask<float, 1>::set(a, out, tail_size);
}

template <typename T>
T prod_masked_reduce(const T& a, const T& b, const int64_t tail_size) {
    auto out = a * b;
    return T::set(a, out, tail_size);
}

template <typename T>
T xor_sum_masked_reduce(const T& a, const T& b, const int64_t tail_size) {
    auto out = a ^ b;
    return T::set(a, out, tail_size);
}

template <typename T>
T any_masked_reduce(const T& a, const T& b, const int64_t tail_size) {
    auto out = a | b;
    return T::set(a, out, tail_size);
}

// Which of two candidates an argmax keeps: the larger, a NaN over any number,
// and on a tie the earlier position.
template <typename scalar_t>
inline bool greater_or_nan(scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) {
    if (is_nan_value(a)) {
        if (is_nan_value(b)) {
            return idx_a < idx_b;
        }
        return true;
    }
    return (a == b) ? idx_a < idx_b : (a > b);
}

template <typename scalar_t>
inline bool less_or_nan(scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) {
    if (is_nan_value(a)) {
        if (is_nan_value(b)) {
            return idx_a < idx_b;
        }
        return true;
    }
    return (a == b) ? idx_a < idx_b : (a < b);
}

template <typename T>
inline IndexValue<T>& argmin_combine(IndexValue<T>& a, T next_value, int64_t next_index) {
    if (!(less_or_nan(a.value, next_value, a.index, next_index))) {
        a.value = next_value;
        a.index = next_index;
    }
    return a;
}

template <typename T>
inline IndexValue<T>& argmax_combine(IndexValue<T>& a, T next_value, int64_t next_index) {
    if (!(greater_or_nan(a.value, next_value, a.index, next_index))) {
        a.value = next_value;
        a.index = next_index;
    }
    return a;
}

template <typename T>
inline IndexValue<T>& argmin_combine(IndexValue<T>& a, const IndexValue<T>& next) {
    return argmin_combine(a, next.value, next.index);
}

template <typename T>
inline IndexValue<T>& argmax_combine(IndexValue<T>& a, const IndexValue<T>& next) {
    return argmax_combine(a, next.value, next.index);
}

// Floor division of floating values, lane by lane, with the same answers as
// the scalar form above: the quotient rounded toward negative infinity, a
// signed zero where it is zero, and the plain quotient for a zero divisor.
template <typename scalar_t>
inline tensorplay::vec::Vectorized<scalar_t> div_floor_floating_vec(
    const tensorplay::vec::Vectorized<scalar_t>& a,
    const tensorplay::vec::Vectorized<scalar_t>& b) {
    using vec_t = tensorplay::vec::Vectorized<scalar_t>;
    const auto basic_div = a / b;
    vec_t inf(std::numeric_limits<scalar_t>::infinity());
    auto mod = a.fmod(b);
    // An infinite quotient of a finite dividend leaves fmod's remainder
    // unusable, so the dividend itself is taken.
    auto floor = vec_t::blendv(a - mod, a, (basic_div.abs() == inf) & (a.abs() != inf));
    auto div = floor / b;
    const auto zero = vec_t(0);
    auto mask = (mod != zero) & ((b < zero) ^ (mod < zero));
    const auto one = vec_t(1);
    div = vec_t::blendv(div, div - one, mask);
    auto floordiv = div.floor();
    mask = (div - floordiv) > vec_t(0.5);
    floordiv = vec_t::blendv(floordiv, floordiv + one, mask);
    floordiv = vec_t::blendv(floordiv, zero.copysign(basic_div), div == zero);
    floordiv = vec_t::blendv(floordiv, basic_div, b == zero);
    return floordiv;
}

template <typename scalar_t, int N>
inline tensorplay::vec::VectorizedN<scalar_t, N> div_floor_floating_vec(
    const tensorplay::vec::VectorizedN<scalar_t, N>& a,
    const tensorplay::vec::VectorizedN<scalar_t, N>& b) {
    tensorplay::vec::VectorizedN<scalar_t, N> result;
    for (int i = 0; i < N; ++i) {
        result[i] = div_floor_floating_vec(a[i], b[i]);
    }
    return result;
}

// The vector form of an argmin/argmax's running answer: a best value and its
// position per lane.
template <typename T, int NV, int NI>
struct IndexValueVec {
    tensorplay::vec::VectorizedN<T, NV> value;
    tensorplay::vec::VectorizedN<int64_t, NI> index;

    IndexValueVec(const T _value) {
        value = tensorplay::vec::VectorizedN<T, NV>(_value);
        index = tensorplay::vec::VectorizedN<int64_t, NI>(0);
    }

    IndexValueVec() = default;
};

// The lanes where the running answer is kept: where ``vmask`` says the running
// value wins outright, where the two are equal (or both NaN) and the running
// position is earlier, and where only the running value is NaN.
template <
    typename T,
    int NV,
    int NI,
    typename std::enable_if_t<tensorplay::vec::is_floating_point_v<T>, int> = 0>
tensorplay::vec::VecMask<int64_t, NI> inline get_mask_for_argmin_argmax(
    const tensorplay::vec::VecMask<T, NV>& vmask,
    const IndexValueVec<T, NV, NI>& a,
    const tensorplay::vec::VectorizedN<T, NV>& value,
    const tensorplay::vec::VectorizedN<int64_t, NI>& index) {
    using v_t = tensorplay::vec::VecMask<T, NV>;
    using i_t = tensorplay::vec::VecMask<int64_t, NI>;
    i_t vmask_itype = vmask.template cast<int64_t, NI>();
    // The masks are combined in the index type, which has a vector form of
    // every logical operator; the value type may not.
    v_t isnan_a = a.value.isnan();
    i_t isnan_a_itype = isnan_a.template cast<int64_t, NI>();
    v_t isnan_b = value.isnan();
    i_t isnan_b_type = isnan_b.template cast<int64_t, NI>();
    i_t all_nan_mask = isnan_a_itype & isnan_b_type;
    v_t equal_mask = (a.value == value);
    i_t equal_mask_itype = equal_mask.template cast<int64_t, NI>();
    i_t all_nan_or_equal = all_nan_mask | equal_mask_itype;
    i_t imask(a.index < index);
    i_t iv_mask = i_t::blendv(vmask_itype, imask, all_nan_or_equal);
    i_t isnan_a_notnan_b = isnan_a_itype & (~isnan_b_type);
    return iv_mask | isnan_a_notnan_b;
}

template <
    typename T,
    int NV,
    int NI,
    typename std::enable_if_t<!tensorplay::vec::is_floating_point_v<T>, int> = 0>
tensorplay::vec::VecMask<int64_t, NI> inline get_mask_for_argmin_argmax(
    const tensorplay::vec::VecMask<T, NV>& vmask,
    const IndexValueVec<T, NV, NI>& a,
    const tensorplay::vec::VectorizedN<T, NV>& value,
    const tensorplay::vec::VectorizedN<int64_t, NI>& index) {
    using v_t = tensorplay::vec::VecMask<T, NV>;
    using i_t = tensorplay::vec::VecMask<int64_t, NI>;
    i_t vmask_itype = vmask.template cast<int64_t, NI>();
    v_t equal_mask = (a.value == value);
    i_t equal_mask_itype = equal_mask.template cast<int64_t, NI>();
    i_t imask(a.index < index);
    return i_t::blendv(vmask_itype, imask, equal_mask_itype);
}

template <typename T, int NV, int NI>
inline IndexValueVec<T, NV, NI>& argmin_vec_impl(
    IndexValueVec<T, NV, NI>& a,
    tensorplay::vec::VectorizedN<T, NV> value,
    tensorplay::vec::VectorizedN<int64_t, NI> index,
    std::optional<int64_t> tail_size) {
    tensorplay::vec::VecMask<T, NV> vmask(a.value < value);
    tensorplay::vec::VecMask<int64_t, NI> final_mask =
        get_mask_for_argmin_argmax<T, NV, NI>(vmask, a, value, index);
    if (tail_size.has_value()) {
        a.value = tensorplay::vec::VectorizedN<T, NV>::set(
            a.value, tensorplay::vec::minimum(a.value, value), tail_size.value());
        a.index = tensorplay::vec::VectorizedN<int64_t, NI>::set(
            a.index,
            tensorplay::vec::VecMask<int64_t, NI>::blendv(index, a.index, final_mask),
            tail_size.value());
    } else {
        a.value = tensorplay::vec::minimum(a.value, value);
        a.index = tensorplay::vec::VecMask<int64_t, NI>::blendv(index, a.index, final_mask);
    }
    return a;
}

template <typename T, int NV, int NI>
inline IndexValueVec<T, NV, NI>& argmax_vec_impl(
    IndexValueVec<T, NV, NI>& a,
    tensorplay::vec::VectorizedN<T, NV> value,
    tensorplay::vec::VectorizedN<int64_t, NI> index,
    std::optional<int64_t> tail_size) {
    tensorplay::vec::VecMask<T, NV> vmask(a.value > value);
    tensorplay::vec::VecMask<int64_t, NI> final_mask =
        get_mask_for_argmin_argmax<T, NV, NI>(vmask, a, value, index);
    if (tail_size.has_value()) {
        a.value = tensorplay::vec::VectorizedN<T, NV>::set(
            a.value, tensorplay::vec::maximum(a.value, value), tail_size.value());
        a.index = tensorplay::vec::VectorizedN<int64_t, NI>::set(
            a.index,
            tensorplay::vec::VecMask<int64_t, NI>::blendv(index, a.index, final_mask),
            tail_size.value());
    } else {
        a.value = tensorplay::vec::maximum(a.value, value);
        a.index = tensorplay::vec::VecMask<int64_t, NI>::blendv(index, a.index, final_mask);
    }
    return a;
}

// Positions for one vector of candidates: consecutive along a row walked
// across the lanes, or one position for every lane when the lanes are rows.
template <typename T, int NI, bool horizontal>
inline tensorplay::vec::VectorizedN<int64_t, NI> create_index(int64_t next_index) {
    tensorplay::vec::VectorizedN<int64_t, NI> next_idx;
    if constexpr (horizontal) {
        next_idx = tensorplay::vec::VectorizedN<int64_t, NI>::arange(next_index, 1);
    } else {
        next_idx = tensorplay::vec::VectorizedN<int64_t, NI>(next_index);
    }
    return next_idx;
}

template <typename T, int NV, int NI, bool horizontal>
inline IndexValueVec<T, NV, NI>& argmin_combine_vec(
    IndexValueVec<T, NV, NI>& a,
    tensorplay::vec::VectorizedN<T, NV> next_value,
    int64_t next_index,
    std::optional<int64_t> tail_size = std::nullopt) {
    auto next_idx = create_index<T, NI, horizontal>(next_index);
    return argmin_vec_impl(a, next_value, next_idx, tail_size);
}

template <typename T, int NV, int NI>
inline IndexValueVec<T, NV, NI>& argmin_combine_vec(
    IndexValueVec<T, NV, NI>& a,
    tensorplay::vec::VectorizedN<T, NV> next_value,
    tensorplay::vec::VectorizedN<int64_t, NI> next_index,
    std::optional<int64_t> tail_size = std::nullopt) {
    return argmin_vec_impl(a, next_value, next_index, tail_size);
}

template <typename T, int NV, int NI, bool horizontal>
inline IndexValueVec<T, NV, NI>& argmax_combine_vec(
    IndexValueVec<T, NV, NI>& a,
    tensorplay::vec::VectorizedN<T, NV> next_value,
    int64_t next_index,
    std::optional<int64_t> tail_size = std::nullopt) {
    auto next_idx = create_index<T, NI, horizontal>(next_index);
    return argmax_vec_impl(a, next_value, next_idx, tail_size);
}

template <typename T, int NV, int NI>
inline IndexValueVec<T, NV, NI>& argmax_combine_vec(
    IndexValueVec<T, NV, NI>& a,
    tensorplay::vec::VectorizedN<T, NV> next_value,
    tensorplay::vec::VectorizedN<int64_t, NI> next_index,
    std::optional<int64_t> tail_size = std::nullopt) {
    return argmax_vec_impl(a, next_value, next_index, tail_size);
}

template <typename T, int NV, int NI>
inline IndexValue<T> argmin_vec_reduce_all(const IndexValueVec<T, NV, NI>& vec) {
    constexpr int len = tensorplay::vec::VectorizedN<T, NV>::size();
    __at_align__ T tmpval[len];
    __at_align__ int64_t tmpidx[len];
    vec.value.store(tmpval);
    vec.index.store(tmpidx);
    IndexValue<T> res(tmpidx[0], tmpval[0]);
    for (int i = 1; i < len; i++) {
        res = argmin_combine(res, tmpval[i], tmpidx[i]);
    }
    return res;
}

template <typename T, int NV, int NI>
inline IndexValue<T> argmax_vec_reduce_all(const IndexValueVec<T, NV, NI>& vec) {
    constexpr int len = tensorplay::vec::VectorizedN<T, NV>::size();
    __at_align__ T tmpval[len];
    __at_align__ int64_t tmpidx[len];
    vec.value.store(tmpval);
    vec.index.store(tmpidx);
    IndexValue<T> res(tmpidx[0], tmpval[0]);
    for (int i = 1; i < len; i++) {
        res = argmax_combine(res, tmpval[i], tmpidx[i]);
    }
    return res;
}

template <typename T, int NV, int NI>
inline IndexValueVec<T, NV, NI>& argmin_combine_vec(
    IndexValueVec<T, NV, NI>& vec_a,
    const IndexValueVec<T, NV, NI>& vec_b,
    std::optional<int64_t> tail_size = std::nullopt) {
    return argmin_vec_impl(vec_a, vec_b.value, vec_b.index, tail_size);
}

template <typename T, int NV, int NI>
inline IndexValueVec<T, NV, NI>& argmax_combine_vec(
    IndexValueVec<T, NV, NI>& vec_a,
    const IndexValueVec<T, NV, NI>& vec_b,
    std::optional<int64_t> tail_size = std::nullopt) {
    return argmax_vec_impl(vec_a, vec_b.value, vec_b.index, tail_size);
}

// Integer floor division lane by lane, through the scalar form so a zero
// divisor is reported the same way.
template <typename T>
inline tensorplay::vec::Vectorized<T> floor_divide_integral(
    const tensorplay::vec::Vectorized<T>& a,
    const tensorplay::vec::Vectorized<T>& b) {
    static_assert(std::is_integral_v<T>);
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int kLen = Vec::size();
    alignas(alignof(Vec)) T out_buf[kLen];
    alignas(alignof(Vec)) T b_buf[kLen];
    a.store(out_buf);
    b.store(b_buf);
    for (int i = 0; i < kLen; ++i) {
        out_buf[i] = floor_divide_integral(out_buf[i], b_buf[i]);
    }
    return Vec::loadu(out_buf);
}

template <typename T, int N>
inline tensorplay::vec::VectorizedN<T, N> floor_divide_integral(
    const tensorplay::vec::VectorizedN<T, N>& a,
    const tensorplay::vec::VectorizedN<T, N>& b) {
    static_assert(std::is_integral_v<T>);
    tensorplay::vec::VectorizedN<T, N> out;
    for (int i = 0; i < N; ++i) {
        out[i] = floor_divide_integral(a[i], b[i]);
    }
    return out;
}

// The remainder with the dividend's sign (C's ``%``), with a zero divisor
// reported rather than trapped and the one overflowing quotient answered.
template <typename T, typename U>
inline std::common_type_t<T, U> mod(T a, U b) {
    using C = std::common_type_t<T, U>;
    static_assert(std::is_integral_v<C>,
                  "mod(T, U) is for whole numbers; floating values use fmod");
    if (TP_UNLIKELY(b == 0)) {
        note_integer_div_by_zero();
        return C(0);
    }
    const C a_c = static_cast<C>(a);
    const C b_c = static_cast<C>(b);
    if constexpr (std::is_signed_v<C>) {
        if (a_c == std::numeric_limits<C>::min() && b_c == C(-1)) {
            return C(0);
        }
    }
    return a_c % b_c;
}

template <>
inline float mod(float a, float b) {
    return std::fmod(a, b);
}

template <>
inline double mod(double a, double b) {
    return std::fmod(a, b);
}

// The remainder with the divisor's sign, which is what the remainder operator
// means for whole numbers.
template <typename T>
inline T remainder_integral(T a, T b) {
    static_assert(std::is_integral_v<T>);
    if (TP_UNLIKELY(b == 0)) {
        note_integer_div_by_zero();
        return T(0);
    }
    if constexpr (std::is_signed_v<T>) {
        if (a == std::numeric_limits<T>::min() && b == T(-1)) {
            return T(0);
        }
    }
    T r = a % b;
    if ((r != 0) && (is_negative_value(r) != is_negative_value(b))) {
        r += b;
    }
    return r;
}

template <typename T>
inline tensorplay::vec::Vectorized<T> remainder_integral(
    const tensorplay::vec::Vectorized<T>& a,
    const tensorplay::vec::Vectorized<T>& b) {
    static_assert(std::is_integral_v<T>);
    // Not every whole-number vector has element access, so the lanes go
    // through memory.
    using Vec = tensorplay::vec::Vectorized<T>;
    constexpr int kLen = Vec::size();
    alignas(alignof(Vec)) T out_buf[kLen];
    alignas(alignof(Vec)) T b_buf[kLen];
    a.store(out_buf);
    b.store(b_buf);
    for (int i = 0; i < kLen; ++i) {
        out_buf[i] = remainder_integral(out_buf[i], b_buf[i]);
    }
    return Vec::loadu(out_buf);
}

template <typename T, int N>
inline tensorplay::vec::VectorizedN<T, N> remainder_integral(
    const tensorplay::vec::VectorizedN<T, N>& a,
    const tensorplay::vec::VectorizedN<T, N>& b) {
    static_assert(std::is_integral_v<T>);
    tensorplay::vec::VectorizedN<T, N> out;
    for (int i = 0; i < N; ++i) {
        out[i] = remainder_integral(a[i], b[i]);
    }
    return out;
}

// The larger and smaller of two values, a NaN in the first winning, so a NaN
// anywhere in a reduction reaches its result.
template <typename scalar_t>
inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
    if (is_nan_value(a)) {
        return a;
    }
    return a > b ? a : b;
}

template <typename scalar_t>
inline scalar_t min_propagate_nan(scalar_t a, scalar_t b) {
    if (is_nan_value(a)) {
        return a;
    }
    return a < b ? a : b;
}

// A counter-based random stream (Philox 4x32, ten rounds): the value at a
// position depends on the seed and the position alone, so a kernel can read
// the value at any position without walking the stream to it.
class Philox4_32 {
public:
    explicit Philox4_32(uint64_t seed = 67280421310721, uint64_t subsequence = 0,
                        uint64_t offset = 0) {
        key_[0] = static_cast<uint32_t>(seed);
        key_[1] = static_cast<uint32_t>(seed >> 32);
        counter_ = {0, 0, 0, 0};
        counter_[2] = static_cast<uint32_t>(subsequence);
        counter_[3] = static_cast<uint32_t>(subsequence >> 32);
        state_ = 0;
        incr_n(offset);
    }

    uint32_t operator()(int32_t n_rounds = 10) {
        if (state_ == 0) {
            output_ = rand(counter_, key_, n_rounds);
            incr();
        }
        uint32_t ret = output_[state_];
        state_ = (state_ + 1) & 3;
        return ret;
    }

    float randn(uint32_t n_rounds) {
        if (state_ == 0) {
            output_ = rand(counter_, key_, n_rounds);
            incr();
        }
        // Box-Muller over (0, 1]: one minus a value in [0, 1) never feeds a
        // zero to the logarithm.
        float u1 = 1 - uniform(output_[0]);
        float u2 = 1 - uniform(output_[1]);
        return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                                  std::cos(2.0 * 3.14159265358979323846 * u2));
    }

    static constexpr float uniform(uint32_t value) {
        // The largest scale for which the largest value still maps below one.
        constexpr float scale = 4.6566127342e-10f;
        return static_cast<float>(value & 0x7FFFFFFF) * scale;
    }

private:
    using UINT4 = std::array<uint32_t, 4>;
    using UINT2 = std::array<uint32_t, 2>;

    void incr_n(uint64_t n) {
        uint32_t nlo = static_cast<uint32_t>(n);
        uint32_t nhi = static_cast<uint32_t>(n >> 32);
        counter_[0] += nlo;
        if (counter_[0] < nlo) {
            nhi++;
            counter_[1] += nhi;
            if (nhi != 0 && nhi <= counter_[1]) {
                return;
            }
        } else {
            counter_[1] += nhi;
            if (nhi <= counter_[1]) {
                return;
            }
        }
        if (++counter_[2]) {
            return;
        }
        ++counter_[3];
    }

    void incr() {
        if (++counter_[0]) return;
        if (++counter_[1]) return;
        if (++counter_[2]) return;
        ++counter_[3];
    }

    static uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t* result_high) {
        const uint64_t product = static_cast<uint64_t>(a) * b;
        *result_high = static_cast<uint32_t>(product >> 32);
        return static_cast<uint32_t>(product);
    }

    static UINT4 single_round(UINT4 ctr, UINT2 in_key) {
        uint32_t hi0 = 0;
        uint32_t hi1 = 0;
        uint32_t lo0 = mulhilo32(kPhiloxSA, ctr[0], &hi0);
        uint32_t lo1 = mulhilo32(kPhiloxSB, ctr[2], &hi1);
        UINT4 ret;
        ret[0] = hi1 ^ ctr[1] ^ in_key[0];
        ret[1] = lo1;
        ret[2] = hi0 ^ ctr[3] ^ in_key[1];
        ret[3] = lo0;
        return ret;
    }

    static UINT4 rand(UINT4 counter, UINT2 key, uint32_t n_rounds) {
        for (uint32_t round = 0; round < (n_rounds - 1); round++) {
            counter = single_round(counter, key);
            key[0] += kPhilox10A;
            key[1] += kPhilox10B;
        }
        return single_round(counter, key);
    }

    static constexpr uint32_t kPhilox10A = 0x9E3779B9;
    static constexpr uint32_t kPhilox10B = 0xBB67AE85;
    static constexpr uint32_t kPhiloxSA = 0xD2511F53;
    static constexpr uint32_t kPhiloxSB = 0xCD9E8D57;

    UINT4 counter_;
    UINT4 output_;
    UINT2 key_;
    uint32_t state_;
};

inline float normalized_rand_cpu(uint32_t seed, uint32_t offset) {
    return Philox4_32::uniform(Philox4_32(seed, 0, offset)());
}

inline float randn_cpu(uint32_t seed, uint32_t offset) {
    Philox4_32 engine(seed, 0, offset);
    return engine.randn(10);
}

inline int64_t randint64_cpu(uint32_t seed, uint32_t offset, int64_t low, int64_t high) {
    auto gen = Philox4_32(seed, 0, offset);
    uint64_t r0 = gen();
    uint64_t r1 = gen();
    uint64_t result = r0 | (r1 << 32);
    return static_cast<int64_t>(result % (high - low)) + low;
}

}  // namespace generated
}  // namespace tensorplay

// The generated kernel body calls the transposed tile load/store by its
// unqualified name, so expose the vector-layer entry points here.
using tensorplay::vec::atomic_add;
using tensorplay::vec::transpose_mxn;

// The cascade summation accumulator is likewise referenced by its unqualified
// name from generated reduction kernels.
using tensorplay::generated::CascadeSumHelper;
using tensorplay::generated::cascade_sum_combine;
using tensorplay::generated::cascade_sum_final;

// So is the variance reduction's state and its helpers.
using tensorplay::generated::Welford;
using tensorplay::generated::WelfordHelper;
using tensorplay::generated::welford_combine;
using tensorplay::generated::welford_vec_reduce_all;

// The reductions, divisions and random reads a kernel spells by their
// unqualified names.
using tensorplay::generated::IndexValue;
using tensorplay::generated::IndexValueVec;
using tensorplay::generated::argmax_combine;
using tensorplay::generated::argmax_combine_vec;
using tensorplay::generated::argmax_vec_reduce_all;
using tensorplay::generated::argmin_combine;
using tensorplay::generated::argmin_combine_vec;
using tensorplay::generated::argmin_vec_reduce_all;
using tensorplay::generated::any_masked_reduce;
using tensorplay::generated::max_masked_reduce;
using tensorplay::generated::min_masked_reduce;
using tensorplay::generated::prod_masked_reduce;
using tensorplay::generated::sum_masked_reduce;
using tensorplay::generated::xor_sum_masked_reduce;
using tensorplay::generated::div_floor_floating;
using tensorplay::generated::div_floor_floating_vec;
using tensorplay::generated::floor_divide_integral;
using tensorplay::generated::mod;
using tensorplay::generated::remainder_integral;
using tensorplay::generated::max_propagate_nan;
using tensorplay::generated::min_propagate_nan;
using tensorplay::generated::normalized_rand_cpu;
using tensorplay::generated::randn_cpu;
using tensorplay::generated::randint64_cpu;

// Special functions, called by name from the scalar form of a generated
// element-wise operation.
using tensorplay::calc_erfinv;
using tensorplay::calc_igamma;
using tensorplay::calc_igammac;
using tensorplay::special_math::airy_ai_forward;
using tensorplay::special_math::bessel_j0_forward;
using tensorplay::special_math::bessel_j1_forward;
using tensorplay::special_math::bessel_y0_forward;
using tensorplay::special_math::bessel_y1_forward;
using tensorplay::special_math::calc_digamma;
using tensorplay::special_math::calc_erfcx;
using tensorplay::special_math::calc_i0;
using tensorplay::special_math::calc_i0e;
using tensorplay::special_math::calc_i1;
using tensorplay::special_math::calc_i1e;
using tensorplay::special_math::calc_log_ndtr;
using tensorplay::special_math::calc_ndtr;
using tensorplay::special_math::calc_ndtri;
using tensorplay::special_math::calc_polygamma;
using tensorplay::special_math::chebyshev_polynomial_t_forward;
using tensorplay::special_math::chebyshev_polynomial_u_forward;
using tensorplay::special_math::chebyshev_polynomial_v_forward;
using tensorplay::special_math::chebyshev_polynomial_w_forward;
using tensorplay::special_math::hermite_polynomial_h_forward;
using tensorplay::special_math::hermite_polynomial_he_forward;
using tensorplay::special_math::laguerre_polynomial_l_forward;
using tensorplay::special_math::legendre_polynomial_p_forward;
using tensorplay::special_math::modified_bessel_i0_forward;
using tensorplay::special_math::modified_bessel_i1_forward;
using tensorplay::special_math::modified_bessel_k0_forward;
using tensorplay::special_math::modified_bessel_k1_forward;
using tensorplay::special_math::scaled_modified_bessel_k0_forward;
using tensorplay::special_math::scaled_modified_bessel_k1_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_t_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_u_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_v_forward;
using tensorplay::special_math::shifted_chebyshev_polynomial_w_forward;
using tensorplay::special_math::spherical_bessel_j0_forward;
using tensorplay::special_math::trigamma;
using tensorplay::special_math::zeta;

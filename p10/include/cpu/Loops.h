#pragma once

// Minimal port of the parts of ATen/native/cpu/Loops.h needed by cpu/Reduce.h.
//
// basic_loop applies a scalar binary op elementwise as
//   out[i] = op(out[i], in[i])
// over the index range [size0, size1), using pointer `data[2]` as the input.
// All reduction call sites use this 3-pointer form with the output duplicated
// in slots 0 and 1 (stride slot 1 is ignored). function_traits introspects the
// op's argument/result types.

#include <cstdint>
#include <tuple>
#include <type_traits>

namespace tensorplay {

namespace detail {

// ATen parity (c10::function_traits): argument slots are computed
// conditionally so that unary (project) and ternary (reduce with index)
// member functions both introspect cleanly -- eager tuple_element on a
// missing slot is a hard error otherwise.
template <class Tuple, size_t I, bool Has = (I < std::tuple_size_v<Tuple>)>
struct arg_at { using type = void; };

template <class Tuple, size_t I>
struct arg_at<Tuple, I, true> { using type = std::tuple_element_t<I, Tuple>; };

} // namespace detail

template <typename T>
struct function_traits : function_traits<decltype(&T::operator())> {};

template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
  using result_type = R;
  using arg1_t = typename detail::arg_at<std::tuple<Args...>, 0>::type;
  using arg2_t = typename detail::arg_at<std::tuple<Args...>, 1>::type;
  static constexpr size_t arity = sizeof...(Args);
};

template <typename R, typename... Args>
struct function_traits<R(Args...)> : function_traits<R (*)(Args...)> {};

template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...)> : function_traits<R (*)(Args...)> {};

template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const> : function_traits<R (*)(Args...)> {};

template <typename T>
using binary_function_traits = function_traits<T>;

template <typename T>
using unary_function_traits = function_traits<T>;

template <typename func_t>
inline void basic_loop(
    char* data[3],
    const int64_t strides[3],
    int64_t size0,
    int64_t size1,
    const func_t& op) {
  using traits = function_traits<func_t>;
  using scalar_t = typename traits::result_type;
  static_assert(traits::arity == 2, "basic_loop expects a binary op");
  for (int64_t i = size0; i < size1; i++) {
    scalar_t* out_ptr = (scalar_t*)(data[0] + i * strides[0]);
    const scalar_t* in_ptr = (const scalar_t*)(data[2] + i * strides[2]);
    *out_ptr = op(*out_ptr, *in_ptr);
  }
}

} // namespace tensorplay
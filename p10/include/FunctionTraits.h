#pragma once

// Introspection of callables for the elementwise kernel machinery: arity,
// argument types, and the result type of lambdas, function pointers, and
// member functions.  Type-only computation, so it is usable from host code
// that instantiates device kernels.

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace tensorplay {

// Fallback, anything with an operator()
template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

// Pointers to class members that are themselves functors.
template <typename ClassType, typename T>
struct function_traits<T ClassType::*> : public function_traits<T> {};

// Const class member functions
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const>
    : public function_traits<ReturnType(Args...)> {};

// Reference types
template <typename T>
struct function_traits<T&> : public function_traits<T> {};
template <typename T>
struct function_traits<T*> : public function_traits<T> {};

// Free functions
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
  // arity is the number of arguments.
  enum { arity = sizeof...(Args) };

  using ArgsTuple = std::tuple<Args...>;
  using result_type = ReturnType;

  template <size_t i>
  struct arg {
    using type = std::tuple_element_t<i, std::tuple<Args...>>;
  };
};

template <typename T>
struct nullary_function_traits {
  using traits = function_traits<T>;
  using result_type = typename traits::result_type;
};

template <typename T>
struct unary_function_traits {
  using traits = function_traits<T>;
  using result_type = typename traits::result_type;
  using arg1_t = typename traits::template arg<0>::type;
};

template <typename T>
struct binary_function_traits {
  using traits = function_traits<T>;
  using result_type = typename traits::result_type;
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
};

}  // namespace tensorplay

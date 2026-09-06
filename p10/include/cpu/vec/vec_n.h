#pragma once

// A vector of `N * Vectorized<T>::size()` elements, i.e. `N` consecutive
// `Vectorized<T>` registers.  Used where two same-width views of one
// register count must interoperate: a bf16/half register pair covers the
// element count of one float register, an int64 register pair covers one
// int32 register, and so on.  Operations delegate element-wise to
// `Vectorized<T>`.

#include "cpu/vec/vec_base.h"

#include <array>
#include <ostream>

namespace tensorplay::vec {
inline namespace CPU_CAPABILITY {

template <typename T, int N>
class VectorizedN {
 public:
  using value_type = T;
  using size_type = int;

  static constexpr size_type size() {
    return Vectorized<T>::size() * N;
  }

 private:
  std::array<Vectorized<T>, N> values;

 public:
  VectorizedN() = default;

  explicit VectorizedN(T val) {
    for (int i = 0; i < N; ++i) {
      values[i] = Vectorized<T>(val);
    }
  }

  template <int L = N, typename std::enable_if_t<L == 1, int> = 0>
  VectorizedN(const Vectorized<T>& val) : values({val}) {}

  template <int L = N, typename std::enable_if_t<L == 2, int> = 0>
  VectorizedN(const Vectorized<T>& val_0, const Vectorized<T>& val_1)
      : values({val_0, val_1}) {}

  template <int L = N, typename std::enable_if_t<L == 1, int> = 0>
  operator Vectorized<T>() const {
    return values[0];
  }

  const Vectorized<T>& operator[](int i) const {
    return values[i];
  }

  Vectorized<T>& operator[](int i) {
    return values[i];
  }

  template <typename Op>
  inline VectorizedN<T, N> unary_op(Op op) const {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] = op(values[i]);
    }
    return result;
  }

  template <typename Op>
  inline VectorizedN<T, N> binary_op(
      const VectorizedN<T, N>& other,
      Op op) const {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] = op(values[i], other.values[i]);
    }
    return result;
  }

  template <int64_t mask>
  static VectorizedN<T, N> blend(
      const VectorizedN<T, N>& a,
      const VectorizedN<T, N>& b) {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] =
          Vectorized<T>::template blend<mask>(a.values[i], b.values[i]);
    }
    return result;
  }

  static VectorizedN<T, N> blendv(
      const VectorizedN<T, N>& a,
      const VectorizedN<T, N>& b,
      const VectorizedN<T, N>& mask) {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] =
          Vectorized<T>::blendv(a.values[i], b.values[i], mask.values[i]);
    }
    return result;
  }

  template <typename step_t>
  static VectorizedN<T, N> arange(
      T base = static_cast<T>(0),
      step_t step = static_cast<step_t>(1)) {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] = Vectorized<T>::arange(base, step);
      base += step * Vectorized<T>::size();
    }
    return result;
  }

  static VectorizedN<T, N> set(
      const VectorizedN<T, N>& a,
      const VectorizedN<T, N>& b,
      int64_t count = size()) {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      if (count > 0) {
        result.values[i] = Vectorized<T>::set(
            a.values[i],
            b.values[i],
            std::min<int64_t>(count, Vectorized<T>::size()));
        count -= Vectorized<T>::size();
      } else {
        result.values[i] = a.values[i];
      }
    }
    return result;
  }

  static VectorizedN<T, N> loadu(const void* ptr) {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] = Vectorized<T>::loadu(ptr);
      ptr = static_cast<const T*>(ptr) + Vectorized<T>::size();
    }
    return result;
  }

  static VectorizedN<T, N> loadu(const void* ptr, int64_t count) {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      if (count > 0) {
        result.values[i] = Vectorized<T>::loadu(
            ptr, std::min<int64_t>(count, Vectorized<T>::size()));
        ptr = static_cast<const T*>(ptr) + Vectorized<T>::size();
        count -= Vectorized<T>::size();
      } else {
        result.values[i] = Vectorized<T>((T)1);
      }
    }
    return result;
  }

  void store(void* ptr) const {
    for (int i = 0; i < N; ++i) {
      values[i].store(ptr);
      ptr = static_cast<T*>(ptr) + Vectorized<T>::size();
    }
  }

  void store(void* ptr, int count) const {
    for (int i = 0; i < N; ++i) {
      values[i].store(ptr, std::min(count, (int)Vectorized<T>::size()));
      ptr = static_cast<T*>(ptr) + Vectorized<T>::size();
      count -= Vectorized<T>::size();
      if (count <= 0) {
        break;
      }
    }
  }

  bool has_inf_nan() const {
    for (int i = 0; i < N; ++i) {
      if (values[i].has_inf_nan()) {
        return true;
      }
    }
    return false;
  }

  VectorizedN<T, N> map(T (*const f)(T)) const {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] = values[i].map(f);
    }
    return result;
  }

  VectorizedN<T, N> map(T (*const f)(const T&)) const {
    VectorizedN<T, N> result;
    for (int i = 0; i < N; ++i) {
      result.values[i] = values[i].map(f);
    }
    return result;
  }

#define TP_VECTORIZED_N_UNARY_OP(op)                                 \
  VectorizedN<T, N> op() const {                                     \
    return unary_op([](const Vectorized<T>& a) { return a.op(); }); \
  }

  TP_VECTORIZED_N_UNARY_OP(isnan)
  TP_VECTORIZED_N_UNARY_OP(abs)
  TP_VECTORIZED_N_UNARY_OP(angle)
  TP_VECTORIZED_N_UNARY_OP(real)
  TP_VECTORIZED_N_UNARY_OP(imag)
  TP_VECTORIZED_N_UNARY_OP(conj)
  TP_VECTORIZED_N_UNARY_OP(acos)
  TP_VECTORIZED_N_UNARY_OP(acosh)
  TP_VECTORIZED_N_UNARY_OP(asin)
  TP_VECTORIZED_N_UNARY_OP(asinh)
  TP_VECTORIZED_N_UNARY_OP(atan)
  TP_VECTORIZED_N_UNARY_OP(atanh)
  TP_VECTORIZED_N_UNARY_OP(erf)
  TP_VECTORIZED_N_UNARY_OP(erfc)
  TP_VECTORIZED_N_UNARY_OP(erfinv)
  TP_VECTORIZED_N_UNARY_OP(exp)
  TP_VECTORIZED_N_UNARY_OP(exp2)
  TP_VECTORIZED_N_UNARY_OP(expm1)
  TP_VECTORIZED_N_UNARY_OP(exp_u20)
  TP_VECTORIZED_N_UNARY_OP(fexp_u20)
  TP_VECTORIZED_N_UNARY_OP(frac)
  TP_VECTORIZED_N_UNARY_OP(log)
  TP_VECTORIZED_N_UNARY_OP(log10)
  TP_VECTORIZED_N_UNARY_OP(log1p)
  TP_VECTORIZED_N_UNARY_OP(log2)
  TP_VECTORIZED_N_UNARY_OP(ceil)
  TP_VECTORIZED_N_UNARY_OP(cos)
  TP_VECTORIZED_N_UNARY_OP(cosh)
  TP_VECTORIZED_N_UNARY_OP(floor)
  TP_VECTORIZED_N_UNARY_OP(neg)
  TP_VECTORIZED_N_UNARY_OP(round)
  TP_VECTORIZED_N_UNARY_OP(sin)
  TP_VECTORIZED_N_UNARY_OP(sinh)
  TP_VECTORIZED_N_UNARY_OP(tan)
  TP_VECTORIZED_N_UNARY_OP(tanh)
  TP_VECTORIZED_N_UNARY_OP(trunc)
  TP_VECTORIZED_N_UNARY_OP(lgamma)
  TP_VECTORIZED_N_UNARY_OP(sqrt)
  TP_VECTORIZED_N_UNARY_OP(reciprocal)
  TP_VECTORIZED_N_UNARY_OP(rsqrt)

#undef TP_VECTORIZED_N_UNARY_OP

#define TP_VECTORIZED_N_BINARY_OP(op)                                    \
  VectorizedN<T, N> op(const VectorizedN<T, N>& other) const {           \
    return binary_op(                                                    \
        other, [](const Vectorized<T>& a, const Vectorized<T>& b) {     \
          return a.op(b);                                               \
        });                                                             \
  }

  TP_VECTORIZED_N_BINARY_OP(atan2)
  TP_VECTORIZED_N_BINARY_OP(copysign)
  TP_VECTORIZED_N_BINARY_OP(fmod)
  TP_VECTORIZED_N_BINARY_OP(hypot)
  TP_VECTORIZED_N_BINARY_OP(nextafter)
  TP_VECTORIZED_N_BINARY_OP(pow)
  TP_VECTORIZED_N_BINARY_OP(operator==)
  TP_VECTORIZED_N_BINARY_OP(operator!=)
  TP_VECTORIZED_N_BINARY_OP(operator>=)
  TP_VECTORIZED_N_BINARY_OP(operator<=)
  TP_VECTORIZED_N_BINARY_OP(operator>)
  TP_VECTORIZED_N_BINARY_OP(operator<)
  TP_VECTORIZED_N_BINARY_OP(eq)
  TP_VECTORIZED_N_BINARY_OP(ne)
  TP_VECTORIZED_N_BINARY_OP(gt)
  TP_VECTORIZED_N_BINARY_OP(ge)
  TP_VECTORIZED_N_BINARY_OP(lt)
  TP_VECTORIZED_N_BINARY_OP(le)

#undef TP_VECTORIZED_N_BINARY_OP
};

#define TP_VECTORIZED_N_UNARY_OP_GLOBAL(op)                          \
  template <typename T, int N>                                       \
  inline VectorizedN<T, N> op(const VectorizedN<T, N>& a) {          \
    return a.unary_op([](const Vectorized<T>& a) { return op(a); }); \
  }

#define TP_VECTORIZED_N_BINARY_OP_GLOBAL(op)                                   \
  template <typename T, int N>                                                 \
  inline VectorizedN<T, N> op(                                                 \
      const VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {                \
    return a.binary_op(b, [](const Vectorized<T>& a, const Vectorized<T>& b) { \
      return op(a, b);                                                         \
    });                                                                        \
  }

TP_VECTORIZED_N_BINARY_OP_GLOBAL(operator+)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(operator-)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(operator*)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(operator/)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(operator%)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(operator||)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(operator<<)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(operator>>)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(maximum)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(minimum)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(clamp_max)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(clamp_min)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(operator&)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(operator|)
TP_VECTORIZED_N_BINARY_OP_GLOBAL(operator^)

template <typename T, int N>
inline VectorizedN<T, N> operator~(const VectorizedN<T, N>& a) {
  return a.unary_op([](const Vectorized<T>& a) { return ~a; });
}

template <typename T, int N>
inline VectorizedN<T, N> operator&&(const VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {
  return a & b;
}

template <typename T, int N>
inline VectorizedN<T, N> clamp(
    const VectorizedN<T, N>& a,
    const VectorizedN<T, N>& min_vec,
    const VectorizedN<T, N>& max_vec) {
  VectorizedN<T, N> result;
  for (int i = 0; i < N; ++i) {
    result[i] = clamp(a[i], min_vec[i], max_vec[i]);
  }
  return result;
}

template <typename T, int N>
inline VectorizedN<T, N> fmadd(
    const VectorizedN<T, N>& a,
    const VectorizedN<T, N>& b,
    const VectorizedN<T, N>& c) {
  VectorizedN<T, N> result;
  for (int i = 0; i < N; ++i) {
    result[i] = fmadd(a[i], b[i], c[i]);
  }
  return result;
}

template <typename T, int N>
inline VectorizedN<T, N> fmsub(
    const VectorizedN<T, N>& a,
    const VectorizedN<T, N>& b,
    const VectorizedN<T, N>& c) {
  VectorizedN<T, N> result;
  for (int i = 0; i < N; ++i) {
    result[i] = fmsub(a[i], b[i], c[i]);
  }
  return result;
}

template <typename T, int N>
inline VectorizedN<T, N> fnmadd(
    const VectorizedN<T, N>& a,
    const VectorizedN<T, N>& b,
    const VectorizedN<T, N>& c) {
  VectorizedN<T, N> result;
  for (int i = 0; i < N; ++i) {
    result[i] = fnmadd(a[i], b[i], c[i]);
  }
  return result;
}

template <typename T, int N>
inline VectorizedN<T, N> fnmsub(
    const VectorizedN<T, N>& a,
    const VectorizedN<T, N>& b,
    const VectorizedN<T, N>& c) {
  VectorizedN<T, N> result;
  for (int i = 0; i < N; ++i) {
    result[i] = fnmsub(a[i], b[i], c[i]);
  }
  return result;
}

template <typename T, int N>
inline VectorizedN<T, N>& operator+=(VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {
  a = a + b;
  return a;
}
template <typename T, int N>
inline VectorizedN<T, N>& operator-=(VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {
  a = a - b;
  return a;
}
template <typename T, int N>
inline VectorizedN<T, N>& operator*=(VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {
  a = a * b;
  return a;
}
template <typename T, int N>
inline VectorizedN<T, N>& operator/=(VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {
  a = a / b;
  return a;
}
template <typename T, int N>
inline VectorizedN<T, N>& operator%=(VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {
  a = a % b;
  return a;
}
template <typename T, int N>
inline VectorizedN<T, N>& operator<<=(VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {
  a = a << b;
  return a;
}
template <typename T, int N>
inline VectorizedN<T, N>& operator>>=(VectorizedN<T, N>& a, const VectorizedN<T, N>& b) {
  a = a >> b;
  return a;
}

#undef TP_VECTORIZED_N_UNARY_OP_GLOBAL
#undef TP_VECTORIZED_N_BINARY_OP_GLOBAL

template <typename T, int N>
std::ostream& operator<<(std::ostream& stream, const VectorizedN<T, N>& vec_n) {
  stream << "vec_n[";
  for (int i = 0; i < N; ++i) {
    if (i != 0) {
      stream << ", ";
    }
    stream << vec_n[i];
  }
  stream << ']';
  return stream;
}

} // namespace tensorplay::vec::inline CPU_CAPABILITY
} // namespace tensorplay::vec

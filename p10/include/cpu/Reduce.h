#pragma once

// pieces needed by the CPU reduction kernels:
//   - binary_kernel_reduce: ops-based (reduce/combine/project) reduction over
//     foreach_reduced_elt, with per-thread partials combined at the end.
//   - binary_kernel_reduce_vec: vectorized reduction over parallel_reduce,
//     accumulating in the (single) data type with 4 accumulators.
//
// This header is included by multi-version-compiled kernel TUs, so the vec
// layer is selected by the CPU_CAPABILITY inline namespace automatically.

#include "cpu/Loops.h"
#include "cpu/vec/vec.h"
#include "Parallel.h"
#include "TensorIterator.h"
#include "load.h"
#include "irange.h"
#include <type_traits>
#include <vector>

namespace tensorplay {
inline namespace CPU_CAPABILITY {

using namespace vec;

#define VEC_LOOP_HEADER(func_t, data) \
  using scalar_t = typename function_traits<func_t>::result_type; \
  using Vec = Vectorized<scalar_t>; \
  char* out_ptr = data[0]; \
  (void) out_ptr;

// reduction that is contiguous over the input in dim 0
template <typename traits>
inline bool is_contiguous_reduction(const int64_t* strides) {
  return strides[0] == 0 &&
         strides[1] == sizeof(typename traits::arg2_t);
}

// reduction that is contiguous over the input in dim 1
template <typename traits>
inline bool is_outer_reduction(const int64_t* strides) {
  return strides[0] == 0 &&
         strides[2] == sizeof(typename traits::result_type) &&
         strides[3] == sizeof(typename traits::arg2_t);
}

template <typename func_t, typename vec_func_t>
inline void vectorized_reduction(char** data, int64_t n, int64_t stride,
                                        func_t op, vec_func_t vop, bool reduce) {
  VEC_LOOP_HEADER(func_t, data)
  const char* in1_ptr = data[1];
  Vec acc[4];
  for (const auto j : irange(4)) {
    acc[j] = Vec::loadu(in1_ptr + j * Vec::size() * sizeof(scalar_t));
  }
  for (const auto i : irange<int64_t>(1, n)) {
    const char* ptr = in1_ptr + stride * i;
    acc[0] = vop(acc[0], Vec::loadu(ptr + (0 * Vec::size() * sizeof(scalar_t))));
    acc[1] = vop(acc[1], Vec::loadu(ptr + (1 * Vec::size() * sizeof(scalar_t))));
    acc[2] = vop(acc[2], Vec::loadu(ptr + (2 * Vec::size() * sizeof(scalar_t))));
    acc[3] = vop(acc[3], Vec::loadu(ptr + (3 * Vec::size() * sizeof(scalar_t))));
  }
  if (reduce) {
    scalar_t buffer[Vec::size()];
    acc[0] = vop(vop(acc[0], acc[1]), vop(acc[2], acc[3]));
    acc[0].store(buffer);
    for (const auto j : irange(1, Vec::size())) {
      buffer[0] = op(buffer[0], buffer[j]);
    }
    auto dst = (scalar_t*)out_ptr;
    *dst = op(*dst, buffer[0]);
  } else {
    for (const auto j : irange(4)) {
      auto dst = out_ptr + j * Vec::size() * sizeof(scalar_t);
      acc[j] = vop(acc[j], Vec::loadu(dst));
      acc[j].store(dst);
    }
  }
}

template <typename F>
inline void UNARY_OUTER_LOOP(char* data[2], const int64_t strides[2], int64_t n, F f) {
  for ([[maybe_unused]] const auto j : irange(n)) {
    f();
    data[0] += strides[0];
    data[1] += strides[1];
  }
}

// computes the reduction out = op(out, in)
template <typename func_t, typename vec_func_t>
inline void vectorized_inner_reduction(char** data, int64_t n, func_t op, vec_func_t vop) {
  VEC_LOOP_HEADER(func_t, data)
  constexpr int64_t vector_stride = 4 * Vec::size() * sizeof(scalar_t);
  int64_t count = n / (4 * Vec::size());
  if (count > 0) {
    vectorized_reduction(data, count, vector_stride, op, vop, /*reduce=*/true);
  }
  char* ptrs[3] = { data[0], data[0], data[1] };
  int64_t strides[] = { 0, 0, sizeof(scalar_t) };
  basic_loop(ptrs, strides, count * 4 * Vec::size(), n, op);
}

// computes the reduction out = op(out, in)
template <typename func_t, typename vec_func_t>
inline void vectorized_outer_reduction(char** data, int64_t inner_stride, int64_t size0, int64_t size1, func_t op, vec_func_t vop) {
  VEC_LOOP_HEADER(func_t, data)

  // reduce down each column of 4 * Vec::size() elements.
  constexpr int64_t vector_stride = 4 * Vec::size() * sizeof(scalar_t);
  int64_t outer_stride[2] = { vector_stride, vector_stride };
  UNARY_OUTER_LOOP(data, outer_stride, size1 / (4 * Vec::size()), [&] {
    vectorized_reduction(data, size0, inner_stride, op, vop, /*reduce=*/false);
  });

  // reduce down the remaining columns
  int64_t step[] = { sizeof(scalar_t), sizeof(scalar_t) };
  int64_t remaining = size1 % (4 * Vec::size());
  UNARY_OUTER_LOOP(data, step, remaining, [&] {
    char* ptrs[3] = { data[0], data[0], data[1] };
    int64_t strides[] = { 0, 0, inner_stride };
    basic_loop(ptrs, strides, 0, size0, op);
  });
}

template<typename traits, typename res_t>
static void set_result(const int index, const res_t result, const TensorIteratorBase &iter, const int num_outputs) {
  if (index < num_outputs) {
    char *out = (char *) iter.data_ptr(index);
    *(res_t *) out = result;
  }
}

template<typename traits, typename res_t>
static void set_results(const res_t result, const TensorIteratorBase &iter, const int num_outputs) {
  TP_CHECK(num_outputs == 1, "expected exactly one output");
  set_result<traits>(0, result, iter, num_outputs);
}

// data_t is the input/output data type.
// acc_t is a type that contains all the necessary data
// to continue reducing.
//
// ops_t is such that &ops_t::reduce, &ops_t::combine, and &ops_t::project exist
// and satisfy the following:
// reduce: (acc_t, data_t, index_t) -> acc_t adds one data point to the accumulated value.
// combine: (acc_t, acc_t) -> acc_t combines two accumulated values into one.
// project: acc_t -> out_t finishes the reduction, getting the required output.
//
// Additionally, acc_t must be default-constructible:
// acc_t {} is an identity for combine,
// and project(acc_t {}) is the value of the operation on zero elements.
//
// The point of `combine` is to support parallelization -
// the idea is to one sequence of `reduce` calls per thread of execution,
// and then to combine them at the end with `combine`.

template <typename ops_t, typename init_t>
void binary_kernel_reduce(TensorIteratorBase& iter, ops_t ops, init_t init) {
  using rf_t = decltype(&ops_t::reduce);
  using cf_t = decltype(&ops_t::combine);
  using pf_t = decltype(&ops_t::project);
  using r_traits = binary_function_traits<rf_t>;
  using c_traits = binary_function_traits<cf_t>;
  using p_traits = unary_function_traits<pf_t>;
  using acc_t = typename p_traits::arg1_t;
  using data_t = typename r_traits::arg2_t;
  static_assert(
    std::is_same_v<acc_t, init_t> &&
    std::is_same_v<acc_t, typename r_traits::arg1_t> &&
    std::is_same_v<acc_t, typename r_traits::result_type> &&
    std::is_same_v<acc_t, typename c_traits::arg1_t> &&
    std::is_same_v<acc_t, typename c_traits::arg2_t> &&
    std::is_same_v<acc_t, typename c_traits::result_type>,
    "all accumulate types must match");
  static_assert(
    std::is_default_constructible_v<acc_t>,
    "the accumulate type must be default-constructible"
  );
  const int num_outputs = iter.noutputs();
  iter.foreach_reduced_elt([&ops, &init, num_outputs](TensorIteratorBase &sub_iter) {
    auto reduction_body = [&ops, &sub_iter, num_outputs](acc_t acc, int64_t begin, int64_t end) -> acc_t {
      int ntensors = sub_iter.ntensors();
      sub_iter.serial_for_each([&acc, &ops, num_outputs, ntensors, begin](char** data, const int64_t* strides, int64_t size) {
        TP_CHECK(ntensors - num_outputs == 1, "expected one input");
        char *in = data[ntensors - 1];
        int64_t stride = strides[ntensors - 1];
        for (const auto i : irange(size)) {
          acc = ops.reduce(acc, load<data_t>(in), begin + i);
          in += stride;
        }
      }, {begin, end});
      return ops.translate_idx(acc, sub_iter.view_offsets()[0]);
    };
    acc_t total_acc = init;
    auto numel = sub_iter.numel();
    if (numel < parallel::GRAIN_SIZE || parallel::get_num_threads() == 1 ||
        parallel::in_parallel_region()) {
      total_acc = reduction_body(total_acc, 0, numel);
    } else {
      int max_threads = parallel::get_num_threads();
      TP_CHECK(max_threads > 0, "no threads");
      static_assert(
        !std::is_same_v<acc_t, bool>,
        "Concurrently modifying different references into std::vector<bool> is UB."
      );
      std::vector<acc_t> buffer((unsigned)max_threads, init);
      parallel::parallel_for(0, numel, parallel::GRAIN_SIZE,
        [&](int64_t begin, int64_t end) {
          auto& acc = buffer[parallel::get_thread_num()];
          acc = reduction_body(acc, begin, end);
        }
      );
      for (const auto i : irange(max_threads)) {
        total_acc = ops.combine(total_acc, buffer[i]);
      }
    }
    set_results<r_traits>(ops.project(total_acc), sub_iter, num_outputs);
  });
}

template <typename func_t, typename vec_func_t>
void binary_kernel_reduce_vec(TensorIteratorBase& iter, func_t op, vec_func_t vop, double ident = 0) {
  using traits = binary_function_traits<func_t>;
  static_assert(
    std::is_same_v<typename traits::result_type, typename traits::arg1_t> &&
    std::is_same_v<typename traits::result_type, typename traits::arg2_t>,
    "all types must match");

  iter.output_base().fill_(ident);
  iter.parallel_reduce([&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
    int64_t outer_strides[] = { strides[2], strides[3] };
    if (is_contiguous_reduction<traits>(strides)) {
      // input is contiguous in dim 0, output is reduced in dim 0
      UNARY_OUTER_LOOP(data, outer_strides, size1, [&] {
        vectorized_inner_reduction(data, size0, op, vop);
      });
    } else if (is_outer_reduction<traits>(strides)) {
      // input and output are contiguous in dim 1
      int64_t inner_stride = strides[1]; // stride of input in dim 0
      vectorized_outer_reduction(data, inner_stride, size0, size1, op, vop);
    } else {
      UNARY_OUTER_LOOP(data, outer_strides, size1, [&] {
        char* ptrs[3] = { data[0], data[0], data[1] };
        int64_t inner_strides[3] = { strides[0], strides[0], strides[1] };
        basic_loop(ptrs, inner_strides, 0, size0, op);
      });
    }
  });
}

// ==========================================================================
// Dual-input fused reduction (one output, two inputs)
// --------------------------------------------------------------------------
// A single pass over both inputs computes out = op(acc, a, b) and accumulates
// into the output, so fused metrics such as sum((a-b)^2) avoid materialising
// the intermediate difference/power tensors and their extra kernels.
//
//   op   : (acc, a, b) -> acc    scalar reduction (ternary)
//   vop  : (Vec, Vec, Vec) -> Vec vectorized reduction
//   cop  : (acc, acc) -> acc      scalar combine of partial results
//   vcop : (Vec, Vec) -> Vec      vectorized combine of partial results
// ==========================================================================

template <typename func_t, typename vec_func_t, typename combine_t, typename vec_combine_t>
inline void vectorized_reduction2(
    char** data, int64_t n, int64_t stride1, int64_t stride2,
    func_t op, vec_func_t vop, combine_t cop, vec_combine_t vcop, bool reduce) {
  using scalar_t = typename function_traits<func_t>::result_type;
  using Vec = Vectorized<scalar_t>;
  char* out_ptr = data[0];
  const char* in1_ptr = data[1];
  const char* in2_ptr = data[2];
  constexpr int64_t vsize = Vec::size();
  Vec acc[4];
  for (const auto j : irange(4)) {
    acc[j] = vop(
        Vec(0),
        Vec::loadu(in1_ptr + j * vsize * sizeof(scalar_t)),
        Vec::loadu(in2_ptr + j * vsize * sizeof(scalar_t)));
  }
  for (const auto i : irange<int64_t>(1, n)) {
    const char* p1 = in1_ptr + stride1 * i;
    const char* p2 = in2_ptr + stride2 * i;
    acc[0] = vop(acc[0],
                Vec::loadu(p1 + 0 * vsize * sizeof(scalar_t)),
                Vec::loadu(p2 + 0 * vsize * sizeof(scalar_t)));
    acc[1] = vop(acc[1],
                Vec::loadu(p1 + 1 * vsize * sizeof(scalar_t)),
                Vec::loadu(p2 + 1 * vsize * sizeof(scalar_t)));
    acc[2] = vop(acc[2],
                Vec::loadu(p1 + 2 * vsize * sizeof(scalar_t)),
                Vec::loadu(p2 + 2 * vsize * sizeof(scalar_t)));
    acc[3] = vop(acc[3],
                Vec::loadu(p1 + 3 * vsize * sizeof(scalar_t)),
                Vec::loadu(p2 + 3 * vsize * sizeof(scalar_t)));
  }
  if (reduce) {
    scalar_t buffer[vsize];
    acc[0] = vcop(vcop(acc[0], acc[1]), vcop(acc[2], acc[3]));
    acc[0].store(buffer);
    for (const auto j : irange<int64_t>(1, vsize)) {
      buffer[0] = cop(buffer[0], buffer[j]);
    }
    auto dst = (scalar_t*)out_ptr;
    *dst = cop(*dst, buffer[0]);
  } else {
    for (const auto j : irange(4)) {
      auto dst = out_ptr + j * vsize * sizeof(scalar_t);
      acc[j] = vcop(acc[j], Vec::loadu(dst));
      acc[j].store(dst);
    }
  }
}

template <typename func_t, typename vec_func_t, typename combine_t, typename vec_combine_t>
inline void vectorized_inner_reduction2(
    char** data, int64_t n,
    func_t op, vec_func_t vop, combine_t cop, vec_combine_t vcop) {
  using scalar_t = typename function_traits<func_t>::result_type;
  using Vec = Vectorized<scalar_t>;
  constexpr int64_t vector_stride = 4 * Vec::size() * sizeof(scalar_t);
  int64_t count = n / (4 * Vec::size());
  if (count > 0) {
    vectorized_reduction2(
        data, count, vector_stride, vector_stride, op, vop, cop, vcop, /*reduce=*/true);
  }
  char* ptrs[4] = { data[0], data[1], data[2], data[2] };
  int64_t strides[4] = { 0, sizeof(scalar_t), sizeof(scalar_t), sizeof(scalar_t) };
  basic_loop2(ptrs, strides, count * 4 * Vec::size(), n, op);
}

template <typename func_t, typename vec_func_t, typename combine_t, typename vec_combine_t>
inline void vectorized_outer_reduction2(
    char** data, int64_t inner_stride1, int64_t inner_stride2, int64_t size0, int64_t size1,
    func_t op, vec_func_t vop, combine_t cop, vec_combine_t vcop) {
  using scalar_t = typename function_traits<func_t>::result_type;
  using Vec = Vectorized<scalar_t>;
  constexpr int64_t vector_stride = 4 * Vec::size() * sizeof(scalar_t);
  int64_t outer_count = size1 / (4 * Vec::size());
  for (const auto k : irange(outer_count)) {
    vectorized_reduction2(
        data, size0, inner_stride1, inner_stride2, op, vop, cop, vcop, /*reduce=*/false);
    data[0] += vector_stride;
    data[1] += vector_stride;
    data[2] += vector_stride;
  }
  int64_t remaining = size1 % (4 * Vec::size());
  for (const auto k : irange(remaining)) {
    char* ptrs[4] = { data[0], data[1], data[2], data[2] };
    int64_t strides[4] = { 0, inner_stride1, inner_stride2, inner_stride2 };
    basic_loop2(ptrs, strides, 0, size0, op);
    data[0] += sizeof(scalar_t);
    data[1] += sizeof(scalar_t);
    data[2] += sizeof(scalar_t);
  }
}

template <typename func_t, typename vec_func_t, typename combine_t, typename vec_combine_t>
void binary_kernel_reduce2_vec(
    TensorIteratorBase& iter,
    func_t op, vec_func_t vop, combine_t cop, vec_combine_t vcop,
    double ident = 0) {
  using scalar_t = typename function_traits<func_t>::result_type;
  iter.output_base().fill_(ident);
  iter.for_each([&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
    const int64_t out_red = strides[0];
    const int64_t a_red = strides[1];
    const int64_t b_red = strides[2];
    const int64_t out_outer = strides[3];
    const int64_t a_outer = strides[4];
    const int64_t b_outer = strides[5];
    if (out_red == 0 && a_red == sizeof(scalar_t) && b_red == sizeof(scalar_t)) {
      for (const auto k : irange(size1)) {
        vectorized_inner_reduction2(data, size0, op, vop, cop, vcop);
        data[0] += out_outer;
        data[1] += a_outer;
        data[2] += b_outer;
      }
    } else if (out_red == 0 &&
               out_outer == sizeof(scalar_t) &&
               a_outer == sizeof(scalar_t) &&
               b_outer == sizeof(scalar_t)) {
      vectorized_outer_reduction2(data, a_red, b_red, size0, size1, op, vop, cop, vcop);
    } else {
      for (const auto k : irange(size1)) {
        char* ptrs[4] = { data[0], data[1], data[2], data[2] };
        int64_t inner_strides[4] = { 0, a_red, b_red, b_red };
        basic_loop2(ptrs, inner_strides, 0, size0, op);
        data[0] += out_outer;
        data[1] += a_outer;
        data[2] += b_outer;
      }
    }
  });
}

} // namespace tensorplay::inline CPU_CAPABILITY
} // namespace tensorplay
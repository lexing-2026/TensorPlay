#pragma once

// TensorIterator-driven GPU elementwise launch machinery:
//
//   gpu_kernel(iter, <lambda>)
//   gpu_kernel_with_scalars(iter, <lambda>)
//   gpu_kernel_multiple_outputs(iter, <lambda>)
//
// The iterator supplies the coalesced iteration shape and per-operand byte
// strides.  Execution paths, in selection order:
//
//   1. vectorized - trivial 1-D iteration with every operand contiguous;
//      each thread moves aligned vectors of scalars;
//   2. unrolled   - contiguous operands; a tail block below a full block of
//      work is handled by a per-element inbounds check;
//   3. legacy     - arbitrary strides via an OffsetCalculator.
//
// Dynamic casting: when operand dtypes differ from the functor's static
// argument types, loads and stores go through fetch_and_cast /
// cast_and_store, so one kernel instantiation covers every promoted
// combination instead of an N x N x N grid of per-dtype kernels.

#include <array>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "Tensor.h"
#include "TensorIterator.h"
#include "FunctionTraits.h"
#include "DynamicCast.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "OffsetCalculator.cuh"

#include <cuda_runtime.h>

namespace tensorplay {
namespace cuda {

constexpr uint32_t kLoopNumThreads = 128;
constexpr int kLoopThreadWorkSize = 4;
constexpr int kLoopBlockWorkSize =
    kLoopThreadWorkSize * static_cast<int>(kLoopNumThreads);

// Compile-time constants exposed as constexpr functions: device code
// forwards them through forwarding references, and a named static member
// would make nvcc demand an out-of-line definition that does not exist.
constexpr int loop_num_threads() { return static_cast<int>(kLoopNumThreads); }
constexpr int loop_thread_work_size() { return kLoopThreadWorkSize; }
constexpr int loop_block_work_size() {
    return loop_thread_work_size() * loop_num_threads();
}

#define TP_LOOP_LAUNCH_BOUNDS(n) __launch_bounds__(n)

// Lambdas passed to the loop dispatchers must carry both specifiers: a
// host-side definition with only __device__ would be wrapped by nvcc into
// an internal closure type whose call operator cannot be introspected.
#define GPU_LAMBDA __host__ __device__

inline void loop_launch_check(const char* what) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TP_THROW(RuntimeError, std::string(what) + ": " +
                                   cudaGetErrorString(err));
    }
}

namespace memory {

// aligned_vector compiles to one wide load/store instruction.
template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
    scalar_t val[vec_size];
};

// Highest vector width whose alignment requirement the pointer satisfies.
template <typename scalar_t>
inline int can_vectorize_up_to(const char* pointer) {
    const uint64_t address = reinterpret_cast<uint64_t>(pointer);
    constexpr int vec2_alignment =
        std::alignment_of_v<aligned_vector<scalar_t, 2>>;
    constexpr int vec4_alignment =
        std::alignment_of_v<aligned_vector<scalar_t, 4>>;
    constexpr int vec8_alignment =
        std::alignment_of_v<aligned_vector<scalar_t, 8>>;
    if (address % vec8_alignment == 0) {
        return 8;
    }
    if (address % vec4_alignment == 0) {
        return 4;
    }
    if (address % vec2_alignment == 0) {
        return 2;
    }
    return 1;
}

template <typename scalar_t>
inline int can_vectorize_up_to(char* pointer) {
    return can_vectorize_up_to<scalar_t>(static_cast<const char*>(pointer));
}

template <int i>
struct can_vectorize_up_to_helper {
    template <typename array_t, typename traits>
    static void apply(int& result, array_t pointers, traits) {
        using arg_t = typename traits::template arg<i>::type;
        result = std::min<int>(
            result, can_vectorize_up_to<arg_t>(pointers[i + 1]));
    }
};

// Vector width limited by every operand's pointer alignment.
template <typename func_t, typename array_t>
inline int can_vectorize_up_to(array_t pointers) {
    using traits = tensorplay::function_traits<func_t>;
    using return_t = typename traits::result_type;
    constexpr int arity = traits::arity;
    int result = can_vectorize_up_to<return_t>(pointers[0]);
    // Local host/device expansion over argument slots.
    [&]<size_t... I>(std::index_sequence<I...>) {
        ((result = std::min<int>(
              result,
              can_vectorize_up_to<typename traits::template arg<I>::type>(
                  pointers[I + 1]))),
         ...);
    }(std::make_index_sequence<arity>{});
    return result;
}

// Pass-through load: pointer arithmetic in element units.
struct LoadWithoutCast {
    template <typename scalar_t>
    __device__ scalar_t load(char* base_ptr, uint32_t offset, int) {
        return tensorplay::detail::dynamic_load<scalar_t>(
            base_ptr + sizeof(scalar_t) * offset);
    }
};

// Loads through fetch_and_cast: the iterator's operand dtype is only known
// at runtime.
template <int N>
struct LoadWithCast {
    using array_t = std::array<ScalarType, std::max<int>(N, 1)>;
    using size_array_t = std::array<uint32_t, std::max<int>(N, 1)>;

    array_t dtypes;
    size_array_t element_sizes;

    LoadWithCast(const TensorIteratorBase& iter) {
        TP_CHECK(iter.ninputs() == N, "LoadWithCast arity mismatch");
        for (int i = 0; i < N; ++i) {
            dtypes[i] = iter.dtype(i + iter.noutputs());
            element_sizes[i] = static_cast<uint32_t>(
                elementSize(iter.dtype(i + iter.noutputs())));
        }
    }

    template <typename scalar_t>
    __device__ scalar_t load(char* base_ptr, uint32_t offset, int arg) {
        void* ptr = base_ptr + element_sizes[arg] * offset;
        return tensorplay::fetch_and_cast<scalar_t>(dtypes[arg], ptr);
    }
};

struct StoreWithoutCast {
    template <typename scalar_t>
    __device__ void store(scalar_t value, char* base_ptr, uint32_t offset,
                          int = 0) {
        *reinterpret_cast<scalar_t*>(base_ptr + sizeof(scalar_t) * offset) =
            value;
    }
};

template <int N = 1>
struct StoreWithCast {
    using array_t = std::array<ScalarType, std::max<int>(N, 1)>;
    using size_array_t = std::array<uint32_t, std::max<int>(N, 1)>;

    array_t dtypes;
    size_array_t element_sizes;

    StoreWithCast(const TensorIteratorBase& iter) {
        TP_CHECK(iter.noutputs() == N, "StoreWithCast arity mismatch");
        for (int i = 0; i < N; ++i) {
            dtypes[i] = iter.dtype(i);
            element_sizes[i] =
                static_cast<uint32_t>(elementSize(iter.dtype(i)));
        }
    }

    template <typename scalar_t>
    __device__ void store(scalar_t value, char* base_ptr, uint32_t offset,
                          int arg = 0) {
        void* ptr = base_ptr + element_sizes[arg] * offset;
        tensorplay::cast_and_store<scalar_t>(dtypes[arg], ptr, value);
    }
};

}  // namespace memory

namespace detail {

// Compile-time unroll: expands to func<I>::apply(...) for I in [0, end).
// Callable from both host launchers and device kernels.
template <template <int i> typename func, int end, int current = 0>
struct static_unroll {
    template <typename... Args>
    static inline __host__ __device__ void with_args(Args&&... args) {
        func<current>::apply(std::forward<Args>(args)...);
        static_unroll<func, end, current + 1>::with_args(args...);
    }
};

template <template <int i> typename func, int end>
struct static_unroll<func, end, end> {
    template <typename... Args>
    static inline __host__ __device__ void with_args(Args...) {}
};

// Loads argument slot arg_index of unroll slot j through the loader policy.
template <int arg_index>
struct unroll_load_helper {
    template <typename args_t, typename policy_t, typename offset_t,
              typename loader_t>
    static __device__ void apply(policy_t& self, args_t* args,
                                 offset_t offset, loader_t loader, int j,
                                 int num_outputs) {
        using arg_t = std::tuple_element_t<arg_index, args_t>;
        // data[0] is the output; inputs start at slot num_outputs.
        std::get<arg_index>(args[j]) = loader.template load<arg_t>(
            self.data[arg_index + num_outputs], offset[arg_index], arg_index);
    }
};

// Loads argument slot arg_index as one aligned vector in the vectorized
// policy: a thread's slot is one vector apart from the next lane's.
template <int arg_index>
struct vectorized_load_helper {
    template <typename args_t, typename policy_t>
    static __device__ void apply(policy_t& self, args_t* args, int idx,
                                 int block_work) {
        using arg_t = std::tuple_element_t<arg_index, args_t>;
        auto ptr = reinterpret_cast<arg_t*>(self.data[arg_index + 1]) +
                   block_work * idx;
        auto args_accessor = [&args] __device__(int thread_unroll_idx)
            -> arg_t& { return std::get<arg_index>(args[thread_unroll_idx]); };
        self.load_single_arg(args_accessor, ptr);
    }
};

// Stores output slot current of a multi-output functor result.
template <int current>
struct multi_outputs_store_helper {
    template <typename data_t, typename offsets_t, typename tuple_t>
    __device__ static void apply(const data_t& data, const offsets_t& offsets,
                                 const tuple_t& ret) {
        using T = std::tuple_element_t<current, tuple_t>;
        T* to = reinterpret_cast<T*>(data[current]) + offsets[current];
        *to = std::get<current>(ret);
    }
};

}  // namespace detail

namespace memory {
namespace policies {

// Full-block unroll: elems_per_thread elements per thread, one coordinate
// resolution per element, loads/stores through the cast policies.
template <typename data_t, typename inp_calc_t, typename out_calc_t,
          typename loader_t, typename storer_t, int elems_per_thread,
          int num_outputs = 1>
struct unroll_base {
    data_t data;
    int remaining;
    inp_calc_t input_offset_calculator;
    out_calc_t output_offset_calculator;
    loader_t loader;
    storer_t storer;
    static constexpr int tws = elems_per_thread;
    static constexpr int block_work() {
        return elems_per_thread * loop_num_threads();
    }

    __device__ unroll_base(data_t data, int remaining, inp_calc_t ic,
                           out_calc_t oc, loader_t l, storer_t s)
        : data(data),
          remaining(remaining),
          input_offset_calculator(ic),
          output_offset_calculator(oc),
          loader(l),
          storer(s) {}

    __device__ inline bool check_inbounds(int thread_work_elem) {
        return static_cast<int>(threadIdx.x +
                                thread_work_elem * loop_num_threads()) <
               remaining;
    }

    template <typename args_t>
    __device__ inline void load(args_t* args, int idx) {
        constexpr int arity = std::tuple_size_v<args_t>;
        int thread_idx = threadIdx.x;
#pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            if (thread_idx < remaining) {
                int linear_idx = thread_idx + block_work() * idx;
                auto offset = input_offset_calculator.get(linear_idx);
                detail::static_unroll<detail::unroll_load_helper, arity>::
                    with_args(*this, args, offset, loader, i, num_outputs);
                thread_idx += loop_num_threads();
            }
        }
    }

    template <typename scalar_t>
    __device__ inline void store(scalar_t* from, int idx) {
        int thread_idx = threadIdx.x;
#pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            if (thread_idx < remaining) {
                int linear_idx = thread_idx + block_work() * idx;
                int offset = output_offset_calculator.get(linear_idx)[0];
                storer.store(from[i], data[0], offset);
                thread_idx += loop_num_threads();
            }
        }
    }
};

template <typename data_t, typename inp_calc_t, typename out_calc_t,
          typename loader_t, typename storer_t, int elems_per_thread,
          int num_outputs = 1>
using unroll = unroll_base<data_t, inp_calc_t, out_calc_t, loader_t, storer_t,
                           elems_per_thread, num_outputs>;

// Contiguous fast path: each thread's slots form one aligned vector per
// unroll round.
template <int vec_size, typename data_t, int elems_per_thread>
struct vectorized {
    static_assert(elems_per_thread % vec_size == 0,
                  "The workload per thread must be a multiple of vec_size");
    static constexpr int loop_size = elems_per_thread / vec_size;
    static constexpr int tws = elems_per_thread;
    static constexpr int block_work() {
        return elems_per_thread * loop_num_threads();
    }

    data_t data;

    __device__ vectorized(data_t data) : data(data) {}

    __device__ inline constexpr bool check_inbounds(int) { return true; }

    template <typename arg_t, typename accessor_t>
    __device__ inline void load_single_arg(accessor_t to, const arg_t* ptr) {
        using vec_t = aligned_vector<arg_t, vec_size>;
        const auto* from = reinterpret_cast<const vec_t*>(ptr);
        vec_t v = from[threadIdx.x];
#pragma unroll
        for (int j = 0; j < vec_size; ++j) {
            to(vec_size * 0 + j) = v.val[j];
        }
    }

    template <typename args_t>
    __device__ inline void load(args_t* args, int idx) {
        constexpr int arity = std::tuple_size_v<args_t>;
        detail::static_unroll<detail::vectorized_load_helper, arity>::with_args(
            *this, args, idx, elems_per_thread * loop_num_threads());
    }

    template <typename scalar_t>
    __device__ inline void store(scalar_t* from, int idx) {
        using vec_t = aligned_vector<scalar_t, vec_size>;
        scalar_t* to = reinterpret_cast<scalar_t*>(data[0]) +
                       elems_per_thread * loop_num_threads() * idx;
        vec_t* to_ = reinterpret_cast<vec_t*>(to);
        int thread_idx = threadIdx.x;
#pragma unroll
        for (int i = 0; i < loop_size; i++) {
            int index = thread_idx + i * loop_num_threads();
            vec_t v;
#pragma unroll
            for (int j = 0; j < vec_size; j++) {
                v.val[j] = from[vec_size * i + j];
            }
            to_[index] = v;
        }
    }
};

// Multi-output unroll: the functor returns a tuple; each element goes to
// its own output operand.
template <typename data_t, typename inp_calc_t, typename out_calc_t,
          int num_outputs>
struct multi_outputs_unroll {
    data_t data;
    int remaining;
    inp_calc_t input_offset_calculator;
    out_calc_t output_offset_calculator;
    static constexpr int tws = kLoopThreadWorkSize;
    static constexpr int block_work() { return loop_block_work_size(); }

    __device__ multi_outputs_unroll(data_t data, int remaining,
                                    inp_calc_t ic, out_calc_t oc)
        : data(data),
          remaining(remaining),
          input_offset_calculator(ic),
          output_offset_calculator(oc) {}

    __device__ inline bool check_inbounds(int thread_work_elem) {
        return static_cast<int>(threadIdx.x +
                                thread_work_elem * loop_num_threads()) <
               remaining;
    }

    template <typename args_t>
    __device__ inline void load(args_t* args, int idx) {
        constexpr int arity = std::tuple_size_v<args_t>;
        int thread_idx = threadIdx.x;
#pragma unroll
        for (int i = 0; i < kLoopThreadWorkSize; i++) {
            if (thread_idx < remaining) {
                int linear_idx = thread_idx + block_work() * idx;
                auto offsets = input_offset_calculator.get(linear_idx);
                detail::static_unroll<detail::unroll_load_helper, arity>::
                    with_args(*this, args, offsets, LoadWithoutCast(), i,
                               num_outputs);
                thread_idx += loop_num_threads();
            }
        }
    }

    template <typename tuple_t>
    __device__ inline void store(tuple_t* from, int idx) {
        int thread_idx = threadIdx.x;
#pragma unroll
        for (int i = 0; i < kLoopThreadWorkSize; i++) {
            if (thread_idx < remaining) {
                int linear_idx = thread_idx + block_work() * idx;
                auto offsets = output_offset_calculator.get(linear_idx);
                detail::static_unroll<detail::multi_outputs_store_helper,
                                      num_outputs>::with_args(data, offsets,
                                                              from[i]);
                thread_idx += loop_num_threads();
            }
        }
    }
};

}  // namespace policies
}  // namespace memory

namespace detail {

// Applies a functor to a tuple of arguments (index expansion).
template <typename F, typename Tuple, size_t... I>
__device__ inline auto tuple_apply_impl(const F& f, const Tuple& t,
                                        std::index_sequence<I...>) {
    return f(std::get<I>(t)...);
}

template <typename F, typename Tuple>
__device__ inline auto tuple_apply(const F& f, const Tuple& t) {
    return tuple_apply_impl(
        f, t, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

}  // namespace detail

// Compares the types the iterator expects (operand dtypes) with the types
// the functor actually reads and writes; a mismatch means loads and stores
// must go through fetch_and_cast / cast_and_store.
template <typename func_t, int nargs = tensorplay::function_traits<func_t>::arity>
struct needs_dynamic_casting {
    static bool check(TensorIteratorBase& iter) {
        using traits = tensorplay::function_traits<func_t>;
        using cpp_type = typename traits::template arg<nargs - 1>::type;
        using cpp_map = TypeTraits<cpp_type>;

        if (iter.input_dtype(nargs - 1) != cpp_map::scalar_type) {
            return true;
        }
        return needs_dynamic_casting<func_t, nargs - 1>::check(iter);
    }
};

template <typename func_t>
struct needs_dynamic_casting<func_t, 0> {
    static bool check(TensorIteratorBase& iter) {
        using traits = tensorplay::function_traits<func_t>;
        using cpp_type = typename traits::result_type;

        if constexpr (std::is_void_v<cpp_type>) {
            return false;
        } else {
            return iter.dtype(0) != TypeTraits<cpp_type>::scalar_type;
        }
    }
};

// Shared kernel body: load arguments through the policy, evaluate the
// functor once per slot, store results through the policy.
template <int elems_per_thread, typename func_t, typename policy_t>
__device__ inline void elementwise_kernel_helper(func_t f, policy_t policy) {
    using traits = tensorplay::function_traits<func_t>;
    using return_t = typename traits::result_type;
    using args_t = typename traits::ArgsTuple;

    return_t results[elems_per_thread];
    args_t args[elems_per_thread];

    policy.load(args, blockIdx.x);

#pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        if (policy.check_inbounds(i)) {
            results[i] = detail::tuple_apply(f, args[i]);
        }
    }

    policy.store(results, blockIdx.x);
}

namespace detail {

// Runtime dtypes of a binary specialization: {output, input0, input1}.
constexpr std::array<std::array<ScalarType, 3>, 6> rt_binary_specializations = {
    {{TypeTraits<float>::scalar_type, TypeTraits<float>::scalar_type,
      TypeTraits<BFloat16>::scalar_type},
     {TypeTraits<float>::scalar_type, TypeTraits<BFloat16>::scalar_type,
      TypeTraits<float>::scalar_type},
     {TypeTraits<BFloat16>::scalar_type, TypeTraits<BFloat16>::scalar_type,
      TypeTraits<float>::scalar_type},
     {TypeTraits<float>::scalar_type, TypeTraits<float>::scalar_type,
      TypeTraits<Half>::scalar_type},
     {TypeTraits<float>::scalar_type, TypeTraits<Half>::scalar_type,
      TypeTraits<float>::scalar_type},
     {TypeTraits<Half>::scalar_type, TypeTraits<Half>::scalar_type,
      TypeTraits<float>::scalar_type}}};

// True when the functor's argument types match the specialization types
// (checked recursively through its ArgsTuple), so the runtime dtype switch
// can be hoisted to the host.
template <typename func_t, typename FirstParamTy, typename SecondParamTy,
          size_t arity, size_t arg_num = 0>
struct check_binary_functor_types_for_specialization {
    static constexpr bool check() {
        if constexpr (arity != 2) {
            return false;
        } else if constexpr (arg_num == 0) {
            using SelectedType =
                std::tuple_element_t<0, typename tensorplay::function_traits<func_t>::ArgsTuple>;
            if constexpr (std::is_same_v<FirstParamTy, SelectedType>) {
                return check_binary_functor_types_for_specialization<
                    func_t, FirstParamTy, SecondParamTy, arity, 1>::check();
            }
        } else if constexpr (arg_num == 1) {
            using SelectedType2 =
                std::tuple_element_t<1, typename tensorplay::function_traits<func_t>::ArgsTuple>;
            if constexpr (std::is_same_v<SecondParamTy, SelectedType2>) {
                return check_binary_functor_types_for_specialization<
                    func_t, FirstParamTy, SecondParamTy, arity, 2>::check();
            }
        }
        return false;
    }
};

// Bottom case: all argument slots matched the specialization types.
template <typename func_t, typename FirstParamTy, typename SecondParamTy,
          size_t arity>
struct check_binary_functor_types_for_specialization<
    func_t, FirstParamTy, SecondParamTy, arity, arity> {
    static constexpr bool check() {
        if constexpr (arity != 0) {
            return true;
        }
        return false;
    }
};

// Nullary functor: never matches a binary specialization.
template <typename func_t, typename FirstParamTy, typename SecondParamTy>
struct check_binary_functor_types_for_specialization<
    func_t, FirstParamTy, SecondParamTy, 0, 0> {
    static constexpr bool check() { return false; }
};

inline bool check_binary_rt_types_for_specialization(
    const TensorIteratorBase& iter) {
    // The specialized bodies invoke the functor with two converted
    // operands, so scalar-folded unary wrappers never take this path; they
    // run through the generic dynamic-cast kernel below.
    if (iter.ninputs() != 2) return false;
    if (iter.is_cpu_scalar(1) || iter.is_cpu_scalar(2)) return false;
    for (const auto& spec : rt_binary_specializations) {
        if (iter.dtype(0) == spec[0] && iter.input_dtype(0) == spec[1] &&
            iter.input_dtype(1) == spec[2]) {
            return true;
        }
    }
    return false;
}

}  // namespace detail

// -----------------------------------------------------------------------
// Kernel bodies
// -----------------------------------------------------------------------

template <int vec_size, int tws, typename func_t, typename array_t>
__device__ inline void vectorized_elementwise_kernel_impl(int N, func_t f,
                                                          array_t data) {
    int remaining = N - kLoopBlockWorkSize * blockIdx.x;

    if (remaining < kLoopBlockWorkSize) {
        // Tail block: unroll machinery with a per-element inbounds check.
        constexpr int kArity =
            tensorplay::function_traits<func_t>::arity;
        auto input_calc = TrivialOffsetCalculator<kArity>();
        auto output_calc = TrivialOffsetCalculator<1>();
        auto loader = memory::LoadWithoutCast();
        auto storer = memory::StoreWithoutCast();
        elementwise_kernel_helper<tws>(
            f,
            memory::policies::unroll<array_t, decltype(input_calc),
                                     decltype(output_calc),
                                     memory::LoadWithoutCast,
                                     memory::StoreWithoutCast, tws>(
                data, remaining, input_calc, output_calc, loader, storer));
    } else {
        elementwise_kernel_helper<tws>(
            f,
            memory::policies::vectorized<vec_size, array_t, tws>(data));
    }
}

template <int vec_size, typename func_t, typename array_t>
TP_LOOP_LAUNCH_BOUNDS(kLoopNumThreads)
__global__ void vectorized_elementwise_kernel(int N, func_t f, array_t data) {
    vectorized_elementwise_kernel_impl<vec_size, kLoopThreadWorkSize>(
        N, f, data);
}

template <int elems_per_thread, typename func_t, typename array_t,
          typename inp_calc_t, typename out_calc_t, typename loader_t,
          typename storer_t>
TP_LOOP_LAUNCH_BOUNDS(kLoopNumThreads)
__global__ void unrolled_elementwise_kernel(int N, func_t f, array_t data,
                                            inp_calc_t ic, out_calc_t oc,
                                            loader_t l, storer_t s) {
    int remaining =
        N - elems_per_thread * loop_num_threads() * blockIdx.x;
    elementwise_kernel_helper<elems_per_thread>(
        f,
        memory::policies::unroll<array_t, inp_calc_t, out_calc_t, loader_t,
                                 storer_t, elems_per_thread>(data, remaining,
                                                             ic, oc, l, s));
}

// Trivial 1-D iteration, no dynamic casting: vec_size chosen by the
// alignment of the operand pointers.
template <typename func_t, typename array_t>
inline void launch_vectorized_kernel(int64_t N, const func_t& f,
                                     array_t data) {
    TP_CHECK(N > 0 && N <= std::numeric_limits<int32_t>::max(),
             "vectorized kernel range must fit 32-bit indexing");
    constexpr int tws = loop_thread_work_size();
    constexpr int threads = loop_num_threads();
    int64_t grid = (N + tws * static_cast<int64_t>(threads) - 1) /
                   (tws * static_cast<int64_t>(threads));
    auto stream = getCurrentCUDAStream().stream();
    // vec_size must divide the per-thread workload exactly (the vectorized
    // policy slices each thread's slots into whole vectors), so cap the
    // alignment-derived width at the fixed thread work size.
    int vec_size =
        std::min(memory::can_vectorize_up_to<func_t>(data), tws);
    switch (vec_size) {
        case 4:
            vectorized_elementwise_kernel<4, func_t, array_t>
                <<<static_cast<unsigned>(grid), kLoopNumThreads, 0, stream>>>(
                    static_cast<int>(N), f, data);
            loop_launch_check("vectorized_elementwise_kernel<4>");
            break;
        case 2:
            vectorized_elementwise_kernel<2, func_t, array_t>
                <<<static_cast<unsigned>(grid), kLoopNumThreads, 0, stream>>>(
                    static_cast<int>(N), f, data);
            loop_launch_check("vectorized_elementwise_kernel<2>");
            break;
        case 1: {
            auto input_calc = TrivialOffsetCalculator<2>();
            auto output_calc = TrivialOffsetCalculator<1>();
            auto loader = memory::LoadWithoutCast();
            auto storer = memory::StoreWithoutCast();
            int64_t grid_unrolled = (N + kLoopBlockWorkSize - 1) /
                                    kLoopBlockWorkSize;
            unrolled_elementwise_kernel<kLoopThreadWorkSize, func_t, array_t,
                                        decltype(input_calc),
                                        decltype(output_calc),
                                        memory::LoadWithoutCast,
                                        memory::StoreWithoutCast>
                <<<static_cast<unsigned>(grid_unrolled), kLoopNumThreads, 0,
                   stream>>>(static_cast<int>(N), f, data, input_calc,
                             output_calc, loader, storer);
            loop_launch_check("unrolled_elementwise_kernel");
            break;
        }
        default:
            TP_CHECK(false, "Unexpected vectorization size");
    }
}

template <int elems_per_thread, typename func_t, typename array_t,
          typename inp_calc_t, typename out_calc_t, typename loader_t,
          typename storer_t>
inline void launch_unrolled_kernel(int64_t N, const func_t& f, array_t data,
                                   inp_calc_t ic, out_calc_t oc, loader_t l,
                                   storer_t s) {
    TP_CHECK(N > 0 && N <= std::numeric_limits<int32_t>::max(),
             "unrolled kernel range must fit 32-bit indexing");
    int64_t grid = (N + kLoopBlockWorkSize - 1) / kLoopBlockWorkSize;
    unrolled_elementwise_kernel<elems_per_thread, func_t, array_t,
                                inp_calc_t, out_calc_t, loader_t, storer_t>
        <<<static_cast<unsigned>(grid), kLoopNumThreads, 0,
           getCurrentCUDAStream().stream()>>>(static_cast<int>(N), f, data, ic,
                                              oc, l, s);
    loop_launch_check("unrolled_elementwise_kernel");
}

template <int nt, int vt, typename func_t>
TP_LOOP_LAUNCH_BOUNDS(nt)
__global__ void elementwise_kernel(int N, func_t f) {
    int tid = threadIdx.x;
    int nv = nt * vt;
    int idx = nv * blockIdx.x + tid;
#pragma unroll
    for (int i = 0; i < vt; i++) {
        if (idx < N) {
            f(idx);
            idx += nt;
        }
    }
}

template <int nt, int vt, typename func_t>
inline void launch_legacy_kernel(int64_t N, const func_t& f) {
    TP_CHECK(N >= 0 && N <= std::numeric_limits<int32_t>::max(),
             "legacy kernel range must fit 32-bit indexing");
    if (N == 0) {
        return;
    }
    dim3 block(nt);
    dim3 grid(static_cast<unsigned>((N + block.x * vt - 1) / (block.x * vt)));
    elementwise_kernel<nt, vt, func_t>
        <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            static_cast<int>(N), f);
    loop_launch_check("elementwise_kernel");
}

template <typename index_t, typename func_t>
TP_LOOP_LAUNCH_BOUNDS(kLoopNumThreads)
__global__ void index_elementwise_kernel(
        index_t N, func_t f,
        typename tensorplay::function_traits<func_t>::result_type* data) {
    index_t index = static_cast<index_t>(blockIdx.x) *
                    static_cast<index_t>(kLoopBlockWorkSize) +
                    static_cast<index_t>(threadIdx.x);
#pragma unroll
    for (int i = 0; i < kLoopThreadWorkSize; ++i) {
        if (index < N) {
            data[index] = f(index);
            index += static_cast<index_t>(kLoopNumThreads);
        }
    }
}

template <typename func_t>
void gpu_kernel_with_index(Tensor& output, const func_t& f) {
    const int64_t N = output.numel();
    if (N == 0) {
        return;
    }

    using scalar_t = typename tensorplay::function_traits<func_t>::result_type;
    const int64_t grid =
        (N + kLoopBlockWorkSize - 1) / kLoopBlockWorkSize;
    auto stream = getCurrentCUDAStream().stream();
    if (N <= std::numeric_limits<int32_t>::max()) {
        index_elementwise_kernel<int, func_t>
            <<<static_cast<unsigned>(grid), kLoopNumThreads, 0, stream>>>(
                static_cast<int>(N), f, output.data_ptr<scalar_t>());
    } else {
        index_elementwise_kernel<int64_t, func_t>
            <<<static_cast<unsigned>(grid), kLoopNumThreads, 0, stream>>>(
                N, f, output.data_ptr<scalar_t>());
    }
    loop_launch_check("index_elementwise_kernel");
}

// Manual unroll: the full-block body issues vt groups of independent loads
// up front so the compiler interleaves them.
template <int nt, int vt, typename func_t>
TP_LOOP_LAUNCH_BOUNDS(nt)
__global__ void elementwise_kernel_manual_unroll(int N, func_t f) {
    int tid = threadIdx.x;
    constexpr int nv = nt * vt;
    int idx = nv * blockIdx.x + tid;
    if ((idx + nt * (vt - 1)) < N) {
        f(idx, true);
    } else {
#pragma unroll
        for (int i = 0; i < vt; i++) {
            if (idx < N) {
                f(idx, false);
                idx += nt;
            }
        }
    }
}

template <int nt, int vt, typename func_t>
inline void launch_legacy_kernel_manual_unroll(int64_t N, const func_t& f) {
    TP_CHECK(N >= 0 && N <= std::numeric_limits<int32_t>::max(),
             "legacy kernel range must fit 32-bit indexing");
    if (N == 0) {
        return;
    }
    dim3 block(nt);
    dim3 grid(static_cast<unsigned>((N + block.x * vt - 1) / (block.x * vt)));
    elementwise_kernel_manual_unroll<nt, vt, func_t>
        <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            static_cast<int>(N), f);
    loop_launch_check("elementwise_kernel_manual_unroll");
}

// -----------------------------------------------------------------------
// Invocation with and without dynamic casting
// -----------------------------------------------------------------------

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
__device__ inline typename traits::result_type invoke_impl(
    const func_t& f, char* const data[], const index_t strides[], int i,
    std::index_sequence<INDEX...>) {
    return f(tensorplay::detail::dynamic_load<
             typename traits::template arg<INDEX>::type>(
        data[INDEX] + i * strides[INDEX])...);
}

template <typename func_t, typename index_t,
          typename traits = tensorplay::function_traits<func_t>>
__device__ inline typename traits::result_type invoke(
    const func_t& f, char* const data[], const index_t strides[], int i) {
    using Indices = std::make_index_sequence<traits::arity>;
    return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <typename traits, typename func_t, typename index_t, size_t... I>
__device__ inline typename traits::result_type invoke_impl(
    const func_t& f, char* const data[], const index_t strides[],
    const ScalarType dtypes[], int i, std::index_sequence<I...>) {
    return f(tensorplay::fetch_and_cast<typename traits::template arg<I>::type>(
        dtypes[I], data[I] + i * strides[I])...);
}

template <typename func_t, typename index_t,
          typename traits = tensorplay::function_traits<func_t>>
__device__ inline typename traits::result_type invoke(
    const func_t& f, char* const data[], const index_t strides[],
    const ScalarType dtypes[], int i) {
    using Indices = std::make_index_sequence<traits::arity>;
    return invoke_impl<traits>(f, data, strides, dtypes, i, Indices{});
}

template <int NARGS>
inline OffsetCalculator<NARGS> make_offset_calculator(
    const TensorIteratorBase& iter) {
    constexpr int array_size = std::max<int>(NARGS, 1);
    std::array<const int64_t*, array_size> strides;
    int64_t element_sizes[array_size];
    for (int i = 0; i < NARGS; i++) {
        strides[i] = iter.strides(i).data();
        element_sizes[i] = iter.element_size(i);
    }
    return OffsetCalculator<NARGS>(iter.ndim(), iter.shape().data(),
                                   strides.data(), element_sizes);
}

// -----------------------------------------------------------------------
// gpu_kernel_impl
// -----------------------------------------------------------------------

template <typename func_t>
void gpu_kernel_impl_nocast(TensorIteratorBase& iter, const func_t& f) {
    using traits = tensorplay::function_traits<func_t>;
    using arg0_t = typename traits::result_type;
    constexpr int ntensors = traits::arity + 1;

    TP_CHECK(iter.can_use_32bit_indexing(),
             "iterator must be split for 32-bit indexing");
    TP_CHECK(iter.ninputs() == traits::arity, "input count mismatch");
    TP_CHECK(iter.noutputs() == 1, "single output required");

    std::array<char*, ntensors> data;
    for (int i = 0; i < ntensors; i++) {
        data[i] = static_cast<char*>(iter.data_ptr(i));
    }

    int64_t numel = iter.numel();

    bool contiguous = iter.is_contiguous();

    if (contiguous) {
        launch_vectorized_kernel(numel, f, data);
        return;
    }
    auto offset_calc = make_offset_calculator<traits::arity + 1>(iter);
    constexpr int unroll_factor = sizeof(arg0_t) >= 4 ? 4 : 8;
    constexpr int grp_sz = 128;
    launch_legacy_kernel_manual_unroll<grp_sz, unroll_factor>(
        numel, [=] __device__(int idx, bool unrl) {
            // Runtime branch over the compile-time constant: nvcc rejects
            // `if constexpr` here because the extended-lambda first-capture
            // rule trips inside the discarded-statement context.
            if (unroll_factor == 4) {
                auto offsets0 = offset_calc.get(idx);
                auto offsets1 = offset_calc.get(idx + grp_sz);
                auto offsets2 = offset_calc.get(idx + grp_sz * 2);
                auto offsets3 = offset_calc.get(idx + grp_sz * 3);
                arg0_t* out0 =
                    reinterpret_cast<arg0_t*>(data[0] + offsets0[0]);
                arg0_t* out1 =
                    reinterpret_cast<arg0_t*>(data[0] + offsets1[0]);
                arg0_t* out2 =
                    reinterpret_cast<arg0_t*>(data[0] + offsets2[0]);
                arg0_t* out3 =
                    reinterpret_cast<arg0_t*>(data[0] + offsets3[0]);
                auto tmp0 = invoke(f, &data[1], &offsets0[1], 1);
                auto tmp1 = invoke(f, &data[1], &offsets1[1], 1);
                auto tmp2 = invoke(f, &data[1], &offsets2[1], 1);
                auto tmp3 = invoke(f, &data[1], &offsets3[1], 1);
                *out0 = tmp0;
                *out1 = tmp1;
                *out2 = tmp2;
                *out3 = tmp3;
            } else {
                constexpr int ngroups = unroll_factor;
                arg0_t results[ngroups];
                auto offsets0 = offset_calc.get(idx);
                for (int g = 0; g < ngroups; g++) {
                    auto offsets = (g == 0)
                                       ? offsets0
                                       : offset_calc.get(idx + grp_sz * g);
                    results[g] = invoke(f, &data[1], &offsets[1], 1);
                }
                for (int g = 0; g < ngroups; g++) {
                    auto offsets = (g == 0)
                                       ? offsets0
                                       : offset_calc.get(idx + grp_sz * g);
                    *reinterpret_cast<arg0_t*>(data[0] + offsets[0]) =
                        results[g];
                }
            }
        });
}

namespace detail {

template <int arg_index>
struct type_specialized_broadcast_kernel_launcher {
    template <typename func_t, typename array_t, typename dtypes_t,
              typename calc_t>
    static void apply(int64_t numel, const func_t& f, array_t data,
                      dtypes_t dtypes, calc_t offset_calc) {
        using traits = tensorplay::function_traits<func_t>;
        using ret_t = typename traits::result_type;
        using arg0_t = typename traits::template arg<0>::type;
        using arg1_t = typename traits::template arg<1>::type;
        if (dtypes[0] == rt_binary_specializations[arg_index][0] &&
            dtypes[1] == rt_binary_specializations[arg_index][1] &&
            dtypes[2] == rt_binary_specializations[arg_index][2]) {
            using ret_cpp_t = typename TypeTraitsOf<
                rt_binary_specializations[arg_index][0]>::type;
            using arg0_cpp_t = typename TypeTraitsOf<
                rt_binary_specializations[arg_index][1]>::type;
            using arg1_cpp_t = typename TypeTraitsOf<
                rt_binary_specializations[arg_index][2]>::type;
            constexpr int grp_sz = 128;
            launch_legacy_kernel_manual_unroll<grp_sz, 4>(
                numel, [=] __device__(int idx, bool unrl) {
                    if (unrl) {
                        auto offsets0 = offset_calc.get(idx);
                        auto offsets1 = offset_calc.get(idx + grp_sz);
                        auto offsets2 = offset_calc.get(idx + grp_sz * 2);
                        auto offsets3 = offset_calc.get(idx + grp_sz * 3);
                        void* out0 = data[0] + offsets0[0];
                        void* out1 = data[0] + offsets1[0];
                        void* out2 = data[0] + offsets2[0];
                        void* out3 = data[0] + offsets3[0];
                        auto u = tensorplay::detail::dynamic_load<arg0_cpp_t>(
                            data[1] + offsets0[1]);
                        auto v = tensorplay::detail::dynamic_load<arg1_cpp_t>(
                            data[2] + offsets0[2]);
                        ret_t result0 =
                            f(tensorplay::detail::scalar_convert<arg0_t>(u),
                              tensorplay::detail::scalar_convert<arg1_t>(v));
                        auto u1 = tensorplay::detail::dynamic_load<arg0_cpp_t>(
                            data[1] + offsets1[1]);
                        auto v1 = tensorplay::detail::dynamic_load<arg1_cpp_t>(
                            data[2] + offsets1[2]);
                        ret_t result1 =
                            f(tensorplay::detail::scalar_convert<arg0_t>(u1),
                              tensorplay::detail::scalar_convert<arg1_t>(v1));
                        auto u2 = tensorplay::detail::dynamic_load<arg0_cpp_t>(
                            data[1] + offsets2[1]);
                        auto v2 = tensorplay::detail::dynamic_load<arg1_cpp_t>(
                            data[2] + offsets2[2]);
                        ret_t result2 =
                            f(tensorplay::detail::scalar_convert<arg0_t>(u2),
                              tensorplay::detail::scalar_convert<arg1_t>(v2));
                        auto u3 = tensorplay::detail::dynamic_load<arg0_cpp_t>(
                            data[1] + offsets3[1]);
                        auto v3 = tensorplay::detail::dynamic_load<arg1_cpp_t>(
                            data[2] + offsets3[2]);
                        ret_t result3 =
                            f(tensorplay::detail::scalar_convert<arg0_t>(u3),
                              tensorplay::detail::scalar_convert<arg1_t>(v3));
                        *reinterpret_cast<ret_cpp_t*>(out0) =
                            tensorplay::detail::scalar_convert<ret_cpp_t>(
                                result0);
                        *reinterpret_cast<ret_cpp_t*>(out1) =
                            tensorplay::detail::scalar_convert<ret_cpp_t>(
                                result1);
                        *reinterpret_cast<ret_cpp_t*>(out2) =
                            tensorplay::detail::scalar_convert<ret_cpp_t>(
                                result2);
                        *reinterpret_cast<ret_cpp_t*>(out3) =
                            tensorplay::detail::scalar_convert<ret_cpp_t>(
                                result3);
                    } else {
                        auto offsets = offset_calc.get(idx);
                        void* out = data[0] + offsets[0];
                        auto u = tensorplay::detail::dynamic_load<arg0_cpp_t>(
                            data[1] + offsets[1]);
                        auto v = tensorplay::detail::dynamic_load<arg1_cpp_t>(
                            data[2] + offsets[2]);
                        ret_t result =
                            f(tensorplay::detail::scalar_convert<arg0_t>(u),
                              tensorplay::detail::scalar_convert<arg1_t>(v));
                        *reinterpret_cast<ret_cpp_t*>(out) =
                            tensorplay::detail::scalar_convert<ret_cpp_t>(
                                result);
                    }
                });
        }
    }
};

}  // namespace detail

// Host-side expansion over the specialization table: expands to
// Launcher<I>::apply(...) for every table row.
template <template <int i> typename Launcher, int I, int END>
struct detail_for_loop {
    template <typename... Args>
    static void apply(Args&&... args) {
        Launcher<I>::apply(std::forward<Args>(args)...);
        detail_for_loop<Launcher, I + 1, END>::apply(
            std::forward<Args>(args)...);
    }
};

template <template <int i> typename Launcher, int END>
struct detail_for_loop<Launcher, END, END> {
    template <typename... Args>
    static void apply(Args...) {}
};

template <typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
    if (!needs_dynamic_casting<func_t>::check(iter)) {
        return gpu_kernel_impl_nocast(iter, f);
    }
    using traits = tensorplay::function_traits<func_t>;
    using arg0_t = typename traits::result_type;
    constexpr int ntensors = traits::arity + 1;

    TP_CHECK(iter.can_use_32bit_indexing(),
             "iterator must be split for 32-bit indexing");
    TP_CHECK(iter.ninputs() == traits::arity, "input count mismatch");
    TP_CHECK(iter.noutputs() == 1, "single output required");

    std::array<char*, ntensors> data;
    for (int i = 0; i < ntensors; i++) {
        data[i] = static_cast<char*>(iter.data_ptr(i));
    }

    int64_t numel = iter.numel();

    bool contiguous = iter.is_contiguous();

    if (contiguous) {
        std::array<ScalarType, ntensors> dtypes;
        auto inner_strides = iter.get_inner_strides();
        std::array<int, ntensors> strides;
        for (int i = 0; i < ntensors; i++) {
            dtypes[i] = iter.dtype(i);
            strides[i] = static_cast<int>(inner_strides[i]);
        }
        constexpr int grp_sz = 128;
        launch_legacy_kernel_manual_unroll<grp_sz, 4>(
            numel, [=] __device__(int idx, bool unrl) {
                if (unrl) {
                    void* out0 = data[0] + strides[0] * idx;
                    void* out1 = data[0] + strides[0] * (idx + grp_sz);
                    void* out2 = data[0] + strides[0] * (idx + grp_sz * 2);
                    void* out3 = data[0] + strides[0] * (idx + grp_sz * 3);
                    arg0_t result0 =
                        invoke(f, &data[1], &strides[1], &dtypes[1], idx);
                    arg0_t result1 = invoke(f, &data[1], &strides[1],
                                            &dtypes[1], (idx + grp_sz));
                    arg0_t result2 = invoke(f, &data[1], &strides[1],
                                            &dtypes[1], (idx + grp_sz * 2));
                    arg0_t result3 = invoke(f, &data[1], &strides[1],
                                            &dtypes[1], (idx + grp_sz * 3));
                    tensorplay::cast_and_store<arg0_t>(dtypes[0], out0,
                                                       result0);
                    tensorplay::cast_and_store<arg0_t>(dtypes[0], out1,
                                                       result1);
                    tensorplay::cast_and_store<arg0_t>(dtypes[0], out2,
                                                       result2);
                    tensorplay::cast_and_store<arg0_t>(dtypes[0], out3,
                                                       result3);
                } else {
                    void* out = data[0] + strides[0] * idx;
                    arg0_t result =
                        invoke(f, &data[1], &strides[1], &dtypes[1], idx);
                    tensorplay::cast_and_store<arg0_t>(dtypes[0], out,
                                                       result);
                }
            });
    } else {
        std::array<ScalarType, ntensors> dtypes;
        for (int i = 0; i < ntensors; i++) {
            dtypes[i] = iter.dtype(i);
        }
        auto offset_calc = make_offset_calculator<traits::arity + 1>(iter);
        // The specialized bodies invoke the functor with two operands, so
        // the arity test is compile-time: scalar-folded unary wrappers (and
        // any other arity) never instantiate this branch and keep the
        // generic dynamic-cast kernel below.
        if constexpr (traits::arity == 2) {
            if (detail::check_binary_rt_types_for_specialization(iter)) {
                detail_for_loop<detail::type_specialized_broadcast_kernel_launcher,
                                0,
                                static_cast<int>(
                                    detail::rt_binary_specializations.size())>::
                    apply(numel, f, data, dtypes, offset_calc);
                return;
            }
        }
        constexpr int grp_sz = 128;
        launch_legacy_kernel_manual_unroll<grp_sz, 4>(
            numel, [=] __device__(int idx, bool unrl) {
                if (unrl) {
                    auto offsets0 = offset_calc.get(idx);
                    auto offsets1 = offset_calc.get(idx + grp_sz);
                    auto offsets2 = offset_calc.get(idx + grp_sz * 2);
                    auto offsets3 = offset_calc.get(idx + grp_sz * 3);
                    void* out0 = data[0] + offsets0[0];
                    void* out1 = data[0] + offsets1[0];
                    void* out2 = data[0] + offsets2[0];
                    void* out3 = data[0] + offsets3[0];
                    arg0_t result0 =
                        invoke(f, &data[1], &offsets0[1], &dtypes[1], 1);
                    arg0_t result1 =
                        invoke(f, &data[1], &offsets1[1], &dtypes[1], 1);
                    arg0_t result2 =
                        invoke(f, &data[1], &offsets2[1], &dtypes[1], 1);
                    arg0_t result3 =
                        invoke(f, &data[1], &offsets3[1], &dtypes[1], 1);
                    tensorplay::cast_and_store<arg0_t>(dtypes[0], out0,
                                                       result0);
                    tensorplay::cast_and_store<arg0_t>(dtypes[0], out1,
                                                       result1);
                    tensorplay::cast_and_store<arg0_t>(dtypes[0], out2,
                                                       result2);
                    tensorplay::cast_and_store<arg0_t>(dtypes[0], out3,
                                                       result3);
                } else {
                    auto offsets = offset_calc.get(idx);
                    void* out = data[0] + offsets[0];
                    arg0_t result =
                        invoke(f, &data[1], &offsets[1], &dtypes[1], 1);
                    tensorplay::cast_and_store<arg0_t>(dtypes[0], out,
                                                       result);
                }
            });
    }
}

// -----------------------------------------------------------------------
// gpu_kernel entry points
// -----------------------------------------------------------------------

template <int N>
inline OffsetCalculator<N> make_input_offset_calculator(
    const TensorIteratorBase& iter) {
    constexpr int array_size = std::max<int>(N, 1);
    std::array<const int64_t*, array_size> strides;
    int64_t element_sizes[array_size];
    for (int i = 0; i < N; i++) {
        strides[i] = iter.strides(i + iter.noutputs()).data();
        element_sizes[i] = iter.element_size(i + iter.noutputs());
    }
    return OffsetCalculator<N>(iter.ndim(), iter.shape().data(),
                               strides.data(), element_sizes);
}

template <int num_outputs = 1>
inline OffsetCalculator<num_outputs> make_output_offset_calculator(
    const TensorIteratorBase& iter) {
    std::array<const int64_t*, num_outputs> strides;
    int64_t element_sizes[num_outputs];
    for (int i = 0; i < num_outputs; i++) {
        strides[i] = iter.strides(i).data();
        element_sizes[i] = iter.element_size(i);
    }
    return OffsetCalculator<num_outputs>(iter.ndim(), iter.shape().data(),
                                         strides.data(), element_sizes);
}

template <typename func_t>
void gpu_kernel_nocast(TensorIteratorBase& iter, const func_t& f) {
    for (int arg = 0; arg < iter.ntensors(); arg++) {
        TP_CHECK(iter.device(arg).type() == DeviceType::CUDA,
                 "expected a CUDA device for argument ", arg);
    }

    if (iter.numel() == 0) {
        return;
    }

    if (!iter.can_use_32bit_indexing()) {
        for (auto& sub_iter : iter.with_32bit_indexing()) {
            gpu_kernel_nocast(sub_iter, f);
        }
        return;
    }

    gpu_kernel_impl_nocast(iter, f);
}

template <typename func_t>
void gpu_kernel(TensorIteratorBase& iter, const func_t& f) {
    for (int arg = 0; arg < iter.ntensors(); arg++) {
        TP_CHECK(iter.device(arg).type() == DeviceType::CUDA,
                 "expected a CUDA device for argument ", arg);
    }

    if (iter.numel() == 0) {
        return;
    }

    if (!iter.can_use_32bit_indexing()) {
        for (auto& sub_iter : iter.with_32bit_indexing()) {
            gpu_kernel(sub_iter, f);
        }
        return;
    }

    gpu_kernel_impl(iter, f);
}

// Scalar-folded functors: a CPU scalar operand is lifted into a kernel
// parameter (kept in the functor's compute precision) and the operand is
// removed from the iteration.
template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct AUnaryFunctor {
    using traits = tensorplay::function_traits<func_t>;
    using opmath_arg1_t = typename traits::template arg<0>::type;
    __device__ return_t operator()(arg2_t b) const { return f(a, b); }
    // The scalar is stored in the higher compute precision.
    AUnaryFunctor(func_t f_, opmath_arg1_t a_) : f(f_), a(a_) {}

  private:
    func_t f;
    opmath_arg1_t a;
};

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BUnaryFunctor {
    using traits = tensorplay::function_traits<func_t>;
    using opmath_arg2_t = typename traits::template arg<1>::type;
    __device__ return_t operator()(arg1_t a) const { return f(a, b); }
    // The scalar is stored in the higher compute precision.
    BUnaryFunctor(func_t f_, opmath_arg2_t b_) : f(f_), b(b_) {}

  private:
    func_t f;
    opmath_arg2_t b;
};

// Inserts casts from arg1_t/arg2_t into the functor's own argument types
// (which may be higher precision), and into return_t.
template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BinaryFunctor {
    __device__ return_t operator()(arg1_t a, arg2_t b) const {
        return f(a, b);
    }
    BinaryFunctor(func_t f_) : f(f_) {}

  private:
    func_t f;
};

template <typename arg1_t, typename arg2_t = arg1_t, typename return_t = arg1_t,
          typename func_t>
void opmath_gpu_kernel_with_scalars(TensorIteratorBase& iter,
                                    const func_t& f) {
    TP_CHECK(iter.ntensors() == 3, "expected one output and two inputs");

    using traits = tensorplay::function_traits<func_t>;
    using opmath_arg1_t = typename traits::template arg<0>::type;
    using opmath_arg2_t = typename traits::template arg<1>::type;
    static_assert(traits::arity == 2,
                  "gpu_kernel_with_scalars only supports two input arguments");

    if (iter.is_cpu_scalar(1)) {
        AUnaryFunctor<arg1_t, arg2_t, return_t, func_t> af(
            f, iter.scalar_value<opmath_arg1_t>(1));
        iter.remove_operand(1);
        gpu_kernel(iter, af);
    } else if (iter.is_cpu_scalar(2)) {
        BUnaryFunctor<arg1_t, arg2_t, return_t, func_t> bf(
            f, iter.scalar_value<opmath_arg2_t>(2));
        iter.remove_operand(2);
        gpu_kernel(iter, bf);
    } else {
        gpu_kernel(iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
    }
}

template <typename scalar_t, typename return_t = scalar_t, typename func_t>
void opmath_symmetric_gpu_kernel_with_scalars(TensorIteratorBase& iter,
                                              const func_t& f) {
    // Requires f(a, b) == f(b, a); halves the generated kernel count.
    TP_CHECK(iter.ntensors() == 3, "expected one output and two inputs");

    using traits = tensorplay::function_traits<func_t>;
    using opmath_arg_t = typename traits::template arg<0>::type;
    static_assert(traits::arity == 2,
                  "gpu_kernel_with_scalars only supports two input arguments");
    static_assert(
        std::is_same_v<opmath_arg_t, typename traits::template arg<1>::type>,
        "f is not symmetric");

    opmath_arg_t scalar_val{};

    if (iter.is_cpu_scalar(1)) {
        scalar_val = iter.scalar_value<opmath_arg_t>(1);
        iter.remove_operand(1);
    } else if (iter.is_cpu_scalar(2)) {
        scalar_val = iter.scalar_value<opmath_arg_t>(2);
        iter.remove_operand(2);
    }

    if (iter.ninputs() == 2) {
        gpu_kernel(iter, BinaryFunctor<scalar_t, scalar_t, return_t, func_t>(f));
    } else {
        AUnaryFunctor<scalar_t, scalar_t, return_t, func_t> unary_f(f,
                                                                    scalar_val);
        gpu_kernel(iter, unary_f);
    }
}

// Legacy variant assuming func_t's argument types already match the memory
// types to load.
template <typename func_t>
void gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
    using traits = tensorplay::function_traits<func_t>;
    static_assert(traits::arity == 2,
                  "gpu_kernel_with_scalars only supports two input arguments");
    using arg1_t = typename traits::template arg<0>::type;
    using arg2_t = typename traits::template arg<1>::type;
    using return_t = typename traits::result_type;
    opmath_gpu_kernel_with_scalars<arg1_t, arg2_t, return_t, func_t>(iter, f);
}

// -----------------------------------------------------------------------
// Multiple outputs
// -----------------------------------------------------------------------

namespace gpu_kernel_multiple_outputs_detail {

template <typename T>
struct is_tuple : std::false_type {};

template <typename... T>
struct is_tuple<std::tuple<T...>> : std::true_type {};

template <int num_outputs, typename func_t, typename array_t,
          typename inp_calc_t, typename out_calc_t>
TP_LOOP_LAUNCH_BOUNDS(kLoopNumThreads)
__global__ void unrolled_elementwise_kernel_for_multi_outputs(
    int N, func_t f, array_t data, inp_calc_t ic, out_calc_t oc) {
    int remaining = N - kLoopBlockWorkSize * blockIdx.x;
    elementwise_kernel_helper<kLoopThreadWorkSize>(
        f, memory::policies::multi_outputs_unroll<array_t, inp_calc_t,
                                                  out_calc_t, num_outputs>(
               data, remaining, ic, oc));
}

template <int num_outputs, typename func_t, typename array_t,
          typename inp_calc_t, typename out_calc_t>
inline void launch_unrolled_kernel_for_multi_outputs(
    int64_t N, const func_t& f, array_t data, inp_calc_t ic, out_calc_t oc) {
    TP_CHECK(N > 0 && N <= std::numeric_limits<int32_t>::max(),
             "kernel range must fit 32-bit indexing");
    int64_t grid = (N + kLoopBlockWorkSize - 1) / kLoopBlockWorkSize;
    unrolled_elementwise_kernel_for_multi_outputs<
        num_outputs, func_t, array_t, inp_calc_t, out_calc_t>
        <<<static_cast<unsigned>(grid), kLoopNumThreads, 0,
           getCurrentCUDAStream().stream()>>>(static_cast<int>(N), f, data, ic,
                                              oc);
    loop_launch_check("unrolled_elementwise_kernel_for_multi_outputs");
}

}  // namespace gpu_kernel_multiple_outputs_detail

template <typename func_t>
void gpu_kernel_multiple_outputs_impl(TensorIteratorBase& iter,
                                      const func_t& f) {
    using traits = tensorplay::function_traits<func_t>;
    using output_t = typename traits::result_type;
    static_assert(
        gpu_kernel_multiple_outputs_detail::is_tuple<output_t>::value,
        "f's return type must be std::tuple");
    constexpr int num_outputs = std::tuple_size_v<output_t>;
    constexpr int num_inputs = traits::arity;
    constexpr int ntensors = num_outputs + num_inputs;

    TP_CHECK(iter.can_use_32bit_indexing(),
             "iterator must be split for 32-bit indexing");
    TP_CHECK(iter.ntensors() == ntensors, "operand count mismatch");

    std::array<char*, ntensors> data;
    for (int i = 0; i < ntensors; i++) {
        data[i] = static_cast<char*>(iter.data_ptr(i));
    }

    int64_t numel = iter.numel();

    if (iter.is_contiguous()) {
        auto input_calc = TrivialOffsetCalculator<num_inputs>();
        auto output_calc = TrivialOffsetCalculator<num_outputs>();
        gpu_kernel_multiple_outputs_detail::
            launch_unrolled_kernel_for_multi_outputs<num_outputs>(
                numel, f, data, input_calc, output_calc);
        return;
    }
    auto input_calc = make_input_offset_calculator<num_inputs>(iter);
    auto output_calc = make_output_offset_calculator<num_outputs>(iter);
    gpu_kernel_multiple_outputs_detail::
        launch_unrolled_kernel_for_multi_outputs<num_outputs>(
            numel, f, data, input_calc, output_calc);
}

template <typename func_t>
void gpu_kernel_multiple_outputs(TensorIteratorBase& iter, const func_t& f) {
    if (iter.numel() == 0) {
        return;
    }
    if (!iter.can_use_32bit_indexing()) {
        for (auto& sub_iter : iter.with_32bit_indexing()) {
            gpu_kernel_multiple_outputs(sub_iter, f);
        }
        return;
    }
    gpu_kernel_multiple_outputs_impl(iter, f);
}

}  // namespace cuda
}  // namespace tensorplay

/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template<>
__forceinline__ __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
        "\t and.b32 %0, sela, %1;\n" 
        "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
    return res;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
__forceinline__ __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
    asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template<typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(static_cast<unsigned long long>(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(static_cast<unsigned long long>(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

// Split one float fragment into the big and small tf32 fragments a tensor
// core's fast-f32 path consumes: the big part is the float rounded to tf32,
// the small part is the residual, and the product of two floats is recovered
// as big*big + big*small + small*big.
template <typename TensorF, typename TensorHi, typename TensorLo>
__forceinline__ __device__ void split_f32_fragment(TensorF const& f,
                                                   TensorHi& hi,
                                                   TensorLo& lo) {
  cutlass::NumericConverterFastF32<> convert_;
#pragma unroll
  for (int i = 0; i < cute::size(f); ++i) {
    auto r = convert_(f.data()[i]);
    hi.data()[i] = r[0];
    lo.data()[i] = r[1];
  }
}

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
using SrcType = std::remove_cv_t<typename Tensor3::value_type>;
    if constexpr (std::is_same_v<SrcType, float>) {
        // The wide precision has no fragment form of its own, so the operand
        // fragments stay float until they are split into tf32 parts at the
        // register level and three products are accumulated per operand pair.
        // The hi/lo parts are per-K-slice so the register pressure of the
        // wide precision's larger fragments stays bounded.
        auto tCrA_f = recast<float>(tCrA);
        auto tCrB_f = recast<float>(tCrB);
        Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA_f);
        CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));        // M
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB_f);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));        // N
        if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
        if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
        #pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
                if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
            }
            auto a_slice = tCrA_f(_, _, i);
            auto b_slice = tCrB_f(_, _, i);
            auto a_hi = make_tensor_like(a_slice);
            auto a_lo = make_tensor_like(a_slice);
            auto b_hi = make_tensor_like(b_slice);
            auto b_lo = make_tensor_like(b_slice);
            split_f32_fragment(a_slice, a_hi, a_lo);
            split_f32_fragment(b_slice, b_hi, b_lo);
            cute::gemm(tiled_mma, a_hi, b_hi, acc);
            cute::gemm(tiled_mma, a_hi, b_lo, acc);
            cute::gemm(tiled_mma, a_lo, b_hi, acc);
        }
        return;
    }
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert a probability fragment stored in the accumulator layout into the
// A-operand register layout of the same tiled MMA.  For the 8-deep
// wide-precision instruction the two layouts assign the fragment's rows and
// columns to different lanes and registers, so the conversion is a register
// permutation within each group of four lanes.  The fragment holds four
// elements per mma step (one element pair per row), and the number of such
// groups varies with the tile's N extent, so the shuffle pattern is applied to
// each group of four linear elements in turn.
template <typename Tensor>
__forceinline__ __device__ auto convert_acc_to_Aregs(Tensor const& tCrA) {
    using value_type = typename Tensor::value_type;
    static_assert(std::is_same_v<value_type, float>);
    static_assert(decltype(size<0>(tCrA))::value == 4);
    constexpr int numel = decltype(size(tCrA))::value;
    static_assert(numel % 4 == 0);
    auto out = make_tensor_like(tCrA);
    const int lane = threadIdx.x % 32;
    const int d = lane % 4;
    const int lq = lane - d;
    #pragma unroll
    for (int g = 0; g < numel / 4; ++g) {
        float f00 = tCrA(4 * g + 0);
        float f02 = tCrA(4 * g + 2);
        float f01 = tCrA(4 * g + 1);
        float f03 = tCrA(4 * g + 3);
        unsigned long long e0 = ((unsigned long long)__float_as_uint(f00) << 32) | (unsigned long long)__float_as_uint(f02);
        unsigned long long e1 = ((unsigned long long)__float_as_uint(f01) << 32) | (unsigned long long)__float_as_uint(f03);
        unsigned long long v0 = __shfl_sync(0xffffffffffffffffull, e0, lq + d / 2);
        unsigned long long v1 = __shfl_sync(0xffffffffffffffffull, e1, lq + d / 2);
        unsigned long long v2 = __shfl_sync(0xffffffffffffffffull, e0, lq + d / 2 + 2);
        unsigned long long v3 = __shfl_sync(0xffffffffffffffffull, e1, lq + d / 2 + 2);
        unsigned long long sel_lo = (d % 2 == 0) ? v0 : v1;
        unsigned long long sel_hi = (d % 2 == 0) ? v2 : v3;
        out(4 * g + 0) = __uint_as_float(uint32_t(sel_lo >> 32));
        out(4 * g + 1) = __uint_as_float(uint32_t(sel_lo & 0xffffffff));
        out(4 * g + 2) = __uint_as_float(uint32_t(sel_hi >> 32));
        out(4 * g + 3) = __uint_as_float(uint32_t(sel_hi & 0xffffffff));
    }
    return out;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    using SrcType = std::remove_cv_t<typename Tensor3::value_type>;
    if constexpr (std::is_same_v<SrcType, float>) {
        // The probability fragment is already float; the value fragment is
        // loaded from smem and both are split into tf32 parts, with the three
        // tensor-core products accumulated into the output.  The hi/lo parts
        // are per-K-slice to keep the register pressure bounded.  The
        // probability fragment lives in the accumulator layout, so it is
        // permuted into the A-operand register layout first.
        auto tCrA_A = convert_acc_to_Aregs(tCrA);
        auto tCrB_f = recast<float>(tCrB);
        Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB_f);
        CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));        // N
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
        #pragma unroll
        for (int i = 0; i < size<2>(tCrA); ++i) {
            if (i < size<2>(tCrA) - 1) {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
            auto a_slice = tCrA_A(_, _, i);
            auto b_slice = tCrB_f(_, _, i);
            auto a_hi = make_tensor_like(recast<tfloat32_t>(a_slice));
            auto a_lo = make_tensor_like(recast<tfloat32_t>(a_slice));
            // The value fragment's slice layout is a strided view whose
            // codomain reaches past the per-thread fragment storage, so it is
            // copied into a compact tensor first; splitting the compact
            // tensor keeps every element inside the tensor-core operand.
            auto b_compact = make_tensor<float>(make_shape(Int<size<0>(b_slice)>{}, Int<size<1>(b_slice)>{}),
                                                make_stride(Int<size<1>(b_slice)>{}, _1{}));
            cute::copy(b_slice, b_compact);
            auto b_hi = make_tensor_like(b_compact);
            auto b_lo = make_tensor_like(b_compact);
            split_f32_fragment(a_slice, a_hi, a_lo);
            split_f32_fragment(b_compact, b_hi, b_lo);
            cute::gemm(tiled_mma, a_hi, b_hi, acc);
            cute::gemm(tiled_mma, a_hi, b_lo, acc);
            cute::gemm(tiled_mma, a_lo, b_hi, acc);
        }
        return;
    }
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    if constexpr (std::is_same_v<To_type, From_type>) {
        return tensor;
    } else {
        constexpr int numel = decltype(size(tensor))::value;
        cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
        // HACK: this requires tensor to be "contiguous"
        auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
        return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout> &tensor) {
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
    using value_t = typename Engine::value_type;
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_uint32 = recast<uint32_t>(tensor);
    #pragma unroll
    for (int i = 0; i < size(tensor_uint32); ++i) {
        tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// On SM80 and above, we can fuse fp32 -> fp16/bf16 conversion and relu into 1 instruction
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static_assert(std::is_same_v<To_type, cutlass::half_t> || std::is_same_v<To_type, cutlass::bfloat16_t>);
    static_assert(std::is_same_v<float, From_type>);
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_float2 = recast<float2>(tensor);
    Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
    #pragma unroll
    for (int i = 0; i < size(out_uint32); ++i) {
        out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
    }
    Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else
    Tensor out = FLASH_NAMESPACE::convert_type<To_type>(tensor);
    FLASH_NAMESPACE::relu_(out);
#endif
    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
    // TD [2023-04-13]: Strange that the code below can cause race condition.
    // I think it's because the copies are under an if statement.
    // if (Is_even_K) {
    //     #pragma unroll
    //     for (int m = 0; m < size<1>(S); ++m) {
    //         if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //             copy(tiled_copy, S(_, m, _), D(_, m, _));
    //         } else if (Clear_OOB_MN) {
    //             clear(D(_, m, _));
    //         }
    //     }
    // } else {  // It's slightly faster in this case if iterate over K first
    //     #pragma unroll
    //     for (int k = 0; k < size<2>(S); ++k) {
    //         if (predicate_K(k)) {
    //             #pragma unroll
    //             for (int m = 0; m < size<1>(S); ++m) {
    //                 if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //                     copy(tiled_copy, S(_, m, k), D(_, m, k));
    //                 } else if (Clear_OOB_MN) {
    //                     clear(D(_, m, k));
    //                 }
    //             }
    //         } else if (Clear_OOB_K) {  // There's no case where !Clear_OOB_K && Clear_OOB_MN
    //             if (Clear_OOB_MN || Is_even_MN) {
    //                 clear(D(_, _, k));
    //             } else {
    //                 #pragma unroll
    //                 for (int m = 0; m < size<1>(S); ++m) {
    //                     if (!(Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN)) {
    //                         clear(D(_, m, k));
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S,
                                      Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                      Tensor<Engine3, Layout3> const &predicate_K,
                                      const int max_MN=0, const int min_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, max_MN = %d, min_MN = %d\n", blockIdx.y, max_MN, min_MN); }
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("Inner loop, blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_softcap(Tensor<Engine, Layout> &tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
    }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void calculate_dtanh(Tensor<Engine0, Layout0> &src_tensor, Tensor<Engine1, Layout1> &dst_tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(src_tensor); ++i) {
        dst_tensor(i) = (1.f - (src_tensor(i) * src_tensor(i))) * softcap;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE

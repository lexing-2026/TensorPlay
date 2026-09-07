#pragma once

// Per-argument offset computation for strided elementwise kernels: a linear
// index over the coalesced iteration space is decomposed into per-dimension
// coordinates (each via one multiply-shift divmod) and recombined into the
// element offset of every operand.  Broadcast dimensions carry stride zero,
// so one operand coordinate stays fixed while the others advance.
//
// Strides are normalized from bytes to elements at construction time using
// the per-operand element sizes; pass nullptr to keep byte strides.

#include "TensorIterator.h"
#include "IntegerDivider.cuh"


#include <array>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIP__)
#define TP_OC_HOST_DEVICE __host__ __device__
#else
#define TP_OC_HOST_DEVICE
#endif

namespace tensorplay {
namespace cuda {

#if defined(USE_ROCM)
constexpr int MAX_DIMS = 16;
#else
constexpr int MAX_DIMS = 25;
#endif

template <int NARGS, typename index_t = uint32_t, bool signed_strides = false>
struct OffsetCalculator {
  // Negative strides implement views with reversed direction; keep them
  // representable when requested.
  using stride_t = std::conditional_t<signed_strides,
                                      std::make_signed_t<index_t>,
                                      index_t>;
  using offset_type = std::array<stride_t, std::max<int>(NARGS, 1)>;

  OffsetCalculator(int dims, const int64_t* sizes,
                   const int64_t* const* strides,
                   const int64_t* element_sizes = nullptr)
      : dims(dims) {
    TP_CHECK(dims <= MAX_DIMS, "tensor has too many (>", MAX_DIMS, ") dims");
    for (int i = 0; i < dims; i++) {
      sizes_[i] = detail::IntDividerU32(static_cast<uint32_t>(sizes[i]));
      for (int arg = 0; arg < NARGS; arg++) {
        int64_t element_size =
            (element_sizes == nullptr ? 1LL : element_sizes[arg]);
        strides_[i][arg] = static_cast<stride_t>(strides[arg][i] / element_size);
      }
    }
  }

  TP_OC_HOST_DEVICE offset_type get(index_t linear_idx) const {
    offset_type offsets;

#if defined(USE_ROCM)
    if (dims > 0 && dims <= 2) {
      auto divmod = sizes_[0].divmod(linear_idx);
#pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] = divmod.mod * strides_[0][arg];
      }
      if (dims == 2) {
        divmod = sizes_[1].divmod(divmod.div);
#pragma unroll
        for (int arg = 0; arg < NARGS; arg++) {
          offsets[arg] += divmod.mod * strides_[1][arg];
        }
      }
      return offsets;
    }
#endif

#pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = 0;
    }
#pragma unroll
    for (int dim = 0; dim < MAX_DIMS; ++dim) {
      if (dim == dims) {
        break;
      }
      auto divmod = sizes_[dim].divmod(linear_idx);
      linear_idx = divmod.div;
#pragma unroll
      for (int arg = 0; arg < NARGS; arg++) {
        offsets[arg] += divmod.mod * strides_[dim][arg];
      }
    }
    return offsets;
  }

  int dims;
  cuda::detail::IntDividerU32 sizes_[MAX_DIMS];
  stride_t strides_[MAX_DIMS][std::max<int>(NARGS, 1)];
};

template <int NARGS, typename index_t = uint32_t>
struct TrivialOffsetCalculator {
  using offset_type = std::array<index_t, std::max<int>(NARGS, 1)>;

  TrivialOffsetCalculator() = default;
  TrivialOffsetCalculator(const OffsetCalculator<NARGS, index_t>&) {}

  TP_OC_HOST_DEVICE offset_type get(index_t linear_idx) const {
    offset_type offsets;
#pragma unroll
    for (int arg = 0; arg < NARGS; arg++) {
      offsets[arg] = linear_idx;
    }
    return offsets;
  }
};

}  // namespace cuda
}  // namespace tensorplay

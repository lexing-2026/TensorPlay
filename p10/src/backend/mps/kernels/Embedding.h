#pragma once
#include "common.h"

template <typename idx_type_t = uint32_t>
struct EmbeddingDenseBackwardParams {
  ::p10::metal::array<idx_type_t, ::p10::metal::max_ndim> outer_sizes;
  ::p10::metal::array<idx_type_t, ::p10::metal::max_ndim> grad_outer_strides;
  ::p10::metal::array<idx_type_t, ::p10::metal::max_ndim> indices_strides;
  idx_type_t grad_feature_stride;
  idx_type_t outer_ndim;
  idx_type_t feature_size;
  int64_t padding_idx;
  bool scale_grad_by_freq;
};
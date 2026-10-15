#pragma once
#include "common.h"

enum class EmbeddingBagMode { SUM = 0, MEAN, MAX };

template <typename idx_type_t = uint32_t>
struct EmbeddingBagParams {
  ::p10::metal::array<idx_type_t, 2> weight_strides;
  ::p10::metal::array<idx_type_t, 2> output_strides;
  ::p10::metal::array<idx_type_t, 2> max_indices_strides;

  bool use_per_sample_weights;
  idx_type_t per_sample_weights_stride;

  idx_type_t num_indices;
  idx_type_t num_bags;
  idx_type_t feature_size;
  idx_type_t num_weights;

  EmbeddingBagMode mode;
  int64_t padding_idx;
};

template <typename idx_type_t = uint32_t>
struct EmbeddingBagBackwardParams {
  ::p10::metal::array<idx_type_t, 2> weight_grad_strides;
  ::p10::metal::array<idx_type_t, 2> output_grad_strides;
  ::p10::metal::array<idx_type_t, 2> max_indices_strides;
  bool use_per_sample_weights;
  idx_type_t per_sample_weights_stride;
  idx_type_t feature_size;
  EmbeddingBagMode mode;
  int64_t padding_idx;
};

template <typename idx_type_t = uint32_t>
struct EmbeddingBagPerSampleWeightsBackwardParams {
  ::p10::metal::array<idx_type_t, 2> output_grad_strides;
  ::p10::metal::array<idx_type_t, 2> weight_strides;
  idx_type_t per_sample_weights_grad_stride;
  idx_type_t feature_size;
  int64_t padding_idx;
};
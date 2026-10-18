#pragma once
// The Metal Standard Library provides the fixed-width integer types; the host
// translation unit gets them from the standard header.
#ifndef __METAL__
#include <cstdint>
#endif

#include "common.h"

// Geometry of one group-normalization launch. The input is read as
// (batch, group, channel-in-group, spatial); a group is the last two
// dimensions together, so `elements_per_group` is the length each threadgroup
// reduces and `hxw` is the spatial length inside it.
struct GroupNormParams {
  uint32_t hxw;
  uint32_t channels_per_group;
  uint32_t elements_per_group;
  uint32_t num_groups;
};

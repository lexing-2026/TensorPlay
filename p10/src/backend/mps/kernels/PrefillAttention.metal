// Flash-attention prefill shaders (long query sequence).  Uses
// simdgroup-matrix MMA; supports causal/additive/bool masks and GQA/MQA.

#include "utils.h"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace p10::metal;
#include "PrefillAttention.h"
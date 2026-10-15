// Decode-phase scaled dot product attention shaders (short Q sequence).
// The kernel templates and instantiations live in DecodeAttention.h.

#include "utils.h"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
#include "DecodeAttention.h"
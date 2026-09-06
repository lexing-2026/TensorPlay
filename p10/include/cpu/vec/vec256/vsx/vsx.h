#pragma once

// Umbrella for the VSX vec layer: helpers plus the float/double/int
// specializations. Include via cpu/vec/vec.h; the caller has already pulled
// in <altivec.h> through cpu/vec/intrinsics.h.

#include "cpu/vec/vec256/vsx/vsx_helpers.h"
#include "cpu/vec/SleefShims.h"
#include "cpu/vec/vec256/vsx/float_vsx.h"
#include "cpu/vec/vec256/vsx/double_vsx.h"
#include "cpu/vec/vec256/vsx/int_vsx.h"

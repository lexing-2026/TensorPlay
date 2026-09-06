#pragma once

// Umbrella for the ZVECTOR vec layer: helpers plus the float/double/int
// specializations. Include via cpu/vec/vec.h; <vecintrin.h> is pulled in by
// cpu/vec/intrinsics.h when the vector facility is enabled.

#include "cpu/vec/vec256/zarch/zarch_helpers.h"
#include "cpu/vec/SleefShims.h"
#include "cpu/vec/vec256/zarch/float_zarch.h"
#include "cpu/vec/vec256/zarch/double_zarch.h"
#include "cpu/vec/vec256/zarch/int_zarch.h"
#include "cpu/vec/vec256/zarch/complex_zarch.h"

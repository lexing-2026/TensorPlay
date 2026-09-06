#pragma once

// Umbrella for the aarch64 NEON vec tier: helpers plus the float/double/
// int specializations. Include via cpu/vec/vec.h; <arm_neon.h> is pulled in
// by cpu/vec/intrinsics.h on __aarch64__.

#include "cpu/vec/vec128/neon_helpers.h"
#include "cpu/vec/SleefShims.h"
#include "cpu/vec/vec128/float_neon.h"
#include "cpu/vec/vec128/double_neon.h"
#include "cpu/vec/vec128/int_neon.h"

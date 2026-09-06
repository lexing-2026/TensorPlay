#pragma once

// Umbrella for the SVE vec layer: helpers plus the float/double/int
// specializations. Include via cpu/vec/vec.h; <arm_sve.h> is pulled in by
// cpu/vec/intrinsics.h when __ARM_FEATURE_SVE is set.

#include "cpu/vec/sve/sve_helpers.h"
#include "cpu/vec/SleefShims.h"
#include "cpu/vec/sve/float_sve.h"
#include "cpu/vec/sve/double_sve.h"
#include "cpu/vec/sve/int_sve.h"

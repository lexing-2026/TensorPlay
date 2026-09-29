#pragma once

// The forward parameters, and the two leaves the dispatcher calls.
//
// Only the parameter structure and the leaf declarations are needed to write
// the dispatcher, and both are in the reference's own shared header.  Nothing
// that defines a kernel is reachable from here, which is what lets the
// dispatcher be a host translation unit.

// The parameter header names the CUDA stream type and the cutlass element
// types, and assumes a toolchain that has already declared them.
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>

#define FLASH_NAMESPACE tensorplay_native_flash
#include "flash/flash.h"
#include "flash/static_switch.h"
#undef FLASH_NAMESPACE

#pragma once

// The error check the copied schedule headers use.
//
// They are copies, so their checks are rewritten to this tree's: the names they
// reach for belong to a build this is not, and spelling them here keeps the
// copies free of any name that is not ours.  A CUDA result that is not success
// is an error, and it is reported the way this tree reports errors.

#include <cuda_runtime.h>
#include <string>

#include "Exception.h"

#define TP_FLASH_CHECK(condition)                                          \
  do {                                                                     \
    cudaError_t flash_error = condition;                                   \
    if (flash_error != cudaSuccess) {                                      \
      TP_THROW(RuntimeError,                                               \
               std::string("CUDA Error: ") + cudaGetErrorString(flash_error)); \
    }                                                                      \
  } while (0)

#define TP_FLASH_LAUNCH_CHECK() TP_FLASH_CHECK(cudaGetLastError())

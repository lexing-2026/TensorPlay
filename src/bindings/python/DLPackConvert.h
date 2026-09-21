#pragma once

// DLPack conversion helpers shared by the tensor export path and the
// environment allocator. Defined in Tensor.cpp.

#include "dlpack_types.h"

#include "DType.h"
#include "Device.h"

DLDataType to_dlpack_dtype(DType dtype);
DType from_dlpack_dtype(DLDataType dt);
DLDevice to_dlpack_device(Device device);
Device from_dlpack_device(DLDevice d);

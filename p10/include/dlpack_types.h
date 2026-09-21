#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * DLPack: Open In Memory Tensor Structure
 * based on https://github.com/dmlc/dlpack
 */

typedef enum {
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLOpenCL = 4,
  kDLVulkan = 7,
  kDLMetal = 8,
  kDLVPI = 9,
  kDLROCM = 10,
  kDLROCMHost = 11,
  kDLExtDev = 12,
  kDLCUDAManaged = 13,
  kDLOneAPI = 14,
  kDLWebGPU = 15,
  kDLHexagon = 16,
} DLDeviceType;

typedef struct {
  int device_type;
  int device_id;
} DLDevice;

typedef enum {
  kDLInt = 0U,
  kDLUInt = 1U,
  kDLFloat = 2U,
  kDLOpaqueHandle = 3U,
  kDLBfloat = 4U,
  kDLComplex = 5U,
  kDLBool = 6U,
} DLDataTypeCode;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

typedef struct {
  void* data;
  DLDevice device;
  int ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

/* Version tag carried by DLManagedTensorVersioned. A consumer that
 * disagrees with the major version must release the tensor via the
 * deleter without touching any other field. */
typedef struct {
  uint32_t major;
  uint32_t minor;
} DLPackVersion;

/* Versioned exchange wrapper: the flags field adds read-only and
 * copy semantics on top of the plain managed tensor. */
typedef struct DLManagedTensorVersioned {
  DLPackVersion version;
  void* manager_ctx;
  void (*deleter)(struct DLManagedTensorVersioned* self);
  uint64_t flags;
  DLTensor dl_tensor;
} DLManagedTensorVersioned;

/* Allocation callback contract used by exchange consumers that let a
 * foreign kernel request a tensor from the host environment: the
 * prototype carries dtype, ndim, shape and device; the callee either
 * stores a versioned wrapper in *out and returns 0, or reports through
 * set_error and returns nonzero. */
typedef int (*DLPackManagedTensorAllocatorFunction)(
    DLTensor* prototype, DLManagedTensorVersioned** out, void* error_ctx,
    void (*set_error)(void* error_ctx, const char* kind, const char* message));

#ifdef __cplusplus
}  // extern "C"
#endif

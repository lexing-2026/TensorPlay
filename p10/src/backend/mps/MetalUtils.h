#pragma once

// Shared helpers for dispatching Metal compute kernels from the MPS backend.

#include "Tensor.h"
#include "Exception.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSStream.h"
#include "backend/mps/OperationUtils.h"

#include <Metal/Metal.h>

#include <array>
#include <exception>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorplay {
namespace mps {

// Maps a storage dtype to the suffix used in Metal kernel host names.
inline std::string scalarToMetalTypeString(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return "float";
    case DType::Float16:
      return "half";
    case DType::BFloat16:
      return "bfloat";
    case DType::Int64:
      return "long";
    case DType::Int32:
      return "int";
    case DType::Int16:
      return "short";
    case DType::Int8:
      return "char";
    case DType::UInt8:
      return "uchar";
    case DType::Bool:
      return "bool";
    default:
      TP_THROW(RuntimeError, "unsupported dtype for Metal shader: ",
               toString(dtype));
  }
}

// Binds a contiguous, offset-0 tensor's storage to a shader buffer slot.
inline void set_shader_arg(id<MTLComputeCommandEncoder> encoder,
                           const Tensor& t, unsigned idx) {
  const Tensor c = (t.is_contiguous() && t.storage_offset() == 0)
      ? t
      : materializeForGraph(t);
  void* buf = getMTLBufferFromPointer(c.data_ptr());
  if (!buf) {
    TP_THROW(RuntimeError,
             "MPS tensor storage is not managed by the MPS allocator");
  }
  [encoder setBuffer:(__bridge id<MTLBuffer>)buf offset:0 atIndex:idx];
}

inline void set_shader_arg(id<MTLComputeCommandEncoder> encoder,
                           id<MTLBuffer> val, unsigned idx) {
  [encoder setBuffer:val offset:0 atIndex:idx];
}

template <typename T>
inline void set_shader_arg(id<MTLComputeCommandEncoder> encoder, const T& val,
                           unsigned idx) {
  static_assert(std::is_trivially_copyable_v<T>, "shader arg must be POD");
  [encoder setBytes:&val length:sizeof(T) atIndex:idx];
}

template <typename T, size_t N>
inline void set_shader_arg(id<MTLComputeCommandEncoder> encoder,
                           const std::array<T, N>& values, unsigned idx) {
  [encoder setBytes:values.data() length:sizeof(T) * values.size() atIndex:idx];
}

template <typename T, typename Alloc>
inline void set_shader_arg(id<MTLComputeCommandEncoder> encoder,
                           const std::vector<T, Alloc>& values, unsigned idx) {
  [encoder setBytes:values.data() length:sizeof(T) * values.size() atIndex:idx];
}

template <unsigned idx = 0, typename T>
inline void mtl_setArgs(id<MTLComputeCommandEncoder> encoder, const T& val) {
  set_shader_arg(encoder, val, idx);
}

template <unsigned idx = 0, typename T, typename... Args>
inline void mtl_setArgs(id<MTLComputeCommandEncoder> encoder, const T& val,
                        Args&&... args) {
  set_shader_arg(encoder, val, idx);
  mtl_setArgs<idx + 1>(encoder, std::forward<Args>(args)...);
}

inline void mtl_dispatch1DJob(id<MTLComputeCommandEncoder> encoder,
                              id<MTLComputePipelineState> cplState,
                              NSUInteger length) {
  const auto maxThreadsPerGroup = [cplState maxTotalThreadsPerThreadgroup];
  const NSUInteger size = std::max<NSUInteger>(length, 1);
  const NSUInteger tg = std::min<NSUInteger>(size, maxThreadsPerGroup);
  [encoder dispatchThreads:MTLSizeMake(size, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// Whether this GPU vends the argument encoders that foreach kernels depend on.
// Such a kernel reads the tensors it updates out of an argument buffer, and an
// encoder is the only way to put a buffer's device address into one: the
// addresses differ from the pointers the host uses, so the buffer cannot be
// filled by hand.  The image shipped with macOS 14 has no encoders at all, and
// asking for one raises rather than returning nil, so the request is the probe.
inline bool vendsArgumentEncoder(id<MTLFunction> func) {
  @try {
    id<MTLArgumentEncoder> encoder = [func newArgumentEncoderWithBufferIndex:0];
    const bool vends = encoder != nil;
    [encoder release];
    return vends;
  } @catch (NSException* exception) {
    return false;
  }
}

// The capability belongs to the GPU device, so the first answer holds for the
// whole process.
inline bool argumentEncodersAvailable(id<MTLFunction> func) {
  static const bool available = vendsArgumentEncoder(func);
  return available;
}

// Binds the tensor pointers a foreach kernel reads out of an argument buffer.
// Only usable once argumentEncodersAvailable() has reported a device that vends
// an encoder.
class ForeachArguments {
 public:
  // The accessor hands back an owned encoder.
  ForeachArguments(id<MTLFunction> func, id<MTLDevice> device)
      : device_(device), encoder_([func newArgumentEncoderWithBufferIndex:0]) {
    bytes_ = [encoder_ encodedLength];
  }
  ~ForeachArguments() {
    [encoder_ release];
  }
  ForeachArguments(const ForeachArguments&) = delete;
  ForeachArguments& operator=(const ForeachArguments&) = delete;

  // A fresh buffer for one launch's bindings.
  id<MTLBuffer> buffer() const {
    id<MTLBuffer> argument_buffer =
        [[device_ newBufferWithLength:bytes_ options:0] autorelease];
    [encoder_ setArgumentBuffer:argument_buffer offset:0];
    return argument_buffer;
  }

  // Binds one tensor into the buffer most recently handed out.
  void bind(NSUInteger index, id<MTLBuffer> storage) const {
    [encoder_ setBuffer:storage offset:0 atIndex:index];
  }

 private:
  id<MTLDevice> device_ = nil;
  id<MTLArgumentEncoder> encoder_ = nil;
  NSUInteger bytes_ = 0;
};

// Runs a block on the MPS stream's serial queue, rethrowing C++ exceptions
// across the dispatch boundary.
inline void dispatch_sync_with_rethrow(dispatch_queue_t queue,
                                       void (^block)()) {
  __block std::optional<std::exception_ptr> block_exception;
  dispatch_sync(queue, ^() {
    try {
      block();
    } catch (...) {
      block_exception = std::current_exception();
    }
  });
  if (block_exception) {
    std::rethrow_exception(*block_exception);
  }
}

} // namespace mps
} // namespace tensorplay
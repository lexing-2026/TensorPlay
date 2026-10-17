#pragma once

// Shared helpers for dispatching Metal compute kernels from the MPS backend.

#include "Tensor.h"
#include "Exception.h"

#include "backend/mps/MPSAllocator.h"
#include "backend/mps/MPSStream.h"
#include "backend/mps/OperationUtils.h"

#include <Metal/Metal.h>

#include <array>
#include <cstring>
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

// Binds the tensor pointers a foreach kernel reads out of an argument buffer.
// Such a buffer holds one 8-byte device address per argument id, which is what
// the pointer arrays in a kernel's argument struct resolve to.  Current GPU
// images vend an encoder that writes those addresses; the image shipped with
// macOS 14 has no argument encoders at all, so the same layout is written
// directly there.
class ForeachArguments {
 public:
  ForeachArguments(MTLFunction_t func, id<MTLDevice> device, NSUInteger slots)
      : device_(device) {
    if (encoders_available(func)) {
      // The accessor hands back an owned encoder.
      encoder_ = [func newArgumentEncoderWithBufferIndex:0];
    }
    bytes_ = encoder_ ? [encoder_ encodedLength] : slots * sizeof(uintptr_t);
  }
  ~ForeachArguments() {
    [encoder_ release];
  }
  ForeachArguments(const ForeachArguments&) = delete;
  ForeachArguments& operator=(const ForeachArguments&) = delete;

  // A fresh buffer whose unbound slots read as null pointers.
  id<MTLBuffer> buffer() const {
    id<MTLBuffer> argument_buffer =
        [[device_ newBufferWithLength:bytes_ options:0] autorelease];
    memset(argument_buffer.contents, 0, bytes_);
    [encoder_ setArgumentBuffer:argument_buffer offset:0];
    return argument_buffer;
  }

  void bind(id<MTLBuffer> slot, NSUInteger index, id<MTLBuffer> storage) const {
    if (encoder_) {
      [encoder_ setBuffer:storage offset:0 atIndex:index];
      return;
    }
    static_cast<uintptr_t*>(slot.contents)[index] =
        reinterpret_cast<uintptr_t>(storage.contents);
  }

 private:
  // The capability belongs to the GPU device, so one probe covers every
  // function; asking for an encoder is harmless where it is supported.
  static bool encoders_available(MTLFunction_t func) {
    static const bool available = [func]() {
      @try {
        id<MTLArgumentEncoder> encoder =
            [func newArgumentEncoderWithBufferIndex:0];
        const bool vends = (encoder != nil);
        [encoder release];
        return vends;
      } @catch (NSException* exception) {
        return false;
      }
    }();
    return available;
  }

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
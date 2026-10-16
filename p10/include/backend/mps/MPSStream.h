#pragma once

#include <cstdint>
#include <utility>

#include "Device.h"
#include "Macros.h"
#include "Stream.h"
#include "backend/mps/MPSDevice.h"

#ifdef __OBJC__
// Apple framework headers emit deprecation warnings from CarbonCore and
// missing-attribute warnings from MPSGraph on recent macOS SDKs; the
// diagnostics are scoped to the includes and popped immediately.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wobjc-property-no-attribute"
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#pragma clang diagnostic pop
#pragma clang diagnostic pop
typedef MPSCommandBuffer* MPSCommandBuffer_t;
typedef id<MTLCommandQueue> MTLCommandQueue_t;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;
typedef id<MTLSharedEvent> MTLSharedEvent_t;
typedef id<MTLDevice> MTLDevice_t;
typedef id<MTLBuffer> MTLBuffer_t;
#else
#include <dispatch/dispatch.h>
typedef void* MPSCommandBuffer_t;
typedef void* MPSGraph;
typedef void* MPSGraphExecutionDescriptor;
typedef void* MPSGraphCompilationDescriptor;
typedef void* MTLCommandQueue_t;
typedef void* MTLComputeCommandEncoder_t;
typedef void* MTLSharedEvent_t;
typedef void* MTLDevice_t;
typedef void* MTLBuffer_t;
typedef void* MTLCommandBufferHandler;
typedef void* NSDictionary;
#define nil NULL
#endif

namespace tensorplay {
namespace mps {

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

enum class SyncType {
  NONE,              // no commit to command buffer
  COMMIT,            // commit and flush the command buffer
  COMMIT_AND_WAIT,   // flush and wait for command buffer execution to finish
  COMMIT_AND_CONTINUE, // commit and continue with a new underlying command buffer
  COMMIT_ADAPTIVE,   // commit adaptively based on available memory
};

class P10_API MPSStream {
 public:
  enum Unchecked { UNCHECKED };

  // Construct a MPSStream from a Stream.  This construction is checked,
  // and will raise an error if the Stream is not, in fact, a MPS stream.
  explicit MPSStream(Stream stream);

  ~MPSStream();

  MTLCommandQueue_t commandQueue() const {
    return _commandQueue;
  }

  dispatch_queue_t queue() const {
    return _serialQueue;
  }

  MPSCommandBuffer_t commandBuffer();
  MTLComputeCommandEncoder_t commandEncoder();
  void endKernelCoalescing();
  void synchronize(SyncType syncType);
  void copy(MTLBuffer_t srcBuffer,
            MTLBuffer_t dstBuffer,
            size_t length,
            size_t srcOffset,
            size_t dstOffset,
            SyncType syncType = SyncType::NONE);
  void executeMPSGraph(MPSGraph* mpsGraph,
                       NSDictionary* feeds,
                       NSDictionary* results,
                       SyncType syncType = SyncType::NONE);
  void addCompletedHandler(MTLCommandBufferHandler block);

  // Commits the current command buffer, waits for completion, and returns
  // the GPU duration in milliseconds from the command buffer timestamps.
  double commitAndWaitGetGpuTimeMs();

  /// Get the MPS device index that this stream is associated with.
  int device_index() const {
    return _stream.device_index();
  }

  MTLCommandQueue_t stream() const {
    return _commandQueue;
  }

  MTLDevice_t device() const;

  /// Explicit conversion to Stream.
  Stream unwrap() const {
    return _stream;
  }

 private:
  Stream _stream;
  MTLCommandQueue_t _commandQueue = nil;
  MPSCommandBuffer_t _commandBuffer = nil;
  MPSCommandBuffer_t _prevCommandBuffer = nil;
  MTLComputeCommandEncoder_t _commandEncoder = nil;
  MPSGraphExecutionDescriptor* _executionDescriptor = nil;
  MPSGraphCompilationDescriptor* _compilationDescriptor = nil;
  dispatch_queue_t _serialQueue = nullptr;
  // CommitAndContinue is enabled by default.
  bool _enableCommitAndContinue = true;

  // Use synchronize() to access any of these commit functions outside MPSStream.
  void commit();
  void commitAndWait();
  void commitAndContinue();
  void flush();
};

/**
 * Get the current MPS stream for this thread. Returns the default stream if no
 * other stream has been set with `setCurrentMPSStream()`.
 */
P10_API MPSStream* getCurrentMPSStream();

/**
 * Set the current MPS stream for this thread. Kernels that call
 * `getCurrentMPSStream()` will enqueue their work onto this stream. Passing
 * nullptr sets to the default stream.
 */
P10_API void setCurrentMPSStream(MPSStream* stream);

/**
 * Get the default MPS stream.
 */
P10_API MPSStream* getDefaultMPSStream();

/**
 * Get a stream from the pool. There are 32 streams in the pool which live for
 * the lifetime of a process. The stream returned by this function is chosen
 * in round-robin order. Note: The default stream is not in the pool.
 */
P10_API MPSStream* getStreamFromPool();

/**
 * Resolve a backend-agnostic Stream value to the matching MPSStream. The
 * default stream (id 0) and pool streams (ids 1..32) are recognized; any
 * other id falls back to the default stream.
 */
P10_API MPSStream* getStreamFromStream(const Stream& stream);

/**
 * Synchronize the default stream and any pool streams created so far.
 */
P10_API void synchronizeAllMPSStreams(SyncType syncType);

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

class P10_API MPSStreamImpl {
 public:
  /**
   * Gets single instance of the MPSStream.
   */
  static MPSStream* getInstance();

 private:
  static MPSStream* _stream;
  MPSStreamImpl();
};

} // namespace mps
} // namespace tensorplay
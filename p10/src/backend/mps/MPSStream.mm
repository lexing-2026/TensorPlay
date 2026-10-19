#include "backend/mps/MPSStream.h"

#include "backend/mps/MetalUtils.h"
#include "backend/mps/MPSAllocator.h"
#include "Exception.h"

#include <array>
#include <atomic>
#include <cstring>
#include <mutex>

// The public header exposes enableCommitAndContinue as readonly; the backend
// needs to toggle it per-stream, so redeclare the property as readwrite in a
// private class extension.
@interface MPSGraphExecutionDescriptor ()
@property(readwrite, atomic) BOOL enableCommitAndContinue;
@end

namespace tensorplay {
namespace mps {
//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

MPSStream::MPSStream(Stream stream) : _stream(stream) {
  _commandQueue = [MPSDevice::getInstance()->device() newCommandQueue];
  if (_stream.device_type() != DeviceType::MPS) {
    TP_THROW(RuntimeError, "MPSStream constructed with a non-MPS stream");
  }
  _serialQueue = dispatch_queue_create("tensorplay mps stream", nullptr);
  // Launches serialize against this lock rather than against the queue's
  // worker thread; it is kept for as long as the queue itself.
  dispatch_set_context(_serialQueue, new std::mutex());
  _errorBuffer = [MPSDevice::getInstance()->device()
      newBufferWithLength:4 * sizeof(int64_t)
                options:MTLResourceStorageModeShared];
  std::memset([_errorBuffer contents], 0, 4 * sizeof(int64_t));
  _executionDescriptor = [MPSGraphExecutionDescriptor new];
  _compilationDescriptor = [MPSGraphCompilationDescriptor new];
  // A launch commits the stream's command buffer as one stage and encoding
  // carries on in the next, so queueing work costs no host round trip while a
  // later wait still covers every stage encoded before it.
  _enableCommitAndContinue = true;
  _executionDescriptor.enableCommitAndContinue = _enableCommitAndContinue;
  // Choose level which optimizes for GPU.
  _compilationDescriptor.optimizationLevel = MPSGraphOptimizationLevel0;
  _executionDescriptor.compilationDescriptor = _compilationDescriptor;
}

MPSStream::~MPSStream() {
  [_commandQueue release];
  _commandQueue = nil;
  [_executionDescriptor release];
  _executionDescriptor = nil;
  [_compilationDescriptor release];
  _compilationDescriptor = nil;
  if (_errorBuffer) {
    [_errorBuffer release];
    _errorBuffer = nil;
  }
  if (_commandBuffer) {
    [_commandBuffer release];
    _commandBuffer = nil;
  }
  if (_prevCommandBuffer) {
    [_prevCommandBuffer release];
    _prevCommandBuffer = nil;
  }
  if (_commandEncoder) {
    [_commandEncoder release];
    _commandEncoder = nil;
  }
  if (_serialQueue) {
    delete static_cast<std::mutex*>(dispatch_get_context(_serialQueue));
    dispatch_set_context(_serialQueue, nullptr);
  }
}

MPSCommandBuffer* MPSStream::commandBuffer() {
  if (!_commandBuffer) {
    _commandBuffer = [[MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue] retain];
    _pendingWork.store(true, std::memory_order_relaxed);
  }
  return _commandBuffer;
}

id<MTLDevice> MPSStream::device() const {
  return [_commandQueue device];
}

id<MTLComputeCommandEncoder> MPSStream::commandEncoder() {
  if (!_commandEncoder) {
    _commandEncoder = [[commandBuffer() computeCommandEncoder] retain];
  }
  return _commandEncoder;
}

void MPSStream::synchronize(SyncType syncType) {
  endKernelCoalescing();
  switch (syncType) {
    case SyncType::NONE:
      // typically in GPU to GPU copies we won't commit explicitly
      break;
    case SyncType::COMMIT:
      commit();
      break;
    case SyncType::COMMIT_ADAPTIVE:
      // Encoding stays inside the command buffer the stream is building, so a
      // run of launches pays one commit. The buffer is only closed early once
      // the pool has run up to its low watermark and holding more work there
      // would add to the pressure.
      if (getIMPSAllocator()->getLowWatermarkValue() <= 1) {
        commit();
      }
      break;
    case SyncType::COMMIT_AND_WAIT:
      commitAndWait();
      break;
    case SyncType::COMMIT_AND_CONTINUE:
      commitAndContinue();
      break;
  }
}

void MPSStream::commit() {
  if (_enableCommitAndContinue) {
    [commandBuffer() commitAndContinue];
  } else {
    flush();
  }
}

void MPSStream::commitAndWait() {
  if (_prevCommandBuffer) {
    // The previous command buffer (if exists) has already been committed,
    // so we just wait until it's completed and then dispose it.
    [_prevCommandBuffer waitUntilCompleted];
    [_prevCommandBuffer release];
    _prevCommandBuffer = nil;
  }

  if (_commandBuffer) {
    [_commandBuffer commit];
    [_commandBuffer waitUntilCompleted];
    [_commandBuffer release];
    _commandBuffer = nil;
  }
  _pendingWork.store(false, std::memory_order_relaxed);
  checkError();
}

void MPSStream::checkError() {
  if (!_errorBuffer) {
    return;
  }
  auto* report = static_cast<int64_t*>([_errorBuffer contents]);
  if (report[0] == 0) {
    return;
  }
  const int64_t value = report[1];
  const int64_t dim_size = report[2];
  report[0] = 0;
  TP_THROW(IndexError, "index ", value, " is out of bounds for dimension ",
           dim_size);
}

void MPSStream::commitAndContinue() {
  if (!_commandBuffer) {
    return;
  }
  [_commandBuffer commitAndContinue];
}

void MPSStream::endKernelCoalescing() {
  if (_commandEncoder) {
    [_commandEncoder endEncoding];
    [_commandEncoder release];
    _commandEncoder = nil;
  }
}

void MPSStream::flush() {
  if (_commandBuffer) {
    [_commandBuffer commit];
    if (!_enableCommitAndContinue) {
      _prevCommandBuffer = _commandBuffer;
    } else {
      [_commandBuffer release];
    }
    _commandBuffer = nil;
  }
}

void MPSStream::addCompletedHandler(MTLCommandBufferHandler block) {
  run_serialized(_serialQueue, ^() {
    @autoreleasepool {
      [commandBuffer() addCompletedHandler:block];
    }
  });
}

void MPSStream::copy(MTLBuffer_t srcBuffer,
                     MTLBuffer_t dstBuffer,
                     size_t length,
                     size_t srcOffset,
                     size_t dstOffset,
                     SyncType syncType) {
  run_serialized(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

      // Large blits are split into 2 GB chunks: a single copyFromBuffer for
      // sizes past 4 GB can fail without surfacing an error.
      constexpr size_t max_copy_size = 0x80000000; // 2GB
      size_t bytes_copied = 0;
      size_t bytes_remains = length;
      while (bytes_remains > 0) {
        NSUInteger bytes_to_copy = std::min(max_copy_size, bytes_remains);
        [blitEncoder copyFromBuffer:srcBuffer
                       sourceOffset:(NSUInteger)srcOffset + bytes_copied
                           toBuffer:dstBuffer
                  destinationOffset:(NSUInteger)dstOffset + bytes_copied
                               size:bytes_to_copy];
        bytes_copied += bytes_to_copy;
        bytes_remains -= bytes_to_copy;
      }
      [blitEncoder endEncoding];

      synchronize(syncType);
    }
  });
}

void MPSStream::executeMPSGraph(MPSGraph* mpsGraph,
                                NSDictionary* feeds,
                                NSDictionary* results,
                                SyncType syncType) {
  run_serialized(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      @try {
        [mpsGraph encodeToCommandBuffer:commandBuffer()
                                  feeds:feeds
                       targetOperations:nil
                      resultsDictionary:results
                    executionDescriptor:_executionDescriptor];
      } @catch (NSException* e) {
        std::string reason =
            [e reason] ? std::string([[e reason] UTF8String]) : std::string("unknown");
        throw tensorplay::RuntimeError(
            {__FILE__, __func__, __LINE__},
            tensorplay::detail::format_msg("MPSGraph encode failed: ", reason));
      }
      SyncType effective = syncType;
      if (!_enableCommitAndContinue && syncType != SyncType::COMMIT_AND_WAIT) {
        effective = SyncType::COMMIT;
      }
      synchronize(effective);
    }
  });
}

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

MPSStream* MPSStreamImpl::getInstance() {
  static MPSStream* _stream = new MPSStream(
      Stream(Stream::UNSAFE, Device(DeviceType::MPS, 0), 0));
  return _stream;
}

MPSStream* getDefaultMPSStream() {
  return MPSStreamImpl::getInstance();
}

namespace {

// Thread-local current stream. The default stream is used when no explicit
// stream has been set for the calling thread.
thread_local MPSStream* tls_current_stream = nullptr;

// Stream pool with 32 streams, chosen in round-robin order.
constexpr int kStreamsPerPool = 32;
std::array<MPSStream*, kStreamsPerPool> streamPool{};
std::atomic<uint32_t> streamPoolIndex{0};
std::once_flag streamPoolOnce;
// Set once the pool's streams exist. A synchronize that arrives before then
// must not run the one-time setup just to drain streams nobody asked for, and
// must not consume the once flag either: the flag belongs to the first caller
// that actually needs a stream.
std::atomic<bool> streamPoolInitialized{false};

void initStreamPool() {
  for (int i = 0; i < kStreamsPerPool; ++i) {
    streamPool[i] = new MPSStream(
        Stream(Stream::UNSAFE, Device(DeviceType::MPS, 0), i + 1));
  }
  streamPoolInitialized.store(true, std::memory_order_release);
}

} // namespace

MPSStream* getCurrentMPSStream() {
  if (tls_current_stream == nullptr) {
    tls_current_stream = getDefaultMPSStream();
  }
  return tls_current_stream;
}

void setCurrentMPSStream(MPSStream* stream) {
  tls_current_stream = stream;
}

void synchronizeHostAccess() {
  MPSStream* stream = getCurrentMPSStream();
  // Draining under the stream's launch lock keeps the wait in step with the
  // launches that are still being encoded.
  run_serialized(stream->queue(), ^() {
    @autoreleasepool {
      stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
  });
}

MPSStream* getStreamFromPool() {
  std::call_once(streamPoolOnce, initStreamPool);
  const uint32_t idx = streamPoolIndex.fetch_add(1, std::memory_order_relaxed);
  return streamPool[idx % kStreamsPerPool];
}

MPSStream* getStreamFromStream(const Stream& stream) {
  if (stream.device_type() != DeviceType::MPS) {
    return getDefaultMPSStream();
  }
  if (stream.id() == 0) {
    return getDefaultMPSStream();
  }
  std::call_once(streamPoolOnce, initStreamPool);
  const uint64_t idx = stream.id() - 1;
  if (idx < kStreamsPerPool) {
    return streamPool[idx];
  }
  return getDefaultMPSStream();
}

void synchronizeAllMPSStreams(SyncType syncType) {
  // Every launch is encoded under the stream's lock before its caller
  // continues, so the default stream holds all of this thread's work by the
  // time the wait starts and is waited on directly. A pool stream may have
  // been used from another thread, so it is drained under its own lock, and
  // an idle one is skipped: it has nothing to wait for.
  getDefaultMPSStream()->synchronize(syncType);
  if (streamPoolInitialized.load(std::memory_order_acquire)) {
    for (int i = 0; i < kStreamsPerPool; ++i) {
      MPSStream* stream = streamPool[i];
      if (!stream->hasPendingWork()) {
        continue;
      }
      run_serialized(stream->queue(), ^() {
        stream->synchronize(syncType);
      });
    }
  }
}

} // namespace mps
} // namespace tensorplay
#pragma once

// NCCL communicator context, mirroring the core of torch.distributed's
// ProcessGroupNCCL: communicator lifecycle plus collectives on raw device
// buffers. Group/rendezvous policy lives in the Python layer
// (tensorplay/distributed), matching torch's c10d/distributed_c10d split.

#include "DType.h"
#include "Macros.h"

#include <cstdint>
#include <cstddef>

namespace tensorplay {
namespace nccl {

using Comm = void*;

static constexpr int kUniqueIdBytes = 128;

enum class ReduceOp : int {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
};

// Returns true when NCCL support was compiled in.
P10_API bool available();

// NCCL version string "maj.min.rev" (compile-time constant).
P10_API const char* version();

P10_API void getUniqueId(uint8_t* out128);

P10_API Comm commInitRank(int rank, int world_size, const uint8_t* uid128);
P10_API void commDestroy(Comm comm);
P10_API void commAbort(Comm comm);
P10_API int commCount(Comm comm);

// All collectives enqueue on `stream` and return once enqueued (async),
// mirroring NCCL semantics; synchronization is the caller's job.

P10_API void allReduce(void* buffer, size_t count, DType dtype, ReduceOp op,
                       Comm comm, void* stream);
P10_API void broadcast(void* buffer, size_t count, DType dtype, int root,
                       Comm comm, void* stream);
P10_API void reduce(void* buffer, size_t count, DType dtype, ReduceOp op,
                    int root, Comm comm, void* stream);
P10_API void allGather(void* sendbuff, void* recvbuff, size_t count,
                       DType dtype, Comm comm, void* stream);
P10_API void reduceScatter(void* sendbuff, void* recvbuff, size_t count,
                           DType dtype, ReduceOp op, Comm comm, void* stream);
// `recvbuff` may be null on non-root ranks; `sendbuff` may be null off root.
P10_API void gather(const void* sendbuff, void* recvbuff, size_t count,
                    DType dtype, int root, Comm comm, void* stream);
P10_API void scatter(const void* sendbuff, void* recvbuff, size_t count,
                     DType dtype, int root, Comm comm, void* stream);
P10_API void send(const void* buffer, size_t count, DType dtype, int peer,
                  Comm comm, void* stream);
P10_API void recv(void* buffer, size_t count, DType dtype, int peer,
                  Comm comm, void* stream);

} // namespace nccl
} // namespace tensorplay

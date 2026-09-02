#pragma once

// ProcessGroupNCCL: communicator lifecycle plus collectives on raw device
// buffers. Group/rendezvous policy lives in the Python layer

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
    Avg = 1,
    Prod = 2,
    Min = 3,
    Max = 4,
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
// matching NCCL semantics; synchronization is the caller's job.

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

// multiple p2p ops between a start/end pair so they enqueue as one NCCL
// group. Must bracket matching send/recv pairs on every rank.
P10_API void groupStart();
P10_API void groupEnd();

// buffer, each rank exchanges `count = numel / world_size` elements.
P10_API void allToAllSingleEqualSplit(const void* sendbuff, void* recvbuff,
                                      size_t count_total, DType dtype,
                                      Comm comm, void* stream);
P10_API void allToAllSingleUnequalSplit(
    const void* sendbuff, const size_t* sendcounts, const size_t* senddispls,
    void* recvbuff, const size_t* recvcounts, const size_t* recvdispls,
    size_t element_size, DType dtype, Comm comm, void* stream);

} // namespace nccl
} // namespace tensorplay

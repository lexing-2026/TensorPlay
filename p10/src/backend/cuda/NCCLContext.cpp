#include "NCCLContext.h"

#include "Exception.h"

// Older NCCL releases predate NCCL_VERSION_STRING; fall back to expanding
// NCCL_VERSION_CODE (mirrors torch's version handling).
#ifndef NCCL_VERSION_STRING
#define TP_STRINGIZE_IMPL(x) #x
#define TP_STRINGIZE(x) TP_STRINGIZE_IMPL(x)
#define NCCL_VERSION_STRING "NCCL version " TP_STRINGIZE(NCCL_VERSION_CODE)
#endif

#ifndef USE_NCCL

// Mirrors pytorch: without NCCL the whole implementation compiles out and
// every entry point degrades to a runtime error.

namespace tensorplay {
namespace nccl {

bool available() { return false; }
const char* version() { return ""; }

void getUniqueId(uint8_t*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
Comm commInitRank(int, int, const uint8_t*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void commDestroy(Comm) {}
void commAbort(Comm) {}
int commCount(Comm) { return 0; }

void allReduce(void*, size_t, DType, ReduceOp, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void broadcast(void*, size_t, DType, int, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void reduce(void*, size_t, DType, ReduceOp, int, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void allGather(void*, void*, size_t, DType, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void reduceScatter(void*, void*, size_t, DType, ReduceOp, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void gather(const void*, void*, size_t, DType, int, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void scatter(const void*, void*, size_t, DType, int, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void send(const void*, size_t, DType, int, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void recv(void*, size_t, DType, int, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void groupStart() {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void groupEnd() {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void allToAllSingleEqualSplit(const void*, void*, size_t, DType, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}
void allToAllSingleUnequalSplit(const void*, const size_t*, const size_t*,
                                void*, const size_t*, const size_t*, size_t,
                                DType, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without NCCL support");
}

} // namespace nccl
} // namespace tensorplay

#else // USE_NCCL

#include "CUDARuntime.h"

#include <nccl.h>

#include <string>

namespace tensorplay {
namespace nccl {
namespace {

void checkNccl(ncclResult_t result, const char* operation) {
    if (result != ncclSuccess) {
        TP_THROW(RuntimeError,
                 std::string("NCCL error in ") + operation + ": " +
                 ncclGetErrorString(result) + " (ncclResult = " +
                 std::to_string(static_cast<int>(result)) + ")");
    }
}

ncclDataType_t toNcclDType(DType dtype) {
    switch (dtype) {
        case DType::Int8: return ncclInt8;
        case DType::UInt8: return ncclUint8;
        case DType::Int32: return ncclInt32;
        case DType::Int64: return ncclInt64;
        case DType::Float16: return ncclFloat16;
        case DType::Float32: return ncclFloat32;
        case DType::Float64: return ncclFloat64;
        case DType::BFloat16: return ncclBfloat16;
        default:
            TP_THROW(RuntimeError,
                     "NCCL does not support dtype " +
                     std::to_string(static_cast<int>(dtype)) +
                     "; supported: int8, uint8, int32, int64, float16, "
                     "bfloat16, float32, float64");
    }
}

ncclRedOp_t toNcclRedOp(ReduceOp op) {
    switch (op) {
        case ReduceOp::Sum: return ncclSum;
        case ReduceOp::Prod: return ncclProd;
        case ReduceOp::Max: return ncclMax;
        case ReduceOp::Min: return ncclMin;
        case ReduceOp::Avg: return ncclAvg;
        default:
            TP_THROW(RuntimeError, "Invalid NCCL reduce op");
    }
}

} // namespace

bool available() { return true; }

const char* version() {
    // Runtime version, same source of truth c10d uses. NCCL packs the
    // version as 1000 * major + 100 * minor + patch.
    static std::string v;
    static bool cached = false;
    if (!cached) {
        int code = 0;
        ncclGetVersion(&code);
        v = std::to_string(code / 1000) + "." +
            std::to_string((code / 100) % 10) + "." +
            std::to_string(code % 100);
        cached = true;
    }
    return v.c_str();
}

void getUniqueId(uint8_t* out128) {
    ncclUniqueId uid;
    checkNccl(ncclGetUniqueId(&uid), "ncclGetUniqueId");
    static_assert(sizeof(uid.internal) == kUniqueIdBytes, "unexpected uid size");
    for (int i = 0; i < kUniqueIdBytes; ++i) out128[i] = uid.internal[i];
}

Comm commInitRank(int rank, int world_size, const uint8_t* uid128) {
    ncclUniqueId uid;
    for (int i = 0; i < kUniqueIdBytes; ++i) uid.internal[i] = uid128[i];
    ncclComm_t comm = nullptr;
    // Mirrors ProcessGroupNCCL: init can take minutes on slow networks.
    checkNccl(ncclCommInitRank(&comm, world_size, uid, rank), "ncclCommInitRank");
    return static_cast<Comm>(comm);
}

void commDestroy(Comm comm) {
    if (comm) checkNccl(ncclCommDestroy(static_cast<ncclComm_t>(comm)), "ncclCommDestroy");
}

void commAbort(Comm comm) {
    if (comm) checkNccl(ncclCommAbort(static_cast<ncclComm_t>(comm)), "ncclCommAbort");
}

int commCount(Comm comm) {
    int count = 0;
    checkNccl(ncclCommCount(static_cast<ncclComm_t>(comm), &count), "ncclCommCount");
    return count;
}

void allReduce(void* buffer, size_t count, DType dtype, ReduceOp op,
               Comm comm, void* stream) {
    checkNccl(ncclAllReduce(buffer, buffer, count, toNcclDType(dtype),
                            toNcclRedOp(op), static_cast<ncclComm_t>(comm),
                            reinterpret_cast<cudaStream_t>(stream)),
              "ncclAllReduce");
}

void broadcast(void* buffer, size_t count, DType dtype, int root,
               Comm comm, void* stream) {
    checkNccl(ncclBroadcast(buffer, buffer, count, toNcclDType(dtype), root,
                            static_cast<ncclComm_t>(comm),
                            reinterpret_cast<cudaStream_t>(stream)),
              "ncclBroadcast");
}

void reduce(void* buffer, size_t count, DType dtype, ReduceOp op, int root,
            Comm comm, void* stream) {
    checkNccl(ncclReduce(buffer, buffer, count, toNcclDType(dtype),
                         toNcclRedOp(op), root, static_cast<ncclComm_t>(comm),
                         reinterpret_cast<cudaStream_t>(stream)),
              "ncclReduce");
}

void allGather(void* sendbuff, void* recvbuff, size_t count, DType dtype,
               Comm comm, void* stream) {
    checkNccl(ncclAllGather(sendbuff, recvbuff, count, toNcclDType(dtype),
                            static_cast<ncclComm_t>(comm),
                            reinterpret_cast<cudaStream_t>(stream)),
              "ncclAllGather");
}

void reduceScatter(void* sendbuff, void* recvbuff, size_t count, DType dtype,
                   ReduceOp op, Comm comm, void* stream) {
    checkNccl(ncclReduceScatter(sendbuff, recvbuff, count, toNcclDType(dtype),
                                toNcclRedOp(op), static_cast<ncclComm_t>(comm),
                                reinterpret_cast<cudaStream_t>(stream)),
              "ncclReduceScatter");
}

void gather(const void* sendbuff, void* recvbuff, size_t count, DType dtype,
            int root, Comm comm, void* stream) {
    checkNccl(ncclGather(const_cast<void*>(sendbuff), recvbuff, count,
                         toNcclDType(dtype), root,
                         static_cast<ncclComm_t>(comm),
                         reinterpret_cast<cudaStream_t>(stream)),
              "ncclGather");
}

void scatter(const void* sendbuff, void* recvbuff, size_t count, DType dtype,
             int root, Comm comm, void* stream) {
    checkNccl(ncclScatter(const_cast<void*>(sendbuff), recvbuff, count,
                          toNcclDType(dtype), root,
                          static_cast<ncclComm_t>(comm),
                          reinterpret_cast<cudaStream_t>(stream)),
              "ncclScatter");
}

void send(const void* buffer, size_t count, DType dtype, int peer,
          Comm comm, void* stream) {
    checkNccl(ncclSend(const_cast<void*>(buffer), count, toNcclDType(dtype),
                       peer, static_cast<ncclComm_t>(comm),
                       reinterpret_cast<cudaStream_t>(stream)),
              "ncclSend");
}

void recv(void* buffer, size_t count, DType dtype, int peer,
          Comm comm, void* stream) {
    checkNccl(ncclRecv(buffer, count, toNcclDType(dtype), peer,
                       static_cast<ncclComm_t>(comm),
                       reinterpret_cast<cudaStream_t>(stream)),
              "ncclRecv");
}

// torch's _nccl_should_send_recv: skip zero-size p2p legs (NCCL errors on
// zero-count send/recv).
namespace {
inline bool shouldSendRecv(size_t count) { return count > 0; }
} // namespace

void groupStart() { checkNccl(ncclGroupStart(), "ncclGroupStart"); }

void groupEnd() { checkNccl(ncclGroupEnd(), "ncclGroupEnd"); }

void allToAllSingleEqualSplit(const void* sendbuff, void* recvbuff,
                              size_t count_total, DType dtype,
                              Comm comm, void* stream) {
    // torch::cuda::nccl::all2all_single_equal_split
#if defined(NCCL_ALLTOALL_SUPPORTED) || \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    int num_ranks = 0;
    checkNccl(ncclCommCount(static_cast<ncclComm_t>(comm), &num_ranks),
              "ncclCommCount");
    size_t count = count_total / static_cast<size_t>(num_ranks);
    if (count * static_cast<size_t>(num_ranks) != count_total) {
        TP_THROW(RuntimeError,
                 "all_to_all_single with equal splits requires the input to be "
                 "evenly divisible by world size");
    }
    checkNccl(ncclAlltoAll(sendbuff, recvbuff, count, toNcclDType(dtype),
                           static_cast<ncclComm_t>(comm),
                           reinterpret_cast<cudaStream_t>(stream)),
              "ncclAlltoAll");
#else
    int num_ranks = 0;
    checkNccl(ncclCommCount(static_cast<ncclComm_t>(comm), &num_ranks),
              "ncclCommCount");
    size_t rankdiff = count_total / static_cast<size_t>(num_ranks);
    auto type = toNcclDType(dtype);
    const char* sbuf = static_cast<const char*>(sendbuff);
    char* rbuf = static_cast<char*>(recvbuff);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    checkNccl(ncclGroupStart(), "ncclGroupStart");
    for (int r = 0; r < num_ranks; ++r) {
        checkNccl(ncclSend(sbuf + r * rankdiff, rankdiff, type, r,
                           static_cast<ncclComm_t>(comm), s), "ncclSend");
        checkNccl(ncclRecv(rbuf + r * rankdiff, rankdiff, type, r,
                           static_cast<ncclComm_t>(comm), s), "ncclRecv");
    }
    checkNccl(ncclGroupEnd(), "ncclGroupEnd");
#endif
}

void allToAllSingleUnequalSplit(
    const void* sendbuff, const size_t* sendcounts, const size_t* senddispls,
    void* recvbuff, const size_t* recvcounts, const size_t* recvdispls,
    size_t element_size, DType dtype, Comm comm, void* stream) {
    // torch::cuda::nccl::all2all_single_unequal_split (send/recv group form;
    // used whenever NCCL lacks ncclAlltoAllv, i.e. always on stock builds).
    auto type = toNcclDType(dtype);
    ncclComm_t c = static_cast<ncclComm_t>(comm);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int num_ranks = 0;
    checkNccl(ncclCommCount(c, &num_ranks), "ncclCommCount");
    checkNccl(ncclGroupStart(), "ncclGroupStart");
    for (int r = 0; r < num_ranks; ++r) {
        if (shouldSendRecv(sendcounts[r])) {
            checkNccl(ncclSend(
                static_cast<const char*>(sendbuff) + senddispls[r] * element_size,
                sendcounts[r], type, r, c, s), "ncclSend");
        }
        if (shouldSendRecv(recvcounts[r])) {
            checkNccl(ncclRecv(
                static_cast<char*>(recvbuff) + recvdispls[r] * element_size,
                recvcounts[r], type, r, c, s), "ncclRecv");
        }
    }
    checkNccl(ncclGroupEnd(), "ncclGroupEnd");
}

} // namespace nccl
} // namespace tensorplay

#endif // TENSORPLAY_NO_NCCL

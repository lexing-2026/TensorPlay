#include "NCCLContext.h"

#include "Exception.h"

#if !defined(USE_NCCL) && !defined(USE_RCCL)

namespace tensorplay {
namespace nccl {

bool available() { return false; }
const char* version() { return ""; }

void getUniqueId(uint8_t*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
Comm commInitRank(int, int, const uint8_t*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void commDestroy(Comm) {}
void commAbort(Comm) {}
int commCount(Comm) { return 0; }

void allReduce(void*, size_t, DType, ReduceOp, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void broadcast(void*, size_t, DType, int, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void reduce(void*, size_t, DType, ReduceOp, int, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void allGather(void*, void*, size_t, DType, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void reduceScatter(void*, void*, size_t, DType, ReduceOp, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void gather(const void*, void*, size_t, DType, int, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void scatter(const void*, void*, size_t, DType, int, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void send(const void*, size_t, DType, int, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void recv(void*, size_t, DType, int, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void groupStart() {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void groupEnd() {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void allToAllSingleEqualSplit(const void*, void*, size_t, DType, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}
void allToAllSingleUnequalSplit(const void*, const size_t*, const size_t*,
                                void*, const size_t*, const size_t*, size_t,
                                DType, Comm, void*) {
    TP_THROW(RuntimeError, "TensorPlay was built without collective support");
}

} // namespace nccl
} // namespace tensorplay

#else

#include "CUDARuntime.h"

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
using CommStream = hipStream_t;
#else
#include <cuda_runtime.h>
#include <nccl.h>
using CommStream = cudaStream_t;
#endif

#include <string>

namespace tensorplay {
namespace nccl {
namespace {

void checkNccl(ncclResult_t result, const char* operation) {
    if (result != ncclSuccess) {
#ifdef USE_RCCL
        const char* library = "RCCL";
#else
        const char* library = "NCCL";
#endif
        TP_THROW(RuntimeError,
                 std::string(library) + " error in " + operation + ": " +
                     ncclGetErrorString(result) + " (result = " +
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
#ifdef USE_RCCL
            const char* library = "RCCL";
#else
            const char* library = "NCCL";
#endif
            TP_THROW(RuntimeError,
                     std::string(library) + " does not support dtype " +
                         std::to_string(static_cast<int>(dtype)));
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
            TP_THROW(RuntimeError, "Invalid collective reduce op");
    }
}

inline bool shouldSendRecv(size_t count) { return count != 0; }

size_t ncclElemSize(ncclDataType_t type) {
    switch (type) {
        case ncclInt8:
        case ncclUint8: return 1;
        case ncclInt32:
        case ncclUint32: return 4;
        case ncclInt64:
        case ncclUint64: return 8;
        case ncclFloat16:
        case ncclBfloat16: return 2;
        case ncclFloat32: return 4;
        case ncclFloat64: return 8;
        default:
            TP_THROW(RuntimeError, "unsupported collective dtype");
    }
}

void copyDeviceAsync(void* dst, const void* src, size_t bytes,
                     CommStream stream) {
#ifdef USE_ROCM
    hipError_t error = hipMemcpyAsync(dst, src, bytes,
                                      hipMemcpyDeviceToDevice, stream);
    if (error != hipSuccess) {
        TP_THROW(RuntimeError, std::string("hipMemcpyAsync: ") +
                                   hipGetErrorString(error));
    }
#else
    cudaError_t error = cudaMemcpyAsync(dst, src, bytes,
                                        cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("cudaMemcpyAsync: ") +
                                   cudaGetErrorString(error));
    }
#endif
}

} // namespace

bool available() { return true; }

const char* version() {
    static std::string value;
    static bool initialized = false;
    if (!initialized) {
        int code = 0;
        checkNccl(ncclGetVersion(&code), "ncclGetVersion");
        const int major_base = code < 2900 ? 1000 : 10000;
        value = std::to_string(code / major_base) + "." +
                std::to_string((code % major_base) / 100) + "." +
                std::to_string(code % 100);
        initialized = true;
    }
    return value.c_str();
}

void getUniqueId(uint8_t* out128) {
    ncclUniqueId uid;
    checkNccl(ncclGetUniqueId(&uid), "ncclGetUniqueId");
    static_assert(sizeof(uid.internal) == kUniqueIdBytes,
                  "unexpected collective id size");
    for (int i = 0; i < kUniqueIdBytes; ++i) out128[i] = uid.internal[i];
}

Comm commInitRank(int rank, int world_size, const uint8_t* uid128) {
    ncclUniqueId uid;
    for (int i = 0; i < kUniqueIdBytes; ++i) uid.internal[i] = uid128[i];
    ncclComm_t comm = nullptr;
    checkNccl(ncclCommInitRank(&comm, world_size, uid, rank),
              "ncclCommInitRank");
    return static_cast<Comm>(comm);
}

void commDestroy(Comm comm) {
    if (comm) {
        checkNccl(ncclCommDestroy(static_cast<ncclComm_t>(comm)),
                  "ncclCommDestroy");
    }
}

void commAbort(Comm comm) {
    if (comm) {
        checkNccl(ncclCommAbort(static_cast<ncclComm_t>(comm)),
                  "ncclCommAbort");
    }
}

int commCount(Comm comm) {
    int count = 0;
    checkNccl(ncclCommCount(static_cast<ncclComm_t>(comm), &count),
              "ncclCommCount");
    return count;
}

void allReduce(void* buffer, size_t count, DType dtype, ReduceOp op,
               Comm comm, void* stream) {
    checkNccl(ncclAllReduce(buffer, buffer, count, toNcclDType(dtype),
                            toNcclRedOp(op), static_cast<ncclComm_t>(comm),
                            reinterpret_cast<CommStream>(stream)),
              "ncclAllReduce");
}

void broadcast(void* buffer, size_t count, DType dtype, int root,
               Comm comm, void* stream) {
    checkNccl(ncclBroadcast(buffer, buffer, count, toNcclDType(dtype), root,
                            static_cast<ncclComm_t>(comm),
                            reinterpret_cast<CommStream>(stream)),
              "ncclBroadcast");
}

void reduce(void* buffer, size_t count, DType dtype, ReduceOp op, int root,
            Comm comm, void* stream) {
    checkNccl(ncclReduce(buffer, buffer, count, toNcclDType(dtype),
                         toNcclRedOp(op), root, static_cast<ncclComm_t>(comm),
                         reinterpret_cast<CommStream>(stream)),
              "ncclReduce");
}

void allGather(void* sendbuff, void* recvbuff, size_t count, DType dtype,
               Comm comm, void* stream) {
    checkNccl(ncclAllGather(sendbuff, recvbuff, count, toNcclDType(dtype),
                            static_cast<ncclComm_t>(comm),
                            reinterpret_cast<CommStream>(stream)),
              "ncclAllGather");
}

void reduceScatter(void* sendbuff, void* recvbuff, size_t count, DType dtype,
                   ReduceOp op, Comm comm, void* stream) {
    checkNccl(ncclReduceScatter(sendbuff, recvbuff, count, toNcclDType(dtype),
                                toNcclRedOp(op), static_cast<ncclComm_t>(comm),
                                reinterpret_cast<CommStream>(stream)),
              "ncclReduceScatter");
}

void gather(const void* sendbuff, void* recvbuff, size_t count, DType dtype,
            int root, Comm comm, void* stream) {
#ifdef USE_RCCL
    checkNccl(ncclGather(sendbuff, recvbuff, count, toNcclDType(dtype), root,
                         static_cast<ncclComm_t>(comm),
                         reinterpret_cast<CommStream>(stream)),
              "ncclGather");
#else
    ncclComm_t c = static_cast<ncclComm_t>(comm);
    CommStream s = reinterpret_cast<CommStream>(stream);
    int num_ranks = 0;
    int cur_rank = 0;
    checkNccl(ncclCommCount(c, &num_ranks), "ncclCommCount");
    checkNccl(ncclCommUserRank(c, &cur_rank), "ncclCommUserRank");
    auto type = toNcclDType(dtype);
    const size_t esz = ncclElemSize(type);
    char* rbuf = static_cast<char*>(recvbuff);
    checkNccl(ncclGroupStart(), "ncclGroupStart");
    if (cur_rank == root) {
        for (int r = 0; r < num_ranks; ++r) {
            if (r != root) {
                if (shouldSendRecv(count)) {
                    checkNccl(ncclRecv(rbuf + r * count * esz, count, type, r,
                                       c, s),
                              "ncclRecv");
                }
            } else if (count > 0) {
                copyDeviceAsync(
                    rbuf + static_cast<size_t>(root) * count * esz, sendbuff,
                    count * esz, s);
            }
        }
    } else if (shouldSendRecv(count)) {
        checkNccl(ncclSend(sendbuff, count, type, root, c, s), "ncclSend");
    }
    checkNccl(ncclGroupEnd(), "ncclGroupEnd");
#endif
}

void scatter(const void* sendbuff, void* recvbuff, size_t count, DType dtype,
             int root, Comm comm, void* stream) {
#ifdef USE_RCCL
    checkNccl(ncclScatter(sendbuff, recvbuff, count, toNcclDType(dtype), root,
                          static_cast<ncclComm_t>(comm),
                          reinterpret_cast<CommStream>(stream)),
              "ncclScatter");
#else
    ncclComm_t c = static_cast<ncclComm_t>(comm);
    CommStream s = reinterpret_cast<CommStream>(stream);
    int num_ranks = 0;
    int cur_rank = 0;
    checkNccl(ncclCommCount(c, &num_ranks), "ncclCommCount");
    checkNccl(ncclCommUserRank(c, &cur_rank), "ncclCommUserRank");
    auto type = toNcclDType(dtype);
    const size_t esz = ncclElemSize(type);
    char* rbuf = static_cast<char*>(recvbuff);
    const char* sbuf = static_cast<const char*>(sendbuff);
    checkNccl(ncclGroupStart(), "ncclGroupStart");
    if (cur_rank == root) {
        for (int r = 0; r < num_ranks; ++r) {
            if (r != root) {
                if (shouldSendRecv(count)) {
                    checkNccl(ncclSend(sbuf + r * count * esz, count, type, r,
                                       c, s),
                              "ncclSend");
                }
            } else if (count > 0) {
                copyDeviceAsync(
                    rbuf, sbuf + static_cast<size_t>(root) * count * esz,
                    count * esz, s);
            }
        }
    } else if (shouldSendRecv(count)) {
        checkNccl(ncclRecv(rbuf, count, type, root, c, s), "ncclRecv");
    }
    checkNccl(ncclGroupEnd(), "ncclGroupEnd");
#endif
}

void send(const void* buffer, size_t count, DType dtype, int peer,
          Comm comm, void* stream) {
    checkNccl(ncclSend(buffer, count, toNcclDType(dtype), peer,
                       static_cast<ncclComm_t>(comm),
                       reinterpret_cast<CommStream>(stream)),
              "ncclSend");
}

void recv(void* buffer, size_t count, DType dtype, int peer,
          Comm comm, void* stream) {
    checkNccl(ncclRecv(buffer, count, toNcclDType(dtype), peer,
                       static_cast<ncclComm_t>(comm),
                       reinterpret_cast<CommStream>(stream)),
              "ncclRecv");
}

void groupStart() { checkNccl(ncclGroupStart(), "ncclGroupStart"); }
void groupEnd() { checkNccl(ncclGroupEnd(), "ncclGroupEnd"); }

void allToAllSingleEqualSplit(const void* sendbuff, void* recvbuff,
                              size_t count_total, DType dtype, Comm comm,
                              void* stream) {
    int num_ranks = 0;
    checkNccl(ncclCommCount(static_cast<ncclComm_t>(comm), &num_ranks),
              "ncclCommCount");
    if (num_ranks <= 0) {
        TP_THROW(RuntimeError, "collective communicator has no ranks");
    }
    if (count_total % static_cast<size_t>(num_ranks) != 0) {
        TP_THROW(RuntimeError,
                 "all_to_all_single with equal splits requires the input to be "
                 "evenly divisible by world size");
    }
    const size_t count = count_total / static_cast<size_t>(num_ranks);
#ifdef USE_RCCL
    checkNccl(ncclAllToAll(sendbuff, recvbuff, count, toNcclDType(dtype),
                           static_cast<ncclComm_t>(comm),
                           reinterpret_cast<CommStream>(stream)),
              "ncclAllToAll");
#else
#if defined(NCCL_ALLTOALL_SUPPORTED) || \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
    checkNccl(ncclAlltoAll(sendbuff, recvbuff, count, toNcclDType(dtype),
                           static_cast<ncclComm_t>(comm),
                           reinterpret_cast<CommStream>(stream)),
              "ncclAlltoAll");
#else
    auto type = toNcclDType(dtype);
    const size_t rankdiff = count * ncclElemSize(type);
    const char* sbuf = static_cast<const char*>(sendbuff);
    char* rbuf = static_cast<char*>(recvbuff);
    CommStream s = reinterpret_cast<CommStream>(stream);
    checkNccl(ncclGroupStart(), "ncclGroupStart");
    for (int r = 0; r < num_ranks; ++r) {
        if (shouldSendRecv(count)) {
            checkNccl(ncclSend(sbuf + r * rankdiff, count, type, r,
                               static_cast<ncclComm_t>(comm), s),
                      "ncclSend");
            checkNccl(ncclRecv(rbuf + r * rankdiff, count, type, r,
                               static_cast<ncclComm_t>(comm), s),
                      "ncclRecv");
        }
    }
    checkNccl(ncclGroupEnd(), "ncclGroupEnd");
#endif
#endif
}

void allToAllSingleUnequalSplit(
    const void* sendbuff, const size_t* sendcounts, const size_t* senddispls,
    void* recvbuff, const size_t* recvcounts, const size_t* recvdispls,
    size_t element_size, DType dtype, Comm comm, void* stream) {
#ifdef USE_RCCL
    (void)element_size;
    checkNccl(ncclAllToAllv(
                  sendbuff, sendcounts, senddispls, recvbuff, recvcounts,
                  recvdispls, toNcclDType(dtype),
                  static_cast<ncclComm_t>(comm),
                  reinterpret_cast<CommStream>(stream)),
              "ncclAllToAllv");
#else
    auto type = toNcclDType(dtype);
    ncclComm_t c = static_cast<ncclComm_t>(comm);
    CommStream s = reinterpret_cast<CommStream>(stream);
    int num_ranks = 0;
    checkNccl(ncclCommCount(c, &num_ranks), "ncclCommCount");
    checkNccl(ncclGroupStart(), "ncclGroupStart");
    for (int r = 0; r < num_ranks; ++r) {
        if (shouldSendRecv(sendcounts[r])) {
            checkNccl(ncclSend(
                          static_cast<const char*>(sendbuff) +
                              senddispls[r] * element_size,
                          sendcounts[r], type, r, c, s),
                      "ncclSend");
        }
        if (shouldSendRecv(recvcounts[r])) {
            checkNccl(ncclRecv(
                          static_cast<char*>(recvbuff) +
                              recvdispls[r] * element_size,
                          recvcounts[r], type, r, c, s),
                      "ncclRecv");
        }
    }
    checkNccl(ncclGroupEnd(), "ncclGroupEnd");
#endif
}

} // namespace nccl
} // namespace tensorplay

#endif

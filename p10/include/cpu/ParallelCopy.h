#pragma once

// Moving a contiguous run of bytes across the intra-op pool.
//
// A run is cut into pieces that are handed to different threads once it is
// long enough to gain from them, and every piece moves with memcpy.  A short
// run stays on the calling thread, where the pool's dispatch would cost more
// than the move itself.

#include "Parallel.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace tensorplay {
namespace cpu {

inline void parallel_copy_bytes(void* dst, const void* src, size_t nbytes) {
    constexpr size_t kPieceBytes = 256 * 1024;
    // Overlapping ranges are moved in one pass: a piece written first could
    // otherwise land on bytes a later piece still has to read.
    const auto dst_addr = reinterpret_cast<uintptr_t>(dst);
    const auto src_addr = reinterpret_cast<uintptr_t>(src);
    const bool disjoint =
        dst_addr + nbytes <= src_addr || src_addr + nbytes <= dst_addr;
    const int64_t pieces = disjoint
        ? static_cast<int64_t>((nbytes + kPieceBytes - 1) / kPieceBytes)
        : 1;
    if (pieces <= 1) {
        std::memcpy(dst, src, nbytes);
        return;
    }
    auto* d = static_cast<char*>(dst);
    const auto* s = static_cast<const char*>(src);
    parallel::parallel_for(0, pieces, 1, [&](int64_t begin, int64_t end) {
        for (int64_t p = begin; p < end; ++p) {
            const size_t off = static_cast<size_t>(p) * kPieceBytes;
            const size_t len = std::min(kPieceBytes, nbytes - off);
            std::memcpy(d + off, s + off, len);
        }
    });
}

} // namespace cpu
} // namespace tensorplay

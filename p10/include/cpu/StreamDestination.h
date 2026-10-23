#pragma once

// Deciding when a pointwise walk writes its destination past the cache.
//
// A store that goes through the cache takes the line in before it puts
// anything there, so that the elements outside the store keep their values.
// A walk that never reads what it writes pays that fetch for nothing: over
// the four packed streams of a ternary map it is a fifth of everything the
// walk moves.  A streamed store writes whole blocks out of a write-combining
// buffer and leaves the fetch out, which is worth having only while the
// destination would not have been wanted in cache again anyway -- so the
// choice is made on whether the walk's working set is larger than the cache
// that would have had to hold it.

#include "Parallel.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace tensorplay {
namespace cpu {
namespace streamdst {

#if defined(__x86_64__)
// One block of eight single-precision answers, laid either through the cache
// or past it.  The streamed form writes a whole block and asks that its
// address begin on one, which is what the walk's leading step of one element
// at a time is for.
template <bool Stream>
__attribute__((target("avx2"), always_inline))
inline void lay_f32_block(float* at, const __m256 answer) {
    if constexpr (Stream) {
        _mm256_stream_ps(at, answer);
    } else {
        _mm256_storeu_ps(at, answer);
    }
}
#endif

// The share of the last-level cache that one of the cores sharing it gets.
// Taken once from the numbers the machine publishes about itself; a part
// that publishes nothing is assumed to hold a megabyte per core.
inline int64_t last_level_bytes_per_core() {
    static const int64_t bytes = []() -> int64_t {
        int64_t size = 0;
        int64_t sharing = 0;
        if (std::FILE* file = std::fopen(
                "/sys/devices/system/cpu/cpu0/cache/index3/size", "re")) {
            char text[32] = {};
            if (std::fgets(text, sizeof(text), file)) {
                char* end = nullptr;
                const long value = std::strtol(text, &end, 10);
                if (end != text && value > 0) {
                    size = value;
                    if (*end == 'K' || *end == 'k') {
                        size *= 1024;
                    } else if (*end == 'M' || *end == 'm') {
                        size *= 1024 * 1024;
                    } else if (*end == 'G' || *end == 'g') {
                        size *= 1024 * 1024 * 1024;
                    }
                }
            }
            std::fclose(file);
        }
        if (std::FILE* file = std::fopen(
                "/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list",
                "re")) {
            char text[4096] = {};
            if (std::fgets(text, sizeof(text), file)) {
                // A list of singles and ranges, "0-3,8,12-15", as the part
                // publishes it.  Each entry is one core, and a range carries
                // the cores between its two ends as well.
                sharing = 0;
                const char* at = text;
                while (*at != '\0') {
                    char* next = nullptr;
                    const long low = std::strtol(at, &next, 10);
                    if (next == at) {
                        break;
                    }
                    ++sharing;
                    if (*next == '-') {
                        const long high = std::strtol(next + 1, &next, 10);
                        if (high > low) {
                            sharing += high - low;
                        }
                    }
                    at = (*next == ',') ? next + 1 : next;
                }
            }
            std::fclose(file);
        }
        if (size <= 0 || sharing <= 0) {
            return 1 << 20;
        }
        return size / sharing;
    }();
    return bytes;
}

inline bool ranges_apart(const void* one, const void* other, int64_t bytes) {
    const auto one_start = reinterpret_cast<uintptr_t>(one);
    const auto other_start = reinterpret_cast<uintptr_t>(other);
    return one_start + bytes <= other_start || other_start + bytes <= one_start;
}

// A ternary walk of three packed operands and one packed destination of the
// same width moves four times n bytes, and the team holds those only while
// they fit in the share of the last-level cache its threads add up to.  Past
// that the lines are chosen against everything else running, and the
// destination is asked for again only by the walk that wrote it -- which
// never reads it.
//
// A destination that is another window over an operand is a different matter:
// there the walk does read its own stores back, and a line streamed away has
// to come from memory for that read.
template <class T>
inline bool ternary_works_past_cache(int64_t n, const T* result, const T* self,
                                     const T* tensor1, const T* tensor2) {
    const int64_t bytes = n * static_cast<int64_t>(sizeof(T));
    if (bytes <= 0) {
        return false;
    }
    const int64_t team = parallel::get_num_threads();
    if (bytes * 4 <= last_level_bytes_per_core() * team) {
        return false;
    }
    return ranges_apart(result, self, bytes) &&
           ranges_apart(result, tensor1, bytes) &&
           ranges_apart(result, tensor2, bytes);
}

}  // namespace streamdst
}  // namespace cpu
}  // namespace tensorplay

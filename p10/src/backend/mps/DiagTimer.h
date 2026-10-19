#pragma once

// TEMPORARY instrumentation: per-call timing windows printed to stderr.

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace tensorplay {
namespace mps {

inline unsigned long long diagNowNs() {
  return static_cast<unsigned long long>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

struct DiagTimer {
  const char* label;
  unsigned long long calls = 0;
  unsigned long long min_ns = ~0ull;
  unsigned long long sum_ns = 0;

  void record(unsigned long long ns) {
    ++calls;
    min_ns = std::min(min_ns, ns);
    sum_ns += ns;
    if (calls % 512 == 0) {
      std::fprintf(stderr, "[mps-t] %s n=%llu min=%.2fus avg=%.2fus\n", label,
                   calls, static_cast<double>(min_ns) / 1000.0,
                   static_cast<double>(sum_ns) / static_cast<double>(calls) /
                       1000.0);
    }
  }
};

} // namespace mps
} // namespace tensorplay

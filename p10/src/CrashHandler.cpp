// Installing a handler for the two signals that mean the process cannot
// continue, so that a worker which dies says where it died.
//
// What a handler is allowed to do here is little, and deliberately so: the
// process is already in a state it cannot recover from, so this prints what
// it can and gets out of the way rather than trying to fix anything.  The
// previous handler is put back first, so a second fault does not re-enter
// this one, and it is then called if it was a real handler, which is what
// keeps a debugger's handler working.
#include <cstddef>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#include <process.h>  // getpid via _getpid
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include "Stacktrace.h"

namespace tensorplay {
namespace {

// The handlers this process had before ours, one slot per signal we take.
// A slot rather than a single variable because the two signals are
// installed independently and each has to be given back its own.
constexpr int kHandledSignals[] = {SIGILL, SIGSEGV};
constexpr size_t kHandledSignalCount =
    sizeof(kHandledSignals) / sizeof(kHandledSignals[0]);
void (*g_previous_handler[kHandledSignalCount])(int) = {};

int slot_for(int signum) {
  for (size_t i = 0; i < kHandledSignalCount; ++i) {
    if (kHandledSignals[i] == signum) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

void crash_handler(int signum) {
  const int slot = slot_for(signum);
  void (*old_action)(int) = nullptr;
  if (slot >= 0) {
    old_action = g_previous_handler[slot];
    g_previous_handler[slot] = nullptr;
  }
  // A second fault while printing must not come back here.
  std::signal(
      signum, (old_action != nullptr) ? old_action : SIG_DFL);

  fprintf(
      stderr,
      "Process %d crashed with signal %s (%d):\n",
      static_cast<int>(getpid()),
      strsignal(signum),
      signum);
  const std::string trace = capture_stacktrace_unconditional();
  fwrite(trace.data(), 1, trace.size(), stderr);
  fflush(stderr);

  if (old_action != nullptr && old_action != SIG_IGN && old_action != SIG_DFL) {
    old_action(signum);
  }
  if (old_action != SIG_IGN) {
    // Leave with a status that says it crashed rather than returned.
    _exit(-1);
  }
}

} // namespace

void init_crash_handler() {
  for (size_t i = 0; i < kHandledSignalCount; ++i) {
    g_previous_handler[i] = std::signal(kHandledSignals[i], crash_handler);
  }
}

} // namespace tensorplay

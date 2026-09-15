#!/usr/bin/env bash
# Prime the compiler-cache server before the build fans out.
#
# The build launches many compiles at once. When no cache server is yet
# listening, every parallel compile client races to bootstrap one under
# a fixed 10-second startup window, and a bootstrap that misses the
# window fails its compile and aborts the build. Starting one server up
# front, while the machine is still idle, turns every later compile
# into a plain connect.
#
# A stale server (possible on reused runners) would keep the cache
# backend configuration it was started with, so it is stopped first.
# "Address in use" from a start attempt means a server is already
# listening, which is the goal as well. A warm-up that cannot get a real
# server response is fatal. Letting compiler clients fall back to
# independent startup attempts recreates the race this script prevents.

set -u

if ! command -v sccache >/dev/null 2>&1; then
    echo "sccache not on PATH; skipping cache server warm-up" >&2
    exit 0
fi

sccache --stop-server >/dev/null 2>&1 || true

start_log="$(mktemp)"
trap 'rm -f "$start_log"' EXIT

started=false
for _ in 1 2 3; do
    if sccache --start-server 2>"$start_log"; then
        started=true
        break
    elif grep -q "Address in use" "$start_log"; then
        started=true
        break
    fi
    echo "sccache server warm-up failed, retrying:" >&2
    cat "$start_log" >&2
    sleep 5
done

if [[ "$started" != true ]]; then
    echo "sccache server did not start" >&2
    if [[ -n "${SCCACHE_ERROR_LOG:-}" && -f "$SCCACHE_ERROR_LOG" ]]; then
        tail -50 "$SCCACHE_ERROR_LOG" >&2
    fi
    exit 1
fi

for _ in 1 2 3 4 5 6; do
    if sccache --zero-stats >/dev/null 2>>"$start_log"; then
        exit 0
    fi
    sleep 5
done

echo "sccache server did not become ready" >&2
cat "$start_log" >&2
if [[ -n "${SCCACHE_ERROR_LOG:-}" && -f "$SCCACHE_ERROR_LOG" ]]; then
    tail -50 "$SCCACHE_ERROR_LOG" >&2
fi
exit 1

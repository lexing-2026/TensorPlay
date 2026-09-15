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
# listening, which is the goal as well. A warm-up that exhausts its
# retries is not fatal: the build then falls back to the on-demand
# bootstrap it would have done without this script.

set -u

if ! command -v sccache >/dev/null 2>&1; then
    echo "sccache not on PATH; skipping cache server warm-up" >&2
    exit 0
fi

sccache --stop-server >/dev/null 2>&1 || true

start_log="$(mktemp)"
trap 'rm -f "$start_log"' EXIT

for _ in 1 2 3 4 5; do
    if sccache --start-server 2>"$start_log"; then
        break
    elif grep -q "Address in use" "$start_log"; then
        break
    fi
    echo "sccache server warm-up failed, retrying:" >&2
    cat "$start_log" >&2
    sleep 5
done

sccache --zero-stats >/dev/null 2>&1 || true
exit 0

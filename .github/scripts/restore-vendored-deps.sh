#!/usr/bin/env bash
# Restore the vendored third_party sources that the wheel build links
# against. CI checkouts do not carry third_party/ (it is gitignored, and
# the workflows check out with submodules: false), so this script clones
# the pinned upstream revisions the local vendored tree was frozen at.
#
# Only the pieces the build actually needs are fetched:
#   - SLEEF (vector math) with its tlfloat submodule -- x86_64 Linux/macOS
#     only; the CMake glue degrades to scalar paths elsewhere
#   - NNPACK with its cpuinfo/FP16/FXdiv/psimd/pthreadpool dependencies
#     and the six/opcodes/PeachPy sources its configure step imports --
#     GCC/Clang on supported architectures only (MSVC turns NNPACK off)
#
# The script is idempotent: an existing checkout with the expected commit
# is left untouched, so it can run on self-hosted runners that already
# populated third_party.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/third_party"
mkdir -p "$DEST"

clone_pin() {
    local dir="$1" url="$2" rev="$3"
    if [[ -d "$DEST/$dir/.git" ]] && [[ "$(git -C "$DEST/$dir" rev-parse HEAD)" == "$rev" ]]; then
        return 0
    fi
    if [[ ! -d "$DEST/$dir/.git" ]]; then
        echo "::group::Clone $dir @ ${rev:0:12}"
        git -C "$DEST" clone --filter=blob:none "$url" "$dir"
        git -C "$DEST/$dir" -c advice.detachedHead=false checkout "$rev"
        echo "::endgroup::"
    fi
}

# --- SLEEF (vector math; static libsleef) ---
# The SleefShims declarations are only referenced by the x86_64 fast paths,
# so any other architecture skips the (long) libsleef build entirely; the
# configure step falls back to scalar libm paths.
if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
    clone_pin sleef https://github.com/shibatch/sleef 7623d6cfa2712462880fa63a4d0f0b5f775d1a83
    if [[ ! -f "$DEST/sleef/submodules/tlfloat/CMakeLists.txt" ]]; then
        echo "::group::Clone sleef tlfloat submodule"
        git -C "$DEST/sleef" submodule update --init --depth 1 submodules/tlfloat
        echo "::endgroup::"
    fi
fi

# --- NNPACK and its vendored dependencies ---
# cmake/External/nnpack.cmake supports x86, x86-64, ARM and ARM64 on
# Linux/macOS with GCC or Clang; MSVC builds disable NNPACK up front.
case "$(uname -s)/$(uname -m)" in
    Linux/*|Darwin/*) nnpack_supported=1 ;;
    *) nnpack_supported=0 ;;
esac
if [[ "$nnpack_supported" == "1" ]]; then
    clone_pin NNPACK https://github.com/Maratyszcza/NNPACK c07e3a0400713d546e0dea2d5466dd22ea389c73
    clone_pin cpuinfo https://github.com/pytorch/cpuinfo bc3c01e230c6974283e4b89421cfb0e232435589
    clone_pin FP16 https://github.com/Maratyszcza/FP16 4dfe081cf6bcd15db339cf2680b9281b8451eeb3
    clone_pin FXdiv https://github.com/Maratyszcza/FXdiv b408327ac2a15ec3e43352421954f5b1967701d1
    clone_pin psimd https://github.com/Maratyszcza/psimd 072586a71b55b7f8c584153d223e95687148a900
    clone_pin pthreadpool https://github.com/google/pthreadpool a56dcd79c699366e7ac6466792c3025883ff7704

    # NNPACK's configure step imports PeachPy (with six and opcodes) from
    # the source directories referenced by cmake/External/nnpack.cmake;
    # vendored Python sources stay under third_party/, never the host env.
    python_pin() {
        local dir="$1" url="$2" rev="$3"
        if [[ ! -d "$DEST/$dir/.git" ]]; then
            echo "::group::Clone $dir @ ${rev:0:12}"
            git -C "$DEST" clone --filter=blob:none "$url" "$dir"
            git -C "$DEST/$dir" -c advice.detachedHead=false checkout "$rev"
            echo "::endgroup::"
        fi
    }
    python_pin python-six https://github.com/benjaminp/six 15e31431af97e5e64b80af0a3f598d382bcdd49a
    python_pin python-opcodes https://github.com/Maratyszcza/opcodes 0e37e4f718d0ad2524b9a7c8147bdb78ff09cdd1
    python_pin python-peachpy https://github.com/malfet/PeachPy f45429b087dd7d5bc78bb40dc7cf06425c252d67
fi

# --- distributed transports (gloo + tensorpipe) ---
clone_pin gloo https://github.com/pytorch/gloo 44651678bdc9ffc837181295acdd142ae7880ad9
clone_pin tensorpipe https://github.com/pytorch/tensorpipe 2b4cd91092d335a697416b2a3cb398283246849d
if [[ -d "$DEST/tensorpipe/.git" && ! -d "$DEST/tensorpipe/third_party/libuv" ]]; then
    echo "::group::Init tensorpipe submodules (libuv/libnop/pybind11)"
    git -C "$DEST/tensorpipe" submodule update --init --depth 1
    echo "::endgroup::"
fi

echo "Vendored dependencies restored under $DEST"

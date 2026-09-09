#pragma once

// Shared im2col / col2im device kernels and the launch-grid helper.  The
// float path (cuDNN-adjacent im2col/col2im entry points) and the fp64
// im2col+GEMM path both instantiate these templates, so they live here to
// keep the two translation units numerically identical.

#include <cuda_runtime.h>

#include <cstdint>

namespace tensorplay {
namespace cuda {
namespace {

inline int cuda_blocks(int64_t n, int threads) {
    return static_cast<int>((n + threads - 1) / threads);
}

template <typename T>
__global__ void im2col_kernel(const T* in, T* col,
                              int64_t N,
                              int64_t C, int64_t H, int64_t W,
                              int64_t kH, int64_t kW,
                              int64_t pH, int64_t pW,
                              int64_t sH, int64_t sW,
                              int64_t dH, int64_t dW,
                              int64_t OH, int64_t OW) {
    const int64_t L = OH * OW;
    const int64_t CP = C * kH * kW;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N * CP * L) return;
    const int64_t n = idx / (CP * L);
    const int64_t rem = idx % (CP * L);
    const int64_t plane = rem / L;
    const int64_t l = rem % L;
    const int64_t ow = l % OW;
    const int64_t oh = l / OW;
    const int64_t kw = plane % kW;
    const int64_t kh = (plane / kW) % kH;
    const int64_t ci = plane / (kW * kH);
    const int64_t ih = oh * sH - pH + kh * dH;
    const int64_t iw = ow * sW - pW + kw * dW;
    T v = static_cast<T>(0);
    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
        v = in[(n * C + ci) * H * W + ih * W + iw];
    col[idx] = v;
}

template <typename T>
__global__ void col2im_kernel(const T* col, T* im,
                              int64_t C, int64_t H, int64_t W,
                              int64_t kH, int64_t kW,
                              int64_t pH, int64_t pW,
                              int64_t sH, int64_t sW,
                              int64_t dH, int64_t dW,
                              int64_t OH, int64_t OW) {
    // Race-free gather formulation: for each im element, the contributing
    // patches are exactly those whose (kh, kw, oh, ow) satisfy
    // ih = oh*sH - pH + kh*dH (and the width twin), so oh can be derived
    // directly instead of scanning all patches.
    const int64_t L = OH * OW;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t frame = C * H * W;
    if (idx >= frame) return;
    const int64_t n = blockIdx.z;
    const int64_t iw = idx % W;
    const int64_t ih = (idx / W) % H;
    const int64_t ci = idx / (W * H);
    T acc = static_cast<T>(0);
    for (int64_t kh = 0; kh < kH; ++kh) {
        const int64_t h_pad = ih + pH - kh * dH;
        if (h_pad < 0 || h_pad % sH != 0) continue;
        const int64_t oh = h_pad / sH;
        if (oh >= OH) continue;
        for (int64_t kw = 0; kw < kW; ++kw) {
            const int64_t w_pad = iw + pW - kw * dW;
            if (w_pad < 0 || w_pad % sW != 0) continue;
            const int64_t ow = w_pad / sW;
            if (ow >= OW) continue;
            const int64_t plane = (ci * kH + kh) * kW + kw;
            acc += col[(n * C * kH * kW + plane) * L + oh * OW + ow];
        }
    }
    im[n * frame + idx] = acc;
}


} // namespace
} // namespace cuda
} // namespace tensorplay

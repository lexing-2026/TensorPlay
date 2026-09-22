// nms / box_iou CUDA kernels.
//
// nms: a pairwise IoU bitmask is computed on the GPU (bit j of row i set when
// box j must be suppressed by box i); the greedy suppression walk then runs
// over the bitmask. Scores are ranked by a stable descending sort.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "Atomic.cuh"
#include "Half.h"
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace tensorplay {
namespace cuda {
namespace {

constexpr int kThreads = 256;

inline int64_t ceil_div_blocks(int64_t n, int64_t threads) {
    int64_t blocks = (n + threads - 1) / threads;
    return blocks > 65535 ? 65535 : blocks;
}

template <typename T>
__device__ inline T iou_single_cuda(const T* a, const T* b) {
    const T ix1 = max(a[0], b[0]);
    const T iy1 = max(a[1], b[1]);
    const T ix2 = min(a[2], b[2]);
    const T iy2 = min(a[3], b[3]);
    const T iw = max(ix2 - ix1, T(0));
    const T ih = max(iy2 - iy1, T(0));
    const T inter = iw * ih;
    const T area_a = (a[2] - a[0]) * (a[3] - a[1]);
    const T area_b = (b[2] - b[0]) * (b[3] - b[1]);
    const T uni = area_a + area_b - inter;
    return uni > T(0) ? inter / uni : T(0);
}

template <typename T>
__global__ void nms_suppression_mask_kernel(
        const int64_t n,
        const T* __restrict__ sorted_boxes,
        const float iou_threshold,
        uint64_t* __restrict__ mask,
        const int64_t words_per_row) {
    const int64_t total = n * words_per_row;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
         index += blockDim.x * gridDim.x) {
        const int64_t i = index / words_per_row;
        const int64_t word = index % words_per_row;
        uint64_t bits = 0;
        const T* bi = sorted_boxes + i * 4;
        const int64_t j_begin = word * 64;
        const int64_t j_end = min(j_begin + 64, n);
        for (int64_t j = j_begin; j < j_end; ++j) {
            if (iou_single_cuda(bi, sorted_boxes + j * 4) > T(iou_threshold)) {
                bits |= (uint64_t(1) << (j - j_begin));
            }
        }
        mask[index] = bits;
    }
}

template <typename T>
__global__ void box_iou_kernel(
        const int64_t n1, const int64_t n2,
        const T* __restrict__ boxes1,
        const T* __restrict__ boxes2,
        T* __restrict__ out) {
    const int64_t total = n1 * n2;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < total;
         index += blockDim.x * gridDim.x) {
        const int64_t i = index / n2;
        const int64_t j = index % n2;
        out[index] = iou_single_cuda(boxes1 + i * 4, boxes2 + j * 4);
    }
}

} // namespace

Tensor nms_cuda(const Tensor& boxes, const Tensor& scores, double iou_threshold) {
    if (boxes.dim() != 2 || boxes.size(1) != 4) {
        TP_THROW(RuntimeError, "nms: boxes must be a 2-D tensor of shape (N, 4)");
    }
    if (scores.dim() != 1 || scores.size(0) != boxes.size(0)) {
        TP_THROW(RuntimeError, "nms: scores must be a 1-D tensor matching the number of boxes");
    }
    const Tensor bc = boxes.contiguous();
    const Tensor sc = scores.contiguous();
    const int64_t n = bc.size(0);
    if (n == 0) {
        return Tensor::empty({0}, DType::Int64, bc.device());
    }
    // Host-side ranking: copy the scores down, stable-sort by descending
    // score, then reorder the boxes on the host and push them back up so the
    // suppression mask rows follow the greedy walk.
    if (bc.dtype() != sc.dtype()) {
        TP_THROW(RuntimeError, "nms: boxes and scores must have the same dtype");
    }
    std::vector<float> host_scores(n);
    std::vector<double> host_scores_f64;
    bool is_f32 = bc.dtype() == DType::Float32;
    if (is_f32) {
        cudaMemcpy(host_scores.data(), sc.data_ptr<float>(), sizeof(float) * n,
                   cudaMemcpyDeviceToHost);
    } else {
        host_scores_f64.resize(n);
        cudaMemcpy(host_scores_f64.data(), sc.data_ptr<double>(), sizeof(double) * n,
                   cudaMemcpyDeviceToHost);
    }
    std::vector<int64_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    if (is_f32) {
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return host_scores[a] > host_scores[b];
        });
    } else {
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return host_scores_f64[a] > host_scores_f64[b];
        });
    }
    Tensor sorted_boxes = Tensor::empty({n, 4}, bc.dtype(), bc.device());
    if (is_f32) {
        std::vector<float> host_boxes(n * 4), sorted(n * 4);
        cudaMemcpy(host_boxes.data(), bc.data_ptr<float>(), sizeof(float) * n * 4,
                   cudaMemcpyDeviceToHost);
        for (int64_t r = 0; r < n; ++r) {
            for (int64_t k = 0; k < 4; ++k) sorted[r * 4 + k] = host_boxes[order[r] * 4 + k];
        }
        cudaMemcpy(sorted_boxes.data_ptr<float>(), sorted.data(), sizeof(float) * n * 4,
                   cudaMemcpyHostToDevice);
    } else {
        std::vector<double> host_boxes(n * 4), sorted(n * 4);
        cudaMemcpy(host_boxes.data(), bc.data_ptr<double>(), sizeof(double) * n * 4,
                   cudaMemcpyDeviceToHost);
        for (int64_t r = 0; r < n; ++r) {
            for (int64_t k = 0; k < 4; ++k) sorted[r * 4 + k] = host_boxes[order[r] * 4 + k];
        }
        cudaMemcpy(sorted_boxes.data_ptr<double>(), sorted.data(), sizeof(double) * n * 4,
                   cudaMemcpyHostToDevice);
    }
    const int64_t words_per_row = (n + 63) / 64;
    Tensor mask = Tensor::empty({n * words_per_row}, DType::Int64, bc.device());
    {
        const int64_t total = n * words_per_row;
        dim3 block(kThreads);
        dim3 grid_dim(ceil_div_blocks(total, kThreads));
        if (bc.dtype() == DType::Float32) {
            nms_suppression_mask_kernel<float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                n, sorted_boxes.data_ptr<float>(), static_cast<float>(iou_threshold),
                reinterpret_cast<uint64_t*>(mask.data_ptr<int64_t>()), words_per_row);
        } else {
            nms_suppression_mask_kernel<double><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                n, sorted_boxes.data_ptr<double>(), iou_threshold,
                reinterpret_cast<uint64_t*>(mask.data_ptr<int64_t>()), words_per_row);
        }
    }
    // Greedy walk over the bitmask rows on the host (cheap: n^2/64 bit tests).
    std::vector<uint64_t> host_mask(n * words_per_row);
    cudaMemcpy(host_mask.data(), mask.data_ptr<int64_t>(),
               sizeof(uint64_t) * host_mask.size(), cudaMemcpyDeviceToHost);
    std::vector<bool> suppressed(n, false);
    std::vector<int64_t> keep;
    keep.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
        if (suppressed[i]) continue;
        keep.push_back(order[i]);
        const uint64_t* row = host_mask.data() + i * words_per_row;
        for (int64_t j = i + 1; j < n; ++j) {
            if (suppressed[j]) continue;
            if ((row[j / 64] >> (j % 64)) & 1) suppressed[j] = true;
        }
    }
    Tensor out = Tensor::empty({static_cast<int64_t>(keep.size())}, DType::Int64, bc.device());
    if (!keep.empty()) {
        cudaMemcpy(out.data_ptr<int64_t>(), keep.data(),
                   sizeof(int64_t) * keep.size(), cudaMemcpyHostToDevice);
    }
    return out;
}

Tensor box_iou_cuda(const Tensor& boxes1, const Tensor& boxes2) {
    if (boxes1.dim() != 2 || boxes1.size(1) != 4 ||
        boxes2.dim() != 2 || boxes2.size(1) != 4) {
        TP_THROW(RuntimeError, "box_iou: boxes must be 2-D tensors of shape (N, 4)");
    }
    const Tensor a = boxes1.contiguous();
    const Tensor b = boxes2.contiguous();
    const int64_t n1 = a.size(0);
    const int64_t n2 = b.size(0);
    Tensor out = Tensor::empty({n1, n2}, a.dtype(), a.device());
    const int64_t total = n1 * n2;
    if (total == 0) return out;
    dim3 block(kThreads);
    dim3 grid_dim(ceil_div_blocks(total, kThreads));
    switch (a.dtype()) {
        case DType::Float32:
            box_iou_kernel<float><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                n1, n2, a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>());
            break;
        case DType::Float64:
            box_iou_kernel<double><<<grid_dim, block, 0, getCurrentCUDAStream().stream()>>>(
                n1, n2, a.data_ptr<double>(), b.data_ptr<double>(), out.data_ptr<double>());
            break;
        default: TP_THROW(TypeError, "box_iou: unsupported dtype");
    }
    return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, NmsKernels) {
    m.impl("nms", nms_cuda);
    m.impl("box_iou", box_iou_cuda);
}

} // namespace cuda
} // namespace tensorplay
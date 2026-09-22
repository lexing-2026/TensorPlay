// nms and box_iou CPU kernels.
//
// Boxes use the xyxy convention (x1, y1, x2, y2), axis-aligned, stored as
// float. Intersection-over-union is computed with the standard
// inter/(area1+area2-inter) formula; degenerate boxes (empty intersection)
// yield IoU 0.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace tensorplay {
namespace cpu {
namespace {

using namespace tensorplay::parallel;

template <typename T>
T box_iou_single(const T* a, const T* b) {
    const T inter_x1 = std::max(a[0], b[0]);
    const T inter_y1 = std::max(a[1], b[1]);
    const T inter_x2 = std::min(a[2], b[2]);
    const T inter_y2 = std::min(a[3], b[3]);
    const T inter_w = std::max(inter_x2 - inter_x1, T(0));
    const T inter_h = std::max(inter_y2 - inter_y1, T(0));
    const T inter = inter_w * inter_h;
    const T area_a = (a[2] - a[0]) * (a[3] - a[1]);
    const T area_b = (b[2] - b[0]) * (b[3] - b[1]);
    const T uni = area_a + area_b - inter;
    if (uni <= T(0)) return T(0);
    return inter / uni;
}

template <typename T>
Tensor box_iou_cpu_impl(const Tensor& boxes1, const Tensor& boxes2) {
    const int64_t N1 = boxes1.size(0);
    const int64_t N2 = boxes2.size(0);
    Tensor out = Tensor::empty({N1, N2}, boxes1.dtype(), boxes1.device());
    if (out.numel() == 0) return out;
    const T* a = boxes1.data_ptr<T>();
    const T* b = boxes2.data_ptr<T>();
    T* o = out.data_ptr<T>();
    parallel_for(0, N1, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const T* ai = a + i * 4;
            T* oi = o + i * N2;
            for (int64_t j = 0; j < N2; ++j) {
                oi[j] = box_iou_single(ai, b + j * 4);
            }
        }
    });
    return out;
}

template <typename T>
Tensor nms_cpu_impl(const Tensor& boxes, const Tensor& scores, T iou_threshold) {
    const int64_t N = boxes.size(0);
    if (N == 0) {
        return Tensor::empty({0}, DType::Int64, boxes.device());
    }
    const T* bp = boxes.data_ptr<T>();
    const T* sp = scores.data_ptr<T>();
    std::vector<int64_t> order(N);
    std::iota(order.begin(), order.end(), 0);
    // Stable ordering by descending score; equal scores keep their input
    // order so the suppressed/kept selection is deterministic.
    std::stable_sort(order.begin(), order.end(),
                     [&](int64_t i, int64_t j) { return sp[i] > sp[j]; });
    std::vector<bool> suppressed(N, false);
    std::vector<int64_t> keep;
    keep.reserve(N);
    for (int64_t i = 0; i < N; ++i) {
        const int64_t cur = order[i];
        if (suppressed[cur]) continue;
        keep.push_back(cur);
        const T* cb = bp + cur * 4;
        for (int64_t j = i + 1; j < N; ++j) {
            const int64_t other = order[j];
            if (suppressed[other]) continue;
            if (box_iou_single(cb, bp + other * 4) > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
    Tensor out = Tensor::empty({static_cast<int64_t>(keep.size())}, DType::Int64, boxes.device());
    int64_t* op = out.data_ptr<int64_t>();
    for (size_t i = 0; i < keep.size(); ++i) op[i] = keep[i];
    return out;
}

} // namespace

Tensor nms_cpu(const Tensor& boxes, const Tensor& scores, double iou_threshold) {
    if (boxes.dim() != 2 || boxes.size(1) != 4) {
        TP_THROW(RuntimeError, "nms: boxes must be a 2-D tensor of shape (N, 4)");
    }
    if (scores.dim() != 1 || scores.size(0) != boxes.size(0)) {
        TP_THROW(RuntimeError, "nms: scores must be a 1-D tensor matching the number of boxes");
    }
    const Tensor bc = boxes.contiguous();
    const Tensor sc = scores.contiguous();
    switch (bc.dtype()) {
        case DType::Float32: return nms_cpu_impl<float>(bc, sc, static_cast<float>(iou_threshold));
        case DType::Float64: return nms_cpu_impl<double>(bc, sc, iou_threshold);
        default: TP_THROW(TypeError, "nms: unsupported dtype");
    }
}

Tensor box_iou_cpu(const Tensor& boxes1, const Tensor& boxes2) {
    if (boxes1.dim() != 2 || boxes1.size(1) != 4 ||
        boxes2.dim() != 2 || boxes2.size(1) != 4) {
        TP_THROW(RuntimeError, "box_iou: boxes must be 2-D tensors of shape (N, 4)");
    }
    const Tensor a = boxes1.contiguous();
    const Tensor b = boxes2.contiguous();
    switch (a.dtype()) {
        case DType::Float32: return box_iou_cpu_impl<float>(a, b);
        case DType::Float64: return box_iou_cpu_impl<double>(a, b);
        default: TP_THROW(TypeError, "box_iou: unsupported dtype");
    }
}

TENSORPLAY_LIBRARY_IMPL(CPU, NmsKernels) {
    m.impl("nms", nms_cpu);
    m.impl("box_iou", box_iou_cpu);
}

} // namespace cpu
} // namespace tensorplay

#pragma once

// Inner loops of the bagged-embedding reduction.
//
// The three stubs below are the only part compiled once per instruction-set
// tier (see TP_CPU_KERNEL_SRCS in p10/CMakeLists.txt).  Everything around them
// -- argument validation, bag layout, output allocation, the index sort that
// gives the backward its lock-free row partition -- is tier-independent and
// lives in EmbeddingBagKernels.cpp.  Each stub therefore takes one plain
// argument pack of raw pointers plus a storage-dtype tag, so a whole operator
// invocation crosses the dispatch boundary exactly once.

#include "DispatchStub.h"
#include "DType.h"

#include <cstdint>

namespace tensorplay {
namespace cpu {

// Reduction selector shared by the forward and backward kernels.
enum BagMode : int64_t { kBagSum = 0, kBagMean = 1, kBagMax = 2 };

// Bags are half-open [starts[b], ends[b]) ranges over the flat index vector.
// Positions no bag covers get offset2bag = -1 and take no part in either
// direction.  `indices` is always Int64 by the time it reaches a stub.
struct BagForwardArgs {
    DType dtype;
    const void* weight;
    const void* per_sample_weights;   // null when absent
    const int64_t* indices;
    const int64_t* starts;
    const int64_t* ends;
    int64_t num_rows;
    int64_t embedding_dim;
    int64_t num_indices;
    int64_t num_bags;
    int64_t mode;
    int64_t padding_idx;
    void* output;
    int64_t* offset2bag;
    int64_t* bag_size;
    int64_t* max_indices;             // only read in max mode
};

// Sum / mean backward walks segments of equal table row: sorted_rows and
// sorted_pos are the index vector sorted by row, and segment_starts has
// num_segments + 1 entries bounding each run.  Every segment owns a distinct
// output row, so the accumulation needs no locks and is reproducible.
// Max backward instead scatters through max_indices and is partitioned by
// column for the same reason.
struct BagDenseBackwardArgs {
    DType dtype;
    const void* grad;
    const void* per_sample_weights;   // null when absent
    const int64_t* sorted_rows;
    const int64_t* sorted_pos;
    const int64_t* segment_starts;
    const int64_t* offset2bag;
    const int64_t* bag_size;
    const int64_t* max_indices;       // only read in max mode
    int64_t num_segments;
    int64_t num_bags;
    int64_t embedding_dim;
    int64_t num_weights;
    int64_t mode;
    int64_t padding_idx;
    bool scale_grad_by_freq;
    void* grad_weight;
};

// One dot product of grad[bag] with weight[row] per index position.
struct BagPerSampleWeightsArgs {
    DType dtype;
    const void* grad;
    const void* weight;
    const int64_t* indices;
    const int64_t* offset2bag;
    int64_t num_rows;
    int64_t num_bags;
    int64_t embedding_dim;
    int64_t num_indices;
    int64_t padding_idx;
    void* output;
};

using bag_forward_fn = void (*)(const BagForwardArgs&);
DECLARE_DISPATCH(bag_forward_fn, bag_forward_stub)

using bag_dense_backward_fn = void (*)(const BagDenseBackwardArgs&);
DECLARE_DISPATCH(bag_dense_backward_fn, bag_dense_backward_stub)

using bag_psw_backward_fn = void (*)(const BagPerSampleWeightsArgs&);
DECLARE_DISPATCH(bag_psw_backward_fn, bag_psw_backward_stub)

} // namespace cpu
} // namespace tensorplay

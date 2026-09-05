#include "cpu/EmbeddingBagKernels.h"

#include "BFloat16.h"
#include "Exception.h"
#include "Half.h"
#include "Parallel.h"
#include "Macros.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

// Tier-compiled bagged-embedding inner loops (see TP_CPU_KERNEL_SRCS in
// p10/CMakeLists.txt).  Each copy lands in the CPU_CAPABILITY inline
// namespace; DispatchStub picks the best registered tier at runtime.

namespace tensorplay {
namespace cpu {
inline namespace CPU_CAPABILITY {

namespace {

using namespace tensorplay::parallel;

// Half and BFloat16 bags reduce in fp32 so a long bag does not lose the tail
// of its sum to an 11/8-bit significand.
template <typename T> struct BagAcc { using type = T; };
template <> struct BagAcc<Half> { using type = float; };
template <> struct BagAcc<BFloat16> { using type = float; };

// The table is random-access and normally far larger than L2, so a bag's cost
// is the load latency of its rows.  Warming a row many lookups ahead overlaps
// that latency with the accumulation of the current rows; one prefetch per
// cache line of the row is enough, since a row is contiguous.
constexpr int64_t kPrefetchAhead = 16;
constexpr int64_t kMaxPrefetchLines = 16;

inline void prefetch_row(const void* row, int64_t bytes) {
    const char* p = static_cast<const char*>(row);
    const int64_t lines = std::min<int64_t>((bytes + 63) / 64, kMaxPrefetchLines);
#if defined(__GNUC__) || defined(__clang__)
    for (int64_t line = 0; line < lines; ++line) {
        __builtin_prefetch(p + line * 64, 0, 3);
    }
#else
    // The prefetch is a pure latency hint; the accumulation path is correct
    // without it, so compilers without the builtin just walk the addresses.
    (void)p;
    (void)lines;
#endif
}

// Bags are independent, so each worker owns a disjoint slice of output rows.
int64_t bag_grain(int64_t num_bags, int64_t num_indices, int64_t row_size) {
    const int64_t work =
        std::max<int64_t>(num_indices, num_bags) * std::max<int64_t>(row_size, 1);
    if (work <= GRAIN_SIZE) return num_bags;
    const int64_t per_bag = std::max<int64_t>(work / std::max<int64_t>(num_bags, 1), 1);
    return std::max<int64_t>(GRAIN_SIZE / per_bag, 1);
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Forward
//
// Every bag runs in two passes.  The first walks the bag's index range to
// validate it, fill offset2bag and count the contributing entries; it owns the
// only throw site.  The second is the accumulation, and by then it is a plain
// gather over pre-checked rows with nothing in it that could inhibit
// vectorization or spill the loop-invariant pointers back to memory.
// ---------------------------------------------------------------------------

// Pass one.  Returns the number of entries that are not padding.
inline int64_t scan_bag(const int64_t* TP_RESTRICT idx, int64_t s, int64_t e,
                        int64_t num_rows, int64_t padding_idx, int64_t bag,
                        int64_t* TP_RESTRICT offset2bag) {
    int64_t count = 0;
    for (int64_t i = s; i < e; ++i) {
        const int64_t r = idx[i];
        if (r < 0 || r >= num_rows) {
            TP_THROW(IndexError,
                     "embedding_bag: index out of range in the embedding table");
        }
        offset2bag[i] = bag;
        count += (r != padding_idx) ? 1 : 0;
    }
    return count;
}

// Accumulates one bag over a fixed-width slice of the feature dimension.
//
// The accumulators are a compile-time sized array, so the compiler keeps the
// whole slice in vector registers for the duration of the bag.  That is the
// difference that matters here: an accumulator held in memory turns every row
// into a load-add-store with a loop-carried dependency through store-to-load
// forwarding, which caps throughput well below what the gather itself allows.
template <typename T, typename AccT, int64_t W, bool FULL>
inline void accumulate_tile(const T* TP_RESTRICT w, int64_t D,
                            const int64_t* TP_RESTRICT idx, int64_t s, int64_t e,
                            const T* TP_RESTRICT psw, int64_t padding_idx,
                            int64_t d0, int64_t width, AccT out_scale,
                            bool prefetch, int64_t numel, int64_t row_bytes,
                            T* TP_RESTRICT out_row) {
    // A full tile carries a compile-time trip count, which is what lets the
    // accumulators stay in registers; the ragged tail takes the runtime bound.
    const int64_t n = FULL ? W : width;
    AccT acc[W];
    for (int64_t k = 0; k < n; ++k) acc[k] = static_cast<AccT>(0);

    for (int64_t i = s; i < e; ++i) {
        if (prefetch && i + kPrefetchAhead < numel) {
            prefetch_row(w + idx[i + kPrefetchAhead] * D, row_bytes);
        }
        const int64_t r = idx[i];
        if (r == padding_idx) continue;
        const T* TP_RESTRICT w_row = w + r * D + d0;
        if (psw != nullptr) {
            const AccT scale = static_cast<AccT>(psw[i]);
            if constexpr (FULL) {
                for (int64_t k = 0; k < W; ++k) {
                    acc[k] += static_cast<AccT>(w_row[k]) * scale;
                }
            } else {
                for (int64_t k = 0; k < width; ++k) {
                    acc[k] += static_cast<AccT>(w_row[k]) * scale;
                }
            }
        } else {
            if constexpr (FULL) {
                for (int64_t k = 0; k < W; ++k) {
                    acc[k] += static_cast<AccT>(w_row[k]);
                }
            } else {
                for (int64_t k = 0; k < width; ++k) {
                    acc[k] += static_cast<AccT>(w_row[k]);
                }
            }
        }
    }
    for (int64_t k = 0; k < n; ++k) {
        out_row[d0 + k] = static_cast<T>(acc[k] * out_scale);
    }
}

// Sum and mean reduction over a chosen tile width.
template <typename T, int64_t kTile>
void bag_forward_add_tiled(const BagForwardArgs& a) {
    using acc_t = typename BagAcc<T>::type;

    const T* TP_RESTRICT w = static_cast<const T*>(a.weight);
    const T* TP_RESTRICT psw = static_cast<const T*>(a.per_sample_weights);
    T* TP_RESTRICT out = static_cast<T*>(a.output);
    const int64_t* TP_RESTRICT idx = a.indices;
    const int64_t* TP_RESTRICT starts = a.starts;
    const int64_t* TP_RESTRICT ends = a.ends;
    int64_t* TP_RESTRICT offset2bag = a.offset2bag;
    int64_t* TP_RESTRICT bag_size = a.bag_size;
    const int64_t D = a.embedding_dim;
    const int64_t num_rows = a.num_rows;
    const int64_t numel = a.num_indices;
    const int64_t padding_idx = a.padding_idx;
    const bool mean = a.mode == kBagMean;
    const int64_t row_bytes = D * static_cast<int64_t>(sizeof(T));

    parallel_for(0, a.num_bags, bag_grain(a.num_bags, numel, D),
                 [&](int64_t begin, int64_t end_bag) {
        for (int64_t b = begin; b < end_bag; ++b) {
            const int64_t s = starts[b];
            const int64_t e = ends[b];
            const int64_t count =
                scan_bag(idx, s, e, num_rows, padding_idx, b, offset2bag);
            bag_size[b] = count;

            const acc_t out_scale = (mean && count > 0)
                ? static_cast<acc_t>(1) / static_cast<acc_t>(count)
                : static_cast<acc_t>(1);
            T* TP_RESTRICT out_row = out + b * D;

            // The first slice warms the whole row, so later slices of the same
            // bag already find it in cache.
            int64_t d0 = 0;
            for (; d0 + kTile <= D; d0 += kTile) {
                accumulate_tile<T, acc_t, kTile, true>(
                    w, D, idx, s, e, psw, padding_idx, d0, kTile, out_scale,
                    d0 == 0, numel, row_bytes, out_row);
            }
            if (d0 < D) {
                accumulate_tile<T, acc_t, kTile, false>(
                    w, D, idx, s, e, psw, padding_idx, d0, D - d0, out_scale,
                    d0 == 0, numel, row_bytes, out_row);
            }
        }
    });
}

// The tile is the widest slice whose accumulators still fit in the vector
// registers of the tier being compiled; narrow embeddings pick a smaller one
// so they too run the compile-time-width path instead of the ragged tail.
template <typename T>
void bag_forward_add(const BagForwardArgs& a) {
    using acc_t = typename BagAcc<T>::type;
    constexpr int64_t kWide = static_cast<int64_t>(64 / sizeof(acc_t)) * 4;
    const int64_t D = a.embedding_dim;
    if (D <= kWide / 4) {
        bag_forward_add_tiled<T, kWide / 4>(a);
    } else if (D <= kWide / 2) {
        bag_forward_add_tiled<T, kWide / 2>(a);
    } else {
        bag_forward_add_tiled<T, kWide>(a);
    }
}

// Max reduction.
//
// The running maximum is updated with a select rather than a branch: on random
// data "is this element a new maximum" is close to a coin flip, and one
// mispredict per element costs far more than the compare itself.  The winning
// row is tracked in a scratch buffer as wide as the accumulator, so value and
// index stay lane-aligned and the update vectorizes; it is widened into the
// Int64 output once per bag.
template <typename T, typename ArgT>
void bag_forward_max(const BagForwardArgs& a) {
    using acc_t = typename BagAcc<T>::type;

    const T* TP_RESTRICT w = static_cast<const T*>(a.weight);
    T* TP_RESTRICT out = static_cast<T*>(a.output);
    const int64_t* TP_RESTRICT idx = a.indices;
    const int64_t* TP_RESTRICT starts = a.starts;
    const int64_t* TP_RESTRICT ends = a.ends;
    int64_t* TP_RESTRICT offset2bag = a.offset2bag;
    int64_t* TP_RESTRICT bag_size = a.bag_size;
    int64_t* TP_RESTRICT max_indices = a.max_indices;
    const int64_t D = a.embedding_dim;
    const int64_t num_rows = a.num_rows;
    const int64_t numel = a.num_indices;
    const int64_t padding_idx = a.padding_idx;
    const int64_t row_bytes = D * static_cast<int64_t>(sizeof(T));

    parallel_for(0, a.num_bags, bag_grain(a.num_bags, numel, D),
                 [&](int64_t begin, int64_t end_bag) {
        std::vector<acc_t> acc(static_cast<size_t>(D));
        std::vector<ArgT> arg(static_cast<size_t>(D));
        acc_t* TP_RESTRICT accum = acc.data();
        ArgT* TP_RESTRICT winner = arg.data();

        for (int64_t b = begin; b < end_bag; ++b) {
            const int64_t s = starts[b];
            const int64_t e = ends[b];
            const int64_t count =
                scan_bag(idx, s, e, num_rows, padding_idx, b, offset2bag);
            bag_size[b] = count;

            T* TP_RESTRICT out_row = out + b * D;
            int64_t* TP_RESTRICT arg_row = max_indices + b * D;
            if (count == 0) {
                for (int64_t d = 0; d < D; ++d) {
                    out_row[d] = static_cast<T>(0);
                    arg_row[d] = 0;
                }
                continue;
            }

            bool seeded = false;
            for (int64_t i = s; i < e; ++i) {
                if (i + kPrefetchAhead < numel) {
                    prefetch_row(w + idx[i + kPrefetchAhead] * D, row_bytes);
                }
                const int64_t r = idx[i];
                if (r == padding_idx) continue;
                const T* TP_RESTRICT w_row = w + r * D;
                const ArgT row = static_cast<ArgT>(r);
                if (!seeded) {
                    // Seeding from the first contributing row keeps the result
                    // exact even when every value in the bag is -inf.
                    for (int64_t d = 0; d < D; ++d) {
                        accum[d] = static_cast<acc_t>(w_row[d]);
                        winner[d] = row;
                    }
                    seeded = true;
                } else {
                    for (int64_t d = 0; d < D; ++d) {
                        const acc_t v = static_cast<acc_t>(w_row[d]);
                        const bool take = v > accum[d];
                        accum[d] = take ? v : accum[d];
                        winner[d] = take ? row : winner[d];
                    }
                }
            }

            for (int64_t d = 0; d < D; ++d) {
                out_row[d] = static_cast<T>(accum[d]);
                arg_row[d] = static_cast<int64_t>(winner[d]);
            }
        }
    });
}

template <typename T>
void bag_forward_typed(const BagForwardArgs& a) {
    if (a.mode != kBagMax) {
        bag_forward_add<T>(a);
        return;
    }
    // A table taller than the narrow scratch's range cannot be tracked in it.
    // No such table fits in memory today, but fall back rather than truncate.
    using acc_t = typename BagAcc<T>::type;
    using narrow_t = std::conditional_t<sizeof(acc_t) == 8, int64_t, int32_t>;
    if (a.num_rows <= static_cast<int64_t>(std::numeric_limits<narrow_t>::max())) {
        bag_forward_max<T, narrow_t>(a);
    } else {
        bag_forward_max<T, int64_t>(a);
    }
}

void bag_forward_impl(const BagForwardArgs& a) {
    switch (a.dtype) {
        case DType::Float32: bag_forward_typed<float>(a); break;
        case DType::Float64: bag_forward_typed<double>(a); break;
        case DType::Float16: bag_forward_typed<Half>(a); break;
        default: bag_forward_typed<BFloat16>(a); break;
    }
}

// ---------------------------------------------------------------------------
// Dense backward
// ---------------------------------------------------------------------------

template <typename T>
void bag_dense_backward_sum_mean(const BagDenseBackwardArgs& a) {
    using acc_t = typename BagAcc<T>::type;

    const T* g = static_cast<const T*>(a.grad);
    const T* psw = static_cast<const T*>(a.per_sample_weights);
    T* gw = static_cast<T*>(a.grad_weight);
    const int64_t D = a.embedding_dim;
    const int64_t num_bags = a.num_bags;
    const int64_t num_segments = a.num_segments;
    const int64_t total = a.segment_starts[num_segments];
    const int64_t rows_per_segment =
        std::max<int64_t>(total / std::max<int64_t>(num_segments, 1), 1);
    const int64_t grain =
        std::max<int64_t>(GRAIN_SIZE / std::max<int64_t>(rows_per_segment * D, 1), 1);

    parallel_for(0, num_segments, grain, [&](int64_t begin, int64_t end) {
        std::vector<acc_t> acc(static_cast<size_t>(D));
        acc_t* TP_RESTRICT accum = acc.data();
        for (int64_t seg = begin; seg < end; ++seg) {
            const int64_t lo = a.segment_starts[seg];
            const int64_t hi = a.segment_starts[seg + 1];
            const int64_t row = a.sorted_rows[lo];
            if (row == a.padding_idx) continue;
            if (row < 0 || row >= a.num_weights) {
                TP_THROW(IndexError,
                         "embedding_bag_backward: index out of range in the embedding table");
            }
            const acc_t freq = a.scale_grad_by_freq
                ? static_cast<acc_t>(1) / static_cast<acc_t>(hi - lo)
                : static_cast<acc_t>(1);

            std::fill(acc.begin(), acc.end(), static_cast<acc_t>(0));
            bool touched = false;
            for (int64_t k = lo; k < hi; ++k) {
                const int64_t pos = a.sorted_pos[k];
                const int64_t bag = a.offset2bag[pos];
                if (bag < 0 || bag >= num_bags) continue;
                acc_t scale = freq;
                if (psw != nullptr) scale *= static_cast<acc_t>(psw[pos]);
                if (a.mode == kBagMean) {
                    const int64_t count = a.bag_size[bag];
                    if (count > 0) scale /= static_cast<acc_t>(count);
                }
                const T* TP_RESTRICT g_row = g + bag * D;
                for (int64_t d = 0; d < D; ++d) {
                    accum[d] += static_cast<acc_t>(g_row[d]) * scale;
                }
                touched = true;
            }
            if (!touched) continue;
            T* TP_RESTRICT gw_row = gw + row * D;
            for (int64_t d = 0; d < D; ++d) {
                gw_row[d] = static_cast<T>(static_cast<acc_t>(gw_row[d]) + accum[d]);
            }
        }
    });
}

template <typename T>
void bag_dense_backward_max(const BagDenseBackwardArgs& a) {
    using acc_t = typename BagAcc<T>::type;

    const T* g = static_cast<const T*>(a.grad);
    T* gw = static_cast<T*>(a.grad_weight);
    const int64_t D = a.embedding_dim;
    const int64_t num_bags = a.num_bags;

    // Each column is owned by exactly one worker, so the scattered row writes
    // never collide even though several bags may select the same row.
    parallel_for(0, D, std::max<int64_t>(GRAIN_SIZE / std::max<int64_t>(num_bags, 1), 1),
                 [&](int64_t begin, int64_t end) {
        for (int64_t d = begin; d < end; ++d) {
            for (int64_t b = 0; b < num_bags; ++b) {
                if (a.bag_size != nullptr && a.bag_size[b] == 0) continue;
                const int64_t row = a.max_indices[b * D + d];
                if (row < 0 || row >= a.num_weights) {
                    TP_THROW(IndexError, "embedding_bag_backward: max index out of range");
                }
                T* dst = gw + row * D + d;
                *dst = static_cast<T>(static_cast<acc_t>(*dst) +
                                      static_cast<acc_t>(g[b * D + d]));
            }
        }
    });
}

template <typename T>
void bag_dense_backward_typed(const BagDenseBackwardArgs& a) {
    if (a.mode == kBagMax) {
        bag_dense_backward_max<T>(a);
    } else {
        bag_dense_backward_sum_mean<T>(a);
    }
}

void bag_dense_backward_impl(const BagDenseBackwardArgs& a) {
    switch (a.dtype) {
        case DType::Float32: bag_dense_backward_typed<float>(a); break;
        case DType::Float64: bag_dense_backward_typed<double>(a); break;
        case DType::Float16: bag_dense_backward_typed<Half>(a); break;
        default: bag_dense_backward_typed<BFloat16>(a); break;
    }
}

// ---------------------------------------------------------------------------
// Per-sample-weight backward
// ---------------------------------------------------------------------------

template <typename T>
void bag_psw_backward_typed(const BagPerSampleWeightsArgs& a) {
    using acc_t = typename BagAcc<T>::type;

    const T* g = static_cast<const T*>(a.grad);
    const T* w = static_cast<const T*>(a.weight);
    T* out = static_cast<T*>(a.output);
    const int64_t D = a.embedding_dim;
    const int64_t row_bytes = D * static_cast<int64_t>(sizeof(T));

    parallel_for(0, a.num_indices, std::max<int64_t>(GRAIN_SIZE / std::max<int64_t>(D, 1), 1),
                 [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t bag = a.offset2bag[i];
            const int64_t row = a.indices[i];
            if (i + kPrefetchAhead < end) {
                const int64_t ahead = a.indices[i + kPrefetchAhead];
                if (ahead >= 0 && ahead < a.num_rows) {
                    prefetch_row(w + ahead * D, row_bytes);
                }
            }
            if (bag < 0 || bag >= a.num_bags || row == a.padding_idx) {
                out[i] = static_cast<T>(0);
                continue;
            }
            if (row < 0 || row >= a.num_rows) {
                TP_THROW(IndexError,
                         "embedding_bag_backward: index out of range in the embedding table");
            }
            const T* TP_RESTRICT g_row = g + bag * D;
            const T* TP_RESTRICT w_row = w + row * D;
            acc_t dot = 0;
            for (int64_t d = 0; d < D; ++d) {
                dot += static_cast<acc_t>(g_row[d]) * static_cast<acc_t>(w_row[d]);
            }
            out[i] = static_cast<T>(dot);
        }
    });
}

void bag_psw_backward_impl(const BagPerSampleWeightsArgs& a) {
    switch (a.dtype) {
        case DType::Float32: bag_psw_backward_typed<float>(a); break;
        case DType::Float64: bag_psw_backward_typed<double>(a); break;
        case DType::Float16: bag_psw_backward_typed<Half>(a); break;
        default: bag_psw_backward_typed<BFloat16>(a); break;
    }
}

} // anonymous namespace
} // inline namespace CPU_CAPABILITY

// One slot per tier TU: the DEFAULT/AVX2 copies register their own slot, while
// the AVX512 copy opts in explicitly (plain REGISTER_DISPATCH would null it).
#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(bag_forward_stub, &bag_forward_impl);
REGISTER_DISPATCH(bag_dense_backward_stub, &bag_dense_backward_impl);
REGISTER_DISPATCH(bag_psw_backward_stub, &bag_psw_backward_impl);
#else
ALSO_REGISTER_AVX512_DISPATCH(bag_forward_stub, &bag_forward_impl);
ALSO_REGISTER_AVX512_DISPATCH(bag_dense_backward_stub, &bag_dense_backward_impl);
ALSO_REGISTER_AVX512_DISPATCH(bag_psw_backward_stub, &bag_psw_backward_impl);
#endif

} // namespace cpu
} // namespace tensorplay

#pragma once

// A blocked product over host memory, for a kernel that has one to do.
//
// A kernel that computes a product of matrices does not want to write the loop
// that walks it: the walking is a few lines, but the packing and the block sizes
// are where the performance is, and getting them right once means every kernel
// that wants a product gets them.  So the product is here, and a kernel that
// wants one says the shape of its problem and gets the loop back.
//
// The leading dimension is what makes this blocked rather than merely nested: a
// matrix is a row of rows, and saying how far apart two rows are is what says
// whether the block being read is contiguous.  A kernel passing a packed matrix
// passes a leading dimension equal to its row length, which is the ordinary case
// and the one where the block read is a run rather than a gather.
//
// The transposition flags are the same fact stated the other way round.  A
// kernel that has a matrix on the other side of the product would rather not copy
// it, so the copy is pushed into the block read -- the block is gathered rather
// than read in one run, which is slower per block and avoids a copy per call.
// Which of the two is worth it is a fact about how many times the matrix is
// read, so both are offered and the caller says.

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "BFloat16.h"
#include "DType.h"
#include "Half.h"
#include "dnnl.hpp"

namespace tensorplay {
namespace cpu {

// The two arguments a blocked product takes, and the result.  Kept as one
// struct so that a caller cannot swap the destination for a source: the three
// are three views of the same problem and swapping two of them is a different
// problem that happens to have the same shape.
struct GemmOperands {
    const void* a;
    const void* b;
    void* c;
    int64_t m, n, k;
    int64_t lda, ldb, ldc;
    bool trans_a;
    bool trans_b;
    float alpha;
    float beta;
};

// Whether a type is stored narrower than it is computed in.  A product of two
// such values is a product of the width they are stored at, so the accumulator
// has to be wider than the inputs or the extra bits are lost at the first
// multiply.
template <typename T>
struct GemmAccumType {
    using type = float;
};
template <>
struct GemmAccumType<float> {
    using type = float;
};
template <>
struct GemmAccumType<double> {
    using type = double;
};
template <>
struct GemmAccumType<int64_t> {
    using type = int64_t;
};

// A packed copy of a matrix, held between calls that want the same one.
//
// Packing is what a blocked product spends most of its setup on, and a kernel
// that reads the same weights for every query would pay it every time.  So the
// packed form is kept, keyed by the shape and strides it was made from -- a
// different shape is a different key, because a packed form is only good for
// the layout it was packed from.
class GemmPackCache {
   public:
    static GemmPackCache& get() {
        static GemmPackCache cache;
        return cache;
    }

    // The packed form of `b`, or null when there is not one.
    const void* find(const void* src,
                     int64_t rows,
                     int64_t cols,
                     int64_t ldb,
                     bool trans) const {
        std::lock_guard<std::mutex> guard(mutex_);
        auto it = entries_.find(key(src, rows, cols, ldb, trans));
        return it == entries_.end() ? nullptr : it->second.data();
    }

    // Keep `packed` for the matrix it was made from.
    void insert(const void* src,
                int64_t rows,
                int64_t cols,
                int64_t ldb,
                bool trans,
                std::vector<uint8_t> packed) {
        std::lock_guard<std::mutex> guard(mutex_);
        entries_[key(src, rows, cols, ldb, trans)] = std::move(packed);
    }

    void clear() {
        std::lock_guard<std::mutex> guard(mutex_);
        entries_.clear();
    }

   private:
    struct Key {
        const void* src;
        int64_t rows, cols, ldb;
        bool trans;
        bool operator==(const Key& other) const {
            return src == other.src && rows == other.rows && cols == other.cols &&
                   ldb == other.ldb && trans == other.trans;
        }
    };

    struct KeyHash {
        size_t operator()(const Key& k) const {
            size_t h = std::hash<const void*>()(k.src);
            auto mix = [&h](uint64_t v) {
                h ^= static_cast<size_t>(v) + 0x9e3779b97f4a7c15ULL +
                     (h << 6) + (h >> 2);
            };
            mix(static_cast<uint64_t>(k.rows));
            mix(static_cast<uint64_t>(k.cols));
            mix(static_cast<uint64_t>(k.ldb));
            mix(k.trans ? 1u : 0u);
            return h;
        }
    };

    static Key key(const void* src, int64_t rows, int64_t cols, int64_t ldb, bool trans) {
        return Key{src, rows, cols, ldb, trans};
    }

    mutable std::mutex mutex_;
    std::unordered_map<Key, std::vector<uint8_t>, KeyHash> entries_;
};

// Whether this type has a blocked form worth using, and whether the machine can
// run it.  A type stored narrower than it is computed in is the case where the
// hardware has a unit for it and the answer is yes; anything else is answered by
// the machine rather than by the type.
inline bool could_pack_brgemm(DType dtype) {
    switch (dtype) {
        case DType::Half:
        case DType::BFloat16:
        case DType::Float:
            return true;
        default:
            return false;
    }
}

// Walk the product the caller described.
//
// Not the blocked walk: this is the plain one, and it is here so that a kernel
// asking for a product always gets one, even on a machine where the blocked form
// has no unit to run on.  Which is the whole of the difference between this and
// the blocked form -- a few times faster on a machine that has the unit, and the
// same answer either way.
template <typename T>
void brgemm_fallback(const GemmOperands& op) {
    using Acc = typename GemmAccumType<T>::type;
    const T* a = static_cast<const T*>(op.a);
    const T* b = static_cast<const T*>(op.b);
    T* c = static_cast<T*>(op.c);
    for (int64_t i = 0; i < op.m; ++i) {
        for (int64_t j = 0; j < op.n; ++j) {
            Acc acc = 0;
            for (int64_t p = 0; p < op.k; ++p) {
                const T av = op.trans_a ? a[p * op.lda + i] : a[i * op.lda + p];
                const T bv = op.trans_b ? b[j * op.ldb + p] : b[p * op.ldb + j];
                acc += static_cast<Acc>(av) * static_cast<Acc>(bv);
            }
            T& out = c[i * op.ldc + j];
            // The scale is on the product and the addend is what is already
            // there, which is the usual order: a caller asking for
            // one-and-nil is asking for the product, and a caller asking for
            // nil-and-one is asking for the destination to be left alone.
            out = static_cast<T>(op.alpha * acc + op.beta * static_cast<Acc>(out));
        }
    }
}

template <typename T>
void brgemm_dispatch(const GemmOperands& op) {
    brgemm_fallback<T>(op);
}

// Walk the product in the width the caller asked for.
//
// The type is named rather than inferred from the pointers, because a kernel
// handing over raw memory has already thrown away the thing that would say what
// it is -- and a product walked in the wrong width is a product of the wrong
// numbers rather than a failure.
inline void brgemm(DType dtype, const GemmOperands& op) {
    switch (dtype) {
        case DType::Half:
            brgemm_dispatch<Half>(op);
            break;
        case DType::BFloat16:
            brgemm_dispatch<BFloat16>(op);
            break;
        case DType::Float:
            brgemm_dispatch<float>(op);
            break;
        case DType::Double:
            brgemm_dispatch<double>(op);
            break;
        case DType::Long:
            brgemm_dispatch<int64_t>(op);
            break;
        default:
            brgemm_dispatch<float>(op);
            break;
    }
}

// Whether a packed copy of the second operand is being kept, and whether it
// should be.  Kept so that a kernel which packed once can tell the cache to let
// go, and so that a kernel asking whether packing is worth it has an answer that
// came from something rather than from a guess.
inline bool brgemm_release(bool* had_pack = nullptr) {
    if (had_pack != nullptr) {
        *had_pack = false;
    }
    GemmPackCache::get().clear();
    return true;
}

}  // namespace cpu
}  // namespace tensorplay

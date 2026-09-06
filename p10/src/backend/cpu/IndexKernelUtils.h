#pragma once

#include "TensorIterator.h"

namespace tensorplay::cpu {

struct Indexer {
    int64_t num_indexers;
    char** indexers;
    const int64_t* indexer_strides;
    const std::vector<int64_t>& original_sizes;
    const std::vector<int64_t>& original_strides;

    int64_t get(int64_t i) const {
        int64_t offset = 0;
        for (int64_t j = 0; j < num_indexers; ++j) {
            int64_t value = *reinterpret_cast<int64_t*>(indexers[j] + i * indexer_strides[j]);
            const int64_t size = original_sizes[j];
            TP_CHECK_INDEX(value >= -size && value < size,
                           "index ", value, " is out of bounds for dimension ", j,
                           " with size ", size);
            if (value < 0) value += size;
            offset += value * original_strides[j];
        }
        return offset;
    }

    int64_t get_1(int64_t i) const {
        int64_t value = *reinterpret_cast<int64_t*>(indexers[0] + i * indexer_strides[0]);
        const int64_t size = original_sizes[0];
        TP_CHECK_INDEX(value >= -size && value < size,
                       "index ", value, " is out of bounds for dimension 0 with size ", size);
        if (value < 0) value += size;
        return value * original_strides[0];
    }
};

template <typename Function>
void cpu_index_kernel(TensorIterator& iter,
                      const std::vector<int64_t>& index_size,
                      const std::vector<int64_t>& index_stride,
                      const Function& function, bool serial = false) {
    const int ntensor = iter.ntensors();
    TP_CHECK(ntensor >= 2 && static_cast<size_t>(ntensor - 2) == index_size.size() &&
             index_size.size() == index_stride.size(), "invalid index iterator metadata");
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        Indexer indexer{ntensor - 2, &data[2], &strides[2], index_size, index_stride};
        bool constant = true;
        for (int arg = 2; arg < ntensor; ++arg) constant &= strides[arg] == 0;
        if (constant) {
            const int64_t offset = indexer.get(0);
            for (int64_t i = 0; i < n; ++i) {
                function(data[0] + strides[0] * i, data[1] + strides[1] * i, offset);
            }
        } else if (indexer.num_indexers == 1) {
            for (int64_t i = 0; i < n; ++i) {
                function(data[0] + strides[0] * i, data[1] + strides[1] * i, indexer.get_1(i));
            }
        } else {
            for (int64_t i = 0; i < n; ++i) {
                function(data[0] + strides[0] * i, data[1] + strides[1] * i, indexer.get(i));
            }
        }
    };
    if (serial) iter.serial_for_each(loop, {0, iter.numel()});
    else iter.for_each(loop, 3000);
}

} // namespace tensorplay::cpu

#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "Utils.h"
#include <cstring>
#include <vector>
#include <algorithm>

namespace tensorplay {
namespace cpu {

Tensor constant_pad_nd_cpu(const Tensor& self, const std::vector<int64_t>& pad, Scalar value) {
    auto self_shape = self.shape();
    auto ndim = self.dim();
    auto pad_len = pad.size();
    
    if (pad_len % 2 != 0) {
         TP_THROW(ValueError, "Length of pad must be even but instead it equals ", pad_len);
    }
    
    int64_t l_pad = pad_len / 2;
    if (l_pad > ndim) {
         TP_THROW(ValueError, "Padding length too large");
    }
    
    // 1. Calculate output shape
    std::vector<int64_t> out_shape = static_cast<std::vector<int64_t>>(self_shape);
    for (size_t i = 0; i < l_pad; ++i) {
        // pad is [pad_left, pad_right, pad_top, pad_bottom, ...]
        // corresponding to dim [ndim-1, ndim-2, ...]
        int64_t pad_idx = i * 2; // 0, 2, 4
        int64_t dim = ndim - 1 - i;
        out_shape[dim] += pad[pad_idx] + pad[pad_idx + 1];
    }
    
    // 2. Allocate output
    Tensor output = Tensor::empty(out_shape, self.dtype(), self.device());
    
    // 3. Fill with value
    output.fill_(value);
    
    // 4. Copy input to output slice
    Tensor out_slice = output;
    for (size_t i = 0; i < l_pad; ++i) {
        int64_t dim = ndim - 1 - i;
        int64_t pad_l = pad[i * 2];
        // narrow/slice logic
        // slice(dim, start, end)
        int64_t start = pad_l;
        int64_t end = start + self_shape[dim];
        out_slice = out_slice.slice(dim, start, end);
    }
    out_slice.copy_(self);
    
    return output;
}

Tensor constant_pad_nd_backward_cpu(const Tensor& grad_output, const std::vector<int64_t>& pad) {
    auto ndim = grad_output.dim();
    auto pad_len = pad.size();
    auto l_pad = pad_len / 2;
    
    Tensor grad_input = grad_output;
    for (size_t i = 0; i < l_pad; ++i) {
        int64_t dim = ndim - 1 - i;
        int64_t pad_l = pad[i * 2];
        int64_t pad_r = pad[i * 2 + 1];
        
        int64_t start = pad_l;
        int64_t end = grad_input.size(dim) - pad_r;
        grad_input = grad_input.slice(dim, start, end);
    }
    
    // Return contiguous copy
    if (grad_input.is_contiguous()) {
        return grad_input;
    } else {
        return grad_input.clone();
    }
}

TENSORPLAY_LIBRARY_IMPL(CPU, PadKernels) {
    m.impl("constant_pad_nd", constant_pad_nd_cpu);
    m.impl("constant_pad_nd_backward", constant_pad_nd_backward_cpu);
}

} // namespace cpu
} // namespace tensorplay

// Optimized 1-D max pooling.

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "GradMode.h"
#include "Parallel.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

namespace tensorplay::cpu {

namespace ops = tensorplay::tpx::ops;

namespace {

int64_t div_rtn(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

int64_t pooling_output_shape(int64_t input, int64_t kernel, int64_t padding,
                             int64_t stride, int64_t dilation, bool ceil_mode) {
    if (stride == 0) TP_THROW(RuntimeError, "stride should not be zero");
    if (padding < 0) TP_THROW(RuntimeError, "padding must be non-negative");
    if (kernel <= 0) TP_THROW(RuntimeError, "kernel_size must be greater than zero");
    if (stride <= 0) TP_THROW(RuntimeError, "stride must be greater than zero");
    if (dilation <= 0) TP_THROW(RuntimeError, "dilation must be greater than zero");
    if (padding > ((kernel - 1) * dilation + 1) / 2) {
        TP_THROW(RuntimeError, "padding should be at most half of effective kernel size");
    }
    int64_t output = div_rtn(
        input + 2 * padding - dilation * (kernel - 1) - 1 +
            (ceil_mode ? stride - 1 : 0),
        stride) + 1;
    if (ceil_mode && (output - 1) * stride >= input + padding) --output;
    return output;
}

void check_max_pool1d_args(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    if (self.dim() != 2 && self.dim() != 3) {
        TP_THROW(RuntimeError, "max_pool1d() expected 2D or 3D input tensor");
    }
    if (kernel_size.size() != 1) {
        TP_THROW(RuntimeError, "max_pool1d() kernel_size must contain one int");
    }
    if (!stride.empty() && stride.size() != 1) {
        TP_THROW(RuntimeError, "max_pool1d() stride must contain one int");
    }
    if (padding.size() != 1 || dilation.size() != 1) {
        TP_THROW(RuntimeError,
                 "max_pool1d() padding and dilation must contain one int");
    }
    const int64_t step = stride.empty() ? kernel_size[0] : stride[0];
    if ((self.dim() == 2 && (self.size(0) == 0 || self.size(1) == 0)) ||
        (self.dim() == 3 && (self.size(1) == 0 || self.size(2) == 0))) {
        TP_THROW(RuntimeError, "max_pool1d(): input dimensions must be non-zero");
    }
    if (kernel_size[0] <= 0) {
        TP_THROW(RuntimeError, "max_pool1d() kernel_size must be greater than zero");
    }
    if (step <= 0) {
        TP_THROW(RuntimeError, "max_pool1d() stride must be greater than zero");
    }
    if (padding[0] < 0) {
        TP_THROW(RuntimeError, "max_pool1d() padding must be non-negative");
    }
    if (padding[0] > kernel_size[0] / 2) {
        TP_THROW(RuntimeError,
                 "max_pool1d() padding should be at most half of kernel size");
    }
    if (dilation[0] <= 0) {
        TP_THROW(RuntimeError, "max_pool1d() dilation must be greater than zero");
    }
    if (pooling_output_shape(self.size(-1), kernel_size[0], padding[0], step,
                             dilation[0], ceil_mode) <= 0) {
        TP_THROW(RuntimeError, "max_pool1d() Invalid computed output size");
    }
}

template <typename scalar_t>
struct max_pool_accum {
    using type = scalar_t;
};

template <>
struct max_pool_accum<tensorplay::Half> {
    using type = float;
};

template <>
struct max_pool_accum<tensorplay::BFloat16> {
    using type = float;
};

template <typename scalar_t>
void max_pool1d_kernel(
    scalar_t* output, const scalar_t* input, int64_t nb, int64_t nc,
    int64_t iw, int64_t ow, int64_t kw, int64_t sj, int64_t pj, int64_t dj) {
    using acc_t = typename max_pool_accum<scalar_t>::type;
    const acc_t fill = -std::numeric_limits<acc_t>::infinity();
    const int64_t total_planes = nb * nc;

    tensorplay::parallel::parallel_for(0, total_planes, 1,
        [&](int64_t begin, int64_t end) {
            for (int64_t plane = begin; plane < end; ++plane) {
                scalar_t* op = output + plane * ow;
                const scalar_t* ip = input + plane * iw;
                std::fill_n(op, ow, static_cast<scalar_t>(fill));

                // position updates only the output interval it can reach.
                for (int64_t kj = 0; kj < kw; ++kj) {
                    int64_t oj = std::max<int64_t>(0, (pj - kj * dj + sj - 1) / sj);
                    int64_t oe = ow;
                    const int64_t last = (ow - 1) * sj + kj * dj - pj;
                    if (last >= iw) oe -= (last - (iw - 1) + sj - 1) / sj;
                    int64_t ij = oj * sj + kj * dj - pj;
                    for (; oj < oe; ++oj, ij += sj) {
                        const acc_t value = static_cast<acc_t>(ip[ij]);
                        const acc_t current = static_cast<acc_t>(op[oj]);
                        if (std::isnan(value) || current < value) {
                            op[oj] = static_cast<scalar_t>(value);
                        }
                    }
                }
            }
        });
}

Tensor max_pool1d_impl(const Tensor& self,
                       const std::vector<int64_t>& kernel_size,
                       const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& dilation, bool ceil_mode) {
    const int64_t nb = self.dim() == 3 ? self.size(-3) : 1;
    const int64_t nc = self.size(-2);
    const int64_t iw = self.size(-1);
    const int64_t kw = kernel_size[0];
    const int64_t sj = stride.empty() ? kw : stride[0];
    const int64_t pj = padding[0];
    const int64_t dj = dilation[0];
    const int64_t ow = pooling_output_shape(iw, kw, pj, sj, dj, ceil_mode);
    if (ow <= 0) TP_THROW(RuntimeError, "max_pool1d() Invalid computed output size");

    const Tensor input = self.is_contiguous() ? self : self.contiguous();
    Tensor output = Tensor::empty({nb, nc, ow}, self.dtype(), self.device());
    switch (self.dtype()) {
        case DType::Float32:
            max_pool1d_kernel(output.data_ptr<float>(), input.data_ptr<float>(),
                              nb, nc, iw, ow, kw, sj, pj, dj);
            break;
        case DType::Float64:
            max_pool1d_kernel(output.data_ptr<double>(), input.data_ptr<double>(),
                              nb, nc, iw, ow, kw, sj, pj, dj);
            break;
        case DType::Float16:
            max_pool1d_kernel(output.data_ptr<tensorplay::Half>(),
                              input.data_ptr<tensorplay::Half>(), nb, nc, iw, ow,
                              kw, sj, pj, dj);
            break;
        case DType::BFloat16:
            max_pool1d_kernel(output.data_ptr<tensorplay::BFloat16>(),
                              input.data_ptr<tensorplay::BFloat16>(), nb, nc, iw,
                              ow, kw, sj, pj, dj);
            break;
        default:
            return std::get<0>(ops::max_pool1d_with_indices(
                self, kernel_size, stride, padding, dilation, ceil_mode));
    }
    return self.dim() == 2 ? ops::squeeze(output, 0) : output;
}

} // namespace

Tensor max_pool1d_native_cpu(
    const Tensor& self, const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation, bool ceil_mode) {
    check_max_pool1d_args(self, kernel_size, stride, padding, dilation, ceil_mode);
    if (self.requires_grad() && tensorplay::GradMode::is_enabled()) {
        return std::get<0>(ops::max_pool1d_with_indices(
            self, kernel_size, stride, padding, dilation, ceil_mode));
    }
    return max_pool1d_impl(self, kernel_size, stride, padding, dilation,
                           ceil_mode);
}

TENSORPLAY_LIBRARY_IMPL(CPU, NativeMaxPooling) {
    m.impl("max_pool1d", max_pool1d_native_cpu);
}

} // namespace tensorplay::cpu

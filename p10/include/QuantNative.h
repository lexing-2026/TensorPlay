#pragma once
#include "QuantElementwise.h"
#include <vector>

namespace tensorplay::quant {
struct PoolGeometry {
    int rank, mode;
    int64_t in[3]{1,1,1}, out[3]{1,1,1}, kernel[3]{1,1,1}, stride[3]{1,1,1}, pad[3]{}, dilation[3]{1,1,1};
    int64_t in_size = 1, out_size = 1, zp, divisor;
    bool include_pad;
    int64_t channels = 1, batch_stride = 0, channel_stride = 0, spatial_strides[3]{};
    bool channels_last = false;
};
struct PoolWindow {
    int64_t start[3]{}, end[3]{};
    uint32_t initial = 0;
    float inverse = 0.f;
};

#define TP_DECLARE_PACKED_RECURRENCE(backend) \
std::tuple<Tensor,Tensor,Tensor> recurrent_packed_##backend(const Tensor&,const Tensor&, \
    const std::optional<Tensor>&,const std::vector<Tensor>&,const std::vector<Tensor>&, \
    const std::vector<Tensor>&,const std::vector<bool>&,int64_t,double,bool,bool,bool, \
    int64_t,const std::optional<Tensor>&);
TP_DECLARE_PACKED_RECURRENCE(cpu)
Tensor std_strided_cpu(const Tensor&, const std::vector<int64_t>&, bool, double);
Tensor moment_inner_cpu(const Tensor&, const std::vector<int64_t>&, bool, bool, double);
Tensor grid_sample_cpu(const Tensor&, const Tensor&, int64_t, int64_t, bool);
Tensor batch_norm_cpu(const Tensor&, const std::optional<Tensor>&, const std::optional<Tensor>&,
    const Tensor&, const Tensor&, double, double, int64_t, bool);
Tensor resize_cpu(const Tensor&, const std::vector<int64_t>&,
    const std::vector<std::optional<double>>&, bool, bool, bool);
Tensor binary_cpu(const Tensor&, const Tensor&, double, int64_t, double, int64_t,
                  double, int64_t, int, bool = false);
Tensor cat_cpu(const std::vector<Tensor>&, int64_t, std::optional<double>,
               std::optional<int64_t>, bool);
Tensor pool_cpu(const Tensor&, const std::vector<int64_t>&,
                const std::vector<int64_t>&, const std::vector<int64_t>&,
                const std::vector<int64_t>&, bool, bool, std::optional<int64_t>, int);
#ifdef USE_CUDA
TP_DECLARE_PACKED_RECURRENCE(cuda)
Tensor std_strided_cuda(const Tensor&, const std::vector<int64_t>&, bool, double);
Tensor moment_inner_cuda(const Tensor&, const std::vector<int64_t>&, bool, bool, double);
Tensor grid_sample_cuda(const Tensor&, const Tensor&, int64_t, int64_t, bool);
Tensor batch_norm_cuda(const Tensor&, const std::optional<Tensor>&, const std::optional<Tensor>&,
    const Tensor&, const Tensor&, double, double, int64_t, bool);
Tensor resize_cuda(const Tensor&, const std::vector<int64_t>&,
    const std::vector<std::optional<double>>&, bool, bool, bool);
Tensor binary_cuda(const Tensor&, const Tensor&, double, int64_t, double, int64_t,
                   double, int64_t, int, bool = false);
Tensor cat_cuda(const std::vector<Tensor>&, int64_t, std::optional<double>,
                std::optional<int64_t>, bool);
Tensor pool_cuda(const Tensor&, const std::vector<int64_t>&,
                 const std::vector<int64_t>&, const std::vector<int64_t>&,
                 const std::vector<int64_t>&, bool, bool, std::optional<int64_t>, int);
#endif
#undef TP_DECLARE_PACKED_RECURRENCE
}

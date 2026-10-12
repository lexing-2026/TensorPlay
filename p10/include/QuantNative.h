#pragma once
#include "QuantElementwise.h"
#include <vector>

namespace tensorplay::quant {
#define TP_DECLARE_PACKED_RECURRENCE(backend) \
std::tuple<Tensor,Tensor,Tensor> recurrent_packed_##backend(const Tensor&,const Tensor&, \
    const std::optional<Tensor>&,const std::vector<Tensor>&,const std::vector<Tensor>&, \
    const std::vector<Tensor>&,const std::vector<bool>&,int64_t,double,bool,bool,bool, \
    int64_t,const std::optional<Tensor>&);
TP_DECLARE_PACKED_RECURRENCE(cpu)
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

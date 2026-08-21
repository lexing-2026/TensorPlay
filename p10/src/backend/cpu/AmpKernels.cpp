#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"

#include <cmath>
#include <vector>

namespace tensorplay {
namespace cpu {
namespace {

template <typename scalar_t>
bool any_non_finite(const std::vector<Tensor>& self) {
    for (const auto& g : self) {
        const scalar_t* p = g.data_ptr<scalar_t>();
        const int64_t n = g.numel();
        for (int64_t i = 0; i < n; ++i) {
            float v = static_cast<float>(p[i]);
            if (std::isnan(v) || std::isinf(v)) return true;
        }
    }
    return false;
}

template <typename scalar_t>
void scale_list(std::vector<Tensor>& self, double inv_scale) {
    for (auto& g : self) {
        scalar_t* p = g.data_ptr<scalar_t>();
        const int64_t n = g.numel();
        for (int64_t i = 0; i < n; ++i) {
            p[i] = static_cast<scalar_t>(static_cast<float>(p[i]) *
                                         static_cast<float>(inv_scale));
        }
    }
}

template <typename F>
void dispatch_float_dtype(const char* op_name, const std::vector<Tensor>& self, F&& fn) {
    if (self.empty()) return;
    switch (self[0].dtype()) {
        case DType::Float32: fn.template operator()<float>(); break;
        case DType::Float64: fn.template operator()<double>(); break;
        case DType::Float16: fn.template operator()<Half>(); break;
        case DType::BFloat16: fn.template operator()<BFloat16>(); break;
        default:
            TP_THROW(NotImplementedError, std::string(op_name) +
                ": unsupported dtype");
    }
}

} // anonymous namespace

void _amp_foreach_non_finite_check_and_unscale_cpu(
    std::vector<Tensor> self, Tensor& found_inf, const Tensor& inv_scale) {
    if (found_inf.data_ptr<float>()[0] != 0.0f) {
        return;
    }
    dispatch_float_dtype(
        "_amp_foreach_non_finite_check_and_unscale_", self,
        [&]<typename scalar_t>() {
            if (any_non_finite<scalar_t>(self)) {
                found_inf.data_ptr<float>()[0] = 1.0f;
                return;
            }
            scale_list<scalar_t>(
                self, static_cast<double>(inv_scale.data_ptr<float>()[0]));
        });
}

void _amp_update_scale_cpu(
    Tensor& self, Tensor& growth_tracker, const Tensor& found_inf,
    double growth_factor, double backoff_factor, int64_t growth_interval) {
    float* scale_ptr = self.data_ptr<float>();
    int32_t* tracker_ptr = growth_tracker.data_ptr<int32_t>();
    if (found_inf.data_ptr<float>()[0] > 0) {
        scale_ptr[0] = static_cast<float>(scale_ptr[0] * backoff_factor);
        tracker_ptr[0] = 0;
    } else {
        tracker_ptr[0] += 1;
        if (tracker_ptr[0] == growth_interval) {
            scale_ptr[0] = static_cast<float>(scale_ptr[0] * growth_factor);
            tracker_ptr[0] = 0;
        }
    }
}

} // namespace cpu

TENSORPLAY_LIBRARY_IMPL(CPU, AmpKernels) {
    m.impl("_amp_foreach_non_finite_check_and_unscale_",
           cpu::_amp_foreach_non_finite_check_and_unscale_cpu);
    m.impl("_amp_update_scale_", cpu::_amp_update_scale_cpu);
}

} // namespace tensorplay

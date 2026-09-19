// Composite kernels: batch-norm entry points expressed through
// native_batch_norm.  Every backend with a native_batch_norm kernel gets
// them; backends with a dedicated kernel register their own.

#include "Dispatcher.h"
#include "Exception.h"
#include "OutWrite.h"
#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <optional>
#include <tuple>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

namespace {

Tensor empty_reserve(const Tensor& input) {
    return Tensor::empty({0}, DType::UInt8, input.device());
}

} // anonymous namespace

// Training normalization that updates the running statistics in place.
std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_with_update_native(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias, Tensor& running_mean,
        Tensor& running_var, double momentum, double eps) {
    auto [output, save_mean, save_invstd] = ops::native_batch_norm(
        input, weight, bias, running_mean, running_var, true, momentum, eps);
    return {output, save_mean, save_invstd, empty_reserve(input)};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_with_update_out_native(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias, Tensor& running_mean,
        Tensor& running_var, double momentum, double eps, Tensor& out,
        Tensor& save_mean, Tensor& save_invstd, Tensor& reserve) {
    auto result = batch_norm_with_update_native(input, weight, bias, running_mean,
                                                running_var, momentum, eps);
    write_out(out, std::get<0>(result));
    write_out(save_mean, std::get<1>(result));
    write_out(save_invstd, std::get<2>(result));
    write_out(reserve, std::get<3>(result));
    return {out, save_mean, save_invstd, reserve};
}

// Evaluation normalization with the running statistics, never updated.
std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_no_update_native(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias,
        const std::optional<Tensor>& running_mean,
        const std::optional<Tensor>& running_var, double momentum, double eps) {
    TP_CHECK(running_mean.has_value() && running_mean->defined(),
             "running_mean must be defined in evaluation mode");
    TP_CHECK(running_var.has_value() && running_var->defined(),
             "running_var must be defined in evaluation mode");
    auto [output, save_mean, save_invstd] = ops::native_batch_norm(
        input, weight, bias, running_mean, running_var, false, momentum, eps);
    return {output, save_mean, save_invstd, empty_reserve(input)};
}

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_legit_no_training_native(
        const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias, const Tensor& running_mean,
        const Tensor& running_var, double momentum, double eps) {
    return ops::native_batch_norm(input, weight, bias, running_mean, running_var,
                                  false, momentum, eps);
}

} // namespace composite

TENSORPLAY_LIBRARY_IMPL(Composite, BatchNormVariantComposite) {
    m.impl("_batch_norm_with_update", composite::batch_norm_with_update_native);
    m.impl("_batch_norm_with_update.out", composite::batch_norm_with_update_out_native);
    m.impl("_batch_norm_no_update", composite::batch_norm_no_update_native);
    m.impl("_native_batch_norm_legit_no_training",
           composite::native_batch_norm_legit_no_training_native);
}

} // namespace tensorplay

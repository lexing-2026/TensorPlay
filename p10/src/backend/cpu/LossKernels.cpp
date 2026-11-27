#include "Tensor.h"
#include "Generator.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "LossReduce.h"
#include "Parallel.h"
#include "cpu/LossVecKernels.h"
#include "Scalar.h"
#include <algorithm>
#include <tuple>
#include <optional>
#include <cmath>
#include <vector>

#include "../composite/AttentionComposite.h"

namespace tensorplay {
namespace cpu {

namespace ops = tensorplay::tpx::ops;

// The tier-compiled walks live in LossVecKernelsImpl.cpp.
DEFINE_DISPATCH(bce_sum_stub);
DEFINE_DISPATCH(bce_into_stub);
DEFINE_DISPATCH(kl_sum_stub);
DEFINE_DISPATCH(kl_into_stub);
DEFINE_DISPATCH(kl_grad_stub);
DEFINE_DISPATCH(mse_sum_stub);
DEFINE_DISPATCH(mse_into_stub);
DEFINE_DISPATCH(mse_grad_stub);
DEFINE_DISPATCH(l1_sum_stub);
DEFINE_DISPATCH(l1_into_stub);
DEFINE_DISPATCH(l1_grad_stub);
DEFINE_DISPATCH(bce_logits_rows_stub);
DEFINE_DISPATCH(huber_sum_stub);
DEFINE_DISPATCH(huber_into_stub);
DEFINE_DISPATCH(smooth_l1_sum_stub);
DEFINE_DISPATCH(smooth_l1_into_stub);
DEFINE_DISPATCH(smooth_l1_grad_stub);
DEFINE_DISPATCH(softmargin_sum_stub);
DEFINE_DISPATCH(softmargin_into_stub);
DEFINE_DISPATCH(softmargin_grad_stub);
DEFINE_DISPATCH(poisson_sum_stub);
DEFINE_DISPATCH(poisson_into_stub);
DEFINE_DISPATCH(poisson_grad_stub);
DEFINE_DISPATCH(hinge_sum_stub);
DEFINE_DISPATCH(hinge_into_stub);
DEFINE_DISPATCH(hinge_grad_stub);
DEFINE_DISPATCH(margin_rank_sum_stub);
DEFINE_DISPATCH(margin_rank_into_stub);
DEFINE_DISPATCH(margin_rank_grad_stub);
DEFINE_DISPATCH(cosine_rows_stub);
DEFINE_DISPATCH(cosine_grad_stub);
DEFINE_DISPATCH(multilabel_rows_stub);
DEFINE_DISPATCH(triplet_rows_stub);
DEFINE_DISPATCH(nll_rows_stub);
DEFINE_DISPATCH(nll_weight_total_stub);
DEFINE_DISPATCH(nll_grad_stub);
DEFINE_DISPATCH(nll2d_rows_stub);
DEFINE_DISPATCH(nll2d_grad_stub);

template <typename scalar_t>
std::tuple<Tensor, Tensor> nll_loss_impl(const Tensor& input_arg, const Tensor& target_arg,
                                         const std::optional<Tensor>& weight_arg,
                                         int64_t reduction, int64_t ignore_index) {
    // reduction: 0=none, 1=mean, 2=sum.  input is (N, C) with target (N), or
    // (C) with a 0-d target.
    const bool no_batch = input_arg.dim() == 1;
    const Tensor input = (no_batch ? input_arg.unsqueeze(0) : input_arg).contiguous();
    const Tensor target = (no_batch ? target_arg.reshape({1}) : target_arg)
                              .to(DType::Int64).contiguous();
    const int64_t n_batch = input.size(0);
    const int64_t n_classes = input.size(1);
    if (target.numel() != n_batch) {
        TP_THROW(ValueError, "nll_loss: expected target of ", n_batch,
                 " elements, got ", target.numel());
    }
    Tensor weight;
    if (weight_arg.has_value() && weight_arg->defined()) {
        if (weight_arg->numel() != n_classes) {
            TP_THROW(RuntimeError, "weight tensor should be defined either for all ",
                     n_classes, " classes or no classes");
        }
        weight = weight_arg->to(input.dtype()).contiguous();
    }

    const scalar_t* input_data = input.data_ptr<scalar_t>();
    const int64_t* target_data = target.data_ptr<int64_t>();
    const scalar_t* weight_data = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;

    // One gathered score per row, and rows are independent: the row is the unit
    // the walk splits on.  The reduce rides along in the same walk -- each chunk
    // keeps its own pair of sums and the tail adds them up -- so a 32K-row loss
    // stops paying for a serial pass with a branch and a double add in it.
    std::vector<scalar_t> output_data(reduction == 0 ? static_cast<size_t>(n_batch)
                                                     : static_cast<size_t>(0));
    constexpr int64_t kRowChunk = 1024;
    const int64_t n_chunks = (n_batch + kRowChunk - 1) / kRowChunk;
    std::vector<double> chunk_sum(static_cast<size_t>(n_chunks), 0.0);
    std::vector<double> chunk_weight(static_cast<size_t>(n_chunks), 0.0);
    int64_t bad_target = -1;
    parallel::parallel_for(0, n_batch, kRowChunk, [&](int64_t b, int64_t e) {
        const size_t ci = static_cast<size_t>(b / kRowChunk);
        double local_sum = 0.0;
        double local_weight = 0.0;
        for (int64_t i = b; i < e; ++i) {
            const int64_t t = target_data[i];
            if (t == ignore_index) {
                if (reduction == 0) output_data[static_cast<size_t>(i)] = 0;
                continue;
            }
            if (t < 0 || t >= n_classes) {
                bad_target = t;
                continue;
            }
            const double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
            const double loss =
                -static_cast<double>(input_data[i * n_classes + t]) * w;
            local_sum += loss;
            local_weight += w;
            if (reduction == 0) output_data[static_cast<size_t>(i)] = static_cast<scalar_t>(loss);
        }
        chunk_sum[ci] = local_sum;
        chunk_weight[ci] = local_weight;
    });
    if (bad_target >= 0) TP_THROW(IndexError, "Target ", bad_target, " is out of bounds.");
    double total_weight = 0;
    double sum = 0;
    for (int64_t c = 0; c < n_chunks; ++c) {
        sum += chunk_sum[static_cast<size_t>(c)];
        total_weight += chunk_weight[static_cast<size_t>(c)];
    }

    const DType dt = input.dtype();
    if (reduction == 0) {
        Tensor per_sample = Tensor::tensor(output_data, dt, input.device());
        if (no_batch) per_sample = per_sample.reshape({});
        // Unreduced batched losses carry no normalizer.
        const double reported = no_batch ? total_weight : 0.0;
        return std::make_tuple(per_sample, Tensor::full({}, Scalar(reported), dt, input.device()));
    }
    if (reduction == 1) {
        sum /= total_weight;  // every target ignored: 0 / 0 is NaN
    } else if (reduction != 2) {
        TP_THROW(ValueError, "Invalid reduction mode");
    }
    return std::make_tuple(Tensor::full({}, Scalar(sum), dt, input.device()),
                           Tensor::full({}, Scalar(total_weight), dt, input.device()));
}

std::tuple<Tensor, Tensor> nll_loss_kernel(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight, int64_t reduction, int64_t ignore_index) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    const bool no_batch = input.dim() == 1;
    const bool has_w = weight.has_value() && weight->defined();
    // The fused walk reads the gathered class scores in parallel; it covers
    // contiguous Float32/Float64 inputs with an Int64 target, and the optional
    // per-class weights in the input dtype.  Everything else falls back.
    const bool fused =
        (input.dtype() == DType::Float32 || input.dtype() == DType::Float64) &&
        input.is_contiguous() && target.is_contiguous() &&
        target.dtype() == DType::Int64 &&
        (!has_w || (weight->dtype() == input.dtype() &&
                    weight->is_contiguous()));
    if (fused) {
        const int64_t rows = no_batch ? 1 : input.size(0);
        const int64_t cols = no_batch ? input.numel() : input.size(1);
        const int64_t* tgt = target.data_ptr<int64_t>();
        for (int64_t i = 0; i < rows; ++i) {
            const int64_t t = tgt[i];
            if (t != ignore_index && (t < 0 || t >= cols))
                TP_THROW(IndexError, "Target ", t, " is out of bounds.");
        }
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        const void* wptr = has_w ? weight->data_ptr() : nullptr;
        const int has_weight = has_w ? 1 : 0;
        if (reduction == 0) {
            Tensor per_row = Tensor::empty({rows}, dt, dev);
            const auto [sum, tw] = nll_rows_stub(
                DeviceType::CPU, input.data_ptr(), target.data_ptr(), wptr,
                per_row.data_ptr(), rows, cols, ignore_index, has_weight,
                dtype);
            (void)sum;
            Tensor result = no_batch ? per_row.reshape({}) : per_row;
            // Unreduced batched losses carry no normalizer; a single-sample
            // call reports its own weight.
            const double reported = no_batch ? tw : 0.0;
            return std::make_tuple(
                result, Tensor::full({}, Scalar(reported), dt, dev));
        }
        const auto [sum, tw] = nll_rows_stub(
            DeviceType::CPU, input.data_ptr(), target.data_ptr(), wptr,
            nullptr, rows, cols, ignore_index, has_weight, dtype);
        double value = sum;
        if (reduction == 1) value /= tw;  // every target ignored: 0 / 0 is NaN
        return std::make_tuple(Tensor::full({}, Scalar(value), dt, dev),
                               Tensor::full({}, Scalar(tw), dt, dev));
    }
    if (input.dtype() == DType::Float32) return nll_loss_impl<float>(input, target, weight, reduction, ignore_index);
    if (input.dtype() == DType::Float64) return nll_loss_impl<double>(input, target, weight, reduction, ignore_index);
    if (input.dtype() == DType::BFloat16) return nll_loss_impl<BFloat16>(input, target, weight, reduction, ignore_index);
    if (input.dtype() == DType::Half) return nll_loss_impl<Half>(input, target, weight, reduction, ignore_index);
    TP_THROW(NotImplementedError, "nll_loss only supports Float32, Float64, BFloat16 and Half");
}

template <typename scalar_t>
Tensor nll_loss_backward_impl(const Tensor& grad_output_arg, const Tensor& input_arg,
                              const Tensor& target_arg,
                              const std::optional<Tensor>& weight_arg, int64_t reduction,
                              int64_t ignore_index, const Tensor& total_weight) {
    // The walk names one entry per row, so every operand is read by position:
    // the seed a sum leaves behind is a single value spread across the whole row
    // axis, and an index or weight arriving from another layout is stored in its
    // own strides.
    // A single sample arrives as a bare class vector with a 0-d target; it is
    // lifted to one row so the walk below has a uniform shape.
    const bool no_batch = input_arg.dim() == 1;
    const Tensor input = (no_batch ? input_arg.unsqueeze(0) : input_arg).contiguous();
    const Tensor grad_output = grad_output_arg.to(input.dtype()).contiguous();
    const Tensor target = (no_batch ? target_arg.reshape({1}) : target_arg)
                              .to(DType::Int64).contiguous();
    Tensor weight;
    if (weight_arg.has_value() && weight_arg->defined()) {
        weight = weight_arg->to(input.dtype()).contiguous();
    }

    int64_t n_batch = input.size(0);
    int64_t n_classes = input.size(1);

    Tensor grad_input = Tensor::zeros({n_batch, n_classes}, input.dtype(), input.device());
    scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();

    const int64_t* target_data = target.data_ptr<int64_t>();
    const scalar_t* weight_data = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;

    double tw = 0;
    if (total_weight.defined()) {
        tw = total_weight.item().to<double>();
    } else if (reduction == 1) {
        for (int64_t i = 0; i < n_batch; ++i) {
            int64_t t = target_data[i];
            if (t != ignore_index) {
                tw += weight_data ? static_cast<double>(weight_data[t]) : 1.0;
            }
        }
    }

    if (reduction == 0) { // None
        const scalar_t* grad_out_data = grad_output.data_ptr<scalar_t>();
        for (int64_t i = 0; i < n_batch; ++i) {
            int64_t t = target_data[i];
            if (t == ignore_index) continue;
            double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
            grad_input_data[i * n_classes + t] = static_cast<scalar_t>(-w * static_cast<double>(grad_out_data[i]));
        }
    } else {
        double grad_val = grad_output.item().to<double>();
        if (reduction == 1) { // Mean
             if (tw > 0) grad_val /= tw;
        }

        for (int64_t i = 0; i < n_batch; ++i) {
            int64_t t = target_data[i];
            if (t == ignore_index) continue;
            double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
            grad_input_data[i * n_classes + t] = static_cast<scalar_t>(-w * grad_val);
        }
    }

    if (no_batch) return grad_input.reshape({n_classes});
    return grad_input;
}

Tensor nll_loss_backward_kernel(const Tensor& grad_output_arg, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight, int64_t reduction, int64_t ignore_index, const Tensor& total_weight) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    // The seed may arrive in a different float width than the score plane; the
    // result is written at the score width, so the seed is read at that width.
    const bool float_seed =
        grad_output_arg.dtype() == DType::Float32 ||
        grad_output_arg.dtype() == DType::Float64;
    const bool float_scores =
        input.dtype() == DType::Float32 || input.dtype() == DType::Float64;
    const Tensor grad_output =
        (float_seed && float_scores && grad_output_arg.dtype() != input.dtype())
            ? grad_output_arg.to(input.dtype())
            : grad_output_arg;
    const bool no_batch = input.dim() == 1;
    const bool has_w = weight.has_value() && weight->defined();
    const int64_t rows = no_batch ? 1 : input.size(0);
    const int64_t cols = no_batch ? input.numel() : input.size(1);
    const bool fused =
        (input.dtype() == DType::Float32 || input.dtype() == DType::Float64) &&
        input.is_contiguous() && target.is_contiguous() &&
        target.dtype() == DType::Int64 &&
        (!has_w || (weight->dtype() == input.dtype() &&
                    weight->is_contiguous())) &&
        grad_output.is_contiguous() && grad_output.dtype() == input.dtype() &&
        (reduction == 0 ? grad_output.numel() == rows
                        : grad_output.numel() == 1);
    if (fused) {
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        const void* wptr = has_w ? weight->data_ptr() : nullptr;
        double tw = 0.0;
        if (total_weight.defined() && total_weight.numel() == 1) {
            tw = total_weight.item().to<double>();
        } else if (reduction == 1) {
            tw = nll_weight_total_stub(DeviceType::CPU, target.data_ptr(),
                                       wptr, rows, ignore_index,
                                       has_w ? 1 : 0, dtype);
        }
        Tensor grad_input = Tensor::zeros({rows, cols}, dt, dev);
        nll_grad_stub(DeviceType::CPU, grad_output.data_ptr(),
                      target.data_ptr(), wptr, grad_input.data_ptr(), rows,
                      cols, ignore_index, reduction, tw, dtype);
        if (no_batch) grad_input = grad_input.reshape({cols});
        return grad_input;
    }
    if (input.dtype() == DType::Float32) return nll_loss_backward_impl<float>(grad_output, input, target, weight, reduction, ignore_index, total_weight);
    if (input.dtype() == DType::Float64) return nll_loss_backward_impl<double>(grad_output, input, target, weight, reduction, ignore_index, total_weight);
    if (input.dtype() == DType::BFloat16) return nll_loss_backward_impl<BFloat16>(grad_output, input, target, weight, reduction, ignore_index, total_weight);
    if (input.dtype() == DType::Half) return nll_loss_backward_impl<Half>(grad_output, input, target, weight, reduction, ignore_index, total_weight);
    TP_THROW(NotImplementedError, "nll_loss backward only supports Float32, Float64, BFloat16 and Half");
}

// (N, C, H, W), target (N, H, W); every spatial position is an independent
// batch row, so the row layout is input[(n * C + t) * H * W + pos].
template <typename scalar_t>
std::tuple<Tensor, Tensor> nll_loss2d_impl(const Tensor& input, const Tensor& target,
                                           const std::optional<Tensor>& weight,
                                           int64_t reduction, int64_t ignore_index) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "nll_loss2d: Expected 4D input");
    if (target.dim() != 3) TP_THROW(RuntimeError, "nll_loss2d: Expected 3D target");
    const int64_t N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    if (target.size(0) != N || target.size(1) != H || target.size(2) != W)
        TP_THROW(RuntimeError, "nll_loss2d: target shape must match input spatial dims");
    const int64_t rows = N * H * W;

    const Tensor input_c = input.contiguous();
    const Tensor target_c = target.contiguous();
    const scalar_t* input_data = input_c.data_ptr<scalar_t>();
    const int64_t* target_data = target_c.data_ptr<int64_t>();
    const scalar_t* weight_data = weight.has_value() && weight->defined() ? weight->data_ptr<scalar_t>() : nullptr;

    std::vector<scalar_t> output_data(reduction == 0 ? rows : static_cast<int64_t>(0));
    constexpr int64_t kRowChunk = 256;
    const int64_t n_chunks = (rows + kRowChunk - 1) / kRowChunk;
    std::vector<double> chunk_sum(static_cast<size_t>(n_chunks), 0.0);
    std::vector<double> chunk_weight(static_cast<size_t>(n_chunks), 0.0);
    const int64_t hw = H * W;
    int64_t bad_target = -1;

    parallel::parallel_for(0, rows, kRowChunk, [&](int64_t b, int64_t e) {
        const size_t ci = static_cast<size_t>(b / kRowChunk);
        double local_sum = 0.0;
        double local_weight = 0.0;
        // The row index walks forward one pixel at a time, so the batch and the
        // in-image offset are carried in registers instead of divided per row.
        int64_t n = b / hw;
        int64_t pos = b % hw;
        for (int64_t i = b; i < e; ++i) {
            const int64_t t = target_data[i];
            if (t == ignore_index) {
                if (reduction == 0) output_data[static_cast<size_t>(i)] = 0;
            } else if (t < 0 || t >= C) {
                bad_target = t;
            } else {
                const double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
                const double loss =
                    -static_cast<double>(input_data[(n * C + t) * hw + pos]) * w;
                local_sum += loss;
                local_weight += w;
                if (reduction == 0) output_data[static_cast<size_t>(i)] = static_cast<scalar_t>(loss);
            }
            if (++pos == hw) {
                pos = 0;
                ++n;
            }
        }
        chunk_sum[ci] = local_sum;
        chunk_weight[ci] = local_weight;
    });
    if (bad_target >= 0) TP_THROW(RuntimeError, "Target out of bounds");

    double total_weight = 0;
    double sum = 0;
    for (int64_t c = 0; c < n_chunks; ++c) {
        sum += chunk_sum[static_cast<size_t>(c)];
        total_weight += chunk_weight[static_cast<size_t>(c)];
    }

    DType dt = input.dtype();
    if (reduction == 0) {
        // Unreduced losses carry no normalizer: total_weight stays zero.
        return std::make_tuple(Tensor::tensor(output_data, dt, input.device()).reshape({N, H, W}),
                               Tensor::full({}, Scalar(0.0), dt, input.device()));
    }
    if (reduction != 1 && reduction != 2) TP_THROW(ValueError, "Invalid reduction mode");
    // Every target ignored under mean reduction: 0 / 0 is NaN.
    if (reduction == 1) sum /= total_weight;
    return std::make_tuple(Tensor::full({}, Scalar(sum), dt, input.device()),
                           Tensor::full({}, Scalar(total_weight), dt, input.device()));
}

// (N, C, H, W) scores with a target per spatial position; the fused walk takes
// the batch axis as its parallel unit and folds the whole sample's positions in
// one sweep.
std::tuple<Tensor, Tensor> nll_loss2d_kernel(const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight, int64_t reduction, int64_t ignore_index) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "nll_loss2d: Expected 4D input");
    if (target.dim() != 3) TP_THROW(RuntimeError, "nll_loss2d: Expected 3D target");
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    const int64_t batch = input.size(0), classes = input.size(1);
    const int64_t h = input.size(2), w = input.size(3);
    if (target.size(0) != batch || target.size(1) != h || target.size(2) != w)
        TP_THROW(RuntimeError, "nll_loss2d: target shape must match input spatial dims");
    const bool has_w = weight.has_value() && weight->defined();
    const bool fused =
        (input.dtype() == DType::Float32 || input.dtype() == DType::Float64) &&
        input.is_contiguous() && target.is_contiguous() &&
        target.dtype() == DType::Int64 &&
        (!has_w || (weight->dtype() == input.dtype() &&
                    weight->is_contiguous()));
    if (fused) {
        const int64_t hw = h * w;
        const int64_t rows = batch * hw;
        const int64_t* tgt = target.data_ptr<int64_t>();
        for (int64_t i = 0; i < rows; ++i) {
            const int64_t t = tgt[i];
            if (t != ignore_index && (t < 0 || t >= classes))
                TP_THROW(IndexError, "Target ", t, " is out of bounds.");
        }
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        const void* wptr = has_w ? weight->data_ptr() : nullptr;
        const int has_weight = has_w ? 1 : 0;
        if (reduction == 0) {
            Tensor per_row = Tensor::empty({batch, h, w}, dt, dev);
            nll2d_rows_stub(DeviceType::CPU, input.data_ptr(),
                            target.data_ptr(), wptr, per_row.data_ptr(), batch,
                            classes, hw, ignore_index, has_weight, dtype);
            // Unreduced losses carry no normalizer: total_weight stays zero.
            return std::make_tuple(
                per_row, Tensor::full({}, Scalar(0.0), dt, dev));
        }
        const double sum = nll2d_rows_stub(
            DeviceType::CPU, input.data_ptr(), target.data_ptr(), wptr, nullptr,
            batch, classes, hw, ignore_index, has_weight, dtype);
        const double tw = nll_weight_total_stub(
            DeviceType::CPU, target.data_ptr(), wptr, rows, ignore_index,
            has_weight, dtype);
        double value = sum;
        if (reduction == 1) value /= tw;  // every target ignored: 0 / 0 is NaN
        return std::make_tuple(Tensor::full({}, Scalar(value), dt, dev),
                               Tensor::full({}, Scalar(tw), dt, dev));
    }
    if (input.dtype() == DType::Float32) return nll_loss2d_impl<float>(input, target, weight, reduction, ignore_index);
    if (input.dtype() == DType::Float64) return nll_loss2d_impl<double>(input, target, weight, reduction, ignore_index);
    if (input.dtype() == DType::BFloat16) return nll_loss2d_impl<BFloat16>(input, target, weight, reduction, ignore_index);
    if (input.dtype() == DType::Half) return nll_loss2d_impl<Half>(input, target, weight, reduction, ignore_index);
    TP_THROW(NotImplementedError, "nll_loss2d only supports Float32, Float64, BFloat16 and Half");
}

template <typename scalar_t>
Tensor nll_loss2d_backward_impl(const Tensor& grad_output, const Tensor& input, const Tensor& target,
                                const std::optional<Tensor>& weight, int64_t reduction,
                                int64_t ignore_index, const Tensor& total_weight) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "nll_loss2d_backward: Expected 4D input");
    if (target.dim() != 3) TP_THROW(RuntimeError, "nll_loss2d_backward: Expected 3D target");
    const int64_t N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    const int64_t rows = N * H * W;

    const Tensor target_c = target.contiguous();
    const int64_t* target_data = target_c.data_ptr<int64_t>();
    Tensor weight_c;
    if (weight.has_value() && weight->defined()) {
        weight_c = weight->contiguous();
    }
    const scalar_t* weight_data = weight_c.defined() ? weight_c.data_ptr<scalar_t>() : nullptr;

    Tensor grad_input = Tensor::zeros({N, C, H, W}, input.dtype(), input.device());
    scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();

    double tw = 0;
    if (total_weight.defined()) {
        tw = total_weight.item().to<double>();
    } else if (reduction == 1) {
        for (int64_t i = 0; i < rows; ++i) {
            const int64_t t = target_data[i];
            if (t != ignore_index) tw += weight_data ? static_cast<double>(weight_data[t]) : 1.0;
        }
    }

    const Tensor grad_out_c = reduction == 0 ? grad_output.contiguous() : Tensor();
    const scalar_t* grad_out_data = reduction == 0 ? grad_out_c.data_ptr<scalar_t>() : nullptr;
    const double grad_scalar = reduction == 0 ? 0.0 : grad_output.item().to<double>();

    for (int64_t i = 0; i < rows; ++i) {
        const int64_t t = target_data[i];
        if (t == ignore_index) continue;
        if (t < 0 || t >= C) TP_THROW(RuntimeError, "Target out of bounds");
        const int64_t n = i / (H * W);
        const int64_t pos = i % (H * W);
        const double w = weight_data ? static_cast<double>(weight_data[t]) : 1.0;
        double g = reduction == 0 ? static_cast<double>(grad_out_data[i]) : grad_scalar;
        if (reduction == 1 && tw > 0) g /= tw;
        grad_input_data[(n * C + t) * H * W + pos] = static_cast<scalar_t>(-w * g);
    }
    return grad_input;
}

Tensor nll_loss2d_backward_kernel(const Tensor& grad_output_arg, const Tensor& input, const Tensor& target, const std::optional<Tensor>& weight, int64_t reduction, int64_t ignore_index, const Tensor& total_weight) {
    if (input.dim() != 4) TP_THROW(RuntimeError, "nll_loss2d_backward: Expected 4D input");
    if (target.dim() != 3) TP_THROW(RuntimeError, "nll_loss2d_backward: Expected 3D target");
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    // The seed may arrive in a narrower float width than the score plane; the
    // result is written at the score width, so the seed is read at that width.
    const bool float_seed =
        grad_output_arg.dtype() == DType::Float32 ||
        grad_output_arg.dtype() == DType::Float64;
    const bool float_scores =
        input.dtype() == DType::Float32 || input.dtype() == DType::Float64;
    const Tensor grad_output =
        (float_seed && float_scores && grad_output_arg.dtype() != input.dtype())
            ? grad_output_arg.to(input.dtype())
            : grad_output_arg;
    const int64_t batch = input.size(0), classes = input.size(1);
    const int64_t h = input.size(2), w = input.size(3);
    const int64_t hw = h * w;
    const int64_t rows = batch * hw;
    const bool has_w = weight.has_value() && weight->defined();
    const bool fused =
        float_scores && input.is_contiguous() && target.is_contiguous() &&
        target.dtype() == DType::Int64 &&
        (!has_w || (weight->dtype() == input.dtype() &&
                    weight->is_contiguous())) &&
        grad_output.is_contiguous() && grad_output.dtype() == input.dtype() &&
        (reduction == 0 ? grad_output.numel() == rows
                        : grad_output.numel() == 1);
    if (fused) {
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        const void* wptr = has_w ? weight->data_ptr() : nullptr;
        const int has_weight = has_w ? 1 : 0;
        double tw = 0.0;
        if (total_weight.defined() && total_weight.numel() == 1) {
            tw = total_weight.item().to<double>();
        } else if (reduction == 1) {
            tw = nll_weight_total_stub(DeviceType::CPU, target.data_ptr(), wptr,
                                       rows, ignore_index, has_weight, dtype);
        }
        Tensor grad_input = Tensor::zeros({batch, classes, h, w}, dt, dev);
        nll2d_grad_stub(DeviceType::CPU, grad_output.data_ptr(),
                        target.data_ptr(), wptr, grad_input.data_ptr(), batch,
                        classes, hw, ignore_index, reduction, tw, dtype);
        return grad_input;
    }
    if (input.dtype() == DType::Float32) return nll_loss2d_backward_impl<float>(grad_output, input, target, weight, reduction, ignore_index, total_weight);
    if (input.dtype() == DType::Float64) return nll_loss2d_backward_impl<double>(grad_output, input, target, weight, reduction, ignore_index, total_weight);
    if (input.dtype() == DType::BFloat16) return nll_loss2d_backward_impl<BFloat16>(grad_output, input, target, weight, reduction, ignore_index, total_weight);
    if (input.dtype() == DType::Half) return nll_loss2d_backward_impl<Half>(grad_output, input, target, weight, reduction, ignore_index, total_weight);
    TP_THROW(NotImplementedError, "nll_loss2d backward only supports Float32, Float64, BFloat16 and Half");
}

// MSE Loss
// The squared deviation is read once per operand and folded in the same walk:
// no deviation plane is materialized, and the mean shares the fold by the
// element count.
Tensor mse_loss_kernel(const Tensor& input, const Tensor& target, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    if (fused_float(input, target)) {
        const int64_t n = input.numel();
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        if (reduction == 0) {
            Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt, dev);
            mse_into_stub(DeviceType::CPU, input.data_ptr(), target.data_ptr(),
                          out.data_ptr(), n, dtype);
            return out;
        }
        const double total = mse_sum_stub(DeviceType::CPU, input.data_ptr(),
                                          target.data_ptr(), n, dtype);
        if (reduction == 1)
            return Tensor::full({}, Scalar(total / static_cast<double>(n)), dt, dev);
        return Tensor::full({}, Scalar(total), dt, dev);
    }
    Tensor diff = input - target;
    Tensor sq_diff = diff * diff;
    if (reduction == 0) return sq_diff;
    if (reduction == 1) return sq_diff.mean();
    return sq_diff.sum();
}

// dL/dx = 2 (x - t) * g, shared by the element count when the forward averaged.
Tensor mse_loss_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    const int64_t n = input.numel();
    const bool fused =
        fused_float(input, target) && grad_output.dtype() == input.dtype() &&
        same_layout(grad_output, input) && grad_output.numel() == n;
    if (fused) {
        const DType dt = input.dtype();
        const int dtype = static_cast<int>(dt);
        const double scale = reduction == 1
                                 ? 2.0 / static_cast<double>(n)
                                 : 2.0;
        Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt,
                   input.device());
        mse_grad_stub(DeviceType::CPU, grad_output.data_ptr(),
                      input.data_ptr(), target.data_ptr(), out.data_ptr(), n,
                      scale, dtype);
        return out;
    }
    Tensor diff = input - target;
    if (reduction == 0) return 2.0 * diff * grad_output;
    const double scale = reduction == 1
                             ? 2.0 / static_cast<double>(n)
                             : 2.0;
    return (scale * diff) * grad_output;
}



// following the mse_loss_kernel house style.
// reduction: 0=none, 1=mean, 2=sum
// -----------------------------------------------------------------------------

namespace {

Tensor loss_reduce(const Tensor& x, int64_t reduction) {
    if (reduction == 0) return x;
    if (reduction == 1) return x.mean();
    if (reduction == 2) return x.sum();
    TP_THROW(ValueError, "Invalid reduction mode");
}

Tensor scale_grad(const Tensor& g, int64_t reduction, int64_t numel) {
    if (reduction == 1) return g / static_cast<double>(numel);
    return g;
}

// Local stand-ins until clamp_min/xlogy/zeros_like land as real Tensor ops.
Tensor zeros_like_shim(const Tensor& t) { return t * 0; }

Tensor clamp_min_shim(const Tensor& x, Scalar s) {
    return Tensor::where(x.lt(s), s, x);
}

Tensor xlogy_shim(const Tensor& x, const Tensor& y) {
    return Tensor::where(x.eq(Scalar(0)), x * 0, x * y.log());
}

} // anonymous namespace

// The absolute deviation read the same way: one load of each operand per lane,
// folded by the walk, and the mean shares the fold by the element count.
Tensor l1_loss_kernel(const Tensor& input, const Tensor& target, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    if (fused_float(input, target)) {
        const int64_t n = input.numel();
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        if (reduction == 0) {
            Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt, dev);
            l1_into_stub(DeviceType::CPU, input.data_ptr(), target.data_ptr(),
                         out.data_ptr(), n, dtype);
            return out;
        }
        const double total = l1_sum_stub(DeviceType::CPU, input.data_ptr(),
                                         target.data_ptr(), n, dtype);
        if (reduction == 1)
            return Tensor::full({}, Scalar(total / static_cast<double>(n)), dt, dev);
        return Tensor::full({}, Scalar(total), dt, dev);
    }
    return loss_reduce((input - target).abs(), reduction);
}

// The side the deviation sits on, carried by the incoming gradient and shared
// by the element count when the forward averaged.
Tensor l1_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                               const Tensor& target, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    const int64_t n = input.numel();
    const bool fused =
        fused_float(input, target) && grad_output.dtype() == input.dtype() &&
        same_layout(grad_output, input) && grad_output.numel() == n;
    if (fused) {
        const DType dt = input.dtype();
        const int dtype = static_cast<int>(dt);
        const double scale =
            reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
        Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt,
                   input.device());
        l1_grad_stub(DeviceType::CPU, grad_output.data_ptr(), input.data_ptr(),
                     target.data_ptr(), out.data_ptr(), n, scale, dtype);
        return out;
    }
    Tensor g = (input - target).sign() * grad_output;
    return scale_grad(g, reduction, n);
}

// z = |x - t|; z < beta ? 0.5 z^2 / beta : z - 0.5 beta. The switch is strict,
// so the quadratic branch holds no lane when the width is nothing and every
// lane answers the deviation itself.  The fused walk reads both operands once
// and never builds the deviation plane.
Tensor smooth_l1_loss_cpu(const Tensor& input, const Tensor& target,
                          int64_t reduction, double beta) {
    if (beta < 0)
        TP_THROW(ValueError, "smooth_l1_loss does not support negative values for beta.");
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    if (fused_float(input, target)) {
        const int64_t n = input.numel();
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        if (reduction == 0) {
            Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt, dev);
            smooth_l1_into_stub(DeviceType::CPU, input.data_ptr(),
                                target.data_ptr(), out.data_ptr(), n, beta,
                                dtype);
            return out;
        }
        const double total = smooth_l1_sum_stub(DeviceType::CPU,
                                                input.data_ptr(),
                                                target.data_ptr(), n, beta,
                                                dtype);
        if (reduction == 1)
            return Tensor::full({}, Scalar(total / static_cast<double>(n)), dt, dev);
        return Tensor::full({}, Scalar(total), dt, dev);
    }
    Tensor diff = input - target;
    Tensor absd = diff.abs();
    Tensor loss = Tensor::where(absd.lt(Scalar(beta)), diff * diff * (0.5 / beta),
                                absd - 0.5 * beta);
    return loss_reduce(loss, reduction);
}

// x = input - target, with norm = 1/numel for mean reduction. The slope is a
// three-way test on the signed deviation, and the two outer tests are reached
// before the division, so a lane sitting on the width answers the full slope
// and a width of nothing never divides.
Tensor smooth_l1_loss_backward_cpu(const Tensor& grad_output, const Tensor& input,
                                   const Tensor& target, int64_t reduction, double beta) {
    if (beta < 0)
        TP_THROW(ValueError, "smooth_l1_loss does not support negative values for beta.");
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    const int64_t n = input.numel();
    const bool fused =
        fused_float(input, target) && grad_output.dtype() == input.dtype() &&
        same_layout(grad_output, input) && grad_output.numel() == n;
    if (fused) {
        const DType dt = input.dtype();
        const int dtype = static_cast<int>(dt);
        const double scale =
            reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
        Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt,
                   input.device());
        smooth_l1_grad_stub(DeviceType::CPU, grad_output.data_ptr(),
                            input.data_ptr(), target.data_ptr(), out.data_ptr(),
                            n, beta, scale, dtype);
        return out;
    }
    Tensor diff = input - target;
    Tensor slope = Tensor::where(
        diff.le(Scalar(-beta)), Scalar(-1.0),
        Tensor::where(diff.ge(Scalar(beta)), Scalar(1.0), diff / beta));
    return scale_grad(slope * grad_output, reduction, input.numel());
}

Tensor huber_loss_cpu(const Tensor& input, const Tensor& target,
                      int64_t reduction, double delta) {
    if (delta <= 0)
        TP_THROW(ValueError, "huber_loss does not support non-positive values for delta.");
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    if (fused_float(input, target) && (reduction == 0 || reduction == 1 ||
                                       reduction == 2)) {
        const int64_t n = input.numel();
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        if (reduction == 0) {
            Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt, dev);
            huber_into_stub(DeviceType::CPU, input.data_ptr(), target.data_ptr(),
                            out.data_ptr(), n, delta, dtype);
            return out;
        }
        const double total = huber_sum_stub(DeviceType::CPU, input.data_ptr(),
                                            target.data_ptr(), n, delta, dtype);
        if (reduction == 1)
            return Tensor::full({}, Scalar(total / static_cast<double>(n)), dt, dev);
        return Tensor::full({}, Scalar(total), dt, dev);
    }
    Tensor diff = input - target;
    Tensor absd = diff.abs();
    Tensor loss = Tensor::where(absd.le(Scalar(delta)), diff * diff * 0.5,
                                delta * (absd - 0.5 * delta));
    return loss_reduce(loss, reduction);
}

// x = input - target; |x| <= delta ? norm * x : norm * delta * sign(x).
Tensor huber_loss_backward_cpu(const Tensor& grad_output, const Tensor& input,
                               const Tensor& target, int64_t reduction, double delta) {
    Tensor diff = input - target;
    Tensor g = Tensor::where(diff.abs().le(Scalar(delta)), diff, delta * diff.sign()) * grad_output;
    return scale_grad(g, reduction, input.numel());
}

// t log t - t x (or exp(t) (t - x) when the target sits in log space), read
// once per operand and folded in the same walk; a zero target contributes
// nothing.  The mean shares the fold by the element count.
Tensor kl_div_kernel(const Tensor& input, const Tensor& target,
                     int64_t reduction, bool log_target) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    if (fused_float(input, target)) {
        const int64_t n = input.numel();
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        const int lt = log_target ? 1 : 0;
        if (reduction == 0) {
            Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt, dev);
            kl_into_stub(DeviceType::CPU, input.data_ptr(), target.data_ptr(),
                         out.data_ptr(), n, lt, dtype);
            return out;
        }
        const double total = kl_sum_stub(DeviceType::CPU, input.data_ptr(),
                                         target.data_ptr(), n, lt, dtype);
        if (reduction == 1)
            return Tensor::full({}, Scalar(total / static_cast<double>(n)), dt, dev);
        return Tensor::full({}, Scalar(total), dt, dev);
    }
    Tensor loss;
    if (!log_target) {
        Tensor nz = target.ne(0).to(input.dtype());
        loss = (xlogy_shim(target, target) - target * input) * nz;
    } else {
        Tensor t = target.exp();
        loss = t * (target - input);
    }
    return loss_reduce(loss, reduction);
}

// d/dinput is the negated target (probability space) or its negated exponent
// (log space), carried by the incoming gradient and shared by the count a mean
// divided with.
Tensor kl_div_backward_kernel(const Tensor& grad_output, const Tensor& input,
                              const Tensor& target, int64_t reduction, bool log_target) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    const int64_t n = input.numel();
    const bool fused =
        fused_float(input, target) && grad_output.dtype() == input.dtype() &&
        same_layout(grad_output, input) && grad_output.numel() == n;
    if (fused) {
        const DType dt = input.dtype();
        const int dtype = static_cast<int>(dt);
        const double scale =
            reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
        Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt,
                   input.device());
        kl_grad_stub(DeviceType::CPU, grad_output.data_ptr(), input.data_ptr(),
                     target.data_ptr(), out.data_ptr(), n,
                     log_target ? 1 : 0, scale, dtype);
        return out;
    }
    Tensor g;
    if (!log_target) {
        Tensor nz = target.ne(0).to(input.dtype());
        g = (-target * nz) * grad_output;   // d/dinput of -target*input
    } else {
        g = -(target.exp()) * grad_output;
    }
    return scale_grad(g, reduction, n);
}

// per-element loss is (t-1)*max(log(1-x),-100) - t*max(log(x),-100), then
// optional weight multiply and reduction.
void bce_check_01(const Tensor& t, const char* what) {
    if (t.numel() == 0) return;
    // Counting only the lanes strictly below or above the bounds lets a lane
    // that is not a number through, since it answers neither comparison.
    // Refusing on the pair of comparisons instead keeps the two bounds and a
    // NaN in the same gate.
    Tensor inside = Tensor::logical_and(t.ge(0.0), t.le(1.0)).all();
    if (!inside.item<bool>())
        TP_THROW(RuntimeError, std::string("all elements of ") + what +
                 " should be between 0 and 1");
}

Tensor binary_cross_entropy_cpu(const Tensor& input, const Tensor& target,
                                const std::optional<Tensor>& weight, int64_t reduction) {
    const bool has_weight = weight.has_value() && weight->defined();
    if (!has_weight && fused_float(input, target) &&
        (reduction == 0 || reduction == 1 || reduction == 2)) {
        int bad_input = 0;
        int bad_target = 0;
        const int64_t n = input.numel();
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        Tensor out;
        double total = 0.0;
        if (reduction == 0) {
            out = Tensor(static_cast<std::vector<int64_t>>(input.sizes()), dt, dev);
            bce_into_stub(DeviceType::CPU, input.data_ptr(), target.data_ptr(),
                          out.data_ptr(), n, &bad_input, &bad_target, dtype);
        } else {
            total = bce_sum_stub(DeviceType::CPU, input.data_ptr(),
                                 target.data_ptr(), n, &bad_input, &bad_target,
                                 dtype);
        }
        if (bad_input)
            TP_THROW(RuntimeError,
                     "all elements of input should be between 0 and 1");
        if (bad_target)
            TP_THROW(RuntimeError,
                     "all elements of target should be between 0 and 1");
        if (reduction == 0) return out;
        if (reduction == 1)
            return Tensor::full({}, Scalar(total / static_cast<double>(n)), dt, dev);
        return Tensor::full({}, Scalar(total), dt, dev);
    }
    bce_check_01(input, "input");
    bce_check_01(target, "target");
    Tensor log_x = clamp_min_shim(input.log(), Scalar(-100.0));
    Tensor log_1mx = clamp_min_shim((-input + 1.0).log(), Scalar(-100.0));
    Tensor loss = (target - 1.0) * log_1mx - target * log_x;
    if (has_weight) loss = loss * weight.value();
    return loss_reduce(loss, reduction);
}

// grad * (x - t) / max((1 - x) * x, 1e-12), then weight multiply, and
// division by input.numel() for mean reduction.
Tensor binary_cross_entropy_backward_cpu(const Tensor& grad_output, const Tensor& input,
                                         const Tensor& target,
                                         const std::optional<Tensor>& weight, int64_t reduction) {
    Tensor denom = clamp_min_shim(input * (-input + 1.0), Scalar(1e-12));
    Tensor g = grad_output * (input - target) / denom;
    if (weight.has_value() && weight->defined()) g = g * weight.value();
    return scale_grad(g, reduction, input.numel());
}

// max(0, margin - (x1 - x2) t), read once per operand and folded in the same
// walk; the mean shares the fold by the element count.
Tensor margin_ranking_loss_kernel(const Tensor& input1, const Tensor& input2,
                                  const Tensor& target, double margin, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    if (fused_float(input1, input2) && target.dtype() == input1.dtype() &&
        same_layout(input2, target)) {
        const int64_t n = input1.numel();
        const DType dt = input1.dtype();
        const Device& dev = input1.device();
        const int dtype = static_cast<int>(dt);
        if (reduction == 0) {
            Tensor out(static_cast<std::vector<int64_t>>(input1.sizes()), dt, dev);
            margin_rank_into_stub(DeviceType::CPU, input1.data_ptr(),
                                  input2.data_ptr(), target.data_ptr(),
                                  out.data_ptr(), n, margin, dtype);
            return out;
        }
        const double total = margin_rank_sum_stub(DeviceType::CPU,
                                                  input1.data_ptr(),
                                                  input2.data_ptr(),
                                                  target.data_ptr(), n,
                                                  margin, dtype);
        if (reduction == 1)
            return Tensor::full({}, Scalar(total / static_cast<double>(n)), dt, dev);
        return Tensor::full({}, Scalar(total), dt, dev);
    }
    Tensor loss = clamp_min_shim(-(input1 - input2) * target + margin, Scalar(0.0));
    return loss_reduce(loss, reduction);
}

// The active mask is whether the margin expression stays positive; the first
// operand receives -(active t) times the incoming gradient and the second its
// negation, shared by the element count when the forward averaged.
std::tuple<Tensor, Tensor> margin_ranking_loss_backward_kernel(
        const Tensor& grad_output, const Tensor& input1, const Tensor& input2,
        const Tensor& target, double margin, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    const int64_t n = input1.numel();
    const bool fused =
        fused_float(input1, input2) && target.dtype() == input1.dtype() &&
        same_layout(input2, target) && grad_output.dtype() == input1.dtype() &&
        same_layout(grad_output, input1) && grad_output.numel() == n;
    if (fused) {
        const DType dt = input1.dtype();
        const int dtype = static_cast<int>(dt);
        const double scale =
            reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
        Tensor out1(static_cast<std::vector<int64_t>>(input1.sizes()), dt,
                    input1.device());
        Tensor out2(static_cast<std::vector<int64_t>>(input2.sizes()), dt,
                    input2.device());
        margin_rank_grad_stub(DeviceType::CPU, grad_output.data_ptr(),
                              input1.data_ptr(), input2.data_ptr(),
                              target.data_ptr(), out1.data_ptr(),
                              out2.data_ptr(), n, margin, scale, dtype);
        return std::make_tuple(out1, out2);
    }
    Tensor active = ((-(input1 - input2) * target + margin).gt(0.0)).to(input1.dtype());
    Tensor g = -active * target * grad_output;
    g = scale_grad(g, reduction, n);
    return std::make_tuple(g, -g);
}

// x when the target lane is 1, max(0, margin - x) when it is -1, and their
// sum for any other lane; read once per operand and folded in the same walk.
Tensor hinge_embedding_loss_kernel(const Tensor& input, const Tensor& target,
                                   double margin, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    if (fused_float(input, target)) {
        const int64_t n = input.numel();
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        if (reduction == 0) {
            Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt, dev);
            hinge_into_stub(DeviceType::CPU, input.data_ptr(),
                            target.data_ptr(), out.data_ptr(), n, margin,
                            dtype);
            return out;
        }
        const double total = hinge_sum_stub(DeviceType::CPU, input.data_ptr(),
                                            target.data_ptr(), n, margin,
                                            dtype);
        if (reduction == 1)
            return Tensor::full({}, Scalar(total / static_cast<double>(n)), dt, dev);
        return Tensor::full({}, Scalar(total), dt, dev);
    }
    //                        + where(t != -1, x, 0)
    Tensor z = zeros_like_shim(input);
    Tensor margin_part = Tensor::where(target.ne(1),
                                       clamp_min_shim(margin - input, Scalar(0.0)), z);
    Tensor self_part = Tensor::where(target.ne(-1), input, z);
    return loss_reduce(margin_part + self_part, reduction);
}

//   y == 1          -> 1
//   y == -1         -> (margin - x > 0) ? -1 : 0
//   otherwise       -> 1 + ((margin - x > 0) ? -1 : 0)
// The slope is folded into the fused walk with the reduction's scale.
Tensor hinge_embedding_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                            const Tensor& target, double margin, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    const int64_t n = input.numel();
    const bool fused =
        fused_float(input, target) && grad_output.dtype() == input.dtype() &&
        same_layout(grad_output, input) && grad_output.numel() == n;
    if (fused) {
        const DType dt = input.dtype();
        const int dtype = static_cast<int>(dt);
        const double scale =
            reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
        Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt,
                   input.device());
        hinge_grad_stub(DeviceType::CPU, grad_output.data_ptr(),
                        input.data_ptr(), target.data_ptr(), out.data_ptr(),
                        n, margin, scale, dtype);
        return out;
    }
    Tensor ones = Tensor::ones_like(input);
    Tensor active = ((margin - input).gt(Scalar(0.0))).to(input.dtype());
    Tensor g = Tensor::where(target.eq(1), ones,
               Tensor::where(target.eq(-1), -active, ones - active));
    g = g * grad_output;
    return scale_grad(g, reduction, n);
}

// log(1 + exp(-target*input)), read once per operand and folded in the same
// walk; the mean shares the fold by the element count.
Tensor soft_margin_loss_kernel(const Tensor& input, const Tensor& target, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    if (fused_float(input, target)) {
        const int64_t n = input.numel();
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        if (reduction == 0) {
            Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt, dev);
            softmargin_into_stub(DeviceType::CPU, input.data_ptr(),
                                 target.data_ptr(), out.data_ptr(), n, dtype);
            return out;
        }
        const double total = softmargin_sum_stub(DeviceType::CPU,
                                                 input.data_ptr(),
                                                 target.data_ptr(), n, dtype);
        if (reduction == 1)
            return Tensor::full({}, Scalar(total / static_cast<double>(n)), dt, dev);
        return Tensor::full({}, Scalar(total), dt, dev);
    }
    Tensor loss = ((input * target) * -1.0).exp().add(1.0).log();
    return loss_reduce(loss, reduction);
}

// d/dx log(1 + exp(-t x)) = -t * sigmoid(-t x), carried by the incoming
// gradient and shared by the element count when the forward averaged.
Tensor soft_margin_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                        const Tensor& target, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    const int64_t n = input.numel();
    const bool fused =
        fused_float(input, target) && grad_output.dtype() == input.dtype() &&
        same_layout(grad_output, input) && grad_output.numel() == n;
    if (fused) {
        const DType dt = input.dtype();
        const int dtype = static_cast<int>(dt);
        const double scale =
            reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
        Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt,
                   input.device());
        softmargin_grad_stub(DeviceType::CPU, grad_output.data_ptr(),
                             input.data_ptr(), target.data_ptr(),
                             out.data_ptr(), n, scale, dtype);
        return out;
    }
    Tensor g = -target * ((input * target) * -1.0).sigmoid() * grad_output;
    return scale_grad(g, reduction, n);
}

// One row's cosine is accumulated in a single walk over the row: the squared
// norms and the dot product share their loads, and the row answer is the
// target's branch on the cosine.
Tensor cosine_embedding_loss_kernel(const Tensor& x1, const Tensor& x2,
                                    const Tensor& target, double margin, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    const bool fused =
        fused_float(x1, x2) && target.dtype() == x1.dtype() &&
        target.numel() == x1.size(0) && target.is_contiguous();
    if (fused) {
        const int64_t rows = x1.size(0);
        const int64_t width = x1.size(1);
        const DType dt = x1.dtype();
        const Device& dev = x1.device();
        const int dtype = static_cast<int>(dt);
        if (reduction == 0) {
            Tensor out({rows}, dt, dev);
            cosine_rows_stub(DeviceType::CPU, x1.data_ptr(), x2.data_ptr(),
                             target.data_ptr(), out.data_ptr(), rows, width,
                             margin, dtype);
            return out;
        }
        const double total = cosine_rows_stub(DeviceType::CPU, x1.data_ptr(),
                                              x2.data_ptr(), target.data_ptr(),
                                              nullptr, rows, width, margin,
                                              dtype);
        if (reduction == 1)
            return Tensor::full({}, Scalar(total / static_cast<double>(rows)), dt, dev);
        return Tensor::full({}, Scalar(total), dt, dev);
    }
    Tensor n1 = x1.pow(2).sum(std::vector<int64_t>{1});
    Tensor n2 = x2.pow(2).sum(std::vector<int64_t>{1});
    Tensor d = (n1 * n2).sqrt();
    Tensor cos = (x1 * x2).sum(std::vector<int64_t>{1}) / d;
    Tensor zero = zeros_like_shim(cos);
    //   t ==  1 -> 1 - cos
    //   t == -1 -> max(0, cos - margin)
    //   else    -> 0
    Tensor loss = Tensor::where(target.eq(1), zero + (1.0 - cos),
                   Tensor::where(target.eq(-1), clamp_min_shim(cos - margin, Scalar(0.0)),
                                 zero));
    return loss_reduce(loss, reduction);
}

// The row's loss slope times the incoming gradient scales both input planes;
// the fused walk folds the mean's 1/rows into the same row slope.
std::tuple<Tensor, Tensor> cosine_embedding_loss_backward_kernel(
        const Tensor& grad_output, const Tensor& x1, const Tensor& x2,
        const Tensor& target, double margin, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    const int64_t rows = x1.size(0);
    const bool fused =
        fused_float(x1, x2) && target.dtype() == x1.dtype() &&
        target.numel() == rows && target.is_contiguous() &&
        grad_output.dtype() == x1.dtype() && grad_output.numel() == rows &&
        grad_output.is_contiguous();
    if (fused) {
        const int64_t width = x1.size(1);
        const DType dt = x1.dtype();
        const int dtype = static_cast<int>(dt);
        const double scale =
            reduction == 1 ? 1.0 / static_cast<double>(rows) : 1.0;
        Tensor out1(static_cast<std::vector<int64_t>>(x1.sizes()), dt,
                    x1.device());
        Tensor out2(static_cast<std::vector<int64_t>>(x2.sizes()), dt,
                    x2.device());
        cosine_grad_stub(DeviceType::CPU, grad_output.data_ptr(), x1.data_ptr(),
                         x2.data_ptr(), target.data_ptr(), out1.data_ptr(),
                         out2.data_ptr(), rows, width, margin, scale, dtype);
        return std::make_tuple(out1, out2);
    }
    Tensor n1 = x1.pow(2).sum(std::vector<int64_t>{1});
    Tensor n2 = x2.pow(2).sum(std::vector<int64_t>{1});
    Tensor d = clamp_min_shim((n1 * n2).sqrt(), Scalar(1e-12));
    Tensor cos = (x1 * x2).sum(std::vector<int64_t>{1}) / d;

    Tensor ones_row = Tensor::ones({x1.size(0)}, x1.dtype(), x1.device());
    Tensor dl_dcos = Tensor::where(target.eq(1), -1.0 * ones_row,
                     Tensor::where(target.eq(-1),
                         ((cos - margin).gt(0.0)).to(x1.dtype()),
                         zeros_like_shim(ones_row)));

    if (reduction == 1) dl_dcos = dl_dcos / static_cast<double>(x1.size(0));

    Tensor c = cos.unsqueeze(1);
    Tensor g1 = (x2 / d.unsqueeze(1)) - c * (x1 / n1.unsqueeze(1));
    Tensor g2 = (x1 / d.unsqueeze(1)) - c * (x2 / n2.unsqueeze(1));
    g1 = g1 * (dl_dcos * grad_output).unsqueeze(1);
    g2 = g2 * (dl_dcos * grad_output).unsqueeze(1);
    return std::make_tuple(g1, g2);
}

// exp(x) - t x (log_input) or x - t log(x + eps), plus the Stirling term when
// full; both operand spaces are read once and folded in the same walk.
Tensor poisson_nll_loss_kernel(const Tensor& input, const Tensor& target,
                               bool log_input, bool full, double eps, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    if (fused_float(input, target)) {
        const int64_t n = input.numel();
        const DType dt = input.dtype();
        const Device& dev = input.device();
        const int dtype = static_cast<int>(dt);
        if (reduction == 0) {
            Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt, dev);
            poisson_into_stub(DeviceType::CPU, input.data_ptr(),
                              target.data_ptr(), out.data_ptr(), n,
                              log_input ? 1 : 0, full ? 1 : 0, eps, dtype);
            return out;
        }
        const double total = poisson_sum_stub(DeviceType::CPU,
                                              input.data_ptr(),
                                              target.data_ptr(), n,
                                              log_input ? 1 : 0,
                                              full ? 1 : 0, eps, dtype);
        if (reduction == 1)
            return Tensor::full({}, Scalar(total / static_cast<double>(n)), dt, dev);
        return Tensor::full({}, Scalar(total), dt, dev);
    }
    Tensor loss;
    if (log_input) {
        loss = input.exp() - target * input;
    } else {
        loss = input - target * (input + eps).log();
    }
    if (full) {
        // Stirling approximation term: t*log(t) - t + 0.5*log(2*pi*t), for t>0
        // The selection takes the truth values; the product takes them as
        // numbers of the loss's type.
        Tensor pos = target.gt(1);
        Tensor t_safe = Tensor::where(pos, target, Tensor::ones_like(target));
        Tensor stirling = (xlogy_shim(target, t_safe) - target +
                           (t_safe * (2.0 * M_PI)).log() * 0.5) * pos.to(input.dtype());
        loss = loss + stirling;
    }
    return loss_reduce(loss, reduction);
}

// d/dx is exp(x) - t (log_input) or 1 - t / (x + eps); the Stirling term does
// not depend on x, so `full` never reaches the gradient.
Tensor poisson_nll_loss_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                        const Tensor& target, bool log_input, bool full,
                                        double eps, int64_t reduction) {
    if (reduction != 0 && reduction != 1 && reduction != 2)
        TP_THROW(ValueError, "Invalid reduction mode");
    (void)full;
    const int64_t n = input.numel();
    const bool fused =
        fused_float(input, target) && grad_output.dtype() == input.dtype() &&
        same_layout(grad_output, input) && grad_output.numel() == n;
    if (fused) {
        const DType dt = input.dtype();
        const int dtype = static_cast<int>(dt);
        const double scale =
            reduction == 1 ? 1.0 / static_cast<double>(n) : 1.0;
        Tensor out(static_cast<std::vector<int64_t>>(input.sizes()), dt,
                   input.device());
        poisson_grad_stub(DeviceType::CPU, grad_output.data_ptr(),
                          input.data_ptr(), target.data_ptr(), out.data_ptr(),
                          n, log_input ? 1 : 0, eps, scale, dtype);
        return out;
    }
    Tensor g;
    if (log_input) {
        g = input.exp() - target;
    } else {
        g = 1.0 - target / (input + eps);
    }
    g = g * grad_output;
    return scale_grad(g, reduction, n);
}

// ---------------------------------------------------------------------------
//
// `_ctc_loss` returns the RAW per-sequence negative log likelihood (entries
// stay +inf for impossible alignments; `zero_infinity` is honored only by
// with the log-alpha table (N, T, 2S+1) consumed by the backward.
// Targets use the batch-first zero-padded layout (N, S); lengths are (N,).

namespace {

inline int64_t ctc_target_prime(const int64_t* targets, int64_t stride,
                                int64_t idx, int64_t blank) {
    if (idx % 2 == 0) return blank;
    return targets[(idx / 2) * stride];
}

template <typename scalar_t>
std::tuple<Tensor, Tensor> ctc_loss_impl(const Tensor& log_probs,
                                         const Tensor& targets,
                                         const std::vector<int64_t>& input_lengths,
                                         const std::vector<int64_t>& target_lengths,
                                         int64_t blank) {
    constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
    int64_t T = log_probs.size(0);
    int64_t N = log_probs.size(1);
    int64_t C = log_probs.size(2);
    int64_t S = targets.size(1);
    int64_t M = 2 * S + 1;

    Tensor neg_log_likelihood = Tensor::empty({N}, log_probs.dtype(), log_probs.device());
    Tensor log_alpha = Tensor::empty({N, T, M}, log_probs.dtype(), log_probs.device());
    const scalar_t* lpp = log_probs.data_ptr<scalar_t>();
    scalar_t* lap = log_alpha.data_ptr<scalar_t>();
    scalar_t* nllp = neg_log_likelihood.data_ptr<scalar_t>();
    const int64_t* tgt = targets.data_ptr<int64_t>();

    for (int64_t b = 0; b < N; ++b) {
        int64_t il = input_lengths[b];
        int64_t tl = target_lengths[b];
        if (il < 0 || il > T)
            TP_THROW(RuntimeError, "_ctc_loss: input_lengths out of range");
        if (tl < 0 || tl > S)
            TP_THROW(RuntimeError, "_ctc_loss: target_lengths out of range");
        const int64_t* tb = tgt + b * S;
        scalar_t* lab = lap + b * T * M;

        if (il == 0) {
            scalar_t log_likelihood = (tl == 0) ? static_cast<scalar_t>(0) : neginf;
            nllp[b] = static_cast<scalar_t>(-log_likelihood);
            for (int64_t t = 0; t < T; ++t)
                for (int64_t s = 0; s < M; ++s) lab[t * M + s] = neginf;
            continue;
        }

        // first row of eq (6): everything starts from -inf
        for (int64_t s = 0; s < M; ++s) lab[s] = neginf;
        const scalar_t* lp0 = lpp + (0 * N + b) * C;
        lab[0] = lp0[blank];
        if (tl > 0)
            lab[1] = lp0[ctc_target_prime(tb, 1, 1, blank)];

        // alpha recursion, eqs (6)/(7)
        for (int64_t t = 1; t < il; ++t) {
            const scalar_t* lpt = lpp + (t * N + b) * C;
            const scalar_t* prev = lab + (t - 1) * M;
            scalar_t* cur = lab + t * M;
            for (int64_t s = 0; s < M; ++s) cur[s] = neginf;
            for (int64_t s = 0; s <= 2 * tl && s < M; ++s) {
                const int64_t current_target_prime =
                    ctc_target_prime(tb, 1, s, blank);
                scalar_t la1 = prev[s];
                scalar_t lamax = la1;
                scalar_t la2 = neginf, la3 = neginf;
                if (s > 0) {
                    la2 = prev[s - 1];
                    if (la2 > lamax) lamax = la2;
                }
                if (s > 1 && ctc_target_prime(tb, 1, s - 2, blank) !=
                                 ctc_target_prime(tb, 1, s, blank)) {
                    la3 = prev[s - 2];
                    if (la3 > lamax) lamax = la3;
                }
                if (lamax == neginf) lamax = 0;  // cannot do neginf - neginf
                cur[s] = std::log(std::exp(la1 - lamax) + std::exp(la2 - lamax) +
                                  std::exp(la3 - lamax)) +
                         lamax + lpt[current_target_prime];
            }
        }
        // deterministic -inf for frames beyond each sample's input length
        for (int64_t t = il; t < T; ++t)
            for (int64_t s = 0; s < M; ++s) lab[t * M + s] = neginf;

        // likelihood: sum of the last two alphas, eq (8)
        if (tl == 0) {
            nllp[b] = -lab[(il - 1) * M];
        } else {
            scalar_t l1 = lab[(il - 1) * M + tl * 2];
            scalar_t l2 = lab[(il - 1) * M + tl * 2 - 1];
            scalar_t m = std::max(l1, l2);
            m = (m == neginf) ? static_cast<scalar_t>(0) : m;
            scalar_t log_likelihood =
                std::log(std::exp(l1 - m) + std::exp(l2 - m)) + m;
            nllp[b] = -log_likelihood;
        }
    }
    return std::make_tuple(neg_log_likelihood, log_alpha);
}

template <typename scalar_t>
Tensor ctc_loss_backward_impl(const Tensor& grad_out, const Tensor& log_probs,
                              const Tensor& targets,
                              const std::vector<int64_t>& input_lengths,
                              const std::vector<int64_t>& target_lengths,
                              const Tensor& neg_log_likelihood,
                              const Tensor& log_alpha, int64_t blank,
                              bool zero_infinity) {
    constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
    int64_t T = log_probs.size(0);
    int64_t N = log_probs.size(1);
    int64_t C = log_probs.size(2);
    int64_t S = targets.size(1);
    int64_t M = 2 * S + 1;

    if (grad_out.numel() != 1 && grad_out.numel() != N)
        TP_THROW(RuntimeError,
                 "_ctc_loss_backward: grad_output must hold one entry per sequence");

    // eq (16) collection detects untouched entries via that sentinel.
    Tensor grad = Tensor::empty_like(log_probs);
    scalar_t* gp_init = grad.data_ptr<scalar_t>();
    std::fill_n(gp_init, static_cast<size_t>(T * N * C), neginf);
    const scalar_t* lpp = log_probs.data_ptr<scalar_t>();
    const scalar_t* lap = log_alpha.data_ptr<scalar_t>();
    const scalar_t* nllp = neg_log_likelihood.data_ptr<scalar_t>();
    const int64_t* tgt = targets.data_ptr<int64_t>();
    const scalar_t* gop = grad_out.data_ptr<scalar_t>();
    scalar_t* gp = grad.data_ptr<scalar_t>();
    std::vector<scalar_t> beta(M);

    for (int64_t b = 0; b < N; ++b) {
        if (zero_infinity && nllp[b] == std::numeric_limits<scalar_t>::infinity()) {
            // zeroed loss: zero this batch item's (strided) grad column
            for (int64_t t = 0; t < T; ++t)
                for (int64_t c = 0; c < C; ++c) gp[(t * N + b) * C + c] = static_cast<scalar_t>(0);
            continue;
        }
        int64_t il = input_lengths[b];
        int64_t tl = target_lengths[b];
        const int64_t* tb = tgt + b * S;
        const scalar_t* lpb = lpp + b * C;          // frame stride N*C
        const scalar_t* lab = lap + b * T * M;
        scalar_t* gb = gp + b * C;

        if (il > 0) {
            // beta initialization at t = input_length - 1
            for (int64_t s = 0; s < M; ++s) beta[s] = neginf;
            const scalar_t* lp_last = lpb + ((il - 1) * N) * C;
            scalar_t* g_last = gb + (il - 1) * N * C;
            const scalar_t* la_last = lab + (il - 1) * M;
            beta[2 * tl] = lp_last[blank];
            g_last[blank] = la_last[2 * tl] + beta[2 * tl];
            if (tl > 0) {
                int64_t prime = ctc_target_prime(tb, 1, 2 * tl - 1, blank);
                beta[2 * tl - 1] = lp_last[prime];
                // first two states are a blank and a non-blank: no log+ needed
                g_last[prime] = la_last[2 * tl - 1] + beta[2 * tl - 1];
            }

            // eq (10)/(11) recursion plus eq (16) collection.
            // beta is a rolling single-row buffer: ascending s keeps
            // beta[s+1]/beta[s+2] holding the previous-row (t+1) values
            // until after they are consumed at step s.
            for (int64_t t = il - 2; t >= 0; --t) {
                const scalar_t* lpt = lpb + (t * N) * C;
                const scalar_t* lan = lab + ((t + 1) * M);
                const scalar_t* lat = lab + (t * M);
                scalar_t* gt = gb + (t * N) * C;
                for (int64_t s = 0; s <= 2 * tl; ++s) {
                    const int64_t current_target_prime =
                        ctc_target_prime(tb, 1, s, blank);
                    scalar_t lb1 = beta[s];
                    scalar_t lbmax = lb1;
                    scalar_t lb2 = neginf, lb3 = neginf;
                    if (s < 2 * tl) {
                        lb2 = beta[s + 1];
                        if (lb2 > lbmax) lbmax = lb2;
                    }
                    if (s < 2 * tl - 1 &&
                        ctc_target_prime(tb, 1, s + 2, blank) != current_target_prime) {
                        lb3 = beta[s + 2];
                        if (lb3 > lbmax) lbmax = lb3;
                    }
                    if (lbmax == neginf) lbmax = 0;
                    beta[s] = std::log(std::exp(lb1 - lbmax) + std::exp(lb2 - lbmax) +
                                       std::exp(lb3 - lbmax)) +
                              lbmax + lpt[current_target_prime];
                    // collected[b, t, target'[s]] "log+=" alpha + beta
                    scalar_t log_alpha_beta = lat[s] + beta[s];
                    scalar_t& lcab = gt[current_target_prime];
                    if (lcab == neginf) {
                        lcab = log_alpha_beta;
                    } else {
                        scalar_t mx = std::max(lcab, log_alpha_beta);
                        lcab = std::log(std::exp(lcab - mx) +
                                        std::exp(log_alpha_beta - mx)) + mx;
                    }
                }
            }
        }

        // wrap up with the remaining items of eq (16); note the dense softmax
        // training-time log_softmax, so unused labels receive exp(lp) * gr.
        // A reduced loss hands over one scalar while a caller that separates
        // each sequence hands over one entry per sequence.  Indexing the
        // scalar by b walked past its storage, which is where the access
        // violation showed up once the batch grew.
        scalar_t gr = grad_out.numel() == 1 ? gop[0] : gop[b];
        for (int64_t t = 0; t < il; ++t) {
            const scalar_t* lpt = lpb + (t * N) * C;
            scalar_t* gt = gb + (t * N) * C;
            for (int64_t c = 0; c < C; ++c) {
                scalar_t res = gt[c];
                scalar_t lpv = lpt[c];
                gt[c] = (std::exp(lpv) - std::exp(res + nllp[b] - lpv)) * gr;
            }
        }
        // zero the remainder (frames beyond input_length)
        for (int64_t t = il; t < T; ++t)
            for (int64_t c = 0; c < C; ++c) gb[(t * N) * C + c] = static_cast<scalar_t>(0);
    }
    return grad;
}

// `_ctc_loss` reads its lengths as integers. The reference takes int32 and
// int64 ranks and rejects everything else with the argument's own name, so the
// unpack converts through int64 instead of viewing the buffer as int64.
std::vector<int64_t> ctc_lengths_vector(const Tensor& lengths, const char* name) {
    if (lengths.dtype() != DType::Int64 && lengths.dtype() != DType::Int32)
        TP_THROW(RuntimeError, std::string(name) + " must be integral");
    auto flat = lengths.to(DType::Int64).contiguous();
    const int64_t* p = flat.data_ptr<int64_t>();
    return std::vector<int64_t>(p, p + flat.numel());
}

}  // namespace

Tensor _ctc_loss_backward_cpu(const Tensor& grad, const Tensor& log_probs,
                              const Tensor& targets, const Tensor& input_lengths,
                              const Tensor& target_lengths,
                              const Tensor& neg_log_likelihood,
                              const Tensor& log_alpha, int64_t blank,
                              bool zero_infinity) {
    std::vector<int64_t> il = ctc_lengths_vector(input_lengths, "input_lengths");
    std::vector<int64_t> tl = ctc_lengths_vector(target_lengths, "target_lengths");
    if (log_probs.dtype() == DType::Float64)
        return ctc_loss_backward_impl<double>(
            grad.contiguous(), log_probs.contiguous(), targets.contiguous(),
            il, tl, neg_log_likelihood.contiguous(), log_alpha.contiguous(),
            blank, zero_infinity);
    if (log_probs.dtype() == DType::Float32)
        return ctc_loss_backward_impl<float>(
            grad.contiguous(), log_probs.contiguous(), targets.contiguous(),
            il, tl, neg_log_likelihood.contiguous(), log_alpha.contiguous(),
            blank, zero_infinity);
    TP_THROW(NotImplementedError, "_ctc_loss_backward only supports Float32/Float64");
}

std::tuple<Tensor, Tensor> _ctc_loss_cpu(const Tensor& log_probs,
                                         const Tensor& targets,
                                         const Tensor& input_lengths,
                                         const Tensor& target_lengths,
                                         int64_t blank, bool zero_infinity) {
    (void)zero_infinity;
    std::vector<int64_t> il = ctc_lengths_vector(input_lengths, "input_lengths");
    std::vector<int64_t> tl = ctc_lengths_vector(target_lengths, "target_lengths");
    if (log_probs.dtype() == DType::Float64)
        return ctc_loss_impl<double>(log_probs.contiguous(),
                                     targets.contiguous(), il, tl, blank);
    if (log_probs.dtype() == DType::Float32)
        return ctc_loss_impl<float>(log_probs.contiguous(),
                                    targets.contiguous(), il, tl, blank);
    TP_THROW(NotImplementedError, "_ctc_loss only supports Float32/Float64");
}


Tensor ctc_loss_intlist_cpu(const Tensor& log_probs, const Tensor& targets,
                            const std::vector<int64_t>& input_lengths,
                            const std::vector<int64_t>& target_lengths,
                            int64_t blank, int64_t reduction,
                            bool zero_infinity) {
    Tensor il = Tensor::tensor(input_lengths, DType::Int64, log_probs.device());
    Tensor tl = Tensor::tensor(target_lengths, DType::Int64, log_probs.device());
    return composite::ctc_loss_compose(log_probs, targets, il, tl, blank, reduction,
                            zero_infinity);
}

Tensor ctc_loss_tensor_cpu(const Tensor& log_probs, const Tensor& targets,
                           const Tensor& input_lengths,
                           const Tensor& target_lengths, int64_t blank,
                           int64_t reduction, bool zero_infinity) {
    if (input_lengths.dtype() != DType::Int64 &&
        input_lengths.dtype() != DType::Int32) {
        TP_THROW(TypeError, "ctc_loss: input_lengths must be integral");
    }
    if (target_lengths.dtype() != DType::Int64 &&
        target_lengths.dtype() != DType::Int32) {
        TP_THROW(TypeError, "ctc_loss: target_lengths must be integral");
    }
    return composite::ctc_loss_compose(log_probs, targets, input_lengths, target_lengths,
                            blank, reduction, zero_infinity);
}

TENSORPLAY_LIBRARY_IMPL(CPU, LossKernels) {
    m.impl("nll_loss_forward", nll_loss_kernel);
    m.impl("nll_loss_backward", nll_loss_backward_kernel);
    m.impl("nll_loss2d_forward", nll_loss2d_kernel);
    m.impl("nll_loss2d_backward", nll_loss2d_backward_kernel);
    m.impl("_ctc_loss", _ctc_loss_cpu);
    m.impl("_ctc_loss_backward", _ctc_loss_backward_cpu);
    m.impl("ctc_loss.IntList", ctc_loss_intlist_cpu);
    m.impl("ctc_loss.Tensor", ctc_loss_tensor_cpu);

    m.impl("mse_loss", mse_loss_kernel);
    m.impl("mse_loss_backward", mse_loss_backward_kernel);

    m.impl("tp_l1_loss", l1_loss_kernel);
    m.impl("tp_l1_loss_backward", l1_loss_backward_kernel);
    m.impl("smooth_l1_loss", smooth_l1_loss_cpu);
    m.impl("smooth_l1_loss_backward", smooth_l1_loss_backward_cpu);
    m.impl("huber_loss", huber_loss_cpu);
    m.impl("huber_loss_backward", huber_loss_backward_cpu);
    m.impl("tp_kl_div", kl_div_kernel);
    m.impl("tp_kl_div_backward", kl_div_backward_kernel);
    m.impl("binary_cross_entropy", binary_cross_entropy_cpu);
    m.impl("binary_cross_entropy_backward", binary_cross_entropy_backward_cpu);
    m.impl("tp_margin_ranking_loss", margin_ranking_loss_kernel);
    m.impl("tp_margin_ranking_loss_backward", margin_ranking_loss_backward_kernel);
    m.impl("tp_hinge_embedding_loss", hinge_embedding_loss_kernel);
    m.impl("tp_hinge_embedding_loss_backward", hinge_embedding_loss_backward_kernel);
    m.impl("tp_soft_margin_loss", soft_margin_loss_kernel);
    m.impl("tp_soft_margin_loss_backward", soft_margin_loss_backward_kernel);
    m.impl("tp_cosine_embedding_loss", cosine_embedding_loss_kernel);
    m.impl("tp_cosine_embedding_loss_backward", cosine_embedding_loss_backward_kernel);
    m.impl("tp_poisson_nll_loss", poisson_nll_loss_kernel);
    m.impl("tp_poisson_nll_loss_backward", poisson_nll_loss_backward_kernel);

    // The plain names answer the same walk with the same reduction argument:
    // the two spellings differ only in that the folded twins were added later.
    m.impl("l1_loss", l1_loss_kernel);
    m.impl("kl_div", kl_div_kernel);
    m.impl("margin_ranking_loss", margin_ranking_loss_kernel);
    m.impl("hinge_embedding_loss", hinge_embedding_loss_kernel);
    m.impl("soft_margin_loss", soft_margin_loss_kernel);
    m.impl("cosine_embedding_loss", cosine_embedding_loss_kernel);
    m.impl("poisson_nll_loss", poisson_nll_loss_kernel);
}

} // namespace cpu
} // namespace tensorplay


// -----------------------------------------------------------------------------

#pragma once

namespace tensorplay::quant::TP_QN_DETAIL {
Tensor batch_norm(const Tensor& input, const std::optional<Tensor>& weight,
        const std::optional<Tensor>& bias, const Tensor& mean, const Tensor& var,
        double eps, double scale, int64_t zp, bool relu) {
    Guard guard(input);
    TP_CHECK(weight && bias, "Normalization requires weight and bias");
    validate(input);
    if (!input.numel()) return input.clone();
    TP_CHECK(input.dim() >= 2 && input.dim() <= 5, "Normalization requires rank two through five");
    const int64_t channels = input.size(1);
    for (const Tensor* p : {&*weight, &*bias, &mean, &var})
        TP_CHECK(p->device() == input.device() && p->dtype() == DType::Float32 && p->numel() == channels,
            "Normalization parameters must contain one floating value per channel on the input device");
    Tensor logical = input;
    while (logical.dim() < 4) logical = logical.unsqueeze(-1);
    const auto format = logical.dim() == 4 ? MemoryFormat::ChannelsLast : MemoryFormat::ChannelsLast3d;
    const Tensor x = logical.contiguous(static_cast<int64_t>(format)), w = weight->contiguous(), b = bias->contiguous(),
        m = mean.contiguous(), v = var.contiguous();
    Tensor y = output(input, logical.shape(), scale, zp);
    y = y.as_strided(logical.shape(), get_channels_last_strides(logical.shape()), 0);
    y.impl()->set_quantizer(make_per_tensor_affine_quantizer(scale, zp, input.dtype()));
    Tensor alpha = Tensor::empty({channels}, DType::Float32, input.device());
    Tensor beta = Tensor::empty({channels}, DType::Float32, input.device());
    auto* ap = alpha.data_ptr<float>(); auto* bp = beta.data_ptr<float>();
    const auto *wp = w.data_ptr<float>(), *biasp = b.data_ptr<float>(),
        *mp = m.data_ptr<float>(), *vp = v.data_ptr<float>();
    const double xs = quantized::q_scale(input);
    const int64_t xz = quantized::q_zero_point(input);
    apply(channels, [=] TP_QN_DEVICE (int64_t c) {
        const float inv = 1. / ::sqrt(vp[c] + static_cast<float>(eps));
        ap[c] = inv * wp[c] * (xs / scale);
        bp[c] = (biasp[c] - mp[c] * inv * wp[c]) / scale;
    });
    codes(input.dtype(), [&]<class T>() {
        const auto* src = x.data_ptr<T>(); auto* dst = y.data_ptr<T>();
#ifdef TP_QUANT_CUDA
        apply(y.numel(), [=] TP_QN_DEVICE (int64_t i) {
            const int64_t c = i % channels;
            float value = ap[c] * static_cast<float>(static_cast<int64_t>(src[i]) - xz) + bp[c];
            if (relu) value = ::fmaxf(value, 0.f);
            dst[i] = encode<T>(value, 1.f, zp);
        });
#else
        const int64_t rows = y.numel() / channels;
        parallel::parallel_for(0, rows, std::max<int64_t>(1, 32768 / channels), [&](int64_t first, int64_t last) {
            for (int64_t row = first; row < last; ++row)
                cpu::quantized_batch_norm_stub(DeviceType::CPU, src+row*channels, dst+row*channels,
                    channels, input.dtype(), ap, bp, xz, zp, relu);
        });
#endif
    });
    if (input.dim() < 4) {
        y = y.contiguous();
        while (y.dim() > input.dim()) y = y.squeeze(-1);
    }
    return y;
}
}

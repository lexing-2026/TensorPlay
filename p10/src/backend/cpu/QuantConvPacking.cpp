#include "QuantConvPacking.h"
#include "Exception.h"

#include <cstring>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

inline int64_t align_up_4(int64_t v) {
    return (v + 3) / 4 * 4;
}

Tensor pad_channels_axis(
    const Tensor& wq, int64_t out_ch, int64_t in_ch, int64_t KH, int64_t KW,
    bool pad_in_channels) {
    // Zero-pads the channel axes to the next multiple of four; each source
    // output-channel row copies into the front of its padded row.  The
    // depthwise form keeps its one-channel rows unpadded.
    const int64_t N_aligned = align_up_4(out_ch);
    const int64_t C_aligned = pad_in_channels ? align_up_4(in_ch) : in_ch;
    Tensor padded = Tensor::zeros(
        {N_aligned, C_aligned, KH, KW}, DType::Float32, wq.device());
    const uint8_t* src =
        static_cast<const uint8_t*>(wq.contiguous().data_ptr());
    uint8_t* dst = static_cast<uint8_t*>(padded.data_ptr());
    const size_t row = static_cast<size_t>(in_ch * KH * KW * sizeof(float));
    const size_t dst_row =
        static_cast<size_t>(C_aligned * KH * KW * sizeof(float));
    for (int64_t o = 0; o < out_ch; ++o) {
        std::memcpy(dst + static_cast<size_t>(o) * dst_row,
                    src + static_cast<size_t>(o) * row, row);
    }
    return padded;
}

Tensor slice_rows(const Tensor& t, int64_t rows, int64_t inner) {
    // Takes the first `rows` of a contiguous [rows_aligned, inner] plane.
    Tensor out = Tensor::empty({rows, inner}, DType::Float32, t.device());
    std::memcpy(
        out.data_ptr(), t.contiguous().data_ptr(),
        static_cast<size_t>(rows * inner * sizeof(float)));
    return out;
}

// Copies the leading `out_rows` x `in_cols` block out of a contiguous
// [rows_aligned, cols_aligned, inner] padded plane; the padded plane's row
// stride is `cols_aligned * inner`, so a prefix copy would pick up padding
// whenever the column count is not a multiple of four.
void copy_channel_rows(
    const Tensor& plane, Tensor& out, int64_t out_rows, int64_t in_cols,
    int64_t cols_aligned, int64_t inner) {
    const float* src = plane.contiguous().data_ptr<float>();
    float* dst = out.data_ptr<float>();
    for (int64_t r = 0; r < out_rows; ++r) {
        std::memcpy(
            dst + r * in_cols * inner,
            src + r * cols_aligned * inner,
            static_cast<size_t>(in_cols * inner * sizeof(float)));
    }
}

} // namespace

std::tuple<Tensor, Tensor> quantized_conv2d_prepack_cpu(
    const Tensor& weight,
    const Tensor& weight_scales,
    const Tensor& weight_zero_points,
    std::optional<Tensor> bias,
    bool transposed) {
    TP_CHECK(
        weight.dim() == 4,
        "quantized conv prepack: expected a 4-D weight");
    TP_CHECK(
        weight_scales.dim() == 1 &&
            weight_zero_points.shape() == weight_scales.shape(),
        "quantized conv prepack: scales/zero_points must be 1-D");

    // Dequantize with per-channel parameters on the host so the compute
    // shaders receive float payloads.  For a regular weight the channel
    // parameter indexes the output channels (dim 0); a transposed weight
    // arrives as [in, out, KH, KW], so its per-channel parameters index the
    // output channels on dim 1 instead.
    const int64_t O_raw = weight.size(0);
    const int64_t C_raw = weight.size(1);
    const int64_t KH = weight.size(2);
    const int64_t KW = weight.size(3);
    const int64_t param_channels = transposed ? C_raw : O_raw;
    TP_CHECK(
        weight_scales.size(0) == param_channels,
        "quantized conv prepack: scales length must match the weight's "
        "output channel count");

    Tensor w = weight.contiguous().to(DType::Float32);
    const float* pw = w.data_ptr<float>();
    Tensor sc = weight_scales.to(DType::Float32).contiguous();
    Tensor zp = weight_zero_points.to(DType::Float32).contiguous();
    const float* psc = sc.data_ptr<float>();
    const float* pzp = zp.data_ptr<float>();
    Tensor wq = Tensor::empty(w.shape(), DType::Float32, w.device());
    float* pwq = wq.data_ptr<float>();
    for (int64_t o = 0; o < O_raw; ++o) {
        for (int64_t c = 0; c < C_raw; ++c) {
            const float s = transposed ? psc[c] : psc[o];
            const float z = transposed ? pzp[c] : pzp[o];
            const int64_t base = (o * C_raw + c) * KH * KW;
            for (int64_t i = 0; i < KH * KW; ++i) {
                pwq[base + i] = (pw[base + i] - z) * s;
            }
        }
    }

    // Bias: float domain, padded to a multiple of four, reshaped {4, 1, L4}.
    const int64_t L_aligned = align_up_4(O_raw);
    Tensor b = bias.has_value() && bias->defined()
        ? bias->to(DType::Float32).contiguous()
        : Tensor::zeros({O_raw}, DType::Float32, w.device());
    Tensor bias_pad = Tensor::zeros(
        {L_aligned}, DType::Float32, w.device());
    std::memcpy(
        bias_pad.data_ptr(), b.data_ptr(),
        static_cast<size_t>(O_raw * sizeof(float)));
    Tensor bias_packed =
        bias_pad.reshape({L_aligned / 4, 4})
            .permute({1, 0})
            .reshape({4, 1, L_aligned / 4})
            .contiguous();

    if (transposed) {
        // Channel roles swap: the packed kernel is organized as
        // [out, in, KH, KW] where the original weight was [in, out, KH, KW],
        // and both spatial axes are flipped.
        wq = wq.permute({1, 0, 2, 3}).contiguous().flip({3}).flip({2});
    }

    const int64_t OC = wq.size(0);
    const int64_t IC = wq.size(1);

    Tensor packed;
    if (!transposed && IC == 1) {
        // Depthwise form: {4, N4*C, H*W}; every NxN filter flattens into one
        // row and groups of four filters stack vertically.
        const int64_t N_aligned = align_up_4(OC);
        Tensor padded =
            pad_channels_axis(wq, OC, IC, KH, KW, /*pad_in_channels=*/false);
        const int64_t N4 = N_aligned / 4;
        packed = padded.reshape({N4, 4, IC, KH * KW})
                     .permute({1, 0, 2, 3})
                     .reshape({4, N4 * IC, KH * KW})
                     .contiguous();
    } else {
        const int64_t N_aligned = align_up_4(OC);
        const int64_t C_aligned = align_up_4(IC);
        Tensor padded =
            pad_channels_axis(wq, OC, IC, KH, KW, /*pad_in_channels=*/true);
        const int64_t N4 = N_aligned / 4;
        const int64_t C4 = C_aligned / 4;
        if (!transposed) {
            // Fold groups of four input channels into the width axis, then
            // stack the output-channel groups vertically.
            packed = padded.reshape({N_aligned, C4, 4, KH, KW})
                         .permute({0, 1, 3, 4, 2})
                         .reshape({N_aligned, C4, KH, 4 * KW})
                         .permute({0, 2, 1, 3})
                         .reshape({N_aligned, KH, C_aligned * KW})
                         .contiguous();
        } else {
            // Transposed form interleaves the groups of four instead of
            // folding them contiguously.
            packed = padded.reshape({N_aligned, C4, 4, KH, KW})
                         .permute({0, 3, 4, 1, 2})
                         .reshape({N_aligned, KH, C_aligned * KW})
                         .contiguous();
        }
        packed = packed.reshape({N4, 4, KH, C_aligned * KW})
                     .permute({1, 0, 2, 3})
                     .reshape({4, N4 * KH, C_aligned * KW})
                     .contiguous();
    }

    return std::make_tuple(packed, bias_packed);
}

std::tuple<Tensor, Tensor> quantized_conv2d_unpack_cpu(
    const Tensor& weight_packed,
    const Tensor& bias_packed,
    const std::vector<int64_t>& weight_sizes,
    bool transposed,
    bool depthwise) {
    // Inverse of the rearrangement: rebuild the float-domain weight and bias
    // so the CPU/CUDA run paths can feed the float convolution.
    TP_CHECK(
        weight_sizes.size() == 4,
        "quantized conv unpack: expected a 4-D weight size list");
    const int64_t O = weight_sizes[0];
    const int64_t C = weight_sizes[1];
    const int64_t KH = weight_sizes[2];
    const int64_t KW = weight_sizes[3];

    Tensor w;
    if (depthwise) {
        const int64_t N_aligned = align_up_4(O);
        const int64_t N4 = N_aligned / 4;
        Tensor padded =
            weight_packed.reshape({4, N4, C, KH * KW})
                .permute({1, 0, 2, 3})
                .reshape({N_aligned, C, KH * KW})
                .contiguous();
        w = slice_rows(padded, O, C * KH * KW)
                .reshape({O, C, KH, KW})
                .contiguous();
    } else if (transposed) {
        // The caller reports the transposed weight sizes in [in, out, KH, KW]
        // order; the packed payload was rearranged after the channel-role
        // swap, so the output-channel role drives the row groups and the
        // input-channel role drives the folded groups of four.
        const int64_t OC = C; // packed output-channel role (was dim 1)
        const int64_t IC = O; // packed input-channel role (was dim 0)
        const int64_t N_aligned = align_up_4(OC);
        const int64_t C_aligned = align_up_4(IC);
        const int64_t N4 = N_aligned / 4;
        const int64_t C4 = C_aligned / 4;

        Tensor mid =
            weight_packed.reshape({4, N4, KH, C_aligned * KW})
                .permute({1, 0, 2, 3})
                .reshape({N_aligned, KH, C_aligned * KW})
                .contiguous();
        // Invert the interleaving (the permute is its own inverse).
        mid = mid.reshape({N_aligned, KH, KW, C4, 4})
                  .permute({0, 3, 4, 1, 2})
                  .reshape({N_aligned, C_aligned, KH, KW})
                  .contiguous();
        // Undo the spatial flips and the channel-role swap: the payload now
        // sits as [in_aligned, out_aligned, KH, KW]; copy the real channel
        // rows out of the padded plane one row at a time.
        mid = mid.flip({3}).flip({2}).permute({1, 0, 2, 3}).contiguous();
        w = Tensor::zeros({O, C, KH, KW}, DType::Float32,
                          weight_packed.device());
        copy_channel_rows(mid, w, O, C, align_up_4(C), KH * KW);
    } else {
        const int64_t N_aligned = align_up_4(O);
        const int64_t C_aligned = align_up_4(C);
        const int64_t N4 = N_aligned / 4;
        const int64_t C4 = C_aligned / 4;

        Tensor mid =
            weight_packed.reshape({4, N4, KH, C_aligned * KW})
                .permute({1, 0, 2, 3})
                .reshape({N_aligned, KH, C_aligned * KW})
                .contiguous();
        // Unfold the input-channel groups of four from the width axis
        // (each packing permute below is inverted step by step).
        mid = mid.reshape({N_aligned, KH, C4, 4 * KW})
                  .permute({0, 2, 1, 3})
                  .reshape({N_aligned, C4, KH, 4 * KW})
                  .contiguous();
        mid = mid.reshape({N_aligned, C4, KH, KW, 4})
                  .permute({0, 1, 4, 2, 3})
                  .reshape({N_aligned, C_aligned, KH, KW})
                  .contiguous();
        w = Tensor::zeros({O, C, KH, KW}, DType::Float32,
                          weight_packed.device());
        copy_channel_rows(mid, w, O, C, align_up_4(C), KH * KW);
    }

    // Bias reshape back to a plain [length] vector: the output-channel count
    // of the weight sizes, i.e. dim 0 for regular weights and dim 1 for a
    // transposed weight.
    const int64_t bias_len = transposed ? C : O;
    const int64_t L_aligned = align_up_4(bias_len);
    Tensor b = bias_packed.reshape({4, L_aligned / 4})
                   .permute({1, 0})
                   .reshape({L_aligned})
                   .contiguous();
    Tensor bias_out = Tensor::empty(
        {bias_len}, DType::Float32, weight_packed.device());
    std::memcpy(
        bias_out.data_ptr(), b.data_ptr(),
        static_cast<size_t>(bias_len * sizeof(float)));
    return std::make_tuple(w, bias_out);
}

} // namespace cpu
} // namespace tensorplay

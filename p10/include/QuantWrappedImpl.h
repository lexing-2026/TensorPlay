#pragma once
#ifndef TP_QUANT_CUDA
#include <cstring>

namespace tensorplay::quant::TP_QN_DETAIL {
struct WrappedLayout {
    int64_t n, k, padded, kind, bias;
    int64_t codes = 64, offsets, scales, zero_points, biases, size;
    WrappedLayout(int64_t outputs, int64_t features, int64_t type, bool has_bias)
        : n(outputs), k(features), padded((features+31)/32*32), kind(type), bias(has_bias) {
        auto align = [](int64_t value) { return (value+7)/8*8; };
        offsets = align(codes+n*(kind==1 ? padded : k*2));
        scales = offsets+n*8;
        zero_points = align(scales+n*4);
        biases = zero_points+n*8;
        size = align(biases+(bias ? n*4 : 0));
    }
};

Tensor wrapped_view(const Tensor& storage, int64_t offset, int64_t bytes, DType dtype) {
    return storage.narrow(0,offset,bytes).view_dtype(dtype);
}

WrappedLayout wrapped_layout(const Tensor& storage, int64_t expected_kind) {
    TP_CHECK(storage.dtype()==DType::UInt8 && storage.device().is_cpu() && storage.dim()==1 && storage.is_contiguous() && storage.numel()>=64,
        "Wrapped parameters require a contiguous byte payload on the host");
    int64_t h[8]; std::memcpy(h,storage.data_ptr(),64);
    TP_CHECK(h[0]==0x5450575241503031LL && h[1]==1 && h[2]>=0 && h[3]>=0 && h[4]==expected_kind && (h[5]==0 || h[5]==1),
        "Wrapped parameter header is invalid");
    WrappedLayout layout(h[2],h[3],h[4],h[5]);
    TP_CHECK(layout.size==storage.numel(), "Wrapped parameter payload has an invalid size");
    return layout;
}

Tensor wrapped_allocate(const WrappedLayout& layout) {
    Tensor storage=Tensor::zeros({layout.size},DType::UInt8,Device(DeviceType::CPU));
    const int64_t h[8]{0x5450575241503031LL,1,layout.n,layout.k,layout.kind,layout.bias,0,0};
    std::memcpy(storage.data_ptr(),h,64);
    return storage;
}

Tensor wrapped_prepack(const Tensor& weight,const Tensor& scale,const Tensor& zp,const Tensor& bias) {
    TP_CHECK(weight.device().is_cpu() && weight.dim()==2 && bias.device()==weight.device() && bias.dtype()==DType::Float32 && bias.dim()==1 && bias.numel()==weight.size(0),
        "Wrapped linear packing requires host matrix weights and a floating output channel bias");
    Tensor qw=make_per_tensor_affine_quantizer(scale.item<double>(),zp.item<int64_t>(),DType::QInt8)->quantize(weight);
    auto [packed,offsets]=cpu::_quantized_weight_prepack_cpu(qw);
    WrappedLayout layout(weight.size(0),weight.size(1),1,true);
    Tensor storage=wrapped_allocate(layout);
    auto* data=storage.data_ptr<uint8_t>();
    if (packed.numel()) std::memcpy(data+layout.codes,packed.data_ptr(),packed.numel());
    if (offsets.numel()) std::memcpy(data+layout.offsets,offsets.data_ptr(),offsets.numel()*8);
    Tensor scales=wrapped_view(storage,layout.scales,layout.n*4,DType::Float32);
    Tensor zps=wrapped_view(storage,layout.zero_points,layout.n*8,DType::Int64);
    for (int64_t n=0;n<layout.n;++n) { scales.data_ptr<float>()[n]=qw.impl()->quantizer()->scale(); zps.data_ptr<int64_t>()[n]=qw.impl()->quantizer()->zero_point(); }
    const Tensor b=bias.contiguous();
    if (layout.n) std::memcpy(data+layout.biases,b.data_ptr(),layout.n*4);
    return storage;
}

Tensor wrapped_linear_prepacked(const Tensor& input,const Tensor& xs,const Tensor& xz,
        const Tensor& storage,const Tensor& ys,const Tensor& yz,int64_t out_channels) {
    const auto layout=wrapped_layout(storage,1);
    const Tensor packed=wrapped_view(storage,layout.codes,layout.n*layout.padded,DType::Int8).reshape({layout.n,layout.padded});
    const Tensor offsets=wrapped_view(storage,layout.offsets,layout.n*8,DType::Int64);
    const Tensor scales=wrapped_view(storage,layout.scales,layout.n*4,DType::Float32);
    const Tensor zps=wrapped_view(storage,layout.zero_points,layout.n*8,DType::Int64);
    const Tensor bias=wrapped_view(storage,layout.biases,layout.n*4,DType::Float32);
    const Tensor qx=make_per_tensor_affine_quantizer(xs.item<double>(),xz.item<int64_t>(),DType::QUInt8)->quantize(input);
    const Tensor result=cpu::_quantized_packed_run_cpu(qx,packed,offsets,scales,zps,bias,{layout.n,layout.k},
        ys.item<float>(),yz.item<int64_t>(),{},{},{},1,false,false,0,0.,std::nullopt);
    return result.impl()->quantizer()->dequantize(result);
}

Tensor wrapped_linear(const Tensor& input,const Tensor& xs,const Tensor& xz,
        const Tensor& weight,const Tensor& ws,const Tensor& wz,const Tensor& bias,
        const Tensor& ys,const Tensor& yz,int64_t out_channels) {
    return wrapped_linear_prepacked(input,xs,xz,wrapped_prepack(weight,ws,wz,bias),ys,yz,out_channels);
}

Tensor wrapped_half_prepack(const Tensor& weight) {
    TP_CHECK(weight.device().is_cpu() && weight.dim()==2, "Wrapped half packing requires a host weight matrix");
    const Tensor half=fp16_prepack(weight);
    WrappedLayout layout(weight.size(0),weight.size(1),2,false);
    Tensor storage=wrapped_allocate(layout);
    if (half.numel()) std::memcpy(storage.data_ptr<uint8_t>()+layout.codes,half.data_ptr(),half.numel()*2);
    return storage;
}

Tensor wrapped_half_linear(const Tensor& input,const Tensor& storage,const Tensor& bias,int64_t out_channels) {
    const auto layout=wrapped_layout(storage,2);
    Tensor half=wrapped_view(storage,layout.codes,layout.n*layout.k*2,DType::Float16).reshape({layout.n,layout.k});
    return fp16_linear(input,half,bias,false);
}
Tensor legacy_integer_pack(const Tensor& weight) {
    TP_CHECK(weight.dim()==2 && weight.dtype()==DType::Int8 && weight.device().is_cpu(),
        "Integer matrix packing requires signed byte host weights");
    Tensor qw=quantized::make_qtensor(weight.contiguous(),make_per_tensor_affine_quantizer(1.,0,DType::QInt8),DType::QInt8);
    auto [packed,offsets]=cpu::_quantized_weight_prepack_cpu(qw);
    WrappedLayout layout(weight.size(0),weight.size(1),1,false);
    Tensor storage=wrapped_allocate(layout);
    if (packed.numel()) std::memcpy(storage.data_ptr<uint8_t>()+layout.codes,packed.data_ptr(),packed.numel());
    if (offsets.numel()) std::memcpy(storage.data_ptr<uint8_t>()+layout.offsets,offsets.data_ptr(),offsets.numel()*8);
    return storage;
}

std::tuple<Tensor,Tensor,double,int64_t> legacy_quantize_weight(const Tensor& weight) {
    TP_CHECK(weight.dim()==2 && weight.dtype()==DType::Float32 && weight.device().is_cpu(),
        "Weight quantization requires a floating host matrix");
    auto extrema = Tensor::aminmax(weight.reshape({weight.numel()}), {}, false);
    float minimum = std::min(std::get<0>(extrema).item().to<float>(), 0.f);
    float maximum = std::max(std::get<1>(extrema).item().to<float>(), 0.f);
    TP_CHECK(minimum <= maximum, "weight quantization requires an ordered finite range");
    double scale = (static_cast<double>(maximum) - minimum) / 255;
    if (static_cast<float>(scale) == 0.f || std::isinf(1.f / static_cast<float>(scale))) scale = .1;
    constexpr float cutoff = 6.1e-5f;
    if (scale < cutoff) {
        const float original = scale;
        scale = cutoff;
        if (minimum == 0.f) maximum = cutoff * 255;
        else if (maximum == 0.f) minimum = -cutoff * 255;
        else {
            const float amplifier = cutoff / original;
            minimum *= amplifier;
            maximum *= amplifier;
        }
    }
    const double from_min = -128 - minimum / scale;
    const double from_max = 127 - maximum / scale;
    const double error_min = 128 - std::abs(minimum / scale);
    const double error_max = 127 - std::abs(maximum / scale);
    const double initial = error_min < error_max ? from_min : from_max;
    const int64_t zero_point = static_cast<int64_t>(std::nearbyint(std::clamp(initial, -128., 127.)));
    scale = static_cast<float>(scale);
    Tensor q=cpu::quantize_per_tensor_cpu(weight,scale,zero_point,DType::QInt8);
    Tensor raw=q.int_repr(),offsets=Tensor::empty({weight.size(0)},DType::Int32,weight.device());
    const Tensor values=raw.contiguous();
    const int64_t k=weight.size(1),zp=quantized::q_zero_point(q);
    for (int64_t n=0;n<weight.size(0);++n) {
        int64_t sum=-k*zp;
        for (int64_t i=0;i<k;++i) sum+=values.data_ptr<int8_t>()[n*k+i];
        offsets.data_ptr<int32_t>()[n]=static_cast<int32_t>(sum);
    }
    return {raw,offsets,scale,zp};
}

Tensor legacy_integer_linear(const Tensor& input,const Tensor& weight,const Tensor& storage,
        const Tensor& col_offsets,const Scalar& weight_scale,const Scalar& weight_zp,const Tensor& bias) {
    const auto layout=wrapped_layout(storage,1);
    TP_CHECK(weight.dim()==2 && weight.size(0)==layout.n && weight.size(1)==layout.k && col_offsets.dtype()==DType::Int32 && col_offsets.numel()==layout.n && col_offsets.device()==input.device(),
        "Integer linear dimensions and column offsets must match the packed matrix");
    TP_CHECK(weight_scale.isFloatingPoint() && weight_zp.isIntegral(false), "Integer linear requires a floating scale and integer zero point");
    Tensor packed=wrapped_view(storage,layout.codes,layout.n*layout.padded,DType::Int8).reshape({layout.n,layout.padded});
    Tensor offsets=Tensor::empty({layout.n},DType::Int64,input.device());
    Tensor scales=Tensor::empty({layout.n},DType::Float32,input.device());
    Tensor zps=Tensor::empty({layout.n},DType::Int64,input.device());
    const Tensor columns=col_offsets.contiguous();
    const int64_t zp=weight_zp.to<int64_t>();
    for (int64_t n=0;n<layout.n;++n) {
        offsets.data_ptr<int64_t>()[n]=static_cast<int64_t>(columns.data_ptr<int32_t>()[n])+zp*layout.k;
        scales.data_ptr<float>()[n]=weight_scale.to<float>(); zps.data_ptr<int64_t>()[n]=zp;
    }
    const Tensor qx=cpu::quantize_per_tensor_dynamic_cpu(input,DType::QUInt8,false),b=bias.contiguous();
    return cpu::_quantized_packed_run_cpu(qx,packed,offsets,scales,zps,b,{layout.n,layout.k},
        1.,0,{},{},{},1,false,true,0,0.,std::nullopt);
}

#define TP_LEGACY_CELL_ARGS \
    const Tensor& wi,const Tensor& wh,const Tensor& bi,const Tensor& bh, \
    const Tensor& pi,const Tensor& ph,const Tensor& ci,const Tensor& ch, \
    const Scalar& si,const Scalar& sh,const Scalar& zi,const Scalar& zh
#define TP_LEGACY_CELL_VALUES wi,wh,bi,bh,pi,ph,ci,ch,si,sh,zi,zh
std::tuple<Tensor,Tensor> legacy_cell(const Tensor& input,const Tensor& hidden,
        const std::optional<Tensor>& cell,TP_LEGACY_CELL_ARGS,int64_t mode) {
    Tensor a=legacy_integer_linear(input,wi,pi,ci,si,zi,bi);
    Tensor b=legacy_integer_linear(hidden,wh,ph,ch,sh,zh,bh);
    return recurrent_step_checked(a,b,hidden,cell,mode);
}
std::tuple<Tensor,Tensor> legacy_lstm_cell(const Tensor& input,const std::vector<Tensor>& hx,TP_LEGACY_CELL_ARGS) {
    TP_CHECK(hx.size()==2, "Long-memory cells require hidden and cell states");
    return legacy_cell(input,hx[0],hx[1],TP_LEGACY_CELL_VALUES,2);
}
Tensor legacy_gru_cell(const Tensor& input,const Tensor& hx,TP_LEGACY_CELL_ARGS) {
    return std::get<0>(legacy_cell(input,hx,std::nullopt,TP_LEGACY_CELL_VALUES,3));
}
Tensor legacy_relu_cell(const Tensor& input,const Tensor& hx,TP_LEGACY_CELL_ARGS) {
    return std::get<0>(legacy_cell(input,hx,std::nullopt,TP_LEGACY_CELL_VALUES,0));
}
Tensor legacy_tanh_cell(const Tensor& input,const Tensor& hx,TP_LEGACY_CELL_ARGS) {
    return std::get<0>(legacy_cell(input,hx,std::nullopt,TP_LEGACY_CELL_VALUES,1));
}
#undef TP_LEGACY_CELL_ARGS
#undef TP_LEGACY_CELL_VALUES

Tensor legacy_integer_pack_sizes(const Tensor& weight,int64_t k,int64_t n) {
    return legacy_integer_pack(weight);
}
Tensor legacy_half_linear(const Tensor& input,const Tensor& storage,const std::optional<Tensor>& bias) {
    const auto layout=wrapped_layout(storage,2);
    Tensor half=wrapped_view(storage,layout.codes,layout.n*layout.k*2,DType::Float16).reshape({layout.n,layout.k});
    return fp16_linear(input,half,bias,false);
}
Tensor legacy_half_linear_bias(const Tensor& input,const Tensor& storage,const Tensor& bias) {
    return legacy_half_linear(input,storage,bias);
}
Tensor legacy_half_linear_out(const Tensor& input,const Tensor& storage,const std::optional<Tensor>& bias,Tensor& out) {
    Tensor result=legacy_half_linear(input,storage,bias);
    TP_CHECK(out.dtype()==result.dtype() && out.device()==result.device(), "Half linear output must match the result dtype and device");
    if (out.shape()!=result.shape()) out.resize_(result.shape());
    out.copy_(result); return out;
}
Tensor legacy_half_linear_bias_out(const Tensor& input,const Tensor& storage,const Tensor& bias,Tensor& out) {
    return legacy_half_linear_out(input,storage,bias,out);
}

}
#endif

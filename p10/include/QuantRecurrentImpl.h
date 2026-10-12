#pragma once
#include "QuantKernels.h"
#include "QuantPacked.h"

#ifdef TP_QUANT_CUDA
#define TP_QR_FN(x) tensorplay::cuda::x##_cuda
#else
#define TP_QR_FN(x) tensorplay::cpu::x##_cpu
#endif

namespace tensorplay::quant::TP_QN_DETAIL {

struct RecurrentLinear {
    Tensor weight, packed, offsets, scales, zps;
    std::optional<Tensor> bias;
    explicit RecurrentLinear(const Tensor& w, const Tensor& b) : weight(w), bias(b.defined() ? std::optional<Tensor>(b) : std::nullopt) {
        TP_CHECK(w.dim() == 2, "Recurrent weights must be matrices");
        if (w.dtype() == DType::Float16) return;
        TP_CHECK(w.dtype() == DType::QInt8 && w.impl()->quantizer(), "Recurrent weights require signed integer grids or half storage");
        struct CachedWeight { Tensor source, packed, offsets; uint32_t version=0; uintptr_t stream=0; };
        thread_local std::array<CachedWeight,16> cache;
        thread_local size_t next=0;
#ifdef TP_QUANT_CUDA
        const uintptr_t stream=reinterpret_cast<uintptr_t>(tensorplay::cuda::getCurrentCUDAStream().stream());
#else
        const uintptr_t stream=0;
#endif
        const bool cacheable=w.impl()->version_counter().is_enabled();
        const uint32_t version=cacheable ? w._version() : 0;
        for (const auto& entry:cache) {
            if (entry.source.defined() && cacheable && entry.source.impl()==w.impl() && entry.version==version && entry.stream==stream) {
                packed=entry.packed; offsets=entry.offsets; break;
            }
        }
        if (!packed.defined()) {
            std::tie(packed, offsets) = TP_QR_FN(_quantized_weight_prepack)(w);
            if (cacheable && packed.numel() <= 4*1024*1024) cache[next++%cache.size()]={w,packed,offsets,version,stream};
        }
        if (w.impl()->quantizer()->qscheme() == kPerTensorAffine) {
            scales = Tensor::empty({w.size(0)}, DType::Float32, w.device());
            zps = Tensor::empty({w.size(0)}, DType::Int64, w.device());
            scales.fill_(quantized::q_scale(w)); zps.fill_(quantized::q_zero_point(w));
        } else {
            TP_CHECK(w.impl()->quantizer()->qscheme() == kPerChannelAffine && quantized::q_per_channel_axis(w) == 0,
                     "Recurrent channel grids require output axis zero");
            scales = quantized::q_per_channel_scales(w).to(DType::Float32).contiguous();
            zps = quantized::q_per_channel_zero_points(w).to(DType::Int64).contiguous();
        }
    }
    Tensor operator()(const Tensor& x, bool reduce_range) const {
        if (weight.dtype() == DType::Float16) return fp16_linear(x, weight, bias, false);
        Tensor qx = x.numel() ? TP_QR_FN(quantize_per_tensor_dynamic)(x, DType::QUInt8, reduce_range)
            : TP_QR_FN(quantize_per_tensor)(x, .1, 0, DType::QUInt8);
        return TP_QR_FN(_quantized_packed_run)(qx, packed, offsets, scales, zps, bias,
            weight.shape(), 1., 0, {}, {}, {}, 1, false, true, 0, 0., std::nullopt);
    }
};

struct RecurrentStep {
    const float *input, *recurrent, *hidden, *cell; float *hy, *cy; int64_t width; int mode;
    TP_QN_DEVICE float sigmoid(float x) const { return 1.f / (1.f + ::expf(-x)); }
    TP_QN_DEVICE void operator()(int64_t i) const {
        const int gates = mode == 2 ? 4 : mode == 3 ? 3 : 1;
        const int64_t base = i / width * gates * width + i % width;
        if (mode == 2) {
            const float ingate = sigmoid(input[base] + recurrent[base]);
            const float forgetgate = sigmoid(input[base+width] + recurrent[base+width]);
            const float candidate = ::tanhf(input[base+2*width] + recurrent[base+2*width]);
            const float outgate = sigmoid(input[base+3*width] + recurrent[base+3*width]);
            const float value = forgetgate * cell[i] + ingate * candidate;
            cy[i] = value; hy[i] = outgate * ::tanhf(value);
        } else if (mode == 3) {
            const float reset = sigmoid(recurrent[base] + input[base]);
            const float update = sigmoid(recurrent[base+width] + input[base+width]);
            const float candidate = ::tanhf(input[base+2*width] + recurrent[base+2*width] * reset);
            hy[i] = (hidden[i] - candidate) * update + candidate;
        } else {
            const float value = input[base] + recurrent[base];
            hy[i] = mode == 0 ? ::fmaxf(value, 0.f) : ::tanhf(value);
        }
    }
};

std::tuple<Tensor,Tensor> recurrent_step(const Tensor& gates, const Tensor& hidden_gates,
        const Tensor& hidden, const Tensor& cell, int64_t mode) {
    Tensor a = gates.contiguous(), b = hidden_gates.contiguous(), h = hidden.contiguous(), c;
    if (mode == 2) c = cell.contiguous();
    Tensor hy = Tensor::empty(hidden.shape(), DType::Float32, hidden.device());
    Tensor cy = mode == 2 ? Tensor::empty(cell.shape(), DType::Float32, cell.device())
                         : Tensor::empty({0}, DType::Float32, hidden.device());
    apply(h.numel(), RecurrentStep{a.data_ptr<float>(),b.data_ptr<float>(),h.data_ptr<float>(),
        mode == 2 ? c.data_ptr<float>() : nullptr,hy.data_ptr<float>(),mode == 2 ? cy.data_ptr<float>() : nullptr,hidden.size(1),static_cast<int>(mode)});
    return {hy,cy};
}

std::tuple<Tensor,Tensor> recurrent_step_checked(const Tensor& gates,const Tensor& hidden_gates,
        const Tensor& hidden,const std::optional<Tensor>& cell,int64_t mode) {
    Guard guard(hidden);
    TP_CHECK(mode>=0 && mode<=3 && hidden.dim()==2 && hidden.dtype()==DType::Float32,
        "Recurrent state requires a floating batch matrix and a valid cell mode");
    const int64_t width=hidden.size(1),count=mode==2 ? 4 : mode==3 ? 3 : 1;
    for (const Tensor* value:{&gates,&hidden_gates})
        TP_CHECK(value->dim()==2 && value->size(0)==hidden.size(0) && value->size(1)==width*count && value->dtype()==hidden.dtype() && value->device()==hidden.device(),
            "Recurrent gates must match the state batch, gate count, dtype and device");
    TP_CHECK(mode!=2 || (cell && cell->shape()==hidden.shape() && cell->device()==hidden.device() && cell->dtype()==hidden.dtype()),
        "Long-memory recurrence requires a matching cell state");
    return recurrent_step(gates,hidden_gates,hidden,cell.value_or(Tensor()),mode);
}

std::tuple<Tensor,Tensor> recurrent_cell(const Tensor& input, const Tensor& hidden, const std::optional<Tensor>& cell,
        const Tensor& wi, const Tensor& wh, const std::optional<Tensor>& bi, const std::optional<Tensor>& bh,
        int64_t mode, bool reduce_range) {
    Guard guard(input);
    const int gates = mode == 2 ? 4 : mode == 3 ? 3 : 1;
    TP_CHECK(mode >= 0 && mode <= 3 && input.dtype() == DType::Float32 && hidden.dtype() == DType::Float32
        && input.dim() == 2 && hidden.dim() == 2 && input.size(0) == hidden.size(0), "Recurrent cells require compatible floating batch matrices");
    TP_CHECK(wi.dim() == 2 && wh.dim() == 2 && wi.size(0) == gates * hidden.size(1) && wh.size(0) == wi.size(0)
        && wi.size(1) == input.size(1) && wh.size(1) == hidden.size(1), "Recurrent cell weight dimensions do not match the states");
    TP_CHECK(input.device() == hidden.device() && input.device() == wi.device() && input.device() == wh.device(), "Recurrent cell operands must share a device");
    TP_CHECK(mode != 2 || (cell && cell->shape() == hidden.shape() && cell->dtype() == input.dtype() && cell->device() == input.device()), "Long-memory cells require a compatible cell state");
    RecurrentLinear ih(wi,bi.value_or(Tensor())), hh(wh,bh.value_or(Tensor()));
    return recurrent_step(ih(input,reduce_range),hh(hidden,reduce_range),hidden,cell.value_or(Tensor()),mode);
}

std::tuple<Tensor,Tensor,Tensor> recurrent(const Tensor& input, const Tensor& hidden,
        const std::optional<Tensor>& cell, const std::vector<Tensor>& weights,
        const std::vector<Tensor>& biases, int64_t layers, double dropout,
        bool training, bool bidirectional, bool batch_first, int64_t mode,
        bool reduce_range, const std::optional<Tensor>& batch_sizes) {
    Guard guard(input);
    TP_CHECK(mode >= 0 && mode <= 3 && layers > 0 && dropout >= 0 && dropout <= 1, "Invalid recurrent configuration");
    const int64_t directions = bidirectional ? 2 : 1, gates = mode == 2 ? 4 : mode == 3 ? 3 : 1;
    const bool packed = batch_sizes.has_value();
    TP_CHECK(input.dtype() == DType::Float32 && hidden.dtype() == DType::Float32 && hidden.dim() == 3
        && input.device() == hidden.device() && input.dim() == (packed ? 2 : 3), "Recurrent sequence and states require compatible floating storage");
    TP_CHECK(hidden.size(0) == layers * directions && weights.size() == static_cast<size_t>(2 * layers * directions)
        && (biases.empty() || biases.size() == weights.size()), "Recurrent layer, direction and parameter counts do not match");
    TP_CHECK(mode != 2 || (cell && cell->shape() == hidden.shape() && cell->dtype() == DType::Float32 && cell->device() == input.device()), "Long-memory recurrence requires matching cell states");
    Tensor source = packed || !batch_first ? input.contiguous() : input.transpose(0,1).contiguous();
    const int64_t width = hidden.size(2), batch = hidden.size(1);
    std::vector<int64_t> sizes, offsets;
    int64_t steps = source.size(0);
    if (packed) {
        TP_CHECK(batch_sizes->device().is_cpu() && batch_sizes->dtype() == DType::Int64 && batch_sizes->dim() == 1,
            "Packed batch sizes require a host integer vector");
        Tensor bs = batch_sizes->contiguous(); steps = bs.numel();
        sizes.assign(bs.data_ptr<int64_t>(),bs.data_ptr<int64_t>()+steps);
        int64_t total = 0, previous = batch;
        for (const auto size : sizes) { TP_CHECK(size > 0 && size <= previous, "Packed batch sizes must be positive and nonincreasing"); offsets.push_back(total); total += size; previous = size; }
        TP_CHECK(total == source.size(0) && (steps == 0 || sizes.front() == batch), "Packed row count does not match batch sizes or initial states");
    } else {
        TP_CHECK(source.size(1) == batch, "Sequence batch does not match initial states");
        sizes.assign(steps,batch);
    }
    TP_CHECK(steps > 0, "Recurrent sequence length must be positive");
    Tensor hy = hidden.clone(), cy = mode == 2 ? cell->clone() : Tensor::empty({0},DType::Float32,input.device());
    for (int64_t layer = 0; layer < layers; ++layer) {
        auto shape = static_cast<std::vector<int64_t>>(source.shape()); shape.back() = width * directions;
        Tensor layer_output = Tensor::empty(shape,DType::Float32,input.device());
        for (int64_t dir = 0; dir < directions; ++dir) {
            const int64_t state = layer * directions + dir, p = 2 * state;
            TP_CHECK(weights[p].dim() == 2 && weights[p+1].dim() == 2 && weights[p].size(0) == gates*width
                && weights[p].size(1) == source.size(source.dim()-1) && weights[p+1].size(0) == gates*width && weights[p+1].size(1) == width,
                "Recurrent sequence weight dimensions do not match layer widths");
            TP_CHECK(weights[p].device() == input.device() && weights[p+1].device() == input.device(), "Recurrent weights must share the sequence device");
            RecurrentLinear ih(weights[p],biases.empty()?Tensor():biases[p]);
            RecurrentLinear hh(weights[p+1],biases.empty()?Tensor():biases[p+1]);
            Tensor input_gates = ih(source,reduce_range);
            Tensor h = hidden.select(0,state).clone(), c = mode == 2 ? cell->select(0,state).clone() : Tensor();
            for (int64_t iteration = 0; iteration < steps; ++iteration) {
                const int64_t t = dir == 0 ? iteration : steps-1-iteration, current = sizes[t];
                Tensor active_h = packed ? h.narrow(0,0,current) : h;
                Tensor active_c = mode == 2 ? (packed ? c.narrow(0,0,current) : c) : Tensor();
                Tensor a = packed ? input_gates.narrow(0,offsets[t],current) : input_gates.select(0,t);
                auto [new_h,new_c] = recurrent_step(a,hh(active_h,reduce_range),active_h,active_c,mode);
                if (packed) { h.narrow(0,0,current).copy_(new_h); if (mode == 2) c.narrow(0,0,current).copy_(new_c); }
                else { h = new_h; if (mode == 2) c = new_c; }
                Tensor out = packed ? layer_output.narrow(0,offsets[t],current) : layer_output.select(0,t);
                out.narrow(1,dir*width,width).copy_(new_h);
            }
            hy.select(0,state).copy_(h); if (mode == 2) cy.select(0,state).copy_(c);
        }
        source = layer_output;
        if (training && dropout > 0 && layer+1 < layers)
            source = DispatchStub<Tensor,const Tensor&,double,bool>::call("dropout",computeDispatchKey(input.device()),source,dropout,true);
    }
    if (!packed && batch_first) source = source.transpose(0,1);
    return {source,hy,cy};
}
}
#undef TP_QR_FN

"""Public operator packets forwarding to native quantized execution."""
from functools import partial
import tensorplay as tp
from tensorplay import _C
from . import functional as F
from .packed import PackedWeight, PackedHalfWeight
from .embedding_ops import EmbeddingPackedParams


class OperatorPacket:
    def __init__(self, name, default, **overloads):
        self.__name__ = name
        self.default = default
        self._overloads = {'default': default, **overloads}
        for key, value in overloads.items():
            setattr(self, key, value)

    def __call__(self, *args, **kwargs):
        return self.default(*args, **kwargs)

    def overloads(self):
        return list(self._overloads)

    def __repr__(self):
        return f'<quantized operator {self.__name__}>'


class PackedConvolution:
    def __init__(self, weight, bias, stride, padding, dilation, groups,
                 transposed=False, output_padding=()):
        rank = weight.dim() - 2
        if rank not in (1,2,3) or groups <= 0 or weight.dtype != tp.qint8:
            raise ValueError('Convolution packing requires signed spatial weights and positive groups')
        self.stride, self.padding, self.dilation = list(stride), list(padding), list(dilation)
        self.output_padding = list(output_padding) if output_padding else [0] * rank
        self.groups, self.transposed = int(groups), bool(transposed)
        if any(len(p) != rank for p in (self.stride,self.padding,self.dilation,self.output_padding)):
            raise ValueError('Convolution parameters must contain one value per spatial dimension')
        if any(v <= 0 for v in self.stride+self.dilation) or any(v < 0 for v in self.padding+self.output_padding):
            raise ValueError('Convolution stride and dilation must be positive and padding nonnegative')
        if transposed:
            if weight.qscheme() != tp.per_tensor_affine:
                raise ValueError('Transposed convolution weights require a tensor affine grid')
            self.weight, self.bias = weight.contiguous(), bias
            self.storage = None
        else:
            self.storage = PackedWeight(weight,bias)
            self.weight, self.bias = self.storage._weight_bias()

    def unpack(self):
        return self.weight, self.bias

    def run(self, input, scale, zero_point, relu=False):
        if self.storage is not None:
            return self.storage.run(input,scale,zero_point,self.stride,self.padding,
                                    self.dilation,self.groups,relu)
        return _C.quantized_convolution(input,self.weight,self.bias,float(scale),int(zero_point),
            self.stride,self.padding,self.dilation,self.groups,True,self.output_padding,relu)

    def run_dynamic(self, input, reduce_range=False):
        if input.dtype != tp.float32:
            raise TypeError('Dynamic convolution input must use float storage')
        quantized = (_C.quantize_per_tensor_dynamic(input,tp.quint8,reduce_range) if input.numel()
                     else _C.quantize_per_tensor(input,.1,0,tp.quint8))
        return self.run(quantized,quantized.q_scale(),quantized.q_zero_point()).dequantize()


def _linear(input, packed_weight, output_scale, output_zero_point, relu=False):
    return packed_weight.run(input,output_scale,output_zero_point,relu=relu)


def _linear_dynamic(input, packed_weight, reduce_range=False, relu=False):
    if isinstance(packed_weight,PackedHalfWeight):
        return packed_weight.run(input,relu=relu)
    quantized = (_C.quantize_per_tensor_dynamic(input,tp.quint8,reduce_range) if input.numel()
                 else _C.quantize_per_tensor(input,.1,0,tp.quint8))
    return packed_weight.run(quantized,relu=relu,output_float=True)


def _linear_float(input, input_scale, input_zero_point, packed_weight, relu=False):
    quantized = _C.quantize_per_tensor(input,float(input_scale),int(input_zero_point),tp.quint8)
    return packed_weight.run(quantized,relu=relu,output_float=True)


def _conv(input, packed_weight, *parameters, relu=False):
    if len(parameters) == 2:
        scale,zero_point = parameters
    elif len(parameters) == 6:
        _,_,_,_,scale,zero_point = parameters
    else:
        raise TypeError('Convolution expects a packed weight and output scale and zero point')
    return packed_weight.run(input,scale,zero_point,relu)


def _legacy(*args,**kwargs):
    raise RuntimeError('Legacy packed tensors require regeneration with the current packing interface')


def _conv_unpack_sizes(packed):
    weight,bias=packed.unpack()
    return (list(weight.shape),None if bias is None else list(bias.shape),
        packed.stride,packed.padding,packed.dilation,packed.groups)


def _conv_add(input,accum,packed,scale,zero_point,relu=False):
    if packed.transposed or len(packed.stride) != 2:
        raise ValueError('Residual convolution requires ordinary two dimensional weights')
    return packed.storage.run(input,scale,zero_point,packed.stride,packed.padding,
        packed.dilation,packed.groups,relu,residual=accum)


def _conv_prepack(rank,weight,bias,stride,padding,dilation,groups):
    if weight.dim() != rank+2:
        raise ValueError('Convolution weight rank does not match the operator')
    return PackedConvolution(weight,bias,stride,padding,dilation,groups)


def _transpose_prepack(rank,weight,bias,stride,padding,output_padding,dilation,groups):
    if weight.dim() != rank+2:
        raise ValueError('Transposed weight rank does not match the operator')
    return PackedConvolution(weight,bias,stride,padding,dilation,groups,True,output_padding)


def _binary(name, a, b, scale=None, zero_point=None, out=None):
    if not isinstance(a,tp.Tensor):
        a,b = b,a
    return getattr(F,name)(a,b,scale,zero_point,out=out)


def _binary_out(name,a,b,out):
    return getattr(F,name)(a,b,out=out)


def _scalar(name,a,b,out=None):
    if isinstance(b,tp.Tensor):
        b=b.item()
    return getattr(F,name)(a,b,out=out)


def _inplace(result,input,inplace):
    if inplace:
        return F._result(result,input)
    return result


def _leaky(input,negative_slope,inplace,output_scale,output_zero_point):
    if inplace and (input.q_scale()!=output_scale or input.q_zero_point()!=output_zero_point):
        raise ValueError('Inplace activation requires the input affine grid')
    return _inplace(F.leaky_relu(input,negative_slope,output_scale,output_zero_point),input,inplace)


def _cat_out(inputs,dim,out,relu=False):
    value=(_C.quantized_cat_relu if relu else _C.quantized_cat)(inputs,dim,out.q_scale(),out.q_zero_point())
    return F._result(value,out)


def _rowpack(bits,weight,optimized_qparams=False,nbins=200,ratio=.16):
    return _C._quantized_rowwise_pack(weight,bits,optimized_qparams,nbins,ratio)


def _rowpack_ranges(bits,weight,rowwise_min_max,optimized_qparams=False,nbins=200,ratio=.16):
    return _C._quantized_rowwise_pack(weight,bits,optimized_qparams,nbins,ratio,rowwise_min_max)


def _embedding_bag(bits,weight,indices,offsets=None,scale_grad_by_freq=False,mode=0,
                   pruned_weights=False,per_sample_weights=None,compressed_indices_mapping=None,
                   include_last_offset=False):
    packed=weight.packed if isinstance(weight,EmbeddingPackedParams) else weight
    return _C._quantized_embedding_bag(packed,indices,bits,offsets,per_sample_weights,
        compressed_indices_mapping,include_last_offset,pruned_weights)


def _embedding(bits,weight,indices,pruned_weights=False):
    packed=weight.packed if isinstance(weight,EmbeddingPackedParams) else weight
    return _C._quantized_embedding(packed,indices,bits)


class CellParams:
    def __init__(self,wi,wh,bi=None,bh=None,reduce_range=False):
        self.wi,self.wh,self.bi,self.bh=wi,wh,bi,bh
        self.reduce_range=bool(reduce_range)

    def linear_ih(self,input):
        return _linear_dynamic(input,self.wi,self.reduce_range)

    def linear_hh(self,input):
        return _linear_dynamic(input,self.wh,self.reduce_range)

    def b_ih(self):
        return self.bi

    def b_hh(self):
        return self.bh


def _recurrent_cell(mode,input,hx,wi,wh,bi,bh):
    h,c=hx if mode==2 else (hx,None)
    result=_C._quantized_recurrent_step(_linear_dynamic(input,wi,True),
        _linear_dynamic(h,wh,True),h,c,mode)
    return result if mode==2 else result[0]


def _cell_params_float(wi,wh,bi,bh):
    return CellParams(PackedWeight(_C.quantize_per_tensor_dynamic(wi,tp.qint8,False),bi),
        PackedWeight(_C.quantize_per_tensor_dynamic(wh,tp.qint8,False),bh),bi,bh,False)


_packets = None


def _build_packets():
    entries={}
    def add(name,fn,**overloads):
        entries[name]=OperatorPacket(name,fn,**overloads)
    for name in ('add','add_relu','mul','mul_relu'):
        scalar_name=name.split('_')[0]+'_scalar'+('_relu' if name.endswith('_relu') else '')
        scalar=partial(_scalar,scalar_name)
        add(name,partial(_binary,name),out=partial(_binary_out,name),Scalar=scalar,
            Scalar2=lambda b,a,fn=scalar:fn(a,b),Scalar_out=scalar)
        add(name+'_out',partial(_binary_out,name))
        add(scalar_name,scalar,Tensor=scalar)
        add(scalar_name+'_out',scalar,Tensor=scalar)
    for name in ('sub','div'):
        add(name,getattr(F,name))
    for name in ('matmul','softmax','prelu','threshold','celu','dropout','sigmoid','tanh','hardswish'):
        add(name,getattr(F,name))
    add('elu',_C.quantized_elu)
    add('relu6',lambda input,inplace=False:_inplace(F.relu6(input),input,inplace))
    add('leaky_relu',_leaky)
    add('clamp',lambda input,min=None,max=None:_C.quantized_unary(input,input.q_scale(),
        input.q_zero_point(),7,float('-inf') if min is None else float(min),float('inf') if max is None else float(max)))
    add('cat',_C.quantized_cat);add('cat_relu',_C.quantized_cat_relu)
    add('cat_out',_cat_out);add('cat_relu_out',partial(_cat_out,relu=True))
    for name in ('layer_norm','group_norm','instance_norm','batch_norm','batch_norm_relu'):
        add(name,getattr(_C,'quantized_'+name))
    for rank in (1,2,3):
        add(f'max_pool{rank}d',getattr(_C,f'quantized_max_pool{rank}d'))
        for relu in ('','_relu'):
            add(f'batch_norm{rank}d{relu}',getattr(_C,'quantized_batch_norm'+relu))
            conv=partial(_conv,relu=bool(relu))
            add(f'conv{rank}d{relu}',conv,new=conv)
        add(f'conv{rank}d_prepack',partial(_conv_prepack,rank))
        add(f'conv{rank}d_unpack',lambda packed:packed.unpack())
        add(f'conv{rank}d_dynamic',lambda input,packed,reduce_range=False:packed.run_dynamic(input,reduce_range))
        add(f'conv_transpose{rank}d_prepack',partial(_transpose_prepack,rank))
        add(f'conv_transpose{rank}d',_conv)
        add(f'conv_transpose{rank}d_unpack',lambda packed:packed.unpack())
        add(f'conv_transpose{rank}d_dynamic',lambda input,packed,reduce_range=False:packed.run_dynamic(input,reduce_range))
        for prefix in (f'conv{rank}d',f'conv_transpose{rank}d'):
            for field in ('stride','padding','output_padding','dilation','groups','transpose'):
                attr='transposed' if field=='transpose' else field
                add(prefix+'_'+field,lambda packed,attr=attr:getattr(packed,attr))
    add('conv2d_add',_conv_add)
    add('conv2d_add_relu',partial(_conv_add,relu=True))
    add('conv_prepack',partial(_conv_prepack,2))
    add('conv_unpack',lambda packed:packed.unpack())
    add('conv2d_unpack_sizes',_conv_unpack_sizes)
    add('linear_prepack_legacy',_legacy);add('linear_prepack_fp16_legacy',_legacy)
    add('_wrapped_linear_prepack',_C._wrapped_linear_prepack)
    add('_wrapped_quantized_linear_prepacked',_C._wrapped_quantized_linear_prepacked)
    add('wrapped_quantized_linear',_C._quantized_wrapped_linear)
    add('wrapped_fbgemm_pack_gemm_matrix_fp16',_C._quantized_wrapped_half_prepack)
    add('wrapped_fbgemm_linear_fp16_weight',_C._quantized_wrapped_half_linear)
    add('int4mm_packed_weight_cpu',_C._quantized_int4_mm)
    add('linear_prepack',PackedWeight);add('linear_prepack_fp16',PackedHalfWeight)
    for suffix in ('','_fp16'):
        add('linear_unpack'+suffix,lambda packed:packed._weight_bias(),legacy=_legacy)
    add('linear',_linear);add('linear_relu',partial(_linear,relu=True))
    add('linear_leaky_relu',lambda input,packed,scale,zp,negative_slope:packed.run(
        input,scale,zp,activation=1,negative_slope=negative_slope))
    add('linear_tanh',lambda input,packed,scale,zp:packed.run(input,scale,zp,activation=2))
    add('linear_dynamic',_linear_dynamic);add('linear_relu_dynamic',partial(_linear_dynamic,relu=True))
    add('linear_dynamic_fp16',_linear_dynamic)
    add('linear_relu_dynamic_fp16',partial(_linear_dynamic,relu=True))
    add('linear_dynamic_fp16_unpacked_weight',lambda input,weight,bias:_C._quantized_fp16_linear(
        input,_C._quantized_fp16_prepack(weight),bias))
    add('linear_with_input_q_dq_qweight_dq_output_fp32',_linear_float)
    add('linear_with_input_q_dq_qweight_dq_relu_output_fp32',partial(_linear_float,relu=True))
    add('embedding_bag_prepack',EmbeddingPackedParams)
    add('embedding_bag_unpack',lambda packed:packed.logical_weight)
    for bits,name in ((8,'byte'),(4,'4bit'),(2,'2bit')):
        prefix='embedding_bag_'+name
        add(prefix+'_prepack',partial(_rowpack,bits))
        add(prefix+'_prepack_with_rowwise_min_max',partial(_rowpack_ranges,bits))
        add(prefix+'_unpack',lambda weight,bits=bits:_C._quantized_rowwise_unpack(weight,bits))
        add(prefix+'_rowwise_offsets',partial(_embedding_bag,bits))
        if bits!=2:
            add(prefix,partial(_embedding_bag,bits))
            add('embedding_'+name,partial(_embedding,bits))
    add('make_quantized_cell_params_dynamic',CellParams)
    add('make_quantized_cell_params_fp16',lambda wi,wh:CellParams(wi,wh))
    add('make_quantized_cell_params',_cell_params_float)
    for mode,name in ((0,'rnn_relu'),(1,'rnn_tanh'),(2,'lstm'),(3,'gru')):
        add('quantized_'+name+'_cell_dynamic',partial(_recurrent_cell,mode))
    return entries


def get_packet(name):
    global _packets
    if _packets is None:
        _packets=_build_packets()
    return _packets.get(name)

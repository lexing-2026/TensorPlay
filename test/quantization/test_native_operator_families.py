import io
import numpy as np
import pytest
import torch
import tensorplay as tp
from tensorplay.ao.nn.quantized import functional as F

DEVICES = ['cpu'] + (['cuda'] if tp.cuda.is_available() else [])
TYPES = [('int8', -11), ('uint8', 119), ('int32', -11)]


def pair(data, dtype, device, scale=.07, zp=None):
    zp = (119 if dtype == 'uint8' else -11) if zp is None else zp
    codes = np.asarray(data).astype(dtype)
    return (tp._C._make_per_tensor_quantized_tensor(tp.tensor(codes, device=device), scale, zp),
            torch._make_per_tensor_quantized_tensor(torch.tensor(codes), scale, zp))


def same(actual, expected, tolerance=0):
    assert tuple(actual.shape) == tuple(expected.shape)
    assert actual.q_scale() == pytest.approx(expected.q_scale(), rel=1e-12)
    assert actual.q_zero_point() == expected.q_zero_point()
    np.testing.assert_allclose(actual.int_repr().cpu().numpy().astype('int64'),
                               expected.int_repr().numpy().astype('int64'), atol=tolerance, rtol=0)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('channel',[False,True])
@pytest.mark.parametrize('strided',[False,True])
def test_quantized_copy_transfers_source_grid(device,channel,strided):
    values=np.arange(30,dtype='float32').reshape(5,6)*.03
    if channel:
        scales=np.array([.02,.04,.06,.08,.1],dtype='float64')
        zps=np.array([-2,-1,0,1,2],dtype='int64')
        source=tp.quantize_per_channel(tp.tensor(values,device=device),
            tp.tensor(scales,device=device),tp.tensor(zps,device=device),0,tp.qint8)
        destination=tp.quantize_per_channel(tp.zeros((5,6),device=device),
            tp.ones(5,device=device,dtype=tp.float64),tp.zeros(5,device=device,dtype=tp.int64),0,tp.qint8)
    else:
        source=tp.quantize_per_tensor(tp.tensor(values,device=device),.03,-3,tp.qint8)
        destination=tp.quantize_per_tensor(tp.zeros((5,6),device=device),1.,0,tp.qint8)
    if strided:
        source=source[:,::2]
        destination=destination[:,::2]
    address=destination.data_ptr()
    destination.copy_(source)
    assert destination.data_ptr()==address
    np.testing.assert_array_equal(destination.int_repr().cpu().numpy(),source.int_repr().cpu().numpy())
    np.testing.assert_array_equal(destination.dequantize().cpu().numpy(),source.dequantize().cpu().numpy())


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('features',[1,31,32,33,8193])
@pytest.mark.parametrize('reduce_range',[False,True])
def test_dynamic_linear_pair_saturation_and_chunk_tails(device,features,reduce_range):
    values=np.linspace(-1,7,3*features,dtype='float32').reshape(3,features)
    codes=np.tile(np.array([[127],[-128],[101]],dtype='int8'),(1,features))
    weight,reference=pair(codes,'int8',device,scale=.015,zp=0)
    bias=np.array([.3,-.4,.5],dtype='float32')
    packed=tp.ops.quantized.linear_prepack(weight,tp.tensor(bias,device=device))
    rp=torch.ops.quantized.linear_prepack(reference,torch.tensor(bias))
    actual=tp.ops.quantized.linear_dynamic(tp.tensor(values,device=device),packed,reduce_range)
    expected=torch.ops.quantized.linear_dynamic(torch.tensor(values),rp,reduce_range)
    np.testing.assert_allclose(actual.cpu().numpy(),expected.numpy(),rtol=0,atol=1e-6)
    grid=tp._C.quantize_per_tensor_dynamic(tp.tensor(values,device=device),tp.quint8,reduce_range)
    q=grid.int_repr().cpu().numpy().astype('int64')
    w=codes.astype('int64')
    accumulator=np.zeros((3,3),dtype='int64')
    for k in range(0,features,2):
        pair_sum=q[:,k,None]*w[None,:,k]
        if k+1<features: pair_sum+=q[:,k+1,None]*w[None,:,k+1]
        accumulator+=np.clip(pair_sum,-32768,32767)
    accumulator-=grid.q_zero_point()*w.sum(axis=1)
    factor=np.float32(np.float32(grid.q_scale())*np.float32(.015))
    exact=(accumulator.astype('float32').astype('float64')*np.float64(factor)+bias.astype('float64')).astype('float32')
    np.testing.assert_array_equal(actual.cpu().numpy(),exact)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype,zp',TYPES[:2])
@pytest.mark.parametrize('relu',[False,True])
@pytest.mark.parametrize('weight_zp',[0,-3])
@pytest.mark.parametrize('rows',[1,2,4,32])
@pytest.mark.parametrize('outputs,features',[(9,33),(128,512),(129,1025)])
def test_large_integer_linear_affine_correction_and_weight_mutation(device,dtype,zp,relu,weight_zp,rows,outputs,features):
    from tensorplay.ao.nn.quantized.packed import PackedWeight
    rng=np.random.default_rng(981)
    values=rng.integers(-110,111,size=(rows,features)).astype(dtype)
    x,_=pair(values,dtype,device,scale=.03,zp=zp)
    biases=rng.normal(0,.3,size=outputs).astype('float32')
    weight_codes=rng.integers(-128,128,size=(outputs,features),dtype='int8')
    weight,_=pair(weight_codes,'int8',device,scale=.02,zp=weight_zp)
    packed=PackedWeight(weight,tp.tensor(biases,device=device))
    for iteration in range(2):
        if iteration:
            weight_zp=-3 if weight_zp==0 else 0
            weight_codes=rng.integers(-128,128,size=(outputs,features),dtype='int8')
            weight,_=pair(weight_codes,'int8',device,scale=.02,zp=weight_zp)
            replacement=PackedWeight(weight,tp.tensor(biases,device=device))
            packed.packed.copy_(replacement.packed)
            packed.offsets.copy_(replacement.offsets)
            packed.zero_points.copy_(replacement.zero_points)
        actual=packed.run(x,.2,zp,relu=relu)
        accumulator=(values.astype('int64')-zp)@(weight_codes.astype('int64')-weight_zp).T
        real=accumulator*.03*np.float64(np.float32(.02))+biases
        if relu: real=np.maximum(real,0.)
        lo=np.iinfo(dtype).min;hi=np.iinfo(dtype).max
        expected=np.clip(np.rint(real/.2)+zp,lo,hi).astype(dtype)
        np.testing.assert_array_equal(actual.int_repr().cpu().numpy(),expected)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype,zp',[('int8',127),('uint8',255)])
@pytest.mark.parametrize('features',[65535,65536,65537])
def test_integer_linear_affine_accumulator_exceeds_signed_word(device,dtype,zp,features):
    from tensorplay.ao.nn.quantized.packed import PackedWeight
    codes=np.full((8,features),np.iinfo(dtype).min,dtype=dtype)
    weight_codes=np.full((5,features),-128,dtype='int8')
    x,_=pair(codes,dtype,device,scale=.01,zp=zp)
    weight,_=pair(weight_codes,'int8',device,scale=.01,zp=127)
    packed=PackedWeight(weight)
    actual=packed.run(x,1e4,0)
    acc=(codes.astype('int64')-zp)@(weight_codes.astype('int64')-127).T
    assert np.all(acc>np.iinfo('int32').max)
    expected=np.clip(np.rint(acc*.01*np.float64(np.float32(.01))/1e4),
                     np.iinfo(dtype).min,np.iinfo(dtype).max).astype(dtype)
    np.testing.assert_array_equal(actual.int_repr().cpu().numpy(),expected)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('value', [-100., -.3, 0., .3, 100.])
@pytest.mark.parametrize('operation', ['add_scalar', 'add_scalar_relu', 'mul_scalar', 'mul_scalar_relu'])
def test_scalar_metadata_and_codes(device, dtype, zp, value, operation):
    codes = np.linspace(np.iinfo(dtype).min, np.iinfo(dtype).max, 97).astype(dtype)
    x, r = pair(codes, dtype, device, zp=zp)
    y = getattr(F, operation)(x, value)
    expected = getattr(torch.ops.quantized, operation)(r, value)
    same(y, expected, 256 if dtype == 'int32' and operation.startswith('add') else 0)
    if device == 'cpu':
        obj = tp.ao.nn.quantized.QFunctional()
        obj.scale, obj.zero_point = .2, 7
        if operation in ('add_scalar', 'mul_scalar'):
            same(getattr(obj, operation)(x, value), expected, 256 if dtype == 'int32' else 0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('operation', ['add', 'mul', 'add_relu', 'mul_relu'])
@pytest.mark.parametrize('layout', ['flat', 'broadcast', 'strided', 'empty'])
def test_binary_grids_broadcast_and_layout(device, dtype, zp, operation, layout):
    a = np.arange(74).reshape(2,37)
    b = np.arange(74, 148).reshape(2,37)
    if layout == 'broadcast': b = b[:1]
    if layout == 'empty': a, b = a[:0], b[:1]
    x,r = pair(a,dtype,device,scale=.03,zp=zp)
    z,s = pair(b,dtype,device,scale=.05,zp=zp)
    if layout == 'strided': x,r,z,s = x.T,r.T,z.T,s.T
    same(getattr(F,operation)(x,z,.08,zp),getattr(torch.ops.quantized,operation)(r,s,.08,zp),1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('op', ['threshold', 'gelu', 'gelu_tanh', 'celu', 'clamp', 'prelu', 'softmax'])
def test_unary_and_axis_operations(device, dtype, zp, op):
    x,r = pair(np.arange(222).reshape(2,3,37),dtype,device,zp=zp)
    x,r = x.transpose(1,2), r.transpose(1,2)
    if op == 'threshold': y,e = F.threshold(x,.21,-.7),torch.ops.quantized.threshold(r,.21,-.7)
    elif op == 'gelu': y,e = F.gelu(x),torch.nn.functional.gelu(r)
    elif op == 'gelu_tanh': y,e = F.gelu(x,'tanh'),torch.nn.functional.gelu(r,approximate='tanh')
    elif op == 'celu': y,e = F.celu(x,.06,zp,1.3),torch.ops.quantized.celu(r,.06,zp,1.3)
    elif op == 'clamp': y,e = F.clamp(x,.07,zp,min=-.2,max=.8),torch.clamp(r,-.2,.8)
    elif op == 'prelu':
        w,s = pair(np.arange(37)+100,dtype,device,scale=.01,zp=zp)
        y,e = F.prelu(x,w,.05,zp),torch.ops.quantized.prelu(r,s,.05,zp)
    else: y,e = F.softmax(x,1,.003,zp),torch.ops.quantized.softmax(r,1,.003,zp)
    same(y,e,1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES[:2])
@pytest.mark.parametrize('shapes', [((7,), (7,)), ((7,), (2,7,5)), ((3,7), (7,)),
                                   ((2,1,3,7), (1,4,7,5)), ((0,3,7), (7,5)), ((3,0), (0,5))])
def test_matmul_rank_and_batch_broadcast(device,dtype,zp,shapes):
    a=np.arange(np.prod(shapes[0])).reshape(shapes[0])%97
    b=np.arange(np.prod(shapes[1])).reshape(shapes[1])%89
    x,r=pair(a,dtype,device,zp=zp);z,s=pair(b,dtype,device,scale=.03,zp=zp)
    same(F.matmul(x,z,.08,zp),torch.ops.quantized.matmul(r,s,.08,zp),1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('rank', [1,2,3])
@pytest.mark.parametrize('batched', [False,True])
@pytest.mark.parametrize('kind', ['avg', 'adaptive', 'max'])
def test_pooling_all_grids(device,dtype,zp,rank,batched,kind):
    shape=((2,3) if batched else (3,))+(5,)*rank
    x,r=pair(np.arange(np.prod(shape)).reshape(shape)%201,dtype,device,zp=zp)
    x,r=x.transpose(-1,-2) if rank>1 else x, r.transpose(-1,-2) if rank>1 else r
    if kind=='avg':
        y=getattr(F,f'avg_pool{rank}d')(x,3,2,1,True,False)
        e=getattr(torch.nn.functional,f'avg_pool{rank}d')(
            r.unsqueeze(0) if rank==3 and not batched else r,3,2,1,True,False)
        if rank==3 and not batched: e=e.squeeze(0)
    elif kind=='adaptive':
        y=getattr(F,f'adaptive_avg_pool{rank}d')(x,(3,)*rank)
        e=getattr(torch.nn.functional,f'adaptive_avg_pool{rank}d')(r,(3,)*rank)
    else:
        y=getattr(tp._C,f'quantized_max_pool{rank}d')(x,[3]*rank,[2]*rank,[1]*rank,[2]*rank,True)
        e=getattr(torch.nn.functional,f'max_pool{rank}d')(
            r.unsqueeze(0) if rank==3 and not batched else r,3,2,1,2,True)
        if rank==3 and not batched: e=e.squeeze(0)
    same(y,e,1 if kind!='max' else 0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
def test_cat_empty_and_differing_grids(device,dtype,zp):
    x,r=pair(np.arange(74).reshape(2,37),dtype,device,zp=zp)
    z,s=pair(np.arange(74).reshape(2,37),dtype,device,scale=.031,zp=zp)
    x,r,z,s=x.T,r.T,z.T,s.T
    for fn in ('cat','cat_relu'):
        y=getattr(tp._C,'quantized_'+fn)([x,z],1,.05,zp)
        e=getattr(torch.ops.quantized,fn)([r,s],1,.05,zp)
        same(y,e,1)
    assert F.dropout(x,.1,zp,1.,True).data_ptr() == x.data_ptr()


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('relu',[False,True])
def test_half_dynamic_linear_storage_and_accumulation(device,relu):
    rng=np.random.default_rng(71)
    w=rng.normal(size=(23,53)).astype('float32')
    w[0,:2]=[70000.,-70000.]
    x=rng.normal(size=(2,3,53)).astype('float32');b=rng.normal(size=23).astype('float32')
    packed=tp._C._quantized_fp16_prepack(tp.tensor(w,device=device))
    assert packed.dtype==tp.float16
    expected=x@np.clip(w,-65504,65504).astype('float16').astype('float32').T+b
    if relu: expected=np.maximum(expected,0)
    y=tp._C._quantized_fp16_linear(tp.tensor(x,device=device),packed,tp.tensor(b,device=device),relu)
    np.testing.assert_allclose(y.cpu().numpy(),expected,rtol=2e-5,atol=.015)


@pytest.mark.parametrize('device',DEVICES)
def test_half_dynamic_conversion_and_checkpoint(device):
    model=tp.nn.Sequential(tp.nn.Linear(17,11)).to(device)
    quantized=tp.ao.quantization.quantize_dynamic(model,{tp.nn.Linear},dtype=tp.float16)
    x=tp.randn(2,17,device=device)
    y=quantized(x)
    assert quantized[0].weight.dtype==tp.float16
    other=tp.ao.quantization.quantize_dynamic(model,{tp.nn.Linear},dtype=tp.float16)
    other.load_state_dict(quantized.state_dict())
    np.testing.assert_array_equal(other(x).cpu().numpy(),y.cpu().numpy())


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('bits',[8,4,2])
@pytest.mark.parametrize('optimized',[False,True])
def test_rowwise_pack_unpack_and_sparse_bags(device,bits,optimized):
    rng=np.random.default_rng(12);w=rng.normal(size=(11,36)).astype('float32')
    w[0]=.3;w[1]=0
    packed=F.rowwise_prepack(tp.tensor(w,device=device),bits,optimized)
    name='byte' if bits==8 else f'{bits}bit'
    args=(torch.tensor(w),) if bits==8 else (torch.tensor(w),optimized,200,.16)
    rp=getattr(torch.ops.quantized,f'embedding_bag_{name}_prepack')(*args)
    np.testing.assert_array_equal(packed.cpu().numpy(),rp.numpy())
    unpacked=F.rowwise_unpack(packed,bits)
    expected=getattr(torch.ops.quantized,f'embedding_bag_{name}_unpack')(rp)
    np.testing.assert_allclose(unpacked.cpu().numpy(),expected.numpy(),rtol=0,atol=1e-6)
    ind=np.array([0,4,2,2,7,10],dtype='int64');off=np.array([0,0,3,6],dtype='int32')
    weights=np.array([.2,.4,-1.,1.,.3,.7],dtype='float32')
    actual=F.embedding_bag(packed,tp.tensor(ind,device=device),tp.tensor(off,device=device),bits,
        per_sample_weights=tp.tensor(weights,device=device),include_last_offset=True)
    e=getattr(torch.ops.quantized,f'embedding_bag_{name}_rowwise_offsets')(
        rp,torch.tensor(ind),torch.tensor(off),False,0,False,torch.tensor(weights),None,True)
    np.testing.assert_allclose(actual.cpu().numpy(),e.numpy(),atol=2e-6,rtol=1e-6)
    mapping=np.arange(11,dtype='int32');mapping[2]=-1
    actual=F.embedding_bag(packed,tp.tensor(ind,device=device),tp.tensor(off,device=device),bits,
        pruned_weights=True,compressed_indices_mapping=tp.tensor(mapping,device=device),include_last_offset=True)
    e=getattr(torch.ops.quantized,f'embedding_bag_{name}_rowwise_offsets')(
        rp,torch.tensor(ind),torch.tensor(off),False,0,True,None,torch.tensor(mapping),True)
    np.testing.assert_allclose(actual.cpu().numpy(),e.numpy(),atol=2e-6,rtol=1e-6)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('bits',[2,4])
def test_rowwise_half_metadata_rounding_and_byte_order(device,bits):
    values=np.array([0.,.3,-.3,.1001,-.1001,2**-24,2**-14,
        2**-14-2**-24,1.+2**-11,1.+3*2**-11,65504.,-65504.],dtype='float32')
    weights=np.repeat(values[:,None],8,axis=1)
    packed=F.rowwise_prepack(tp.tensor(weights,device=device),bits)
    expected=getattr(torch.ops.quantized,f'embedding_bag_{bits}bit_prepack')(torch.tensor(weights))
    np.testing.assert_array_equal(packed.cpu().numpy(),expected.numpy())


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('batched',[False,True])
def test_native_max_pool2d_expands_singleton_kernel(device,batched):
    shape=(2,3,5,7) if batched else (3,5,7)
    x,r=pair(np.arange(np.prod(shape)).reshape(shape)%97,'int8',device)
    actual=tp._C.quantized_max_pool2d(x,[2],[],[0],[1],False)
    expected=torch.nn.functional.max_pool2d(r,2)
    same(actual,expected)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('rank',[1,2,3])
@pytest.mark.parametrize('groups',[1,2])
def test_transposed_convolution_geometry(device,rank,groups):
    rng=np.random.default_rng(41);x=rng.normal(size=(2,4)+(3,)*rank).astype('float32')
    w=rng.normal(size=(4,3)+(2,)*rank).astype('float32');b=rng.normal(size=3*groups).astype('float32')
    qx=tp.quantize_per_tensor(tp.tensor(x,device=device),.05,117,tp.quint8)
    qw=tp.quantize_per_tensor(tp.tensor(w,device=device),.03,-5,tp.qint8)
    y=tp._C.quantized_convolution(qx,qw,tp.tensor(b,device=device),.07,119,
        [2]*rank,[1]*rank,[2]*rank,groups,True,[1]*rank)
    rx=torch.quantize_per_tensor(torch.tensor(x),.05,117,torch.quint8)
    rw=torch.quantize_per_tensor(torch.tensor(w),.03,-5,torch.qint8)
    real=getattr(torch.nn.functional,f'conv_transpose{rank}d')(rx.dequantize(),rw.dequantize(),
        torch.tensor(b),stride=2,padding=1,output_padding=1,groups=groups,dilation=2)
    expected=torch.quantize_per_tensor(real,.07,119,torch.quint8)
    same(y,expected,1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('operation', ['add_scalar', 'add_scalar_relu', 'mul_scalar', 'mul_scalar_relu'])
def test_scalar_out_replaces_grid_preserves_storage(device, dtype, zp, operation):
    x,r = pair(np.arange(74).reshape(2,37), dtype, device, zp=zp)
    out,ref = pair(np.zeros((37,2)), dtype, device, scale=.3, zp=zp)
    out,ref = out.T, ref.T
    pointer = out.data_ptr()
    value = -7.3
    result = getattr(F,operation)(x,value,out=out)
    expected = getattr(torch.ops.quantized,operation+'_out')(r,value,ref)
    assert result.data_ptr() == pointer and out.data_ptr() == pointer
    same(out,expected,1)
    same(result,expected,1)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dtype,zp',TYPES)
@pytest.mark.parametrize('rank,mode',[(2,'nearest'),(2,'nearest-exact'),(2,'bilinear'),
                                    (3,'nearest'),(3,'nearest-exact')])
@pytest.mark.parametrize('layout',['contiguous','strided','channels_last'])
@pytest.mark.parametrize('sizing',['size','scale','recomputed'])
def test_resize_integer_storage_and_layout(device,dtype,zp,rank,mode,layout,sizing):
    shape=(2,37)+(5,7) if rank==2 else (2,5,3,5,7)
    x,r=pair(np.arange(np.prod(shape)).reshape(shape)%173,dtype,device,zp=zp)
    if layout=='strided': x,r=x.transpose(-1,-2),r.transpose(-1,-2)
    if layout=='channels_last':
        x=x.contiguous(memory_format=tp.channels_last if rank==2 else tp.channels_last_3d)
        r=r.contiguous(memory_format=torch.channels_last if rank==2 else torch.channels_last_3d)
    args={'size':tuple(9-d for d in range(rank))} if sizing=='size' else {
        'scale_factor':tuple(1.73+.1*d for d in range(rank)),
        'recompute_scale_factor':sizing=='recomputed'}
    if mode=='bilinear': args['align_corners']=False
    y=tp.nn.functional.interpolate(x,mode=mode,**args)
    expected=torch.nn.functional.interpolate(r,mode=mode,**args)
    same(y,expected,1 if mode=='bilinear' else 0)
    if layout=='channels_last':
        assert y.is_contiguous(memory_format=tp.channels_last if rank==2 else tp.channels_last_3d)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('kind',['LSTM','GRU'])
@pytest.mark.parametrize('dtype',['int8','half'])
@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('bidirectional',[False,True])
def test_recurrent_multilayer_states_and_packed_order(device,kind,dtype,packed,bidirectional):
    rng=np.random.default_rng(172)
    options=dict(num_layers=2,batch_first=True,bidirectional=bidirectional)
    native=getattr(tp.ao.nn.quantized.dynamic,kind)(7,11,**options,
        dtype=tp.qint8 if dtype=='int8' else tp.float16).to(device).eval()
    reference=getattr(torch.ao.nn.quantized.dynamic,kind)(7,11,**options,
        dtype=torch.qint8 if dtype=='int8' else torch.float16).eval()
    parameters={}
    for name in native._weight_names:
        values=rng.normal(0,.15,size=tuple(getattr(native,name).shape)).astype('float32')
        if dtype=='int8':
            codes=np.clip(np.rint(values/.005),-128,127).astype('int8')
            weight,ref=pair(codes,'int8',device,scale=.005,zp=0)
        else:
            weight=tp._C._quantized_fp16_prepack(tp.tensor(values,device=device))
            ref=torch.tensor(values.astype('float16').astype('float32'))
        setattr(native,name,weight)
        parameters[name]=ref
    for name in native._bias_names:
        values=rng.normal(0,.05,size=tuple(getattr(native,name).shape)).astype('float32')
        setattr(native,name,tp.tensor(values,device=device));parameters[name]=torch.tensor(values)
    reference.set_weight_bias(parameters)
    values=rng.normal(size=(3,5,7)).astype('float32')
    x,rx=tp.tensor(values,device=device),torch.tensor(values)
    if packed:
        lengths=[2,5,3]
        x=tp.nn.utils.rnn.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        rx=torch.nn.utils.rnn.pack_padded_sequence(rx,lengths,batch_first=True,enforce_sorted=False)
    shape=(2*(2 if bidirectional else 1),3,11)
    values=rng.normal(size=shape).astype('float32')
    h,rh=tp.tensor(values,device=device),torch.tensor(values)
    if kind=='LSTM':
        values=rng.normal(size=shape).astype('float32')
        h,rh=(h,tp.tensor(values,device=device)),(rh,torch.tensor(values))
    y,state=native(x,h);expected,refstate=reference(rx,rh)
    actual=y.data if packed else y;expected=expected.data if packed else expected
    np.testing.assert_allclose(actual.cpu().numpy(),expected.detach().numpy(),atol=3e-5,rtol=3e-5)
    for actual,expected in zip(state if kind=='LSTM' else (state,),refstate if kind=='LSTM' else (refstate,)):
        np.testing.assert_allclose(actual.cpu().numpy(),expected.detach().numpy(),atol=3e-5,rtol=3e-5)
    other=getattr(tp.ao.nn.quantized.dynamic,kind)(7,11,**options,
        dtype=tp.qint8 if dtype=='int8' else tp.float16).to(device).eval()
    other.load_state_dict(native.state_dict())
    result,_=other(x,h)
    np.testing.assert_array_equal((result.data if packed else result).cpu().numpy(),(y.data if packed else y).cpu().numpy())


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('rank',[1,2,3])
@pytest.mark.parametrize('transposed',[False,True])
@pytest.mark.parametrize('groups',[1,2])
def test_dynamic_convolution_requantizes_input_grid(device,rank,transposed,groups):
    rng=np.random.default_rng(512)
    name=('ConvTranspose' if transposed else 'Conv')+f'{rank}d'
    options=dict(stride=2,padding=1,dilation=2,groups=groups)
    if transposed: options['output_padding']=1
    native=getattr(tp.ao.nn.quantized.dynamic,name)(4,6,2,**options).to(device)
    shape=(4,6//groups)+(2,)*rank if transposed else (6,4//groups)+(2,)*rank
    weight=rng.integers(-37,38,size=shape,dtype='int8')
    qw,rw=pair(weight,'int8',device,scale=.02,zp=-3)
    bias=rng.normal(0,.1,size=6).astype('float32')
    native.set_weight_bias(qw,tp.tensor(bias,device=device))
    x=rng.normal(size=(2,4)+(5,)*rank).astype('float32')
    packed_name=('conv_transpose' if transposed else 'conv')+f'{rank}d_prepack'
    arguments=[rw,torch.tensor(bias),[2]*rank,[1]*rank]
    if transposed: arguments.append([1]*rank)
    arguments.extend([[2]*rank,groups])
    packed=getattr(torch.ops.quantized,packed_name)(*arguments)
    expected=getattr(torch.ops.quantized,('conv_transpose' if transposed else 'conv')+f'{rank}d_dynamic')(
        torch.tensor(x),packed,True)
    result=native(tp.tensor(x,device=device))
    grid=tp._C.quantize_per_tensor_dynamic(tp.tensor(x,device=device),tp.quint8,True)
    np.testing.assert_allclose(result.cpu().numpy(),expected.numpy(),atol=grid.q_scale()+1e-6,rtol=0)
    second=getattr(tp.ao.nn.quantized.dynamic,name)(4,6,2,**options).to(device)
    second.load_state_dict(native.state_dict())
    np.testing.assert_array_equal(second(tp.tensor(x,device=device)).cpu().numpy(),result.cpu().numpy())


@pytest.mark.parametrize('device',DEVICES)
def test_public_operator_packets_and_packed_queries(device):
    x,r=pair(np.arange(74).reshape(2,37),'uint8',device,zp=119)
    for name in ('add','add_relu','mul','mul_relu'):
        fn=getattr(tp.ops.quantized,name);ref=getattr(torch.ops.quantized,name)
        same(fn(x,x,.2,119),ref(r,r,.2,119),1)
        same(fn.Scalar(x,-3.7),ref.Scalar(r,-3.7),1)
        same(fn.Scalar2(-3.7,x),ref.Scalar2(-3.7,r),1)
        out,_=pair(np.zeros((2,37)),'uint8',device,scale=.13,zp=87)
        result=fn.out(x,x,out)
        assert result.data_ptr()==out.data_ptr()
        same(out,ref(r,r,.13,87),1)
    w,rw=pair(np.arange(11*37).reshape(11,37)%73,'int8',device,scale=.03,zp=-2)
    b=tp.zeros(11,device=device)
    packed=tp.ops.quantized.linear_prepack(w,b)
    recovered,bias=tp.ops.quantized.linear_unpack(packed)
    rp=torch.ops.quantized.linear_prepack(rw,torch.zeros(11))
    reference_weight,reference_bias=torch.ops.quantized.linear_unpack(rp)
    same(recovered,reference_weight)
    same(tp.ops.quantized.linear(x,packed,.07,119),torch.ops.quantized.linear(
        r,rp,.07,119),1)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('operation', ['linear_leaky_relu', 'linear_tanh'])
@pytest.mark.parametrize('dtype,zp', TYPES[:2])
@pytest.mark.parametrize('weight_zp', [-3,0])
@pytest.mark.parametrize('rows,outputs,features',[(3,11,37),(32,127,513)])
def test_packed_linear_fused_activation(device, operation, dtype, zp,weight_zp,rows,outputs,features):
    rng = np.random.default_rng(110)
    x, r = pair(rng.integers(0, 70, size=(rows, features)), dtype, device, scale=.05, zp=zp)
    w, rw = pair(rng.integers(-60, 60, size=(outputs, features)), 'int8', device, scale=.03, zp=weight_zp)
    bias = rng.normal(0, .1, size=outputs).astype('float32')
    packed = tp.ops.quantized.linear_prepack(w, tp.tensor(bias, device=device))
    args = (.07, zp, .17) if operation.endswith('leaky_relu') else (.07, zp)
    actual = getattr(tp.ops.quantized, operation)(x, packed, *args)
    real = torch.nn.functional.linear(r.dequantize(), rw.dequantize(), torch.tensor(bias))
    real = torch.nn.functional.leaky_relu(real, .17) if operation.endswith('leaky_relu') else torch.tanh(real)
    expected = torch.quantize_per_tensor(real, .07, zp, r.dtype)
    same(actual, expected, 1)
    if dtype == 'uint8' and weight_zp==0:
        saved = torch.backends.quantized.engine
        try:
            torch.backends.quantized.engine = 'onednn'
            rp = torch.ops.quantized.linear_prepack(rw, torch.tensor(bias))
            same(actual, getattr(torch.ops.quantized, operation)(r, rp, *args), 1)
        finally:
            torch.backends.quantized.engine = saved


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('relu', [False, True])
@pytest.mark.parametrize('channels_last', [False, True])
def test_packed_convolution_fused_residual(device, relu, channels_last):
    rng = np.random.default_rng(118)
    x, r = pair(rng.integers(0, 100, size=(2, 4, 5, 6)), 'uint8', device, scale=.03, zp=51)
    w, rw = pair(rng.integers(-20, 20, size=(6, 2, 3, 3)), 'int8', device, scale=.02, zp=0)
    accum, ra = pair(rng.integers(0, 150, size=(2, 6, 5, 6)), 'uint8', device, scale=.08, zp=75)
    if channels_last:
        accum = accum.contiguous(memory_format=tp.channels_last)
        ra = ra.contiguous(memory_format=torch.channels_last)
    pointer = accum.data_ptr()
    packed = tp.ops.quantized.conv2d_prepack(w, None, [1, 1], [1, 1], [1, 1], 2)
    name = 'conv2d_add_relu' if relu else 'conv2d_add'
    real = torch.nn.functional.conv2d(r.dequantize(), rw.dequantize(), padding=1, groups=2) + ra.dequantize()
    if relu:
        real = real.relu()
    expected = torch.quantize_per_tensor(real, .06, 63, torch.quint8)
    actual = getattr(tp.ops.quantized, name)(x, accum, packed, .06, 63)
    same(actual, expected, 1)
    assert actual.is_contiguous(memory_format=tp.channels_last)
    if channels_last:
        assert actual.data_ptr() == pointer
        assert accum.q_scale() == .06 and accum.q_zero_point() == 63
    saved = torch.backends.quantized.engine
    try:
        torch.backends.quantized.engine = 'onednn'
        rp = torch.ops.quantized.conv2d_prepack(rw, None, [1, 1], [1, 1], [1, 1], 2)
        same(actual, getattr(torch.ops.quantized, name)(r, ra, rp, .06, 63), 1)
    finally:
        torch.backends.quantized.engine = saved


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,zp', TYPES)
@pytest.mark.parametrize('rank', [2, 3, 4, 5])
@pytest.mark.parametrize('relu', [False, True])
@pytest.mark.parametrize('empty', [False, True])
def test_batch_norm_all_grids_ranks_and_empty(device, dtype, zp, rank, relu, empty):
    rng = np.random.default_rng(120)
    shape = (0 if empty else 2, 37) + (3,) * (rank-2)
    x, r = pair(rng.integers(0, 200, size=shape), dtype, device, scale=.03, zp=zp)
    params = [rng.normal(0, .2, size=37).astype('float32') for _ in range(3)]
    params.append(rng.uniform(.1, 2, size=37).astype('float32'))
    native = [tp.tensor(v, device=device) for v in params]
    reference = [torch.tensor(v) for v in params]
    dimensions = 1 if rank < 4 else rank-2
    name = f'batch_norm{dimensions}d' + ('_relu' if relu else '')
    actual = getattr(tp.ops.quantized, name)(x, *native, 1e-5, .07, zp)
    expected = getattr(torch.ops.quantized, name)(r, *reference, 1e-5, .07, zp)
    same(actual, expected, 1)
    if not empty and rank >= 4:
        assert actual.is_contiguous(memory_format=tp.channels_last if rank == 4 else tp.channels_last_3d)
    with pytest.raises(Exception):
        getattr(tp.ops.quantized, name)(x, None, native[1], native[2], native[3], 1e-5, .07, zp)


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
@pytest.mark.parametrize('channels', [16, 32, 48, 64, 80, 128, 256])
@pytest.mark.parametrize('group', [32, 64, 128, 256])
@pytest.mark.parametrize('rows', [1, 5, 8, 16, 32, 33])
def test_native_int4_packed_layout_and_multiply(dtype, channels, group, rows):
    rng = np.random.default_rng(124)
    features = group*2
    codes = rng.integers(0, 16, size=(channels, features), dtype='int32')
    packed = tp._C._convert_weight_to_int4pack_for_cpu(tp.tensor(codes), 2)
    expected_packed = torch._convert_weight_to_int4pack_for_cpu(torch.tensor(codes), 2)
    np.testing.assert_array_equal(packed.numpy(), expected_packed.numpy())
    data = rng.normal(0, .2, size=(rows, features)).astype('float32')
    params = rng.uniform(-.05, .05, size=(2, channels, 2)).astype('float32')
    x = tp.tensor(data, dtype=getattr(tp, dtype))
    r = torch.tensor(data, dtype=getattr(torch, dtype))
    p = tp.tensor(params, dtype=getattr(tp, dtype))
    rp = torch.tensor(params, dtype=getattr(torch, dtype))
    actual = tp.ops.quantized.int4mm_packed_weight_cpu(x, packed, tp.tensor(group), p)
    expected = torch.ops.quantized.int4mm_packed_weight_cpu(r, expected_packed, torch.tensor(group), rp)
    atol, rtol = {'float32': (2e-5, 2e-5), 'float16': (2e-4, 2e-3), 'bfloat16': (2e-3, 2e-2)}[dtype]
    np.testing.assert_allclose(actual.float().numpy(), expected.float().numpy(), atol=atol, rtol=rtol)


def test_wrapped_native_integer_linear_and_persistent_payload():
    rng = np.random.default_rng(125)
    x = rng.uniform(-1, 1, size=(3, 37)).astype('float32')
    w = rng.uniform(-1, 1, size=(11, 37)).astype('float32')
    b = rng.normal(0, .1, size=11).astype('float32')
    native = [tp.tensor(v) for v in (x, .03, 61, w, .03, 0, b, .07, 117)]
    reference = [torch.tensor(v) for v in (x, .03, 61, w, .03, 0, b, .07, 117)]
    actual = tp.ops._quantized.wrapped_quantized_linear(*native, 11)
    expected = torch.ops._quantized.wrapped_quantized_linear(*reference, 11)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=.07, rtol=0)
    packed = tp.ops._quantized._wrapped_linear_prepack(*native[3:7])
    assert isinstance(packed, tp.Tensor) and packed.dtype == tp.uint8
    stream = io.BytesIO(); tp.save(packed, stream); stream.seek(0)
    loaded = tp.load(stream, weights_only=False)
    output = tp.ops._quantized._wrapped_quantized_linear_prepacked(*native[:3], loaded, *native[7:], 11)
    np.testing.assert_array_equal(output.numpy(), actual.numpy())


def test_wrapped_native_half_linear_and_legacy_rejection():
    rng = np.random.default_rng(126)
    x = rng.normal(0, .2, size=(3, 37)).astype('float32')
    w = rng.normal(0, .2, size=(11, 37)).astype('float32')
    b = rng.normal(0, .1, size=11).astype('float32')
    packed = tp.ops._quantized.wrapped_fbgemm_pack_gemm_matrix_fp16(tp.tensor(w))
    rp = torch.ops._quantized.wrapped_fbgemm_pack_gemm_matrix_fp16(torch.tensor(w))
    actual = tp.ops._quantized.wrapped_fbgemm_linear_fp16_weight(tp.tensor(x), packed, tp.tensor(b), 11)
    expected = torch.ops._quantized.wrapped_fbgemm_linear_fp16_weight(torch.tensor(x), rp, torch.tensor(b), 11)
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=2e-5, rtol=2e-5)
    for name in ('linear_prepack_legacy', 'linear_prepack_fp16_legacy'):
        with pytest.raises(RuntimeError):
            getattr(tp.ops.quantized, name)(tp.tensor(w))
    for name in ('linear_unpack', 'linear_unpack_fp16'):
        with pytest.raises(RuntimeError):
            getattr(tp.ops.quantized, name).legacy(packed)


@pytest.mark.parametrize('mode,name,gates', [(0, 'rnn_relu', 1), (1, 'rnn_tanh', 1), (2, 'lstm', 4), (3, 'gru', 3)])
def test_dynamic_cell_uses_persistent_packed_parameters(mode, name, gates):
    rng = np.random.default_rng(127)
    x = rng.normal(0, .2, size=(3, 7)).astype('float32')
    h = rng.normal(0, .2, size=(3, 5)).astype('float32')
    c = rng.normal(0, .2, size=(3, 5)).astype('float32')
    wi, ri = pair(rng.integers(-50, 51, size=(gates*5, 7)), 'int8', 'cpu', scale=.005, zp=0)
    wh, rh = pair(rng.integers(-50, 51, size=(gates*5, 5)), 'int8', 'cpu', scale=.005, zp=0)
    bi = rng.normal(0, .1, size=gates*5).astype('float32')
    bh = rng.normal(0, .1, size=gates*5).astype('float32')
    pi = tp.ops.quantized.linear_prepack(wi, tp.tensor(bi))
    ph = tp.ops.quantized.linear_prepack(wh, tp.tensor(bh))
    rp = torch.ops.quantized.linear_prepack(ri, torch.tensor(bi))
    rq = torch.ops.quantized.linear_prepack(rh, torch.tensor(bh))
    hx = [tp.tensor(h), tp.tensor(c)] if mode == 2 else tp.tensor(h)
    rx = [torch.tensor(h), torch.tensor(c)] if mode == 2 else torch.tensor(h)
    op = 'quantized_' + name + '_cell_dynamic'
    actual = getattr(tp.ops.quantized, op)(tp.tensor(x), hx, pi, ph, tp.tensor(bi), tp.tensor(bh))
    expected = getattr(torch.ops.quantized, op)(torch.tensor(x), rx, rp, rq, torch.tensor(bi), torch.tensor(bh))
    for a, b in zip(actual if mode == 2 else [actual], expected if mode == 2 else [expected]):
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=2e-6, rtol=2e-5)
    params = tp.ops.quantized.make_quantized_cell_params_dynamic(pi, ph, tp.tensor(bi), tp.tensor(bh), True)
    np.testing.assert_array_equal(params.linear_ih(tp.tensor(x)).numpy(), pi.run(
        tp.quantize_per_tensor_dynamic(tp.tensor(x), tp.quint8, True), output_float=True).numpy())


@pytest.mark.parametrize('device', DEVICES)
def test_half_linear_cache_invalidates_after_weight_mutation(device):
    rng = np.random.default_rng(128)
    data = rng.normal(0, .2, size=(7, 37)).astype('float32')
    raw = rng.normal(0, .2, size=(19, 37)).astype('float32')
    x = tp.tensor(data, device=device)
    weight = tp._C._quantized_fp16_prepack(tp.tensor(raw, device=device))
    for factor in (1., -1.):
        weight.copy_(tp.tensor(raw*factor, device=device, dtype=tp.float16))
        actual = tp._C._quantized_fp16_linear(x, weight, None)
        expected = torch.nn.functional.linear(torch.tensor(data), torch.tensor(raw*factor).half().float())
        np.testing.assert_allclose(actual.cpu().numpy(), expected.numpy(), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('device', DEVICES)
def test_recurrent_inference_weights_remain_live_after_mutation(device):
    rng = np.random.default_rng(129)
    x = rng.normal(0, .2, size=(3, 7)).astype('float32')
    h = rng.normal(0, .2, size=(3, 5)).astype('float32')
    a = rng.integers(-50, 51, size=(5, 7), dtype='int8')
    b = rng.integers(-50, 51, size=(5, 5), dtype='int8')
    bias = np.zeros(5, dtype='float32')
    with tp.inference_mode():
        wi, ri = pair(a, 'int8', device, scale=.005, zp=0)
        wh, rh = pair(b, 'int8', device, scale=.005, zp=0)
        for factor in (1, -1):
            update, reference = pair(a*factor, 'int8', device, scale=.005, zp=0)
            wi.copy_(update)
            actual, _ = tp._C._quantized_recurrent_cell(tp.tensor(x, device=device),
                tp.tensor(h, device=device), None, wi, wh, tp.tensor(bias, device=device),
                tp.tensor(bias, device=device), 1, True)
            pi = torch.ops.quantized.linear_prepack(reference, torch.tensor(bias))
            ph = torch.ops.quantized.linear_prepack(rh, torch.tensor(bias))
            expected = torch.ops.quantized.quantized_rnn_tanh_cell_dynamic(
                torch.tensor(x), torch.tensor(h), pi, ph, torch.tensor(bias), torch.tensor(bias))
            np.testing.assert_allclose(actual.cpu().numpy(), expected.numpy(), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('mode,name,gates', [(2,'lstm',4),(3,'gru',3)])
@pytest.mark.parametrize('half', [False,True])
@pytest.mark.parametrize('packed', [False,True])
@pytest.mark.parametrize('bidirectional', [False,True])
@pytest.mark.parametrize('features', [7,37])
def test_native_sequence_accepts_persistent_cell_params(device,mode,name,gates,half,packed,bidirectional,features):
    rng=np.random.default_rng(130)
    layers,width,batch=2,5,3
    directions=2 if bidirectional else 1
    data=rng.normal(0,.2,size=(9,features) if packed else (batch,4,features)).astype('float32')
    hidden=rng.normal(0,.2,size=(layers*directions,batch,width)).astype('float32')
    cell=rng.normal(0,.2,size=hidden.shape).astype('float32')
    params,reference_params=[],[]
    for state in range(layers*directions):
        size=features if state//directions==0 else width*directions
        wi=rng.normal(0,.2,size=(gates*width,size)).astype('float32')
        wh=rng.normal(0,.2,size=(gates*width,width)).astype('float32')
        bi=rng.normal(0,.1,size=gates*width).astype('float32')
        bh=rng.normal(0,.1,size=gates*width).astype('float32')
        if half:
            pi=tp.ops.quantized.linear_prepack_fp16(tp.tensor(wi,device=device),tp.tensor(bi,device=device))
            ph=tp.ops.quantized.linear_prepack_fp16(tp.tensor(wh,device=device),tp.tensor(bh,device=device))
            ri=torch.ops.quantized.linear_prepack_fp16(torch.tensor(wi),torch.tensor(bi))
            rh=torch.ops.quantized.linear_prepack_fp16(torch.tensor(wh),torch.tensor(bh))
            params.append(tp.ops.quantized.make_quantized_cell_params_fp16(pi,ph))
            reference_params.append(torch.ops.quantized.make_quantized_cell_params_fp16(ri,rh))
        else:
            qi,ri=pair(np.rint(wi/.005).clip(-128,127),'int8',device,scale=.005,zp=0)
            qh,rh=pair(np.rint(wh/.005).clip(-128,127),'int8',device,scale=.005,zp=0)
            pi=tp.ops.quantized.linear_prepack(qi,tp.tensor(bi,device=device))
            ph=tp.ops.quantized.linear_prepack(qh,tp.tensor(bh,device=device))
            ri=torch.ops.quantized.linear_prepack(ri,torch.tensor(bi))
            rh=torch.ops.quantized.linear_prepack(rh,torch.tensor(bh))
            params.append(tp.ops.quantized.make_quantized_cell_params_dynamic(
                pi,ph,tp.tensor(bi,device=device),tp.tensor(bh,device=device),bool(state%2)))
            reference_params.append(torch.ops.quantized.make_quantized_cell_params_dynamic(
                ri,rh,torch.tensor(bi),torch.tensor(bh),bool(state%2)))
    hx=[tp.tensor(hidden,device=device),tp.tensor(cell,device=device)] if mode==2 else tp.tensor(hidden,device=device)
    rhx=[torch.tensor(hidden),torch.tensor(cell)] if mode==2 else torch.tensor(hidden)
    op=getattr(tp.ops.tp,'quantized_'+name)
    reference=getattr(torch.ops.aten,'quantized_'+name)
    extra={'dtype':tp.float16 if half else tp.qint8,'use_dynamic':not half} if mode==2 else {}
    rextra={'dtype':torch.float16 if half else torch.qint8,'use_dynamic':not half} if mode==2 else {}
    if packed:
        bs=tp.tensor([3,3,2,1],dtype=tp.int64)
        rbs=torch.tensor([3,3,2,1])
        args=(tp.tensor(data,device=device),bs,hx,params,True,layers,0.,False,bidirectional)
        actual=op.data(*args,**extra)
        expected=reference.data(torch.tensor(data),rbs,rhx,reference_params,True,layers,0.,False,bidirectional,**rextra)
    else:
        args=(tp.tensor(data,device=device),hx,params,True,layers,0.,False,bidirectional,True)
        actual=op.input(*args,**extra)
        expected=reference.input(torch.tensor(data),rhx,reference_params,True,layers,0.,False,bidirectional,True,**rextra)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a.cpu().numpy(),b.numpy(),atol=3e-6,rtol=3e-5)
    top=getattr(tp,'quantized_'+name)(*args,**extra)
    for a,b in zip(top,actual):
        np.testing.assert_array_equal(a.cpu().numpy(),b.cpu().numpy())
    for overload in ('input_legacy','data_legacy'):
        with pytest.raises(RuntimeError):
            getattr(op,overload)(*args,**extra)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('mode', [0,1,2,3])
@pytest.mark.parametrize('width', [1,15,16,17,37,128])
def test_recurrent_gate_vector_tails_and_special_values(device,mode,width):
    rng=np.random.default_rng(131)
    count=4 if mode==2 else 3 if mode==3 else 1
    a=rng.normal(0,2,size=(3,count*width)).astype('float32')
    b=rng.normal(0,2,size=a.shape).astype('float32')
    h=rng.normal(0,.2,size=(3,width)).astype('float32')
    c=rng.normal(0,.2,size=h.shape).astype('float32')
    a[0,0]=np.nan
    a[-1,-1]=np.inf
    x,y,state,cell=(torch.tensor(v) for v in (a,b,h,c))
    if mode==2:
        i,f,g,o=(x+y).chunk(4,1)
        next_c=f.sigmoid()*cell+i.sigmoid()*g.tanh()
        expected=(o.sigmoid()*next_c.tanh(),next_c)
    elif mode==3:
        xi,xu,xn=x.chunk(3,1)
        yi,yu,yn=y.chunk(3,1)
        reset=(xi+yi).sigmoid()
        update=(xu+yu).sigmoid()
        candidate=(xn+yn*reset).tanh()
        expected=((state-candidate)*update+candidate,)
    else:
        expected=(torch.relu(x+y) if mode==0 else torch.tanh(x+y),)
    actual=tp._C._quantized_recurrent_step(tp.tensor(a,device=device),tp.tensor(b,device=device),
        tp.tensor(h,device=device),tp.tensor(c,device=device) if mode==2 else None,mode)
    for value,reference in zip(actual,expected):
        np.testing.assert_allclose(value.cpu().numpy(),reference.numpy(),atol=2e-6,rtol=2e-5,equal_nan=True)

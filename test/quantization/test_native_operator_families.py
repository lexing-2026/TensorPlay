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
        e=getattr(torch.nn.functional,f'avg_pool{rank}d')(r,3,2,1,True,False)
    elif kind=='adaptive':
        y=getattr(F,f'adaptive_avg_pool{rank}d')(x,(3,)*rank)
        e=getattr(torch.nn.functional,f'adaptive_avg_pool{rank}d')(r,(3,)*rank)
    else:
        y=getattr(tp._C,f'quantized_max_pool{rank}d')(x,[3]*rank,[2]*rank,[1]*rank,[2]*rank,True)
        e=getattr(torch.nn.functional,f'max_pool{rank}d')(r,3,2,1,2,True)
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

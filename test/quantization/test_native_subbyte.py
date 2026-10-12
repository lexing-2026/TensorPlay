"""Packed affine codes, grid rounding, transfers and checkpoint reconstruction."""
import io
import ctypes
import numpy as np
import pytest
import torch
import tensorplay as tp

DEVICES=['cpu']+(['cuda'] if tp.cuda.is_available() else [])


@pytest.mark.parametrize('bits',[2,4])
@pytest.mark.parametrize('mode',[0x400,0x800,0xc00])
@pytest.mark.parametrize('size',[65,65537])
def test_subbyte_cpu_calling_thread_rounding_mode(bits,mode,size):
    library=ctypes.CDLL(None)
    library.fegetround.restype=ctypes.c_int
    library.fesetround.argtypes=[ctypes.c_int]
    values=np.resize(np.array([-.5,.25,.5,.75,1.5,2.5,3.5,6.75],dtype='float32'),size)
    dtype=getattr(tp,'quint4x2' if bits==4 else 'quint2x4')
    rdtype=getattr(torch,'quint4x2' if bits==4 else 'quint2x4')
    native,reference=tp.tensor(values),torch.tensor(values)
    previous=library.fegetround()
    try:
        assert library.fesetround(mode)==0
        actual=tp.quantize_per_tensor(native,.3,1,dtype)
        expected=torch.quantize_per_tensor(reference,.3,1,rdtype)
        actual_real=actual.dequantize()
        expected_real=expected.dequantize()
    finally:
        assert library.fesetround(previous)==0
    np.testing.assert_array_equal(actual.int_repr().numpy(),expected.int_repr().numpy())
    np.testing.assert_array_equal(actual_real.numpy(),expected_real.numpy())


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('bits',[2,4])
@pytest.mark.parametrize('shape',[(2,3,2,5),(2,3,2,2,5)])
def test_subbyte_channels_last_layout_clone_transfer_and_checkpoint(device,bits,shape):
    values=np.random.default_rng(510).uniform(-1,4,size=shape).astype('float32')
    format_name='channels_last' if len(shape)==4 else 'channels_last_3d'
    native=tp.tensor(values,device=device).contiguous(memory_format=getattr(tp,format_name))
    reference=torch.tensor(values).contiguous(memory_format=getattr(torch,format_name))
    dtype=getattr(tp,'quint4x2' if bits==4 else 'quint2x4')
    rdtype=getattr(torch,'quint4x2' if bits==4 else 'quint2x4')
    actual=tp.quantize_per_tensor(native,.25,1,dtype)
    expected=torch.quantize_per_tensor(reference,.25,1,rdtype)
    pack=8//bits
    expected_bytes=torch.empty(0,dtype=torch.uint8).set_(expected.untyped_storage(),0,
                         ((expected.numel()+pack-1)//pack,),(1,)).numpy()
    stream=io.BytesIO()
    tp.save(actual,stream)
    stream.seek(0)
    for result in (actual,actual.clone(),actual.cpu(),tp.load(stream)):
        assert tuple(result.stride())==tuple(expected.stride())
        np.testing.assert_array_equal(result.int_repr().cpu().numpy(),expected_bytes)
        decoded=result.dequantize()
        assert tuple(decoded.stride())==tuple(expected.dequantize().stride())
        np.testing.assert_array_equal(decoded.cpu().numpy(),expected.dequantize().numpy())
    if device=='cpu' and tp.cuda.is_available():
        moved=actual.cuda().cpu()
        assert tuple(moved.stride())==tuple(expected.stride())
        np.testing.assert_array_equal(moved.int_repr().numpy(),expected_bytes)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('bits',[2,4])
@pytest.mark.parametrize('shape',[(),(0,),(1,),(3,),(3,5),(2,33),(4097,)])
@pytest.mark.parametrize('zp',[0,1])
def test_subbyte_affine_rounding_tail_and_grid(device,bits,shape,zp):
    rng=np.random.default_rng(2061)
    count=int(np.prod(shape))
    values=rng.uniform(-2,5,size=count).astype('float32')
    values[:min(count,8)]=np.array([-.5,.5,1.5,2.5,3.5,4.5,7.5,15.5],dtype='float32')[:min(count,8)]*.25
    values=values.reshape(shape)
    dtype=getattr(tp,'quint4x2' if bits==4 else 'quint2x4')
    rdtype=getattr(torch,'quint4x2' if bits==4 else 'quint2x4')
    actual=tp.quantize_per_tensor(tp.tensor(values,device=device),.25,zp,dtype)
    expected=torch.quantize_per_tensor(torch.tensor(values),.25,zp,rdtype)
    assert actual.dtype==dtype and tuple(actual.shape)==shape
    assert actual.q_scale()==.25 and actual.q_zero_point()==zp
    np.testing.assert_array_equal(actual.int_repr().cpu().numpy(),expected.int_repr().numpy())
    np.testing.assert_array_equal(actual.dequantize().cpu().numpy(),expected.dequantize().numpy())
    pack=8//bits
    rows=int(np.prod(shape[:-1])) if shape else 1
    expected_size=rows*((shape[-1]+pack-1)//pack) if shape else 1
    assert actual.untyped_storage().nbytes()==expected_size
    for result in [actual.clone(),actual.cpu(),actual.reshape((count,))]:
        assert result.dtype==dtype
        np.testing.assert_array_equal(result.int_repr().cpu().numpy(),expected.int_repr().numpy())
    destination=tp._C._empty_affine_quantized(shape,dtype=dtype,device=device,scale=.5,zero_point=0)
    destination.copy_(actual)
    assert destination.q_scale()==.25 and destination.q_zero_point()==zp
    np.testing.assert_array_equal(destination.int_repr().cpu().numpy(),expected.int_repr().numpy())


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('bits',[2,4])
@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('shape',[(3,5,7),(3,0,7),(0,5,7)])
def test_subbyte_float_channel_grid_and_checkpoint(device,bits,axis,shape):
    rng=np.random.default_rng(723)
    values=rng.normal(size=shape).astype('float32')
    scales=np.linspace(.07,.43,values.shape[axis],dtype='float32')
    zeros=np.linspace(-1.25,4.75,values.shape[axis],dtype='float32')
    dtype=getattr(tp,'quint4x2' if bits==4 else 'quint2x4')
    rdtype=getattr(torch,'quint4x2' if bits==4 else 'quint2x4')
    actual=tp.quantize_per_channel(tp.tensor(values,device=device),tp.tensor(scales,device=device),
                                  tp.tensor(zeros,device=device),axis,dtype)
    expected=torch.quantize_per_channel(torch.tensor(values),torch.tensor(scales),torch.tensor(zeros),axis,rdtype)
    assert actual.qscheme()==tp.per_channel_affine_float_qparams
    np.testing.assert_array_equal(actual.int_repr().cpu().numpy(),expected.int_repr().numpy())
    np.testing.assert_array_equal(actual.dequantize().cpu().numpy(),expected.dequantize().numpy())
    stream=io.BytesIO()
    tp.save({'weight':actual},stream)
    stream.seek(0)
    loaded=tp.load(stream)['weight']
    assert loaded.dtype==dtype and tuple(loaded.shape)==tuple(actual.shape)
    assert loaded.q_per_channel_axis()==axis
    np.testing.assert_array_equal(loaded.int_repr().cpu().numpy(),expected.int_repr().numpy())
    np.testing.assert_array_equal(loaded.dequantize().cpu().numpy(),expected.dequantize().numpy())


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('bits',[2,4])
def test_subbyte_noncontiguous_input_and_per_tensor_checkpoint(device,bits):
    values=np.arange(35,dtype='float32').reshape(5,7)/5
    dtype=getattr(tp,'quint4x2' if bits==4 else 'quint2x4')
    rdtype=getattr(torch,'quint4x2' if bits==4 else 'quint2x4')
    actual=tp.quantize_per_tensor(tp.tensor(values,device=device).t(),.15,1,dtype)
    expected=torch.quantize_per_tensor(torch.tensor(values).t(),.15,1,rdtype)
    stream=io.BytesIO();tp.save(actual,stream);stream.seek(0)
    loaded=tp.load(stream)
    assert loaded.dtype==dtype and tuple(loaded.shape)==(7,5)
    assert loaded.q_scale()==.15 and loaded.q_zero_point()==1
    np.testing.assert_array_equal(loaded.int_repr().cpu().numpy(),expected.int_repr().numpy())
    np.testing.assert_array_equal(loaded.dequantize().cpu().numpy(),expected.dequantize().numpy())


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('bits',[2,4])
def test_subbyte_checkpoint_preserves_shared_storage_and_byte_offset(device,bits):
    dtype=getattr(tp,'quint4x2' if bits==4 else 'quint2x4')
    pack=8//bits
    source=tp.quantize_per_tensor(tp.arange(32,dtype=tp.float32,device=device),1.,0,dtype)
    view=tp._C._empty_affine_quantized((8,),dtype=dtype,device=device,scale=1.,zero_point=0)
    view.set_(source.untyped_storage(),1,(8,),(1,))
    empty=tp._C._empty_affine_quantized((0,),dtype=dtype,device=device,scale=1.,zero_point=0)
    original_bytes=source.untyped_storage().nbytes()
    empty.set_(source.untyped_storage(),100,(0,),(1,))
    assert empty.storage_offset()==100 and source.untyped_storage().nbytes()==original_bytes
    stream=io.BytesIO()
    tp.save([source,view],stream)
    stream.seek(0)
    loaded,alias=tp.load(stream)
    assert alias.storage_offset()==1 and tuple(alias.stride())==(1,)
    raw=tp.empty((0,),dtype=tp.uint8,device=device)
    raw.set_(loaded.untyped_storage(),0,(loaded.untyped_storage().nbytes(),),(1,))
    raw.copy_(tp.zeros_like(raw))
    np.testing.assert_array_equal(alias.int_repr().cpu().numpy(),np.zeros((8+pack-1)//pack,dtype='uint8'))
    np.testing.assert_array_equal(alias.dequantize().cpu().numpy(),np.zeros(8,dtype='float32'))


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('columns',[2,32,66])
@pytest.mark.parametrize('weighted',[False,True])
def test_four_bit_logical_embeddings_and_persistent_weights(device,columns,weighted):
    rng=np.random.default_rng(862)
    values=rng.normal(size=(7,columns)).astype('float32')
    scales=rng.uniform(.05,.4,size=7).astype('float32')
    zeros=rng.uniform(-2,8,size=7).astype('float32')
    weight=tp.quantize_per_channel(tp.tensor(values,device=device),tp.tensor(scales,device=device),
                                  tp.tensor(zeros,device=device),0,tp.quint4x2)
    rw=torch.quantize_per_channel(torch.tensor(values),torch.tensor(scales),torch.tensor(zeros),0,torch.quint4x2)
    rp=torch.ops.quantized.embedding_bag_prepack(rw)
    packed=tp.ops.quantized.embedding_bag_prepack(weight)
    indices=np.array([2,1,5,2,6],dtype='int64')
    offsets=np.array([0,2,2,5],dtype='int64')
    weights=np.array([.7,-.3,1.1,2.,.8],dtype='float32') if weighted else None
    ids=tp.tensor(indices,device=device);off=tp.tensor(offsets,device=device)
    ps=tp.tensor(weights,device=device) if weighted else None
    expected=torch.ops.quantized.embedding_4bit(rp,torch.tensor(indices))
    actual=tp.ops.quantized.embedding_4bit(packed,ids)
    np.testing.assert_allclose(actual.cpu().numpy(),expected.numpy(),rtol=0,atol=1e-6)
    module=tp.ao.nn.quantized.Embedding(7,columns,_weight=weight,dtype=tp.quint4x2)
    np.testing.assert_allclose(module(ids).cpu().numpy(),expected.numpy(),rtol=0,atol=1e-6)
    bag=tp.ao.nn.quantized.EmbeddingBag(7,columns,_weight=weight,dtype=tp.quint4x2,include_last_offset=True)
    expected_bag=torch.ops.quantized.embedding_bag_4bit(rp,torch.tensor(indices),torch.tensor(offsets),
                       False,0,False,torch.tensor(weights) if weighted else None,None,True)
    np.testing.assert_allclose(bag(ids,off,ps).cpu().numpy(),expected_bag.numpy(),rtol=0,atol=2e-6)
    state=io.BytesIO();tp.save(bag.state_dict(),state);state.seek(0)
    restored=tp.ao.nn.quantized.EmbeddingBag(7,columns,dtype=tp.quint4x2,include_last_offset=True).to(device)
    restored.load_state_dict(tp.load(state))
    np.testing.assert_allclose(restored(ids,off,ps).cpu().numpy(),expected_bag.numpy(),rtol=0,atol=2e-6)
    assert restored.weight().dtype==tp.quint4x2


def test_four_bit_embedding_observer_and_float_conversion():
    from tensorplay.ao.quantization import float_qparams_weight_only_qconfig_4bit
    from torch.ao.quantization import float_qparams_weight_only_qconfig_4bit as reference_config
    values=np.random.default_rng(124).normal(size=(5,32)).astype('float32')
    observer=float_qparams_weight_only_qconfig_4bit.weight()
    reference=reference_config.weight()
    observer(tp.tensor(values));reference(torch.tensor(values))
    sc,zp=observer.calculate_qparams();rsc,rzp=reference.calculate_qparams()
    np.testing.assert_allclose(sc.numpy(),rsc.numpy(),rtol=0,atol=1e-7)
    np.testing.assert_allclose(zp.numpy(),rzp.numpy(),rtol=0,atol=1e-6)
    source=tp.nn.Embedding(5,32);source.weight.data.copy_(tp.tensor(values))
    source.qconfig=float_qparams_weight_only_qconfig_4bit
    module=tp.ao.nn.quantized.Embedding.from_float(source)
    assert module.weight().dtype==tp.quint4x2
    rw=torch.quantize_per_channel(torch.tensor(values),rsc,rzp,0,torch.quint4x2)
    packed=torch.ops.quantized.embedding_bag_prepack(rw)
    np.testing.assert_allclose(module(tp.tensor([0,3,1])).numpy(),
          torch.ops.quantized.embedding_4bit(packed,torch.tensor([0,3,1])).numpy(),rtol=0,atol=1e-6)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('bits',[2,4])
@pytest.mark.parametrize('channel',[False,True])
def test_subbyte_nonfinite_and_integer_conversion_limits(device,bits,channel):
    values=np.resize(np.array([np.nan,np.inf,-np.inf,1e30,-1e30,1e18,-1e18,2**32+512],dtype='float32'),(3,65))
    dtype=getattr(tp,'quint4x2' if bits==4 else 'quint2x4')
    rdtype=getattr(torch,'quint4x2' if bits==4 else 'quint2x4')
    if channel:
        actual=tp.quantize_per_channel(tp.tensor(values,device=device),tp.ones(3,device=device),
                 tp.ones(3,device=device),0,dtype)
        expected=torch.quantize_per_channel(torch.tensor(values),torch.ones(3),torch.ones(3),0,rdtype)
    else:
        actual=tp.quantize_per_tensor(tp.tensor(values,device=device),1.,1,dtype)
        expected=torch.quantize_per_tensor(torch.tensor(values),1.,1,rdtype)
    np.testing.assert_array_equal(actual.int_repr().cpu().numpy(),expected.int_repr().numpy())


@pytest.mark.parametrize('explicit',[False,True])
def test_four_bit_dynamic_embedding_conversion(explicit):
    values=np.random.default_rng(165).normal(size=(7,32)).astype('float32')
    source=tp.nn.Sequential(tp.nn.EmbeddingBag(7,32,mode='sum'))
    reference=torch.nn.Sequential(torch.nn.EmbeddingBag(7,32,mode='sum'))
    source[0].weight.data.copy_(tp.tensor(values))
    reference[0].weight.data.copy_(torch.tensor(values))
    actual=tp.ao.quantization.quantize_dynamic(source,{tp.nn.EmbeddingBag} if explicit else None,dtype=tp.quint4x2)
    expected=torch.ao.quantization.quantize_dynamic(reference,{torch.nn.EmbeddingBag} if explicit else None,dtype=torch.quint4x2)
    indices=np.array([[1,4,2],[0,3,6]],dtype='int64')
    assert actual[0].weight().dtype==tp.quint4x2
    np.testing.assert_allclose(actual(tp.tensor(indices)).numpy(),expected(torch.tensor(indices)).numpy(),rtol=0,atol=2e-6)

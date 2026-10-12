"""Rebuild affine quantized tensors from integer storage and grid metadata."""

import tensorplay as tp


def rebuild_quantized_tensor(codes, params):
    scheme = params[0]
    if scheme == 'per_tensor_affine':
        result = tp._C._make_per_tensor_quantized_tensor(codes, float(params[1]), int(params[2]))
    elif scheme in ('per_channel_affine', 'per_channel_affine_float_qparams'):
        scales, zero_points, axis = params[1:]
        axis = int(axis)
        if not 0 <= axis < codes.dim():
            raise ValueError('Quantized checkpoint channel axis exceeds the tensor rank')
        floating = (scheme == 'per_channel_affine_float_qparams' or
                    isinstance(zero_points, tp.Tensor) and zero_points.is_floating_point())
        scales = tp.as_tensor(scales, dtype=tp.float32 if floating else tp.float64, device=codes.device)
        zero_points = tp.as_tensor(zero_points, dtype=tp.float32 if floating else tp.int64, device=codes.device)
        if scales.dim() != 1 or zero_points.dim() != 1 or scales.numel() != codes.size(axis) or zero_points.numel() != codes.size(axis):
            raise ValueError('Quantized checkpoint grid length must match its channel dimension')
        result = tp._C._make_per_channel_quantized_tensor(codes, scales, zero_points, axis)
    else:
        raise ValueError('Unsupported quantized checkpoint grid')
    result.set_(codes.untyped_storage(), codes.storage_offset(), codes.shape, codes.stride())
    return result


def rebuild_qtensor(storage, storage_offset, size, stride, params, requires_grad, backward_hooks):
    from .archive import _rebuild_tensor_v2
    codes = _rebuild_tensor_v2(storage, storage_offset, size, stride, False, {})
    return rebuild_quantized_tensor(codes, params)


def quantized_tensor_reduction(tensor):
    storage_dtype = {tp.qint8: tp.int8, tp.quint8: tp.uint8, tp.qint32: tp.int32}[tensor.dtype]
    codes = tp.empty((0,), dtype=storage_dtype, device=tensor.device)
    codes.set_(tensor.untyped_storage(), tensor.storage_offset(), tensor.shape, tensor.stride())
    scheme = tensor.qscheme()
    if scheme == tp.per_tensor_affine:
        params = ('per_tensor_affine', tensor.q_scale(), tensor.q_zero_point())
    elif scheme in (tp.per_channel_affine, tp.per_channel_affine_float_qparams):
        params = ('per_channel_affine_float_qparams' if scheme == tp.per_channel_affine_float_qparams else 'per_channel_affine',
                  tensor.q_per_channel_scales(), tensor.q_per_channel_zero_points(), tensor.q_per_channel_axis())
    else:
        raise ValueError('Unsupported quantized checkpoint grid')
    return rebuild_quantized_tensor, (codes, params)

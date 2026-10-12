"""Rebuild affine quantized tensors from integer storage and grid metadata."""

import tensorplay as tp


def rebuild_quantized_tensor(codes, params):
    if params[-1] == 'packed_affine':
        scheme, dtype, size, stride, offset, grid, _ = params
        if isinstance(dtype, str):
            try:
                dtype = {'quint2x4': tp.quint2x4, 'quint4x2': tp.quint4x2, 'quint1x8': tp.quint1x8}[dtype]
            except KeyError:
                raise ValueError('Unsupported packed quantized checkpoint dtype') from None
        if scheme == 'per_tensor_affine':
            result = tp._C._empty_affine_quantized(size, dtype=dtype, device=codes.device,
                                               scale=float(grid[0]), zero_point=int(grid[1]))
        elif scheme == 'per_channel_affine_float_qparams':
            scales, zero_points, axis = grid
            axis = int(axis)
            if not 0 <= axis < len(size):
                raise ValueError('Quantized checkpoint channel axis exceeds the tensor rank')
            scales = tp.as_tensor(scales, dtype=tp.float32, device=codes.device)
            zero_points = tp.as_tensor(zero_points, dtype=tp.float32, device=codes.device)
            if scales.dim() != 1 or zero_points.dim() != 1 or scales.numel() != size[axis] or zero_points.numel() != size[axis]:
                raise ValueError('Quantized checkpoint grid length must match its channel dimension')
            result = tp._C._empty_per_channel_affine_quantized(list(size), scales=scales,
                         zero_points=zero_points, axis=axis, dtype=dtype, device=codes.device)
        else:
            raise ValueError('Unsupported packed quantized checkpoint grid')
        result.set_(codes.untyped_storage(), offset, size, stride)
        return result
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
    if tensor.dtype in (tp.quint2x4, tp.quint4x2, tp.quint1x8):
        codes = tp.empty((0,), dtype=tp.uint8, device=tensor.device)
        codes.set_(tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes(),), (1,))
        scheme = tensor.qscheme()
        if scheme == tp.per_tensor_affine:
            name, grid = 'per_tensor_affine', (tensor.q_scale(), tensor.q_zero_point())
        elif scheme == tp.per_channel_affine_float_qparams:
            name, grid = 'per_channel_affine_float_qparams', (tensor.q_per_channel_scales(),
                                  tensor.q_per_channel_zero_points(), tensor.q_per_channel_axis())
        else:
            raise ValueError('Unsupported packed quantized checkpoint grid')
        dtype_name = 'quint4x2' if tensor.dtype == tp.quint4x2 else 'quint2x4' if tensor.dtype == tp.quint2x4 else 'quint1x8'
        params = (name, dtype_name, tuple(tensor.shape), tuple(tensor.stride()),
                  tensor.storage_offset(), grid, 'packed_affine')
        return rebuild_quantized_tensor, (codes, params)
    storage_dtype = {tp.qint8: tp.int8, tp.quint8: tp.uint8, tp.qint32: tp.int32, tp.quint2x4: tp.uint8, tp.quint1x8: tp.uint8}[tensor.dtype]
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

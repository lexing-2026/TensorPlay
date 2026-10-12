"""Simulated quantization with differentiable grid parameters."""
import tensorplay
from tensorplay import _C
from tensorplay.nn.parameter import Parameter
from tensorplay.autograd.function import Function
from .fake_quantize import FakeQuantizeBase


def _dtype_limits(dtype):
    raw = {tensorplay.qint8: tensorplay.int8, tensorplay.quint8: tensorplay.uint8, tensorplay.qint32: tensorplay.int32}.get(dtype, dtype)
    return tensorplay.iinfo(raw)


class _LearnableTensor(Function):
    @staticmethod
    def forward(ctx, x, scale, zp, qmin, qmax, factor):
        ctx.save_for_backward(x, scale, zp)
        ctx.args = (qmin, qmax, factor)
        return _C._fake_quantize_learnable_per_tensor_affine(x, scale, zp, qmin, qmax, factor)

    @staticmethod
    def backward(ctx, grad, *_rest):
        gx, gs, gz = _C._fake_quantize_learnable_per_tensor_affine_backward(grad, *ctx.saved_tensors, *ctx.args)
        return gx, gs, gz, None, None, None


class _LearnableChannel(Function):
    @staticmethod
    def forward(ctx, x, scale, zp, axis, qmin, qmax, factor):
        ctx.save_for_backward(x, scale, zp)
        ctx.args = (axis, qmin, qmax, factor)
        return _C._fake_quantize_learnable_per_channel_affine(x, scale, zp, axis, qmin, qmax, factor)

    @staticmethod
    def backward(ctx, grad, *_rest):
        gx, gs, gz = _C._fake_quantize_learnable_per_channel_affine_backward(grad, *ctx.saved_tensors, *ctx.args)
        return gx, gs, gz, None, None, None, None


class _LearnableFakeQuantize(tensorplay.ao.quantization.FakeQuantizeBase):

    def __init__(self, observer, quant_min=0, quant_max=255, scale=1.0, zero_point=0.0, channel_len=-1, use_grad_scaling=False, **observer_kwargs):
        super().__init__()
        if quant_min >= quant_max:
            raise AssertionError('quant_min must be strictly less than quant_max.')
        self.quant_min = quant_min
        self.quant_max = quant_max
        observer_kwargs['quant_min'] = quant_min
        observer_kwargs['quant_max'] = quant_max
        self.use_grad_scaling = use_grad_scaling
        if channel_len == -1:
            self.scale = Parameter(tensorplay.tensor([scale]))
            self.zero_point = Parameter(tensorplay.tensor([zero_point]))
        else:
            if not (isinstance(channel_len, int) and channel_len > 0):
                raise AssertionError('Channel size must be a positive integer.')
            self.scale = Parameter(tensorplay.tensor([scale] * channel_len))
            self.zero_point = Parameter(tensorplay.tensor([zero_point] * channel_len))
        self.activation_post_process = observer(**observer_kwargs)
        if _dtype_limits(self.activation_post_process.dtype).min > quant_min:
            raise AssertionError('quant_min out of bound')
        if quant_max > _dtype_limits(self.activation_post_process.dtype).max:
            raise AssertionError('quant_max out of bound')
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis if hasattr(self.activation_post_process, 'ch_axis') else -1
        self.register_buffer('fake_quant_enabled', tensorplay.tensor([1], dtype=tensorplay.uint8))
        self.register_buffer('static_enabled', tensorplay.tensor([1], dtype=tensorplay.uint8))
        self.register_buffer('learning_enabled', tensorplay.tensor([0], dtype=tensorplay.uint8))
        bitrange = tensorplay.tensor(quant_max - quant_min + 1).double()
        self.bitwidth = int(tensorplay.log2(bitrange).item())
        self.register_buffer('eps', tensorplay.tensor([tensorplay.finfo(tensorplay.float32).eps]))

    def enable_param_learning(self):
        self.toggle_qparam_learning(enabled=True).toggle_fake_quant(enabled=True).toggle_observer_update(enabled=False)
        return self

    def enable_static_estimate(self):
        self.toggle_qparam_learning(enabled=False).toggle_fake_quant(enabled=True).toggle_observer_update(enabled=True)

    def enable_static_observation(self):
        self.toggle_qparam_learning(enabled=False).toggle_fake_quant(enabled=False).toggle_observer_update(enabled=True)

    def toggle_observer_update(self, enabled=True):
        self.static_enabled[0] = int(enabled)
        return self

    def enable_observer(self, enabled=True):
        self.toggle_observer_update(enabled)

    def toggle_qparam_learning(self, enabled=True):
        self.learning_enabled[0] = int(enabled)
        self.scale.requires_grad = enabled
        self.zero_point.requires_grad = enabled
        return self

    def toggle_fake_quant(self, enabled=True):
        self.fake_quant_enabled[0] = int(enabled)
        return self

    def observe_quant_params(self):
        print(f'_LearnableFakeQuantize Scale: {self.scale.detach()}')
        print(f'_LearnableFakeQuantize Zero Point: {self.zero_point.detach()}')

    def calculate_qparams(self):
        self.scale.data.clamp_(min=self.eps.item())
        scale = self.scale.detach()
        zero_point = self.zero_point.detach().round().clamp(self.quant_min, self.quant_max).long()
        return (scale, zero_point)

    def forward(self, X):
        if self.static_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point)
        else:
            self.scale.data.clamp_(min=self.eps.item())
        if self.fake_quant_enabled[0] == 1:
            if self.qscheme in (tensorplay.per_channel_symmetric, tensorplay.per_tensor_symmetric):
                self.zero_point.data.zero_()
            if self.use_grad_scaling:
                grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
            else:
                grad_factor = 1.0
            if self.qscheme in (tensorplay.per_channel_symmetric, tensorplay.per_channel_affine):
                X = _LearnableChannel.apply(X, self.scale, self.zero_point, self.ch_axis, self.quant_min, self.quant_max, grad_factor)
            else:
                X = _LearnableTensor.apply(X, self.scale, self.zero_point, self.quant_min, self.quant_max, grad_factor)
        return X

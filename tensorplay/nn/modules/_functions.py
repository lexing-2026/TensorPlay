"""Stateful autograd functions backing module implementations."""

import tensorplay
from tensorplay.autograd import Function
from tensorplay.types import Tensor


class CrossMapLRN2d(Function):
    @staticmethod
    def forward(ctx, input, size, alpha=1e-4, beta=0.75, k=1):
        ctx.size = size
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.k = k
        ctx.scale = None

        if input.dim() != 4:
            raise ValueError(
                f"CrossMapLRN2d: Expected input to be 4D, got {input.dim()}D instead."
            )

        ctx.scale = ctx.scale or input.new()
        output = input.new()
        channels = input.size(1)

        output.resize_as_(input)
        ctx.scale.resize_as_(input)

        # use output storage as temporary buffer
        input_square = output
        input_square.copy_(input * input)

        pre_pad = int((ctx.size - 1) / 2 + 1)
        pre_pad_crop = min(pre_pad, channels)

        scale_first = ctx.scale.select(1, 0)
        scale_first.zero_()
        # compute first feature map normalization
        for c in range(pre_pad_crop):
            scale_first.add_(input_square.select(1, c))

        # reuse computations for next feature maps normalization
        # by adding the next feature map and removing the previous
        for c in range(1, channels):
            scale_previous = ctx.scale.select(1, c - 1)
            scale_current = ctx.scale.select(1, c)
            scale_current.copy_(scale_previous)
            if c < channels - pre_pad + 1:
                square_next = input_square.select(1, c + pre_pad - 1)
                scale_current.add_(square_next, alpha=1)

            if c > pre_pad:
                square_previous = input_square.select(1, c - pre_pad)
                scale_current.add_(square_previous, alpha=-1)

        ctx.scale.mul_(ctx.alpha / ctx.size).add_(ctx.k)

        output.copy_(ctx.scale ** (-ctx.beta))
        output.mul_(input)

        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        grad_input = grad_output.new()

        batch_size = input.size(0)
        channels = input.size(1)
        input_height = input.size(2)
        input_width = input.size(3)

        padded_ratio = input.new(channels + ctx.size - 1, input_height, input_width)
        accum_ratio = input.new(input_height, input_width)

        cache_ratio_value = 2 * ctx.alpha * ctx.beta / ctx.size
        inverse_pre_pad = int(ctx.size - (ctx.size - 1) / 2)

        grad_input.resize_as_(input)
        grad_input.copy_(ctx.scale ** (-ctx.beta)).mul_(grad_output)

        padded_ratio.zero_()
        padded_ratio_center = padded_ratio.narrow(0, inverse_pre_pad, channels)
        for n in range(batch_size):
            padded_ratio_center.copy_(grad_output[n] * output[n])
            padded_ratio_center.div_(ctx.scale[n])
            accum_ratio.copy_(
                padded_ratio.narrow(0, 0, ctx.size - 1).sum(0, keepdim=False)
            )
            for c in range(channels):
                accum_ratio.add_(padded_ratio[c + ctx.size - 1])
                grad_input[n][c].addcmul_(
                    input[n][c], accum_ratio, value=-cache_ratio_value
                )
                accum_ratio.add_(padded_ratio[c], alpha=-1)

        return grad_input, None, None, None, None

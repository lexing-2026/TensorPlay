#
# PerChannelMinMax observers (uint8 affine, qmin=0/qmax=255) are embedded
import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay import nn


def _quantize_per_tensor_backend(x, scale, zero_point):
    y = tp.round(x / scale) + zero_point
    y = tp.clamp(y, 0, 255).to(tp.uint8)
    return y


def _dequantize_per_tensor_backend(y, scale, zero_point):
    x = scale * (y.to(tp.float32) - zero_point)
    return x


def _quantize_per_channel_backend(x, scale, zero_point):
    y = tp.zeros(x.shape, device=x.device)
    for i in range(x.size(0)):
        y[i, :] = tp.round(x[i, :] / scale[i]) + zero_point[i]
    y = tp.clamp(y, 0, 255).to(tp.uint8)
    return y


def _dequantize_per_channel_backend(y, scale, zero_point):
    y = y.to(tp.float32).to(y.device)
    x = tp.zeros_like(y)
    for i in range(x.size(0)):
        x[i, :] = scale[i] * (y[i, :] - zero_point[i])
    return x


class _MinMaxObserver:

    def __init__(self):
        self.min_val = None
        self.max_val = None

    def to(self, device):
        return self

    def __call__(self, tensor):
        self.min_val = tensor.min().item()
        self.max_val = tensor.max().item()
        return self

    def calculate_qparams(self):
        qmin, qmax = 0, 255
        if self.max_val == self.min_val:
            scale = 1.0
            zero_point = qmin
        else:
            scale = (self.max_val - self.min_val) / float(qmax - qmin)
            zero_point = int(round(qmin - self.min_val / scale))
            zero_point = max(qmin, min(qmax, zero_point))
        return scale, zero_point


class _PerChannelMinMaxObserver:
    """Per-channel uint8 affine min/max observer."""

    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def to(self, device):
        return self

    def __call__(self, tensor):
        self.min_vals = [tensor[i].min().item() for i in range(tensor.size(0))]
        self.max_vals = [tensor[i].max().item() for i in range(tensor.size(0))]
        return self

    def calculate_qparams(self):
        qmin, qmax = 0, 255
        scales, zero_points = [], []
        for mn, mx in zip(self.min_vals, self.max_vals):
            if mx == mn:
                scales.append(1.0)
                zero_points.append(qmin)
            else:
                s = (mx - mn) / float(qmax - qmin)
                zp = int(round(qmin - mn / s))
                zp = max(qmin, min(qmax, zp))
                scales.append(s)
                zero_points.append(zp)
        return (
            tp.tensor(scales, dtype=tp.float32),
            tp.tensor(zero_points, dtype=tp.int64),
        )


def _get_allgather_out_list(all_gather_in_list, world_size):
    out_list = [
        tp.zeros_like(all_gather_in_list)
        for _ in range(world_size)
    ]
    return out_list


def quantization_pertensor_hook(process_group, bucket: dist.GradBucket):
    """
    Apply ``quantize_per_tensor`` logic to DDP using ``allgather`` protocol.

    Workers first allgather the scale and zero point of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensor, and uses ``allgather`` to communicate these across all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes and
    aggregates each quantized gradient tensor locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, quantization_pertensor_hook)
    """
    group_to_use = process_group if process_group is not None else dist.GroupMember.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    world_size = group_to_use.size()

    tensor = bucket.buffer()

    myObserver = _MinMaxObserver().to(tensor.device)
    myObserver(tensor)

    s, z = myObserver.calculate_qparams()
    s_and_z = tp.tensor([s, z], dtype=tp.float32).to(tensor.device)

    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)

    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True
    ).get_future()

    def quantize_and_allgather(fut):
        # Store scale and zeros across all workers.
        all_ranks_s_and_z = fut.wait()[0]
        # All workers quantize their own ``GradBucket`` tensors.
        quantized_tensor = _quantize_per_tensor_backend(
            tensor,
            all_ranks_s_and_z[rank][0].item(),
            all_ranks_s_and_z[rank][1].item(),
        )
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(quantized_tensor, world_size),
            quantized_tensor,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):
        all_ranks_quantized_tensor = fut.wait()[0]

        aggregated_dequantized_tensor = tp.zeros(
            all_ranks_quantized_tensor[0].shape,
            device=tensor.device,
            dtype=tp.float32,
        )
        # Using previously allgathered scales and zeros, dequantize gradient tensors
        # locally and then aggregate them.
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_tensor_backend(
                quantized_tensor,
                all_ranks_s_and_z[r][0].item(),
                all_ranks_s_and_z[r][1].item(),
            )

        return aggregated_dequantized_tensor / world_size

    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)


def quantization_perchannel_hook(process_group, bucket: dist.GradBucket,
                                 bucket_size=512):
    """
    Apply ``quantize_per_channel`` logic to DDP using ``allgather`` protocol.

    Compared to per-tensor, the main motivation of per-channel is
    for considerably large tensors such as a tensor that contains 6 million
    elements quantizing per a bucket size of 512 (or 128) elements may significantly
    increase the resolution.

    It first splits ``GradBucket`` tensor into multiple chunks (channels) of ``bucket_size``
    elements. Then, workers allgather the scales and zero points of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensor, and uses ``allgather`` to communicate these across all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes, flattens, and
    aggregates each quantized gradient tensor locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.
    """
    group_to_use = process_group if process_group is not None else dist.GroupMember.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    world_size = group_to_use.size()

    tensor = bucket.buffer()

    tensor_in_channels = (
        nn.functional.pad(
            input=tensor,
            pad=(0, bucket_size - len(tensor) % bucket_size),
            mode="constant",
            value=0,
        )
        .view(-1, bucket_size)
        .to(tensor.device)
    )

    myPerChannelObserver = _PerChannelMinMaxObserver().to(tensor.device)
    myPerChannelObserver(tensor_in_channels)

    s_ch, z_ch = myPerChannelObserver.calculate_qparams()
    s_and_z = tp.stack((s_ch, z_ch)).to(tensor.device)

    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)
    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True
    ).get_future()

    def quantize_and_allgather(fut):
        # Store scale and zeros across all workers.
        all_ranks_s_and_z = fut.wait()[0]
        # All workers quantize their corresponding ``GradBucket`` tensors.
        quantized_tensor = _quantize_per_channel_backend(
            tensor_in_channels,
            all_ranks_s_and_z[rank, 0, :],
            all_ranks_s_and_z[rank, 1, :],
        )
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(quantized_tensor, world_size),
            quantized_tensor,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):
        all_ranks_quantized_tensor = fut.wait()[0]

        aggregated_dequantized_tensor = tp.zeros(
            all_ranks_quantized_tensor[0].shape,
            device=tensor.device,
            dtype=tp.float32,
        )
        # Using previously allgathered scales and zeros, dequantize gradient tensors
        # locally and then aggregate them.
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_channel_backend(
                quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1]
            )

        return (
            aggregated_dequantized_tensor.reshape(-1).to(tensor.device)[
                : tensor.size(0)
            ]
            / world_size
        )

    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)

#
# linalg.qr(out=) (copy the Q factor back in place), and no resize_ (the
# batched variant pads via a temporary buffer and returns it).
import logging
import math
from collections import defaultdict

import tensorplay as tp
from tensorplay.linalg import qr as _qr, vector_norm as _vector_norm

import tensorplay.distributed as dist
from tensorplay.distributed import distributed_c10d

from . import default_hooks as default


__all__ = ["PowerSGDState", "powerSGD_hook", "batched_powerSGD_hook"]

logger = logging.getLogger(__name__)


def _not_none(x):
    if x is None:
        raise ValueError("Expected non-None value")
    return x


def _bmm_out(a, b, out):
    batch = a.shape[0]
    for i in range(batch):
        out[i].copy_(a[i] @ b[i])
    return out


def _orthogonalize(matrices, epsilon=0):
    """
    Decide between Gram-Schmidt or QR factorization to orthogonalize a batch of matrices.

    QR factorization doesn't work with half-precision, but it is usually faster with a rank > 2.
    """
    if not (len(matrices.shape) == 3 and matrices.shape[2] <= matrices.shape[1]):
        raise AssertionError

    num_matrices = matrices.shape[0]
    rank = matrices.shape[2]
    dtype = matrices.dtype
    if rank <= 2 or dtype in [tp.float16, tp.bfloat16]:
        _orthogonalize_gram_schmidt(matrices, epsilon=epsilon)
    else:
        q, _ = _qr(matrices)
        matrices.copy_(q)


def _orthogonalize_gram_schmidt(matrices, epsilon=0):
    """
    Apply Gram-Schmidt procedure to orthogonalize a batch of matrices.

    If epsilon is 0, this is equivalent to `linalg.qr(matrices, out=(matrices, _))`,
    """
    num_cols = matrices.shape[2]
    for i in range(num_cols):
        # Normalize the i'th column.
        col = matrices[:, :, i : i + 1]
        # If no epsilon is added here, division by zero may be caused by vanishing gradients.
        # This epsilon is not needed if the input batch of matrices covers the gradients of at least one entire layer
        # in the neural network.
        if epsilon == 0:
            # Note that col ** 2 can underflow/overflow if we use FP16.
            # May need to consider multiplying a scaling factor and dividing it later, or using bfloat16 instead.
            try:
                col.div_(_vector_norm(col, dim=1, keepdim=True))
            except ZeroDivisionError:
                logger.error(
                    "The matrices to be orthogonalized has at least a column of all 0s. Please set a small value such as 1e-8 "
                    "as `orthogonalization_epsilon` in PowerSGD state."
                )
                # Recover the values from NaNs to 0s.
                col.fill_(0.0)
        else:
            col.div_(_vector_norm(col, dim=1, keepdim=True) + epsilon)
        # Project it on the rest and remove it.
        if i + 1 < num_cols:
            rest = matrices[:, :, i + 1 :]
            rest.sub_(tp.sum(col * rest, dim=1, keepdim=True) * col)


def _should_compress(
    num_rows, num_cols, matrix_approximation_rank, min_compression_rate
):
    """
    Recommend if tensor given is worth compressing.

    Returns a recommendation as to whether the 2D tensor described by the arguments is worth compressing,
    including statistics describing the expected savings from compression.  We consider a tensor worth
    compressing when ``min_compression_rate`` < uncompressed size / compressed size, where
    uncompressed size = ``num_rows`` * ``num_cols``,
    and compressed size = (``num_rows`` + ``num_cols``) * ``matrix_approximation_rank``.

    The result of this function is a tuple of the form (compression_recommendation, uncompressed_el_count, compressed_el_count), where:

    compression_recommendation is true if the tensor is worth compressing, and false otherwise (see above);

    uncompressed_el_count is the uncompressed element count, i.e. ``num_rows`` * ``num_cols``; and,

    compress_el_count is the element count after compression, i.e. (``num_rows`` + ``num_cols``) * ``matrix_approximation_rank``.
    """
    uncompressed_size = num_rows * num_cols
    compressed_size = (num_rows + num_cols) * matrix_approximation_rank
    return (
        compressed_size * min_compression_rate < uncompressed_size,
        uncompressed_size,
        compressed_size,
    )


def _report_compression_stats(bucket, state):
    """Report compression stats at frequency of ``compression_stats_logging_frequency`` specified in PowerSGD state."""
    if bucket.is_last() and state.iter >= state.next_stats_report:
        stats = state.compression_stats()
        logger.info(
            "Compression stats: iter %s, total before compression %s, total after compression %s, "
            "rate %s",
            state.iter,
            stats[1],
            stats[2],
            stats[0],
        )
        state.next_stats_report = state.iter + state.compression_stats_logging_frequency


class PowerSGDState:
    r"""
    Store both the algorithm's hyperparameters and internal state for all gradients during training.

    Particularly, ``matrix_approximation_rank`` and ``start_powerSGD_iter`` are the main hyperparameters that should be tuned by the user.
    For performance, we suggest to keep binary hyperparameters ``use_error_feedback`` and ``warm_start`` on.

    1. ``matrix_approximation_rank`` controls the size of compressed low-rank tensors, which determines the compression rate. The lower the rank, the stronger the compression.

    To tune ``matrix_approximation_rank``, we suggest to start from 1 and increase by factors of 2 (like an exponential grid search, 1, 2, 4, ...), until a satisfactory accuracy is reached.

    2. ``start_powerSGD_iter`` defers PowerSGD compression until step ``start_powerSGD_iter``, and vanilla allreduce runs prior to step ``start_powerSGD_iter``.

    3. ``min_compression_rate`` is the minimum compression rate required when a layer is compressed.

    Compression statistics are logged every ``compression_stats_logging_frequency`` iterations once PowerSGD compression starts.

    4. ``orthogonalization_epsilon`` can be a very small value (e.g., 1e-8) added to every normalized matrix column in orthogonalization step, to prevent div-by-zero error if any column has all 0s.

    5. ``batch_tensors_with_same_shape`` controls whether to compress and decompress tensors with same shape in a batched operation to achieve higher parallelism.

    .. warning ::
        If error feedback or warm-up is enabled, the minimum value of ``start_powerSGD_iter`` allowed in DDP is 2.
    """

    __slots__ = [
        "process_group",
        # The fields below are the hyperparameters that often need to be tuned by the user.
        "matrix_approximation_rank",
        "start_powerSGD_iter",
        # The fields below are the hyperparameters that seldom need be tuned by the user.
        "min_compression_rate",
        "orthogonalization_epsilon",
        # The fields below are the binary hyperparameters recommended to be turned on for performance and accuracy.
        "use_error_feedback",
        "warm_start",
        "batch_tensors_with_same_shape",
        # The fields below are internal state.
        "rng",
        "error_dict",
        "p_memory_dict",
        "q_memory_dict",
        "iter",
        # The fields below are for recording compression stats.
        "total_numel_before_compression",
        "total_numel_after_compression",
        "compression_stats_logging_frequency",
        "next_stats_report",
    ]

    def __init__(
        self,
        process_group,
        matrix_approximation_rank=1,
        start_powerSGD_iter=1_000,
        min_compression_rate=2,
        use_error_feedback=True,
        warm_start=True,
        orthogonalization_epsilon=0,
        random_seed=0,
        compression_stats_logging_frequency=10_000,
        batch_tensors_with_same_shape: bool = False,
    ):
        logger.info(
            "PowerSGD config: matrix_approximation_rank = %s; start_powerSGD_iter = %s; "
            "min_compression_rate = %s; orthogonalization_epsilon = %s; use_error_feedback = %s; warm_start = %s; "
            "random_seed = %s; compression_stats_logging_frequency = %s; batch_tensors_with_same_shape = %s",
            matrix_approximation_rank,
            start_powerSGD_iter,
            min_compression_rate,
            orthogonalization_epsilon,
            use_error_feedback,
            warm_start,
            random_seed,
            compression_stats_logging_frequency,
            batch_tensors_with_same_shape,
        )

        self.process_group = process_group
        self.matrix_approximation_rank = matrix_approximation_rank
        if (use_error_feedback or warm_start) and start_powerSGD_iter <= 1:
            raise ValueError(
                "Expect `start_powerSGD_iter` > 1 if `use_error_feedback` or `warm_start` is enabled, "
                "because PowerSGD can only be applied after the first two iterations in DDP."
            )
        self.start_powerSGD_iter = start_powerSGD_iter
        self.min_compression_rate = min_compression_rate
        self.use_error_feedback = use_error_feedback
        self.warm_start = warm_start
        self.orthogonalization_epsilon = orthogonalization_epsilon
        # The purpose of this RNG is to generate different random seeds for initializing Q across iterations,
        # but in the same order for all the DDP replicas.
        import numpy as np

        self.rng = np.random.RandomState(random_seed)
        self.error_dict: dict[int, tp.Tensor] = {}
        self.p_memory_dict: dict[int, tp.Tensor] = {}
        self.q_memory_dict: dict[int, tp.Tensor] = {}
        self.iter = 0
        self.total_numel_before_compression = 0
        self.total_numel_after_compression = 0
        self.compression_stats_logging_frequency = max(
            1, compression_stats_logging_frequency
        )
        self.next_stats_report = 0
        self.batch_tensors_with_same_shape = batch_tensors_with_same_shape

    def __getstate__(self):
        r"""Return picklable state; process group is excluded."""
        logger.warning(
            "NOTE: Process group is not serializable and excluded from a saved state."
        )
        return {
            slot: getattr(self, slot)
            for slot in self.__slots__
            if slot != "process_group"
        }

    def __setstate__(self, state):
        r"""Restore state; process group is set to default."""
        self.process_group = distributed_c10d._get_default_group()
        logger.warning(
            "NOTE: Process group will be set to a default group (i.e. the world size).\
                If a different group is desired, please set `self.process_group` after PowerSGD state is loaded."
        )
        for slot, value in state.items():
            setattr(self, slot, value)

    def maybe_increase_iter(self, bucket):
        """Track iterations and trigger log message at start of local SGD."""
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_last():
            self.iter += 1

        if self.iter == self.start_powerSGD_iter:
            logger.info("Start to apply PowerSGD after %s iterations.", self.iter)

    def compression_stats(self):
        r"""
        Return latest compression statistics as tuple.

        Returns tuple of form (compress_rate, numel_before_compression, numel_after_compression).
        """
        compress_rate = (
            self.total_numel_before_compression / self.total_numel_after_compression
            if self.total_numel_after_compression > 0
            else 0
        )
        return (
            compress_rate,
            self.total_numel_before_compression,
            self.total_numel_after_compression,
        )


def powerSGD_hook(
    state: PowerSGDState, bucket: dist.GradBucket
):
    r"""
    Implement PowerSGD algorithm.

    This DDP communication hook implements PowerSGD gradient compression
    algorithm described in the `paper <https://arxiv.org/abs/1905.13727>`_.

    Note that this communication hook enforces vanilla allreduce for the first ``state.start_powerSGD_iter`` iterations.

    Args:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1,
                                  start_powerSGD_iter=10, min_compression_rate=0.5)
        >>> ddp_model.register_comm_hook(state, powerSGD_hook)
    """
    process_group = state.process_group
    group_to_use = (
        process_group if process_group is not None else _not_none(dist.GroupMember.WORLD)
    )
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()

    # Run vanilla allreduce in the first `start_powerSGD_iter` iterations.
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)

    # Apply PowerSGD after `start_powerSGD_iter` iterations.
    device = input_tensor.device
    dtype = input_tensor.dtype

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.index()
    input_tensor_cp = None
    total_length = input_tensor.shape[0]
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logger.info(
                "A zero tensor of length %s that represents local error is created.",
                total_length,
            )
            state.error_dict[bucket_index] = tp.zeros(
                total_length, device=device, dtype=dtype
            )

        # Keep a copy of the input tensor,
        # so that we can compute the local error caused by compression later,
        # by comparing this copy and the input tensor updated after decompression.
        input_tensor_cp = input_tensor.detach().clone()

    # Unflatten the input tensor into per-parameter tensors, for layer-wise compression.
    tensors = bucket.gradients()

    # Step I: Divide all the tensors into two groups,
    # one will be compressed before allreduce and the other will be directly allreduced without compression.
    tensors_to_compress, uncompressed_tensors = [], []
    total_Ps_size = 0
    total_Qs_size = 0
    for tensor in tensors:
        matrix = tensor.reshape(tensor.shape[0], -1)
        n, m = matrix.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        compress_test = _should_compress(
            n, m, matrix_approximation_rank, state.min_compression_rate
        )
        state.total_numel_before_compression += compress_test[1]
        if compress_test[0]:
            tensors_to_compress.append(matrix)
            total_Ps_size += n * matrix_approximation_rank
            total_Qs_size += m * matrix_approximation_rank
            state.total_numel_after_compression += compress_test[2]
        else:
            uncompressed_tensors.append(tensor)
            state.total_numel_after_compression += compress_test[1]

    _report_compression_stats(bucket, state)

    # Step II: Handle uncompressed tensors.
    # Allocate contiguous memory for these tensors to allreduce efficiently.
    uncompressed_tensors_memory = (
        tp.cat([tensor.reshape(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else tp.tensor([], device=device, dtype=dtype)
    )

    # Step III: Handle the tensors that should be compressed.
    need_randomize_qs = False
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        need_randomize_qs = True
        if state.warm_start:
            logger.info(
                "Allocating contiguous memory of length %s for Ps, and of length %s for Qs, respectively.",
                total_Ps_size,
                total_Qs_size,
            )
        state.p_memory_dict[bucket_index] = tp.empty(
            total_Ps_size, device=device, dtype=dtype
        )
        state.q_memory_dict[bucket_index] = tp.empty(
            total_Qs_size, device=device, dtype=dtype
        )

    # Batch tensors to compress by shape.
    shape_to_tensors = defaultdict(list)
    for tensor in tensors_to_compress:
        shape_to_tensors[tensor.shape].append(tensor)

    def maybe_batched_tensors_to_compress():
        for tensors in shape_to_tensors.values():
            if state.batch_tensors_with_same_shape:
                batch_size = len(tensors)
                if batch_size == 1:
                    # Use the original tensor to avoid copy.
                    yield tensors[0].unsqueeze(0)
                else:
                    yield tp.stack(tensors)
            else:
                for tensor in tensors:
                    yield tensor.unsqueeze(0)

    # Create Ps and Qs that point to the allocated memory.
    tensors_to_compress = []
    ps = []
    qs = []
    p_idx = 0
    q_idx = 0
    for tensor in maybe_batched_tensors_to_compress():
        batch_size, n, m = tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        tensors_to_compress.append(tensor)
        ps.append(
            state.p_memory_dict[bucket_index][
                p_idx : p_idx + batch_size * n * matrix_approximation_rank
            ].view((batch_size, n, matrix_approximation_rank))
        )
        qs.append(
            state.q_memory_dict[bucket_index][
                q_idx : q_idx + batch_size * m * matrix_approximation_rank
            ].view((batch_size, m, matrix_approximation_rank))
        )
        p_idx += batch_size * n * matrix_approximation_rank
        q_idx += batch_size * m * matrix_approximation_rank

    # If warm-start is enabled, reuse Qs from the previous iteration if possible and skip filling random values.
    if not need_randomize_qs:
        for q in qs:
            _orthogonalize(q, state.orthogonalization_epsilon)
    else:
        with tp.fork_rng(devices=[]):
            # Fork this RNG to avoid changing the seed globally and affecting the random sampling anywhere else in the training.
            # The seed makes sure that the initial random values are the same across all the DDP replicas.
            # This seed should differ at every step.
            tp.manual_seed(state.rng.randint(1_000_000_000))
            for q in qs:
                q.copy_(
                    tp.randn(
                        list(q.shape),
                        device="cpu",
                        dtype=dtype,
                    ).to(q.device)
                )
                _orthogonalize(q, state.orthogonalization_epsilon)

    # Compute Ps.
    for tensor, q, p in zip(tensors_to_compress, qs, ps):
        _bmm_out(tensor, q, p)

    # This allreduce is only applied to uncompressed tensors,
    # so it should have been kicked off before the above computation on the compressed tensors to hide more communication costs.
    allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(
        uncompressed_tensors_memory, group=group_to_use, async_op=True
    ).get_future()

    def unpack_uncompressed_tensors_and_allreduce_ps(fut):
        uncompressed_memory = fut.value()[0].div_(world_size)
        idx = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                uncompressed_memory[idx : idx + tensor.numel()].view(tensor.shape)
            )
            idx += tensor.numel()

        # Since these Ps will be orthogonalized later, no need to divide them by world size.
        return (
            dist.all_reduce(
                state.p_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        )

    def compute_qs(fut):
        state.p_memory_dict[bucket_index] = fut.value()
        for p in ps:
            _orthogonalize(p, state.orthogonalization_epsilon)

        # Compute Qs.
        for tensor, p, q in zip(tensors_to_compress, ps, qs):
            _bmm_out(tensor.transpose(1, 2), p, q)

        # Allreduce Qs.
        return (
            dist.all_reduce(
                state.q_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        )

    def decompress(fut):
        state.q_memory_dict[bucket_index] = fut.value().div_(world_size)

        for p, q, tensor in zip(ps, qs, tensors_to_compress):
            _bmm_out(p, q.transpose(1, 2), tensor)

        # Copy batched tensors back to original buffer.
        if state.batch_tensors_with_same_shape:
            for tensor in tensors_to_compress:
                if tensor.shape[0] == 1:
                    # Skip tensor with batch_size == 1 since itself is the original tensor.
                    continue
                original_tensors = shape_to_tensors[tensor.shape[1:]]
                for i, original_tensor in enumerate(original_tensors):
                    original_tensor.copy_(tensor[i])

        if tp.cuda.is_available():
            tp.cuda.synchronize(device.index if device.index >= 0 else None)

        if state.use_error_feedback:
            # Memorize the local errors.
            if input_tensor_cp is None:
                raise AssertionError
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()

        state.maybe_increase_iter(bucket)

        return input_tensor

    return (
        allreduce_contiguous_uncompressed_tensors_fut.then(
            unpack_uncompressed_tensors_and_allreduce_ps
        )
        .then(compute_qs)
        .then(decompress)
    )


def batched_powerSGD_hook(
    state: PowerSGDState, bucket: dist.GradBucket
):
    r"""
    Implement simplified PowerSGD algorithm.

    This DDP communication hook implements a simplified PowerSGD gradient compression
    algorithm described in the `paper <https://arxiv.org/abs/1905.13727>`_.
    This variant does not compress the gradients layer by layer,
    but instead compresses the flattened input tensor that batches all the gradients.
    Therefore, it is **faster** than :meth:`powerSGD_hook`,
    but usually results in a **much lower accuracy**, unless ``matrix_approximation_rank`` is 1.

    .. warning ::
        Increasing ``matrix_approximation_rank`` here may not necessarily increase the accuracy,
        because batching per-parameter tensors without column/row alignment can destroy low-rank structure.

    Args:
        state (PowerSGDState): State information to configure the compression rate and support error feedback, warm start, etc.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1)
        >>> ddp_model.register_comm_hook(state, batched_powerSGD_hook)
    """
    process_group = state.process_group
    group_to_use = (
        process_group if process_group is not None else _not_none(dist.GroupMember.WORLD)
    )
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()

    # Run vanilla allreduce in the first `start_powerSGD_iter` iterations.
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return default._allreduce_fut(group_to_use, input_tensor)

    # Apply PowerSGD after `start_powerSGD_iter` iterations.
    device = input_tensor.device
    total_length = input_tensor.shape[0]
    state.total_numel_before_compression += total_length

    # View the input tensor as a 2D square-shape tensor, and pad 0s if necessary.
    # tp has no resize_; pad through a temporary buffer instead.
    square_side_length = math.ceil(math.sqrt(total_length))
    state.total_numel_after_compression += (
        square_side_length * state.matrix_approximation_rank * 2
    )
    padded_total_length = square_side_length**2
    padded_input = tp.zeros(padded_total_length, device=device,
                            dtype=input_tensor.dtype)
    padded_input[:total_length].copy_(input_tensor)
    input_tensor = padded_input

    _report_compression_stats(bucket, state)

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.index()
    input_tensor_cp = None
    if state.use_error_feedback:
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            logger.info(
                "A zero tensor of length %s that represents local error is created.",
                padded_total_length,
            )
            state.error_dict[bucket_index] = tp.zeros(
                padded_total_length, device=device, dtype=input_tensor.dtype
            )

        input_tensor_cp = input_tensor.detach().clone()
    matrix = input_tensor.view((square_side_length, square_side_length))

    # Reuse P and Q from the previous iteration if possible.
    if not state.warm_start or bucket_index not in state.p_memory_dict:
        if state.warm_start:
            logger.info(
                "Initializing low-rank tensors P and Q, each of which has a shape of %s x %s.",
                square_side_length,
                state.matrix_approximation_rank,
            )

        def create_low_rank_tensor(fill_random_values, rng):
            """Return a low-rank 2D tensor of square_side_length * matrix_approximation_rank."""
            if fill_random_values:
                with tp.fork_rng(devices=[]):
                    tp.manual_seed(rng.randint(1_000_000_000))
                    return tp.randn(
                        [square_side_length, state.matrix_approximation_rank],
                        device="cpu",
                        dtype=input_tensor.dtype,
                    ).to(device)
            else:
                return tp.empty(
                    square_side_length,
                    state.matrix_approximation_rank,
                    device=device,
                    dtype=input_tensor.dtype,
                )

        state.p_memory_dict[bucket_index] = create_low_rank_tensor(
            fill_random_values=False, rng=state.rng
        )
        state.q_memory_dict[bucket_index] = create_low_rank_tensor(
            fill_random_values=True, rng=state.rng
        )
    _orthogonalize(state.q_memory_dict[bucket_index])

    state.p_memory_dict[bucket_index].copy_(
        matrix @ state.q_memory_dict[bucket_index]
    )
    allreduce_p_fut = dist.all_reduce(
        state.p_memory_dict[bucket_index], group=group_to_use, async_op=True
    ).get_future()

    def compute_q(fut):
        state.p_memory_dict[bucket_index] = fut.value()[0]
        _orthogonalize(state.p_memory_dict[bucket_index])

        state.q_memory_dict[bucket_index].copy_(
            matrix.t() @ state.p_memory_dict[bucket_index]
        )

        return (
            dist.all_reduce(
                state.q_memory_dict[bucket_index], group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        )

    def decompress(fut):
        state.q_memory_dict[bucket_index] = fut.value().div_(world_size)
        approx_matrix = (
            state.p_memory_dict[bucket_index] @ state.q_memory_dict[bucket_index].t()
        )

        if state.use_error_feedback:
            # Memorize the local errors.
            if input_tensor_cp is None:
                raise AssertionError
            state.error_dict[bucket_index] = input_tensor_cp - input_tensor
        if tp.cuda.is_available():
            tp.cuda.synchronize(device.index if device.index >= 0 else None)
        if not state.warm_start:
            state.p_memory_dict.clear()
            state.q_memory_dict.clear()
        # Truncate back to the original length; the reducer copies this
        # result buffer back into the parameter gradients.
        ret = approx_matrix.reshape(-1)[:total_length]

        state.maybe_increase_iter(bucket)

        return ret

    return allreduce_p_fut.then(compute_q).then(decompress)

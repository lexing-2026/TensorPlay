r"""Utilities for packed (variable-length) sequences.

``_pad_packed_sequence`` and ``pad_sequence`` helpers for sequence packing.
"""

from typing import Any, Callable, NamedTuple, Optional, TypeVar

import tensorplay as tp
from tensorplay import Tensor

__all__ = [
    "PackedSequence",
    "invert_permutation",
    "pack_padded_sequence",
    "pad_packed_sequence",
    "pack_sequence",
    "pad_sequence",
    "unpad_sequence",
    "unpack_sequence",
]

_T = TypeVar("_T")
_R = TypeVar("_R")


class PackedSequence_(NamedTuple):
    data: Tensor
    batch_sizes: Tensor
    sorted_indices: Optional[Tensor]
    unsorted_indices: Optional[Tensor]


def bind(optional: Optional[_T], fn: Callable[[_T], _R]) -> Optional[_R]:
    if optional is None:
        return None
    return fn(optional)


class PackedSequence(PackedSequence_):
    r"""Holds the data and list of :attr:`batch_sizes` of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

        Batch sizes represent the number elements at each sequence step in
        the batch, not the varying sequence lengths passed to
        :func:`pack_padded_sequence`.  For instance, given data ``abc`` and ``x``
        the :class:`PackedSequence` would contain data ``axbc`` with
        ``batch_sizes=[2,1,1]``.

    Attributes:
        data (Tensor): Tensor containing packed sequence
        batch_sizes (Tensor): Tensor of integers holding
            information about the batch size at each sequence step
        sorted_indices (Tensor, optional): Tensor of integers holding how this
            :class:`PackedSequence` is constructed from sequences.
        unsorted_indices (Tensor, optional): Tensor of integers holding how
            this to recover the original sequences with correct order.

    .. note::
        :attr:`data` can be on arbitrary device and of arbitrary dtype.
        :attr:`sorted_indices` and :attr:`unsorted_indices` must be ``int64``
        tensors on the same device as :attr:`data`.

        However, :attr:`batch_sizes` should always be a CPU ``int64`` tensor.

        This invariant is maintained throughout :class:`PackedSequence` class,
        and all functions that construct a :class:`PackedSequence` in TensorPlay
        (i.e. they only pass in tensors conforming to this constraint).
    """

    def __new__(
        cls,
        data,
        batch_sizes=None,
        sorted_indices=None,
        unsorted_indices=None,
    ):
        return super().__new__(
            cls,
            *_packed_sequence_init_args(
                data, batch_sizes, sorted_indices, unsorted_indices
            ),
        )

    # NOTE [ device and dtype of a PackedSequence ]
    #
    # See the note above in doc string (starting with ":attr:`data` can be on
    # arbitrary device...").
    def pin_memory(self):
        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        return type(self)(
            self.data.pin_memory(),
            self.batch_sizes,
            bind(self.sorted_indices, lambda t: t.pin_memory()),
            bind(self.unsorted_indices, lambda t: t.pin_memory()),
        )

    def to(self, *args: Any, **kwargs: Any):
        r"""Perform dtype and/or device conversion on `self.data`.

        It has similar signature as :meth:`tensorplay.Tensor.to`

        .. note::

            If the ``self.data`` Tensor already has the correct
            :class:`tensorplay.DType` and :class:`tensorplay.Device`, then
            ``self`` is returned.  Otherwise, returns a copy with the desired
            configuration.
        """

        # Why not convert `batch_sizes`?
        # See NOTE [ device and dtype of a PackedSequence ]
        data = self.data.to(*args, **kwargs)
        if data is self.data:
            return self

        # Does not forward device or dtype arg/kwargs, device is set from
        # data.device
        def call_to(t: Tensor) -> Tensor:
            return t.to(data.device)

        sorted_indices = bind(self.sorted_indices, call_to)
        unsorted_indices = bind(self.unsorted_indices, call_to)
        return type(self)(data, self.batch_sizes, sorted_indices, unsorted_indices)

    def cuda(self, *args: Any, **kwargs: Any):
        if self.data.is_cuda:
            return self.to(*args, **kwargs)
        return self.to(tp.device("cuda"), *args, **kwargs)

    def cpu(self, *args: Any, **kwargs: Any):
        if not self.data.is_cuda:
            return self.to(*args, **kwargs)
        return self.to(tp.device("cpu"))

    def double(self):
        return self.to(dtype=tp.float64)

    def float(self):
        return self.to(dtype=tp.float32)

    def half(self):
        return self.to(dtype=tp.float16)

    def bfloat16(self):
        return self.to(dtype=tp.bfloat16)

    def long(self):
        return self.to(dtype=tp.int64)

    def int(self):
        return self.to(dtype=tp.int32)

    def short(self):
        return self.to(dtype=tp.int16)

    def char(self):
        return self.to(dtype=tp.int8)

    def byte(self):
        return self.to(dtype=tp.uint8)

    @property
    def is_cuda(self) -> bool:
        r"""Return true if `self.data` stored on a gpu."""
        return self.data.is_cuda

    def is_pinned(self) -> bool:
        r"""Return true if `self.data` stored on in pinned memory."""
        return self.data.is_pinned()


# method to construct PackedSequence
def _packed_sequence_init_args(
    data,
    batch_sizes=None,
    sorted_indices=None,
    unsorted_indices=None,
):
    # NB: if unsorted_indices is provided, it should be the inverse permutation
    # to sorted_indices. Don't assert it here because the PackedSequence ctor
    # should only be used internally.

    if unsorted_indices is None:
        unsorted_indices = invert_permutation(sorted_indices)

    # support being called as `PackedSequence(data, batch_sizes, sorted_indices)`
    if batch_sizes is not None:
        if batch_sizes.device.type != "cpu":
            raise ValueError(
                "batch_sizes should always be on CPU. "
                "Instances of PackedSequence should never be created manually. "
                "They should be instantiated by functions like pack_sequence "
                "and pack_padded_sequences in nn.utils.rnn. "
            )
        return data, batch_sizes, sorted_indices, unsorted_indices

    # support being called as `PackedSequence((data, batch_sizes), *, sorted_indices)`
    else:
        if not (isinstance(data, (list, tuple)) and len(data) == 2):
            raise AssertionError("Expected data to be a list or tuple of length 2")
        return data[0], data[1], sorted_indices, unsorted_indices


def _packed_sequence_init(
    data,
    batch_sizes=None,
    sorted_indices=None,
    unsorted_indices=None,
) -> PackedSequence:
    data, batch_sizes, sorted_indices, unsorted_indices = _packed_sequence_init_args(
        data, batch_sizes, sorted_indices, unsorted_indices
    )
    return PackedSequence(data, batch_sizes, sorted_indices, unsorted_indices)


def invert_permutation(permutation: Optional[Tensor]) -> Optional[Tensor]:
    """Returns the inverse of ``permutation``.

    This is useful for converting between sorted and unsorted indices in
    a :class:`~nn.utils.rnn.PackedSequence`.

    Args:
        permutation (Tensor, optional): a 1-D tensor of indices to invert
    """
    if permutation is None:
        return None
    output = tp.empty_like(permutation)
    output.scatter_(
        0, permutation, tp.arange(0, permutation.numel(), device=permutation.device)
    )
    return output


def _pack_padded_sequence(input: Tensor, lengths: Tensor, batch_first: bool):
    if input.numel() <= 0:
        raise RuntimeError("Cannot pack empty tensors.")
    input = input.transpose(0, 1) if batch_first else input

    if lengths.dim() != 1 or lengths.device.type != "cpu" or lengths.dtype != tp.int64:
        raise RuntimeError(
            f"'lengths' argument should be a 1D CPU int64 tensor, but got "
            f"{lengths.dim()}D {lengths.device} {lengths.dtype} tensor"
        )

    lengths_list = lengths.tolist()
    batch_size = input.size(1)

    if lengths.size(0) != batch_size:
        raise RuntimeError(
            f"Expected `len(lengths)` to be equal to batch_size, but got "
            f"{lengths.size(0)} (batch_size={batch_size})"
        )
    if lengths_list[batch_size - 1] <= 0:
        raise RuntimeError(
            "Length of all samples has to be greater than 0, but found an element "
            "in 'lengths' that is <= 0"
        )
    for i in range(batch_size - 1):
        if lengths_list[batch_size - 1 - i] > lengths_list[batch_size - 2 - i]:
            raise RuntimeError(
                "`lengths` array must be sorted in decreasing order when "
                "`enforce_sorted` is True. You can pass `enforce_sorted=False` "
                "to pack_padded_sequence and/or pack_sequence to sidestep this "
                "requirement if you do not need ONNX exportability."
            )

    steps = []
    batch_sizes_list = []
    step_shape = [-1, *list(input.shape)[2:]]  # == [-1, *input.shape[2:]]

    # To understand what is going on in this loop imagine that the input is a
    # padded 2D array that looks like this (x = valid entry, . = padding)
    #
    #  1 1 1 1 1
    #  2 2 2 . .
    #  2 2 2 . .
    #  4 . . . .
    #  4 . . . .
    #
    # Where the vertical dimension corresponds to time, and horizontal dim to
    # batch. In this example, the lengths array will be equal to [5, 3, 3, 1, 1],
    # and we will iterate over them in reverse order (from the rightmost column
    # to the left). We want to avoid eager slicing of the input at every time
    # step, and wait for the moments where the length increases. In this
    # example, that will happen at the first, second and fourth steps. Then, we
    # slice out the whole block of the input that corresponds to this length,
    # and hasn't been sliced yet (the steps at which each element is sliced are
    # annotated in the array above).  You can think of this as if we were
    # scanning the sequences from the shortest one, and every time we realize
    # there's more elements below in our column, we lower the counter (prev_l),
    # and append the new block to the output.
    prev_l = 0
    for i in range(batch_size):
        l = lengths_list[batch_size - 1 - i]
        if l > prev_l:
            current_batch_size = batch_size - i
            steps.append(
                input.slice(0, prev_l, l)
                .slice(1, 0, current_batch_size)
                .reshape(step_shape)
            )
            batch_sizes_list.extend([current_batch_size] * (l - prev_l))
            prev_l = l
        if l < prev_l:
            raise RuntimeError("Internal error: lengths not sorted")
    return tp.cat(steps), tp.tensor(batch_sizes_list, dtype=tp.int64)


def _pad_packed_sequence(
    data: Tensor,
    batch_sizes: Tensor,
    batch_first: bool,
    padding_value: float,
    total_length: int,
):
    # tolist(), so no contiguity normalization of batch_sizes is needed.
    if batch_sizes.dim() != 1 or batch_sizes.device.type != "cpu" or batch_sizes.dtype != tp.int64:
        raise RuntimeError(
            f"'batch_sizes' argument should be a 1D CPU int64 tensor, but got "
            f"{batch_sizes.dim()}D {batch_sizes.device} {batch_sizes.dtype} tensor"
        )
    if batch_sizes.numel() <= 0:
        raise RuntimeError("batch_sizes can not be empty")

    batch_sizes_list = batch_sizes.tolist()
    max_batch_size = batch_sizes_list[0]
    max_real_seq_length = len(batch_sizes_list)
    max_seq_length = max_real_seq_length
    if total_length > 0:
        if total_length < max_seq_length:
            raise ValueError(
                "Expected total_length to be at least the length of the longest "
                "sequence in input, but got total_length="
                f"{total_length} and max sequence length being {max_seq_length}"
            )
        max_seq_length = total_length

    output_size = [max_seq_length, max_batch_size, *list(data.shape)[1:]]
    output = tp.full(output_size, padding_value, dtype=data.dtype, device=data.device)

    # lengths are filled from the tail (the batch is sorted by decreasing
    # length, so the sequences that finish first occupy the last positions).
    lengths_list = [0] * max_batch_size
    write_idx = max_batch_size - 1
    data_offset = 0
    prev_batch_size = max_batch_size
    prev_i = 0
    for i in range(max_real_seq_length + 1):
        batch_size = batch_sizes_list[i] if i != max_real_seq_length else 0
        if batch_size != prev_batch_size:
            l = prev_batch_size * (i - prev_i)
            # The lines below are equivalent to this:
            # output[prev_i:i, :prev_batch_size] = tmp.view(i - prev_i, prev_batch_size, *input.shape[2:])
            tmp = data.slice(0, data_offset, data_offset + l)
            tmp_view_size = [i - prev_i, prev_batch_size, *list(data.shape)[1:]]
            output.slice(0, prev_i, i).slice(1, 0, prev_batch_size).copy_(
                tmp.reshape(tmp_view_size)
            )
            data_offset += l
            prev_i = i
        dec = prev_batch_size - batch_size
        if dec > 0:
            for _ in range(dec):
                lengths_list[write_idx] = i
                write_idx -= 1
        prev_batch_size = batch_size

    if batch_first:
        output = output.transpose(0, 1)

    return output, tp.tensor(lengths_list, dtype=tp.int64)


def pack_padded_sequence(
    input: Tensor,
    lengths,
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    r"""Packs a Tensor containing padded sequences of variable length.

    :attr:`input` can be of size ``T x B x *`` (if :attr:`batch_first` is ``False``)
    or ``B x T x *`` (if :attr:`batch_first` is ``True``) where ``T`` is the length
    of the longest sequence, ``B`` is the batch size, and ``*`` is any number of dimensions
    (including 0).

    For unsorted sequences, use `enforce_sorted = False`. If :attr:`enforce_sorted` is
    ``True``, the sequences should be sorted by length in a decreasing order, i.e.
    ``input[:,0]`` should be the longest sequence, and ``input[:,B-1]`` the shortest
    one. `enforce_sorted = True` is only necessary for ONNX export.

    It is an inverse operation to :func:`pad_packed_sequence`, and hence :func:`pad_packed_sequence`
    can be used to recover the underlying tensor packed in :class:`PackedSequence`.

    Note:
        This function accepts any input that has at least two dimensions. You
        can apply it to pack the labels, and use the output of the RNN with
        them to compute the loss directly. A Tensor can be retrieved from
        a :class:`PackedSequence` object by accessing its ``.data`` attribute.

    Args:
        input (Tensor): padded batch of variable length sequences.
        lengths (Tensor or list(int)): list of sequence lengths of each batch
            element (must be on the CPU if provided as a tensor).
        batch_first (bool, optional): if ``True``, the input is expected in ``B x T x *``
            format, ``T x B x *`` otherwise. Default: ``False``.
        enforce_sorted (bool, optional): if ``True``, the input is expected to
            contain sequences sorted by length in a decreasing order. If
            ``False``, the input will get sorted unconditionally. Default: ``True``.

    .. warning::
        The dim of ``input`` tensor will be truncated if its length larger than
        correspond value in ``length``.

    Returns:
        a :class:`PackedSequence` object
    """
    if not isinstance(lengths, Tensor):
        lengths = tp.tensor(lengths, dtype=tp.int64)
    else:
        lengths = lengths.to(tp.int64)

    if enforce_sorted:
        sorted_indices = None
    else:
        lengths, sorted_indices = tp.sort(lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        batch_dim = 0 if batch_first else 1
        input = input.index_select(batch_dim, sorted_indices)

    data, batch_sizes = _pack_padded_sequence(input, lengths, batch_first)
    return _packed_sequence_init(data, batch_sizes, sorted_indices, None)


def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: Optional[int] = None,
):
    r"""Pad a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Tensor's data will be of size ``T x B x *`` (if :attr:`batch_first` is ``False``)
    or ``B x T x *`` (if :attr:`batch_first` is ``True``) , where ``T`` is the length
    of the longest sequence and ``B`` is the batch size.

    Example:
        >>> from tensorplay.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        >>> seq = tp.tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]])
        >>> lens = [2, 1, 3]
        >>> packed = pack_padded_sequence(
        ...     seq, lens, batch_first=True, enforce_sorted=False
        ... )
        >>> packed
        PackedSequence(data=tensor([4, 1, 3, 5, 2, 6]), batch_sizes=tensor([3, 2, 1]),
                       sorted_indices=tensor([2, 0, 1]), unsorted_indices=tensor([1, 2, 0]))
        >>> seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
        >>> seq_unpacked
        tensor([[1, 2, 0],
                [3, 0, 0],
                [4, 5, 6]])
        >>> lens_unpacked
        tensor([2, 1, 3])

    .. note::
        :attr:`total_length` is useful to implement the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        model wrapped in DataParallel.

    Args:
        sequence (PackedSequence): batch to pad
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise.
        padding_value (float, optional): values for padded elements.
        total_length (int, optional): if not ``None``, the output will be padded to
            have length :attr:`total_length`. This method will throw :class:`ValueError`
            if :attr:`total_length` is less than the max sequence length in
            :attr:`sequence`.

    Returns:
        Tuple of Tensor containing the padded sequence, and a Tensor
        containing the list of lengths of each sequence in the batch.
        Batch elements will be re-ordered as they were ordered originally when
        the batch was passed to :func:`pack_padded_sequence` or :func:`pack_sequence`.
    """
    max_seq_length = sequence.batch_sizes.size(0)
    if total_length is not None:
        if total_length < max_seq_length:
            raise ValueError(
                "Expected total_length to be at least the length "
                "of the longest sequence in input, but got "
                f"total_length={total_length} and max sequence length being {max_seq_length}"
            )
        max_seq_length = total_length
    padded_output, lengths = _pad_packed_sequence(
        sequence.data, sequence.batch_sizes, batch_first, padding_value, max_seq_length
    )
    unsorted_indices = sequence.unsorted_indices
    if unsorted_indices is not None:
        batch_dim = 0 if batch_first else 1
        return (
            padded_output.index_select(batch_dim, unsorted_indices),
            lengths[unsorted_indices.cpu()],
        )
    return padded_output, lengths


def pad_sequence(
    sequences,
    batch_first: bool = False,
    padding_value: float = 0.0,
    padding_side: str = "right",
) -> Tensor:
    r"""Pad a list of variable length Tensors with :attr:`padding_value`.

    ``pad_sequence`` stacks a list of Tensors along a new dimension, and pads them
    to equal length.  :attr:`sequences` can be list of sequences with size ``L x *``,
    where `L` is length of the sequence and ``*`` is any number of dimensions
    (including ``0``). If :attr:`batch_first` is ``False``, the output is of size
    ``T x B x *``, and ``B x T x *`` otherwise, where ``B`` is the batch size
    (the number of elements in :attr:`sequences``), ``T`` is the length of the
    longest sequence.

    Example:
        >>> from tensorplay.nn.utils.rnn import pad_sequence
        >>> a = tp.ones(25, 300)
        >>> b = tp.ones(22, 300)
        >>> c = tp.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        tensorplay.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise. Default: ``False``.
        padding_value (float, optional): value for padded elements. Default: ``0``.
        padding_side (str, optional): the side to pad the sequences on.
            Default: ``'right'``.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """
    sequences = tuple(sequences)
    if padding_side not in ("left", "right"):
        raise RuntimeError(
            "Expected padding_side to be one of left or right, but got "
            f"{padding_side}."
        )
    if len(sequences) == 0:
        raise RuntimeError("received an empty list of sequences")

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    trailing_dims = list(sequences[0].shape)[1:]
    max_len = max(seq.size(0) for seq in sequences)

    if batch_first:
        out_dims = [len(sequences), max_len]
    else:
        out_dims = [max_len, len(sequences)]
    out_dims.extend(trailing_dims)

    out = tp.full(
        out_dims, padding_value, dtype=sequences[0].dtype, device=sequences[0].device
    )
    for i, seq in enumerate(sequences):
        length_i = seq.size(0)
        start = max_len - length_i if padding_side == "left" else 0
        if batch_first:
            out.select(0, i).slice(0, start, start + length_i).copy_(seq)
        else:
            out.slice(0, start, start + length_i).select(1, i).copy_(seq)
    return out


def unpad_sequence(
    padded_sequences: Tensor,
    lengths: Tensor,
    batch_first: bool = False,
):
    r"""Unpad padded Tensor into a list of variable length Tensors.

    ``unpad_sequence`` unstacks padded Tensor into a list of variable length Tensors.

    Example:
        >>> from tensorplay.nn.utils.rnn import pad_sequence, unpad_sequence
        >>> a = tp.ones(25, 300)
        >>> b = tp.ones(22, 300)
        >>> c = tp.ones(15, 300)
        >>> sequences = [a, b, c]
        >>> padded_sequences = pad_sequence(sequences)
        >>> lengths = tp.as_tensor([v.size(0) for v in sequences])
        >>> unpadded_sequences = unpad_sequence(padded_sequences, lengths)
        >>> tp.allclose(sequences[0], unpadded_sequences[0])
        True

    Args:
        padded_sequences (Tensor): padded sequences.
        lengths (Tensor): length of original (unpadded) sequences.
        batch_first (bool, optional): whether batch dimension first or not. Default: ``False``.

    Returns:
        a list of :class:`Tensor` objects
    """
    unpadded_sequences = []

    if not batch_first:
        padded_sequences = padded_sequences.transpose(0, 1)

    lengths_list = lengths.tolist() if isinstance(lengths, Tensor) else list(lengths)

    for seq, length in zip(padded_sequences.unbind(0), lengths_list):
        unpadded_sequences.append(seq.slice(0, 0, int(length)))

    return unpadded_sequences


def pack_sequence(
    sequences,
    enforce_sorted: bool = True,
) -> PackedSequence:
    r"""Packs a list of variable length Tensors.

    Consecutive call of the next functions: ``pad_sequence``, ``pack_padded_sequence``.

    ``sequences`` should be a list of Tensors of size ``L x *``, where `L` is
    the length of a sequence and ``*`` is any number of trailing dimensions,
    including ``0``.

    For unsorted sequences, use `enforce_sorted = False`. If :attr:`enforce_sorted` is
    ``True``, the sequences should be sorted in the order of decreasing length.
    `enforce_sorted = True` is only necessary for ONNX export.

    Example:
        >>> from tensorplay.nn.utils.rnn import pack_sequence
        >>> a = tp.tensor([1, 2, 3])
        >>> b = tp.tensor([4, 5])
        >>> c = tp.tensor([6])
        >>> pack_sequence([a, b, c])
        PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)

    Args:
        sequences (list[Tensor]): A list of sequences of decreasing length.
        enforce_sorted (bool, optional): if ``True``, checks that the input
            contains sequences sorted by length in a decreasing order. If
            ``False``, this condition is not checked. Default: ``True``.

    Returns:
        a :class:`PackedSequence` object
    """
    lengths = tp.as_tensor([v.size(0) for v in sequences], dtype=tp.int64)
    return pack_padded_sequence(
        pad_sequence(sequences), lengths, enforce_sorted=enforce_sorted
    )


def unpack_sequence(packed_sequences: PackedSequence):
    r"""Unpack PackedSequence into a list of variable length Tensors.

    ``packed_sequences`` should be a PackedSequence object.

    Example:
        >>> from tensorplay.nn.utils.rnn import pack_sequence, unpack_sequence
        >>> a = tp.tensor([1, 2, 3])
        >>> b = tp.tensor([4, 5])
        >>> c = tp.tensor([6])
        >>> sequences = [a, b, c]
        >>> packed_sequences = pack_sequence(sequences)
        >>> unpacked_sequences = unpack_sequence(packed_sequences)

    Args:
        packed_sequences (PackedSequence): A PackedSequence object.

    Returns:
        a list of :class:`Tensor` objects
    """
    padded_sequences, lengths = pad_packed_sequence(packed_sequences, batch_first=True)
    return unpad_sequence(padded_sequences, lengths, batch_first=True)

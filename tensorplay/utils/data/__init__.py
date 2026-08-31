"""Data loading utilities: Dataset, DataLoader, Sampler, collate.

The public names, validation order, and error messages remain stable; the
implementation is built on ``tensorplay.Tensor``.

The implementation also has native batch-fetch hooks for TensorPlay-backed
the default collate path to return a single indexed batch instead of creating
one Python tensor view per sample.
"""

from __future__ import annotations

import bisect
import copy
import contextlib
import itertools
import math
import multiprocessing
import os
import queue
import random
import re
import sys
import threading
import traceback
import warnings
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence, Sized
from typing import Any, Callable, Generic, Iterable, Iterator, List, Optional, TypeVar

import numpy as np

import tensorplay as tp
from tensorplay import Tensor

__all__ = [
    "BatchSampler",
    "ChainDataset",
    "ConcatDataset",
    "DataLoader",
    "Dataset",
    "DistributedSampler",
    "IterableDataset",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "StackDataset",
    "Subset",
    "SubsetRandomSampler",
    "TensorDataset",
    "WeightedRandomSampler",
    "_DatasetKind",
    "default_collate",
    "default_convert",
    "get_worker_info",
    "random_split",
]

_T_co = TypeVar("_T_co", covariant=True)
_T = TypeVar("_T")


# A private sentinel used by dataset wrappers to say that their zero-copy
# batch path cannot represent the requested batch.  The fetcher then falls back
_FAST_BATCH_UNAVAILABLE = object()


def _make_batch_index(indices, tensor: Tensor) -> Tensor:
    """Build an index tensor on the same device as ``tensor``.

    Keep batch indices in an explicit int64 tensor so the dataset fast paths
    use one ``index_select`` on both CPU and CUDA.  This avoids constructing a
    Python tensor view for every sample even though Tensor now also supports
    the public ``tensor[[...]]`` spelling.
    """
    if isinstance(indices, Tensor):
        values = indices.tolist()
    else:
        values = [int(index) for index in indices]
    return tp.tensor(values, dtype=tp.int64, device=tensor.device)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class Dataset(Generic[_T_co]):
    """An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching
    a data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~tensorplay.utils.data.Sampler` implementations and the default
    options of :class:`~tensorplay.utils.data.DataLoader`.

    .. note::
      :class:`~tensorplay.utils.data.DataLoader` by default constructs an index
      sampler that yields integral indices. To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> _T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __add__(self, other: "Dataset[_T_co]") -> "ConcatDataset[_T_co]":
        return ConcatDataset([self, other])


class IterableDataset(Dataset[_T_co], Iterable[_T_co]):
    """An iterable Dataset.

    All datasets that represent an iterable of data samples should subclass it.
    Such form of datasets is particularly useful when data come from a stream.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this dataset.

    When a subclass is used with :class:`~tensorplay.utils.data.DataLoader`,
    each item in the dataset will be yielded from the DataLoader iterator.
    """

    def __add__(self, other: Dataset[_T_co]):
        return ChainDataset([self, other])


class TensorDataset(Dataset[tuple[Tensor, ...]]):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    tensors: tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        if any(tensors[0].size(0) != tensor.size(0) for tensor in tensors):
            raise AssertionError("Size mismatch between tensors")
        self.tensors = tensors

    def __getitem__(self, index):
        if isinstance(index, list):
            return tuple(
                tp.index_select(tensor, 0, _make_batch_index(index, tensor))
                for tensor in self.tensors
            )
        return tuple(tensor[index] for tensor in self.tensors)

    def _tp_get_batch(self, indices):
        """Return the default-collated batch without per-row Python views."""
        return [
            tp.index_select(tensor, 0, _make_batch_index(indices, tensor))
            for tensor in self.tensors
        ]

    def __getitems__(self, indices: list[int]):
        # default-collate path uses ``_tp_get_batch`` above to avoid rebuilding
        # this list and stacking it a second time.
        batch = self._tp_get_batch(indices)
        return [tuple(tensor[i] for tensor in batch) for i in range(len(indices))]

    def __len__(self) -> int:
        return self.tensors[0].size(0)


class StackDataset(Dataset[_T]):
    """Dataset as a stacking of multiple datasets.

    This class is useful to assemble different parts of complex input data,
    given as datasets.

    Example:
        >>> images = ImageDataset()
        >>> texts = TextDataset()
        >>> tuple_stack = StackDataset(images, texts)
        >>> tuple_stack[0] == (images[0], texts[0])
        >>> dict_stack = StackDataset(image=images, text=texts)
        >>> dict_stack[0] == {"image": images[0], "text": texts[0]}

    Args:
        *args (Dataset): Datasets for stacking returned as tuple.
        **kwargs (Dataset): Datasets for stacking returned as dict.
    """

    datasets: tuple | dict

    def __init__(self, *args: Dataset[_T_co], **kwargs: Dataset[_T_co]) -> None:
        if args:
            if kwargs:
                raise ValueError(
                    "Supported either ``tuple``- (via ``args``) or"
                    "``dict``- (via ``kwargs``) like input/output, but both types are given."
                )
            self._length = len(args[0])
            if any(self._length != len(dataset) for dataset in args):
                raise ValueError("Size mismatch between datasets")
            self.datasets = args
        elif kwargs:
            tmp = list(kwargs.values())
            self._length = len(tmp[0])
            if any(self._length != len(dataset) for dataset in tmp):
                raise ValueError("Size mismatch between datasets")
            self.datasets = kwargs
        else:
            raise ValueError("At least one dataset should be passed")

    def __getitem__(self, index):
        if isinstance(self.datasets, dict):
            return {k: dataset[index] for k, dataset in self.datasets.items()}
        return tuple(dataset[index] for dataset in self.datasets)

    def __getitems__(self, indices: list):
        # add batched sampling support when parent datasets supports it.
        if isinstance(self.datasets, dict):
            dict_batch: list[dict] = [{} for _ in indices]
            for k, dataset in self.datasets.items():
                if callable(getattr(dataset, "__getitems__", None)):
                    items = dataset.__getitems__(indices)
                    if len(items) != len(indices):
                        raise ValueError(
                            "Nested dataset's output size mismatch."
                            f" Expected {len(indices)}, got {len(items)}"
                        )
                    for data, d_sample in zip(items, dict_batch, strict=True):
                        d_sample[k] = data
                else:
                    for idx, d_sample in zip(indices, dict_batch, strict=True):
                        d_sample[k] = dataset[idx]
            return dict_batch

        # tuple data
        list_batch: list[list] = [[] for _ in indices]
        for dataset in self.datasets:
            if callable(getattr(dataset, "__getitems__", None)):
                items = dataset.__getitems__(indices)
                if len(items) != len(indices):
                    raise ValueError(
                        "Nested dataset's output size mismatch."
                        f" Expected {len(indices)}, got {len(items)}"
                    )
                for data, t_sample in zip(items, list_batch, strict=True):
                    t_sample.append(data)
            else:
                for idx, t_sample in zip(indices, list_batch, strict=True):
                    t_sample.append(dataset[idx])
        tuple_batch: list[tuple] = [tuple(sample) for sample in list_batch]
        return tuple_batch

    def _tp_get_batch(self, indices):
        """Compose native batches from children when every child supports it."""
        if isinstance(self.datasets, dict):
            result = {}
            for key, dataset in self.datasets.items():
                getter = getattr(dataset, "_tp_get_batch", None)
                if not callable(getter):
                    return _FAST_BATCH_UNAVAILABLE
                batch = getter(indices)
                if batch is _FAST_BATCH_UNAVAILABLE:
                    return _FAST_BATCH_UNAVAILABLE
                result[key] = batch
            return result

        result = []
        for dataset in self.datasets:
            getter = getattr(dataset, "_tp_get_batch", None)
            if not callable(getter):
                return _FAST_BATCH_UNAVAILABLE
            batch = getter(indices)
            if batch is _FAST_BATCH_UNAVAILABLE:
                return _FAST_BATCH_UNAVAILABLE
            result.append(batch)
        return result

    def __len__(self) -> int:
        return self._length


class ConcatDataset(Dataset[_T_co]):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    datasets: list[Dataset[_T_co]]
    cumulative_sizes: list[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        if len(self.datasets) == 0:
            raise AssertionError("datasets should not be an empty iterable")
        for d in self.datasets:
            if isinstance(d, IterableDataset):
                raise AssertionError("ConcatDataset does not support IterableDataset")
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._resolve_index(idx)
        return self.datasets[dataset_idx][sample_idx]

    def _resolve_index(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx >= len(self.datasets):
            # Match the natural IndexError raised by the scalar path while
            # giving the batch path a useful, deterministic failure.
            raise IndexError("list index out of range")
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def __getitems__(self, indices: list[int]):
        """Fetch a batch while grouping requests by child dataset.

        This preserves the order of the caller's indices, but lets child
        datasets use their own batch-aware ``__getitems__`` implementation.
        It removes a Python-level binary search and dispatch for every child
        sample in the common concatenated-dataset case.
        """
        if not indices:
            return []

        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_position, index in enumerate(indices):
            dataset_idx, sample_idx = self._resolve_index(index)
            grouped.setdefault(dataset_idx, []).append((output_position, sample_idx))

        result = [None] * len(indices)
        for dataset_idx, positions in grouped.items():
            child = self.datasets[dataset_idx]
            child_indices = [sample_idx for _, sample_idx in positions]
            getter = getattr(child, "__getitems__", None)
            if callable(getter):
                child_items = getter(child_indices)
                if len(child_items) != len(child_indices):
                    raise ValueError(
                        "Nested dataset's output size mismatch."
                        f" Expected {len(child_indices)}, got {len(child_items)}"
                    )
            else:
                child_items = [child[index] for index in child_indices]
            for (output_position, _), item in zip(positions, child_items, strict=True):
                result[output_position] = item
        return result

    def _tp_get_batch(self, indices):
        # A batch crossing a child boundary needs structure-aware merging.  The
        # public __getitems__ path handles that case; keep the direct native
        # path for the frequent single-child batch where it is unambiguous.
        if not indices:
            return _FAST_BATCH_UNAVAILABLE
        dataset_idx, sample_idx = self._resolve_index(indices[0])
        local_indices = [sample_idx]
        for index in indices[1:]:
            next_dataset_idx, next_sample_idx = self._resolve_index(index)
            if next_dataset_idx != dataset_idx:
                return _FAST_BATCH_UNAVAILABLE
            local_indices.append(next_sample_idx)
        getter = getattr(self.datasets[dataset_idx], "_tp_get_batch", None)
        if not callable(getter):
            return _FAST_BATCH_UNAVAILABLE
        return getter(local_indices)

    @property
    def cummulative_sizes(self):
        warnings.warn(
            "`cummulative_sizes` attribute is renamed to `cumulative_sizes`",
            FutureWarning,
            stacklevel=2,
        )
        return self.cumulative_sizes


class ChainDataset(IterableDataset):
    """Dataset for chaining multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            if not isinstance(d, IterableDataset):
                raise AssertionError("ChainDataset only supports IterableDataset")
            yield from d

    def __len__(self) -> int:
        total = 0
        for d in self.datasets:
            if not isinstance(d, IterableDataset):
                raise AssertionError("ChainDataset only supports IterableDataset")
            total += len(d)
        return total


class Subset(Dataset[_T_co]):
    """Subset of a dataset at specified indices.

    .. note::
        When subclassing `Subset` and overriding `__getitem__`, you **must** also
        override `__getitems__` to ensure `DataLoader` works correctly with your
        custom logic. If you override only `__getitem__`, a `NotImplementedError`
        will be raised when using `DataLoader`.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset[_T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[_T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

        # Check if __getitem__ is overridden but __getitems__ is not
        if (
            type(self).__getitem__ is not Subset.__getitem__
            and type(self).__getitems__ is Subset.__getitems__
        ):
            raise NotImplementedError(
                f"{type(self).__name__} overrides __getitem__ but not __getitems__. "
                "When subclassing Subset and overriding __getitem__, you must also override "
                "__getitems__ to ensure DataLoader works correctly with your custom logic. "
                "A simple implementation:\n\n"
                "def __getitems__(self, indices):\n"
                "    return [self.__getitem__(idx) for idx in indices]"
            )

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: list[int]) -> list[_T_co]:
        # add batched sampling support when parent dataset supports it.
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def _tp_get_batch(self, indices):
        # Never bypass a subclass' explicitly customized fetch contract.
        if type(self) is not Subset:
            return _FAST_BATCH_UNAVAILABLE
        getter = getattr(self.dataset, "_tp_get_batch", None)
        if not callable(getter):
            return _FAST_BATCH_UNAVAILABLE
        return getter([self.indices[idx] for idx in indices])

    def __len__(self) -> int:
        return len(self.indices)


_GENERATOR_LOCK = threading.RLock()


@contextlib.contextmanager
def _use_generator(generator: Optional["tp.Generator"]):
    """Run TensorPlay's global-generator kernels against ``generator``.

    The native random operators currently consume the process-global CPU
    generator.  Swapping its complete serialized state (and restoring it in a
    explicit-generator APIs: the supplied generator advances, while the
    process-global generator is left untouched.  A foreign generator object
    is retained as a compatibility fallback and is seeded by ``initial_seed``
    when its state tensor cannot be exchanged with TensorPlay.
    """
    if generator is None:
        yield
        return

    with _GENERATOR_LOCK:
        global_state = tp.get_rng_state()
        try:
            generator_state = generator.get_state()
            if not isinstance(generator_state, Tensor):
                raise TypeError("generator state is not a TensorPlay Tensor")
            tp.set_rng_state(generator_state)
        except (AttributeError, TypeError, RuntimeError):
            # Keep the old best-effort behavior for foreign Generator-like
            # objects.  Most callers use tp.Generator, for which the branch
            # above is fully stateful.
            tp.manual_seed(generator.initial_seed())
            try:
                yield
            finally:
                tp.set_rng_state(global_state)
            return

        try:
            yield
        finally:
            generator.set_state(tp.get_rng_state())
            tp.set_rng_state(global_state)


def random_split(
    dataset: Dataset[_T],
    lengths: Sequence[int | float],
    generator: Optional["tp.Generator"] = None,
) -> list[Subset[_T]]:
    """Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given, the lengths will be
    computed automatically as floor(frac * len(dataset)) for each fraction
    provided. After computing the lengths, if there are any remainders, 1 count
    will be distributed in round-robin fashion to the lengths until there are
    no remainders left.

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = math.floor(len(dataset) * frac)
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset.",
                    stacklevel=2,
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    with _use_generator(generator):
        indices = tp.randperm(sum(lengths)).tolist()
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths, strict=True)
    ]


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
class Sampler(Generic[_T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing
    a way to iterate over indices or lists of indices (batches) of dataset
    elements, and may provide a :meth:`__len__` method that returns the length
    of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~tensorplay.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a DataLoader.
    """

    def __iter__(self) -> Iterator[_T_co]:
        raise NotImplementedError


class SequentialSampler(Sampler[int]):
    """Samples elements sequentially, always in the same order.

    Args:
        data_source (Sized): data source to sample from. Must implement __len__.
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    """Samples elements randomly. If without replacement, then sample from a
    shuffled dataset. If with replacement, then user can specify
    :attr:`num_samples` to draw.

    Args:
        data_source (Sized): data source to sample from. Must implement __len__.
        replacement (bool): samples are drawn on-demand with replacement if
            ``True``, default=``False``.
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        generator: Optional["tp.Generator"] = None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if not isinstance(self.replacement, bool):
            raise TypeError(
                f"replacement should be a boolean value, but got replacement={self.replacement}"
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            # Fresh random seed per epoch, drawn from the global generator
            seed = int(tp.empty((), dtype=tp.int64).random_(0, 2**63 - 1).item())
            generator = tp.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # Materialize before yielding so the global-generator swap never
        # remains active while a caller pauses a partially consumed sampler.
        with _use_generator(generator):
            if self.replacement:
                indices = []
                for _ in range(self.num_samples // 32):
                    indices.extend(tp.randint(0, n, (32,)).tolist())
                indices.extend(tp.randint(0, n, (self.num_samples % 32,)).tolist())
            else:
                indices = []
                for _ in range(self.num_samples // n):
                    indices.extend(tp.randperm(n).tolist())
                indices.extend(tp.randperm(n).tolist()[: self.num_samples % n])
        yield from indices

    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler[int]):
    """Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator: Optional["tp.Generator"] = None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        with _use_generator(self.generator):
            permutation = tp.randperm(len(self.indices)).tolist()
        yield from (self.indices[i] for i in permutation)

    def __len__(self) -> int:
        return len(self.indices)


class WeightedRandomSampler(Sampler[int]):
    """Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence): a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        generator (Generator): Generator used in sampling.
    """

    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator: Optional["tp.Generator"] = None,
    ) -> None:
        if (
            not isinstance(num_samples, int)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={num_samples}"
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, but got replacement={replacement}"
            )

        weights_tensor = tp.as_tensor(np.asarray(weights, dtype=np.float64))
        if len(weights_tensor.shape) != 1:
            raise ValueError(
                "weights should be a 1d sequence but given "
                f"weights have shape {tuple(weights_tensor.shape)}"
            )

        self.weights = weights_tensor
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        with _use_generator(self.generator):
            rand_tensor = tp.multinomial(
                self.weights, self.num_samples, self.replacement
            )
            indices = rand_tensor.tolist()
        yield from indices

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler[list[int]]):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],
        batch_size: int,
        drop_last: bool,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        sampler_iter = iter(self.sampler)
        if self.drop_last:
            # Create multiple references to the same iterator
            args = [sampler_iter] * self.batch_size
            for batch_droplast in zip(*args, strict=False):
                yield [*batch_droplast]
        else:
            batch = [*itertools.islice(sampler_iter, self.batch_size)]
            while batch:
                yield batch
                batch = [*itertools.islice(sampler_iter, self.batch_size)]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# Distributed sampling
# ---------------------------------------------------------------------------
def _dist_world_size_and_rank():
    """Best-effort world size / rank from tensorplay.distributed."""
    try:
        import tensorplay.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
    except ImportError:
        pass
    raise RuntimeError("Requires distributed package to be available")


class DistributedSampler(Sampler[_T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with distributed data parallel
    training: each process passes a :class:`DistributedSampler` as its
    :class:`~tensorplay.utils.data.DataLoader` sampler and loads an exclusive
    subset of the dataset.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it
        always returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved
            from the current distributed group.
        rank (int, optional): Rank of the current process within
            :attr:`num_replicas`. By default, :attr:`rank` is retrieved from
            the current distributed group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle
            the indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop
            the tail of the data to make it evenly divisible across the number
            of replicas. If ``False``, the sampler will add extra indices to
            make the data evenly divisible across the replicas. Default:
            ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at the
        beginning of each epoch **before** creating the DataLoader iterator is
        necessary to make shuffling work properly across multiple epochs.
        Otherwise, the same ordering will be always used.
    """

    def __init__(
        self,
        dataset: Dataset[_T_co],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None or rank is None:
            default_world_size, default_rank = _dist_world_size_and_rank()
            if num_replicas is None:
                num_replicas = default_world_size
            if rank is None:
                rank = default_rank
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then
        # there is no need to drop any data, since the dataset will be split
        # equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible. This
            # is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = tp.Generator()
            g.manual_seed(self.seed + self.epoch)
            with _use_generator(g):
                indices = tp.randperm(len(self.dataset)).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        if len(indices) != self.total_size:
            raise AssertionError(
                f"Number of indices ({len(indices)}) does not match total_size ({self.total_size})"
            )

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise AssertionError(
                f"Number of subsampled indices ({len(indices)}) does not match num_samples ({self.num_samples})"
            )

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------
np_str_obj_array_pattern = re.compile(r"[SaUO]")


def default_convert(data):
    """Convert each NumPy array element into a :class:`tensorplay.Tensor`.

    If the input is a `Sequence`, `Collection`, or `Mapping`, it tries to
    convert each element inside to a :class:`tensorplay.Tensor`. If the input is
    not a NumPy array, it is left unchanged. This is used as the default
    function for collation when both `batch_sampler` and `batch_size` are NOT
    defined in :class:`~tensorplay.utils.data.DataLoader`.

    Args:
        data: a single data point to be converted
    """
    elem_type = type(data)
    if isinstance(data, Tensor):
        return data
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        # array of string classes and object
        if (
            elem_type.__name__ == "ndarray"
            and np_str_obj_array_pattern.search(data.dtype.str) is not None
        ):
            return data
        return tp.as_tensor(data)
    elif isinstance(data, Mapping):
        try:
            if isinstance(data, MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(data)
                clone.update({key: default_convert(data[key]) for key in data})
                return clone
            else:
                return elem_type({key: default_convert(data[key]) for key in data})
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, tuple):
        return [default_convert(d) for d in data]  # Backwards compatibility.
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        try:
            if isinstance(data, MutableSequence):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(data)
                for i, d in enumerate(data):
                    clone[i] = default_convert(d)
                return clone
            else:
                return elem_type([default_convert(d) for d in data])
        except TypeError:
            # The sequence type may not support `copy()` / `__setitem__(index, item)`
            # or `__init__(iterable)` (e.g., `range`).
            return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def collate(
    batch,
    *,
    collate_fn_map: Optional[dict] = None,
):
    """General collate function that handles collection type of element within
    each batch. The function also opens function registry to deal with specific
    element types. `default_collate_fn_map` provides default collate functions
    for tensors, numpy arrays, numbers and strings.

    Args:
        batch: a single batch to be collated
        collate_fn_map: Optional dictionary mapping from element type to the
            corresponding collate function.
    """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](
                    batch, collate_fn_map=collate_fn_map
                )

    if isinstance(elem, Mapping):
        try:
            if isinstance(elem, MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(elem)
                clone.update(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
                return clone
            else:
                return elem_type(
                    {
                        key: collate(
                            [d[key] for d in batch], collate_fn_map=collate_fn_map
                        )
                        for key in elem
                    }
                )
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return {
                key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map)
                for key in elem
            }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in zip(*batch, strict=False)
            )
        )
    elif isinstance(elem, Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))

        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(
            zip(*batch, strict=False)
        )  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                collate(samples, collate_fn_map=collate_fn_map)
                for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                if isinstance(elem, MutableSequence):
                    # The sequence type may have extra properties, so we can't just
                    # use `type(data)(...)` to create the new sequence.
                    # Create a clone and update it if the sequence type is mutable.
                    clone = copy.copy(elem)
                    for i, samples in enumerate(transposed):
                        clone[i] = collate(samples, collate_fn_map=collate_fn_map)
                    return clone
                else:
                    return elem_type(
                        [
                            collate(samples, collate_fn_map=collate_fn_map)
                            for samples in transposed
                        ]
                    )
            except TypeError:
                # The sequence type may not support `copy()` / `__setitem__(index, item)`
                # or `__init__(iterable)` (e.g., `range`).
                return [
                    collate(samples, collate_fn_map=collate_fn_map)
                    for samples in transposed
                ]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[dict] = None,
):
    return tp.stack(batch, 0)


def collate_numpy_array_fn(
    batch,
    *,
    collate_fn_map: Optional[dict] = None,
):
    elem = batch[0]
    # array of string classes and object
    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

    return collate([tp.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)


def collate_numpy_scalar_fn(
    batch,
    *,
    collate_fn_map: Optional[dict] = None,
):
    return tp.as_tensor(np.asarray(batch))


def collate_float_fn(
    batch,
    *,
    collate_fn_map: Optional[dict] = None,
):
    return tp.tensor(np.asarray(batch), dtype=tp.float64)


def collate_int_fn(
    batch,
    *,
    collate_fn_map: Optional[dict] = None,
):
    return tp.tensor(np.asarray(batch))


def collate_str_fn(
    batch,
    *,
    collate_fn_map: Optional[dict] = None,
):
    return batch


default_collate_fn_map: dict = {
    Tensor: collate_tensor_fn,
    np.ndarray: collate_numpy_array_fn,
    (np.bool_, np.number, np.object_): collate_numpy_scalar_fn,
    float: collate_float_fn,
    int: collate_int_fn,
    str: collate_str_fn,
    bytes: collate_str_fn,
}


def default_collate(batch):
    """Take in a batch of data and put the elements within the batch into a
    tensor with an additional outer dimension - batch size.

    The exact output type can be a :class:`tensorplay.Tensor`, a `Sequence` of
    :class:`tensorplay.Tensor`, a Collection of :class:`tensorplay.Tensor`, or
    left unchanged, depending on the input type. This is used as the default
    function for collation when `batch_size` or `batch_sampler` is defined in
    :class:`~tensorplay.utils.data.DataLoader`.

    Args:
        batch: a single batch to be collated
    """
    return collate(batch, collate_fn_map=default_collate_fn_map)


# ---------------------------------------------------------------------------
# Worker helpers (multiprocessing)
# ---------------------------------------------------------------------------
class _DatasetKind:
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last):
        if dataset_kind == _DatasetKind.Iterable:
            return _IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)


_MP_STATUS_CHECK_INTERVAL = 5.0

_worker_info: Optional["WorkerInfo"] = None


class WorkerInfo:
    """Information about the current DataLoader worker process.

    Attributes:
        id: The current worker id (0 to num_workers - 1).
        num_workers: Total number of workers.
        seed: Random seed set for this worker (``base_seed + worker_id``).
        dataset: Copy of the dataset object in this worker.
    """

    __slots__ = ("id", "num_workers", "seed", "dataset")

    def __init__(self, id, num_workers, seed, dataset):
        self.id = id
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = dataset

    def __repr__(self):
        return (
            f"<WorkerInfo(id={self.id}, num_workers={self.num_workers}, "
            f"seed={self.seed}, dataset={self.dataset!r})>"
        )


def get_worker_info() -> Optional[WorkerInfo]:
    """Returns the information about the current DataLoader iterator worker
    process. When called in a worker process, returns a WorkerInfo object with
    information about that worker process; otherwise returns ``None``.
    """
    return _worker_info


class _KeyErrorMessage(str):
    # repr() of a str is surrounded by quotes, which is unreadable for KeyError
    def __repr__(self):
        return self


class ExceptionWrapper:
    """Wraps an exception plus traceback to communicate across processes."""

    def __init__(self, exc_info=None, where="in background"):
        # It is important that we don't store exc_info in a variable.
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        msg = f"Caught {self.exc_type.__name__} {self.where}.\nOriginal {self.exc_msg}"
        if self.exc_type is KeyError:
            msg = _KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            raise self.exc_type(message=msg)
        try:
            exception = self.exc_type(msg)
        except Exception:
            raise RuntimeError(msg) from None
        raise exception


# ---------------------------------------------------------------------------
# Dataset fetchers (shared by single- and multi-process loading)
# ---------------------------------------------------------------------------
class _BaseDatasetFetcher:
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None:
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

    def fetch(self, possibly_batched_index):
        raise NotImplementedError


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last) -> None:
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (
                self.drop_last and len(data) < len(possibly_batched_index)
            ):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            # TensorPlay-backed datasets can return the already-collated
            # result of a batch index_select.  This is deliberately gated on
            # the stock collate function: a user collate_fn must still receive
            fast_getter = getattr(self.dataset, "_tp_get_batch", None)
            if self.collate_fn is default_collate and callable(fast_getter):
                data = fast_getter(possibly_batched_index)
                if data is not _FAST_BATCH_UNAVAILABLE:
                    return data

            getitems = getattr(self.dataset, "__getitems__", None)
            if callable(getitems):
                data = getitems(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)


# The function `_generate_state` is adapted from `numpy.random.SeedSequence`
# It generates an array of int32 as the seed for `numpy.random`, in order to
# prevent state collision due to same seed and algorithm for `numpy.random`
# and `random` modules.
def _generate_state(base_seed, worker_id):
    INIT_A = 0x43B0D7E5
    MULT_A = 0x931E8875
    INIT_B = 0x8B51F9DD
    MULT_B = 0x58F38DED
    MIX_MULT_L = 0xCA01F9DD
    MIX_MULT_R = 0x4973F715
    XSHIFT = 4 * 8 // 2
    MASK32 = 0xFFFFFFFF

    entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = [0] * 4

    hash_const_A = INIT_A

    def hash(value):
        nonlocal hash_const_A
        value = (value ^ hash_const_A) & MASK32
        hash_const_A = (hash_const_A * MULT_A) & MASK32
        value = (value * hash_const_A) & MASK32
        value = (value ^ (value >> XSHIFT)) & MASK32
        return value

    def mix(x, y):
        result_x = (MIX_MULT_L * x) & MASK32
        result_y = (MIX_MULT_R * y) & MASK32
        result = (result_x - result_y) & MASK32
        result = (result ^ (result >> XSHIFT)) & MASK32
        return result

    for i in range(len(pool)):
        pool[i] = hash(entropy[i])

    for i_src in range(len(pool)):
        for i_dst in range(len(pool)):
            if i_src != i_dst:
                pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))

    hash_const_B = INIT_B
    state = []
    for i_dst in range(4):
        data_val = pool[i_dst]
        data_val = (data_val ^ hash_const_B) & MASK32
        hash_const_B = (hash_const_B * MULT_B) & MASK32
        data_val = (data_val * hash_const_B) & MASK32
        data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
        state.append(data_val)
    return state


class _IterableDatasetStopIteration:
    """Dummy class used to signal the end of an IterableDataset worker."""

    def __init__(self, worker_id):
        self.worker_id = worker_id


class _ResumeIteration:
    """Dummy class used to resume fetching when worker reuse is enabled."""

    def __init__(self, seed=None):
        self.seed = seed


class _ManagerWatchdog:
    def __init__(self) -> None:
        self.manager_pid = os.getppid()
        self.manager_dead = False

    def is_alive(self) -> bool:
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead


def _worker_loop(
    dataset_kind,
    dataset,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    base_seed,
    init_fn,
    worker_id,
    num_workers,
    persistent_workers,
) -> None:
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] in
    try:
        watchdog = _ManagerWatchdog()

        tp.set_num_threads(1)
        seed = base_seed + worker_id
        random.seed(seed)
        tp.manual_seed(seed)
        np.random.seed(_generate_state(base_seed, worker_id))

        global _worker_info
        _worker_info = WorkerInfo(
            id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset
        )

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = _DatasetKind.create_fetcher(
                dataset_kind, dataset, auto_collation, collate_fn, drop_last
            )
        except Exception:
            init_exception = ExceptionWrapper(
                where=f"in DataLoader worker process {worker_id}"
            )

        # When using Iterable mode, some workers can exit earlier than others
        # due to the IterableDataset behaving differently per worker. Such a
        # worker sends `_IterableDatasetStopIteration` so the main process
        # won't send more tasks to it and will send `None` to exit it.
        iteration_end = False

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=_MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if isinstance(r, _ResumeIteration):
                # Acknowledge the main process
                data_queue.put((r, None))
                iteration_end = False
                # Recreate the fetcher for the worker-reuse policy
                fetcher = _DatasetKind.create_fetcher(
                    dataset_kind, dataset, auto_collation, collate_fn, drop_last
                )
                continue
            elif r is None:
                # Received the final signal
                break
            elif done_event.is_set() or iteration_end:
                # Keep getting until we see the final signal (`None`), skipping
                # the processing steps.
                continue
            idx, index = r
            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = fetcher.fetch(index)
                except Exception as e:
                    if (
                        isinstance(e, StopIteration)
                        and dataset_kind == _DatasetKind.Iterable
                    ):
                        data = _IterableDatasetStopIteration(worker_id)
                        iteration_end = True
                    else:
                        data = ExceptionWrapper(
                            where=f"in DataLoader worker process {worker_id}"
                        )
            data_queue.put((idx, data))
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    finally:
        _worker_info = None
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()


# ---------------------------------------------------------------------------
# Pin memory
# ---------------------------------------------------------------------------
def _pin_memory(data):
    if isinstance(data, Tensor):
        return data.pin_memory()

    if hasattr(data, "pin_memory"):
        return data.pin_memory()

    if isinstance(data, (str, bytes)):
        return data

    if isinstance(data, Mapping):
        try:
            if isinstance(data, MutableMapping):
                clone = copy.copy(data)
                clone.update({k: _pin_memory(sample) for k, sample in data.items()})
                return clone
            else:
                return type(data)({k: _pin_memory(sample) for k, sample in data.items()})
        except TypeError:
            return {k: _pin_memory(sample) for k, sample in data.items()}

    if isinstance(data, tuple):
        if hasattr(data, "_fields"):  # namedtuple
            return type(data)(*(_pin_memory(sample) for sample in data))
        return type(data)(_pin_memory(sample) for sample in data)

    if isinstance(data, Sequence):
        try:
            if isinstance(data, MutableSequence):
                clone = copy.copy(data)
                for i, item in enumerate(data):
                    clone[i] = _pin_memory(item)
                return clone
            return type(data)([_pin_memory(sample) for sample in data])
        except TypeError:
            return [_pin_memory(sample) for sample in data]

    return data


def _pin_memory_loop(in_queue, out_queue, done_event) -> None:
    def do_one_step():
        try:
            r = in_queue.get(timeout=_MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            return
        idx, data = r
        if not done_event.is_set() and not isinstance(data, ExceptionWrapper):
            try:
                data = _pin_memory(data)
            except Exception:
                data = ExceptionWrapper(where="in pin memory thread")
            r = (idx, data)
        while not done_event.is_set():
            try:
                out_queue.put(r, timeout=_MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue

    while not done_event.is_set():
        do_one_step()


def _default_multiprocessing_context():
    try:
        methods = multiprocessing.get_all_start_methods()
    except Exception:
        methods = []
    if "fork" in methods:
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()


def _new_base_seed(generator: Optional["tp.Generator"]) -> int:
    with _use_generator(generator):
        return int(tp.empty((), dtype=tp.int64).random_(0, 2**63 - 1).item())


def _cpu_count() -> Optional[int]:
    """

    Prefers ``os.sched_getaffinity`` (respects cgroups / taskset) and falls
    back to ``os.cpu_count``.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def _worker_rationality_warning_msg(num_worker_suggest, num_worker_created,
                                    cpuset_checked: bool) -> str:
    suggested_max_worker_msg = (
        (
            (
                "Our suggested max number of worker in current system is {}{}, which is smaller "
                "than what this DataLoader is going to create."
            ).format(
                num_worker_suggest,
                "" if cpuset_checked else " (`cpuset` is not taken into account)",
            )
        )
        if num_worker_suggest is not None
        else (
            "DataLoader is not able to compute a suggested max number of worker in current system."
        )
    )
    return (
        f"This DataLoader will create {num_worker_created} worker processes in total. {suggested_max_worker_msg} "
        "Please be aware that excessive worker creation might get DataLoader running slow or even freeze, "
        "lower the worker number to avoid potential slowness/freeze if necessary."
    )


class _InfiniteConstantSampler(Sampler[int]):
    r"""Analogous to ``itertools.repeat(0, None)``, but with a ``__len__`` that
    raises ``TypeError``."""

    def __iter__(self) -> Iterator[int]:
        while True:
            yield 0

    def __len__(self) -> int:
        raise TypeError("Cannot determine the length of an infinite sampler")


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
class _BaseDataLoaderIter:
    def __init__(self, loader: "DataLoader"):
        self.loader = loader
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._sampler_iter = iter(self._index_sampler)
        self._num_workers = loader.num_workers
        self._num_yielded = 0

        # memory is enabled only when an accelerator exists.  An explicit
        # device opts into the requested allocator and therefore keeps the
        # flag even when the current process has no default accelerator.
        if not loader.pin_memory_device:
            if loader.pin_memory and not tp.cuda.is_available():
                warnings.warn(
                    "'pin_memory' argument is set as true but no accelerator is found, "
                    "then device pinned memory won't be used.",
                    stacklevel=2,
                )
            self._pin_memory = bool(loader.pin_memory and tp.cuda.is_available())
            self._pin_memory_device = None
        else:
            if not loader.pin_memory:
                warnings.warn(
                    "'pin_memory_device' is set but 'pin_memory' argument is not set, "
                    "then device pinned memory won't be used."
                    "please set 'pin_memory' to true, if you need to use the device pin memory",
                    stacklevel=2,
                )
            self._pin_memory = bool(loader.pin_memory)
            self._pin_memory_device = loader.pin_memory_device
        self._base_seed = _new_base_seed(loader.generator)

    def __iter__(self) -> "_BaseDataLoaderIter":
        return self

    def _next_data(self):
        raise NotImplementedError

    def __next__(self) -> Any:
        data = self._next_data()
        self._num_yielded += 1
        if (
            self._dataset_kind == _DatasetKind.Iterable
            and self._IterableDataset_len_called is not None
            and self._num_yielded > self._IterableDataset_len_called
        ):
            # See NOTE [ IterableDataset and __len__ ] in ``DataLoader.__len__``.
            warn_msg = (
                f"Length of IterableDataset {self.loader.dataset} was reported to be {self._IterableDataset_len_called}"
                f"(when accessing len(dataloader)), but {self._num_yielded} samples have been fetched. "
            )
            if self.loader.num_workers > 0:
                warn_msg += (
                    "For multiprocessing data-loading, this could be caused by not properly configuring the "
                    "IterableDataset replica at each worker. Please see "
                )
            warnings.warn(warn_msg, stacklevel=2)
        return data


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader: "DataLoader"):
        super().__init__(loader)
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            loader._dataset_kind,
            loader.dataset,
            loader._auto_collation,
            loader.collate_fn,
            loader.drop_last,
        )

    def _next_data(self) -> Any:
        index = next(self._sampler_iter)  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        return self.loader._finalize_batch(data)


class _MultiProcessDataLoaderIter(_BaseDataLoaderIter):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler.

    each worker runs fetch + collate on a per-worker index queue, results are
    streamed back through a shared result queue with bounded prefetch, and an
    optional pin-memory thread of the main process pins the returned batches.
    """

    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Workers are daemonic. The protocol between the main process and the
    # workers is that the main process sets `workers_done_event` (or marks a
    # worker unavailable for IterableDataset exhaustion), then sends `None` on
    # the worker's index queue; workers exit upon receiving `None`. Normal
    # data always travels as a 2-tuple `(idx, data)` where `idx` is the task
    # index assigned by `_try_put_index`, so `None` can never be confused with
    # the full note.

    def __init__(self, loader: "DataLoader"):
        super().__init__(loader)
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._persistent_workers = loader.persistent_workers
        self._in_order = loader.in_order

        if self._num_workers <= 0:
            raise AssertionError(
                "num_workers must be greater than 0 for MultiProcessingDataLoaderIter"
            )
        if self._prefetch_factor <= 0:
            raise AssertionError(
                "prefetch_factor must be greater than 0 for MultiProcessingDataLoaderIter"
            )

        if loader.multiprocessing_context is None:
            multiprocessing_context = _default_multiprocessing_context()
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn

        self._worker_result_queue = multiprocessing_context.Queue()
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            index_queue = multiprocessing_context.Queue()
            # Need to `cancel_join_thread` here! See sections (2) and (3b) of
            # NOTE [ Data Loader Multiprocessing Shutdown Logic ].
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_worker_loop,
                args=(
                    self._dataset_kind,
                    loader.dataset,
                    index_queue,
                    self._worker_result_queue,
                    self._workers_done_event,
                    self._auto_collation,
                    self._collate_fn,
                    self._drop_last,
                    self._base_seed,
                    self._worker_init_fn,
                    i,
                    self._num_workers,
                    self._persistent_workers,
                ),
            )
            w.daemon = True
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # The pin thread consumes the worker result queue and feeds a
            # plain `queue.Queue` that `__next__` reads.
            self._data_queue: Any = queue.Queue()
            pin_memory_thread = threading.Thread(
                target=_pin_memory_loop,
                args=(
                    self._worker_result_queue,
                    self._data_queue,
                    self._pin_memory_thread_done_event,
                ),
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue

        self._reset(loader, first_iter=True)

    def _reset(self, loader: "DataLoader", first_iter: bool = False) -> None:
        self._index_sampler = loader._index_sampler
        self._sampler_iter = iter(self._index_sampler)
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # map: task idx => (worker_id,)     if data isn't fetched (outstanding)
        #                  \ (worker_id, data) if already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = 0
        # Whether each worker still has work to do *for this epoch* (i.e., has
        # not exhausted its IterableDataset replica). All `True`s for map-style
        # datasets.
        self._workers_status = [True for i in range(self._num_workers)]
        # Outstanding-task count per worker; each worker holds at most
        # `prefetch_factor` tasks when `in_order=False` load-balances dispatch.
        self._workers_num_tasks = [0 for i in range(self._num_workers)]
        # Reset the worker queue cycle so a new epoch resumes at worker 0.
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        # We resume the prefetching in case it was enabled (persistent workers).
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(_ResumeIteration())
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                return_idx, return_data = self._get_data()
                if isinstance(return_idx, _ResumeIteration):
                    if return_data is not None:
                        raise AssertionError(
                            "Expected return_data to be None when resuming iteration"
                        )
                    resume_iteration_cnt -= 1
        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _try_get_data(self, timeout=_MP_STATUS_CHECK_INTERVAL):
        # Tries to fetch data from `self._data_queue` once for a given timeout.
        # Raises RuntimeError if any worker died unexpectedly. Returns a
        # 2-tuple (success, data).
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ", ".join(str(w.pid) for w in failed_workers)
                raise RuntimeError(
                    f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
                ) from e
            if isinstance(e, queue.Empty):
                return (False, None)
            raise

    def _get_data(self):
        # Fetches data from `self._data_queue`, checking worker/pin-thread
        # status every `_MP_STATUS_CHECK_INTERVAL` seconds.
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError(
                    f"DataLoader timed out after {self._timeout} seconds"
                )
        elif hasattr(self, "_pin_memory_thread"):
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            # while condition is false, i.e., pin_memory_thread died.
            raise RuntimeError("Pin memory thread exited unexpectedly")
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _next_data(self):
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an
            # IterableDataset), advance `self._rcvd_idx` to find the next
            # valid index.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info.get(self._rcvd_idx, None)
                if info:
                    worker_id = info[0]
                    if len(info) == 2 or self._workers_status[worker_id]:
                        break
                    del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                worker_id, data = self._task_info.pop(self._rcvd_idx)
                self._rcvd_idx += 1
                return self._process_data(data, worker_id)

            if self._shutdown or self._tasks_outstanding <= 0:
                raise AssertionError(
                    "Invalid iterator state: shutdown or no outstanding tasks when fetching next data"
                )
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                if not self._in_order:
                    # don't store it for later, process now; this keeps the
                    # object size manageable
                    worker_id = self._task_info.pop(idx)[0]
                    return self._process_data(data, worker_id)
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                worker_id = self._task_info.pop(idx)[0]
                self._rcvd_idx += 1
                return self._process_data(data, worker_id)

    def _try_put_index(self) -> None:
        max_tasks = self._prefetch_factor * self._num_workers
        if self._tasks_outstanding >= max_tasks:
            raise AssertionError(
                "Number of outstanding tasks exceeded maximum allowed tasks"
            )

        try:
            index = next(self._sampler_iter)
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                if self._in_order:
                    break
                elif self._workers_num_tasks[worker_queue_idx] < max_tasks // sum(
                    self._workers_status
                ):
                    # when in_order is False, distribute work to a worker with
                    # capacity; _workers_status only changes in this thread, so
                    # the sum is guaranteed > 0.
                    break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._workers_num_tasks[worker_queue_idx] += 1
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _process_data(self, data, worker_idx):
        self._workers_num_tasks[worker_idx] -= 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        # Pinning (if enabled) already happened in the pin-memory thread;
        # the optional device move is applied here in the main process.
        return self.loader._apply_device(data)

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False) -> None:
        # Mark a worker as having finished its work e.g., due to exhausting an
        # IterableDataset. Signal termination to that specific worker; joining
        # is deferred to `_shutdown_workers`.
        q = self._index_queues[worker_id]
        q.put(None)
        self._workers_status[worker_id] = False

    def _shutdown_workers(self) -> None:
        # Exit the pin memory thread first because exiting workers may leave
        # corrupted data in `worker_result_queue` which it reads from; then
        # signal and join the workers.
        if self._shutdown:
            return
        self._shutdown = True
        try:
            if hasattr(self, "_pin_memory_thread"):
                self._pin_memory_thread_done_event.set()
                # Send something to wake the pin thread up in case it waits.
                self._worker_result_queue.put((None, None))
                self._pin_memory_thread.join()
                self._worker_result_queue.cancel_join_thread()
                self._worker_result_queue.close()

            self._workers_done_event.set()
            for worker_id in range(len(self._workers)):
                if self._persistent_workers or self._workers_status[worker_id]:
                    self._mark_worker_as_unavailable(worker_id, shutdown=True)
            for w in self._workers:
                w.join(timeout=_MP_STATUS_CHECK_INTERVAL)
            for q in self._index_queues:
                q.cancel_join_thread()
                q.close()
        finally:
            for w in self._workers:
                if w.is_alive():
                    # Existing mechanisms try to make the workers exit
                    # peacefully, but in case we unfortunately reach here, we
                    # kill the worker.
                    w.terminate()
                    w.join(timeout=_MP_STATUS_CHECK_INTERVAL)

    def __del__(self):
        try:
            self._shutdown_workers()
        except Exception:
            pass


class DataLoader(Generic[_T_co]):
    """Data loader combines a dataset and a sampler, and provides an iterable
    over the given dataset.

    The DataLoader supports both map-style and iterable-style datasets with
    single- or multi-process loading, customizing loading order and optional
    automatic batching (collation).

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`, and
            :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main
            process. (default: ``0``)
        collate_fn (Callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s). Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy
            Tensors into CUDA page-locked host memory before returning them.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete
            batch, if the dataset size is not divisible by the batch size.
            (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for
            collecting a batch from workers. Should always be non-negative.
            (default: ``0``)
        worker_init_fn (Callable, optional): If not ``None``, this will be
            called on each worker subprocess with the worker id (an int in
            ``[0, num_workers - 1]``) as input, before data loading.
            (default: ``None``)
        multiprocessing_context (str or context, optional): start method or
            ``multiprocessing`` context used to spawn the workers, e.g.,
            ``"fork"`` or ``"spawn"``. If ``None``, the default context of the
            platform is used. (default: ``None``)
        generator (Generator, optional): If not ``None``, this RNG will be used
            by RandomSampler to generate random indexes. (default: ``None``)
        prefetch_factor (int, optional): Number of batches loaded in advance by
            each worker. ``2`` means there will be a total of
            2 * :attr:`num_workers` batches prefetched across all workers.
            (default: ``2`` when :attr:`num_workers > 0`; otherwise must be
            ``None``)
        persistent_workers (bool, optional): If ``True``, the data loader will
            not shut down the worker processes after a dataset has been
            consumed once. This allows to maintain the workers `Dataset`
            instances alive. (default: ``False``)
        pin_memory_device (str, optional): Deprecated device spelling kept for
            accelerator for pinned host allocations. (default: ``""``)
        in_order (bool, optional): If ``False``, the data loader will not
            enforce that batches returned from multiprocessing workers are
            provided in the order the sampler produced them. This enables
            faster delivery of batches that complete early, at the cost of
            batch order no longer being deterministic. (default: ``True``)
        device (str, optional): device to move batches to after collation.
    """

    dataset: Dataset[_T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    pin_memory_device: str
    drop_last: bool
    timeout: float
    sampler: Sampler | Iterable
    generator: Optional["tp.Generator"]
    __initialized = False

    def __init__(
        self,
        dataset: Dataset[_T_co],
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        sampler: Optional[Sampler | Iterable] = None,
        batch_sampler: Optional[Sampler | Iterable] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        multiprocessing_context=None,
        generator: Optional["tp.Generator"] = None,
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
        in_order: bool = True,
        device: Optional[str] = None,
    ) -> None:
        if num_workers < 0:
            raise ValueError(
                "num_workers option should be non-negative; "
                "use num_workers=0 to disable multiprocessing."
            )

        if timeout < 0:
            raise ValueError("timeout option should be non-negative")

        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError(
                "prefetch_factor option could only be specified in multiprocessing."
                "let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None."
            )
        elif num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError("prefetch_factor option should be non-negative")

        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers option needs num_workers > 0")

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.persistent_workers = persistent_workers
        self.in_order = in_order
        self.generator = generator
        self.device = device

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            # We cannot check `shuffle is not None` here, since previously
            # `shuffle=False` was the default.
            if shuffle not in {False, None}:
                raise ValueError(
                    f"DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle={shuffle}"
                )
            if sampler is not None:
                raise ValueError(
                    f"DataLoader with IterableDataset: expected unspecified sampler option, but got sampler={sampler}"
                )
            elif batch_sampler is not None:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    f"batch_sampler option, but got batch_sampler={batch_sampler}"
                )
        else:
            shuffle = bool(shuffle)
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive "
                    "with batch_size, shuffle, sampler, and "
                    "drop_last"
                )
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError(
                    "batch_size=None option disables auto-batching "
                    "and is mutually exclusive with drop_last"
                )

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset, generator=generator)
                else:
                    sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = default_collate
            else:
                collate_fn = default_convert

        self.collate_fn = collate_fn
        self._IterableDataset_len_called = (
            None  # See NOTE [ IterableDataset and __len__ ] in ``__len__``
        )

        self._iterator = None

        # the sampler/data stream.  Mutating these while a worker iterator is
        # alive can otherwise silently mix epochs or strand queued indices.
        self.__initialized = True

        self.check_worker_number_rationality()

    def check_worker_number_rationality(self) -> None:
        # Warn when num_workers exceeds the logical CPUs available to this
        if not self.num_workers or self.num_workers == 0:
            return
        max_num_worker_suggest = _cpu_count()
        cpuset_checked = hasattr(os, "sched_getaffinity")
        if max_num_worker_suggest is None or self.num_workers > max_num_worker_suggest:
            warnings.warn(
                _worker_rationality_warning_msg(
                    max_num_worker_suggest, self.num_workers, cpuset_checked
                ),
                stacklevel=2,
            )

    @property
    def _auto_collation(self) -> bool:
        return self.batch_sampler is not None

    @property
    def _index_sampler(self) -> Sampler | Iterable:
        # The actual sampler used for generating indices to read data at each
        # time. This would be ``batch_sampler`` if in auto-collation mode, and
        # ``sampler`` otherwise. We can't change ``sampler`` and
        # ``batch_sampler`` attributes for BC reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, str):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            "multiprocessing_context option "
                            f"should specify a valid start method in {valid_start_methods!r}, but got "
                            f"multiprocessing_context={multiprocessing_context!r}"
                        )
                    multiprocessing_context = multiprocessing.get_context(
                        multiprocessing_context
                    )

                if not isinstance(
                    multiprocessing_context, multiprocessing.context.BaseContext
                ):
                    raise TypeError(
                        "multiprocessing_context option should be a valid context "
                        "object or a string specifying the start method, but got "
                        f"multiprocessing_context={multiprocessing_context}"
                    )
            else:
                raise ValueError(
                    "multiprocessing_context can only be used with "
                    "multiprocessing (num_workers > 0)"
                )
        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, value):
        if self.__initialized and attr in (
            "batch_size",
            "batch_sampler",
            "sampler",
            "drop_last",
            "dataset",
            "persistent_workers",
        ):
            raise ValueError(
                f"{attr} attribute should not be set after "
                f"{self.__class__.__name__} is initialized"
            )
        super().__setattr__(attr, value)

    def _get_iterator(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessDataLoaderIter(self)

    def __iter__(self) -> Iterator[Any]:
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                # Create a single copy of this iterator per `DataLoader`
                # instance, so it can be re-used across epochs.
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    def __len__(self) -> int:
        if self._dataset_kind == _DatasetKind.Iterable:
            # NOTE [ IterableDataset and __len__ ]
            #
            # For `IterableDataset`, `__len__` could be inaccurate when one
            # naively does multi-processing data loading, since the samples
            # will be duplicated. However, no real use case should be actually
            # using that behavior, so it should count as a user error. We
            # should generally trust user code to do the proper thing (e.g.,
            # configure each replica differently in `__iter__`), and give us
            # the correct `__len__` if they choose to implement it (this will
            # still throw if the dataset does not implement a `__len__`).
            #
            # To provide a further warning, we track if `__len__` was called on
            # the `DataLoader`, save the returned value in
            # `_IterableDataset_len_called`, and warn if the iterator ends up
            # yielding more than this number of samples.

            # Cannot statically verify that dataset is Sized
            length = self._IterableDataset_len_called = len(self.dataset)
            if (
                self.batch_size is not None
            ):  # IterableDataset doesn't allow custom sampler or batch_sampler
                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = math.ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)

    def _finalize_batch(self, data: Any) -> Any:
        # Post-processing applied in the main process after a batch has been
        # skipped when no accelerator is available.
        if self.pin_memory and tp.cuda.is_available():
            data = _pin_memory(data)
        return self._apply_device(data)

    def _apply_device(self, data: Any) -> Any:
        if self.device is not None:
            data = self._move_to_device(data)
        return data

    def _move_to_device(self, data: Any) -> Any:
        if isinstance(data, Tensor):
            if self.device == "cpu":
                return data.cpu()
            return data.cuda(self.device, non_blocking=self.pin_memory)
        if isinstance(data, tuple):
            return tuple(self._move_to_device(item) for item in data)
        if isinstance(data, list):
            return [self._move_to_device(item) for item in data]
        if isinstance(data, dict):
            return {key: self._move_to_device(value) for key, value in data.items()}
        return data

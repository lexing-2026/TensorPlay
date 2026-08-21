import unittest
import warnings

import numpy as np

import tensorplay as tp
from tensorplay.utils import data as td

try:
    import torch
    from torch.utils import data as torchdata
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _ignore_unraisable(fn):
    """Ignore torch's known unraisable from half-initialized MP iterators."""
    if HAS_TORCH:
        try:
            import pytest
            return pytest.mark.filterwarnings(
                "ignore::pytest.PytestUnraisableExceptionWarning"
            )(fn)
        except ImportError:
            pass
    return fn


class _DS(td.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return tp.tensor([i])


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestDatasetParity(unittest.TestCase):
    def test_dataset_error_message(self):
        class D(td.Dataset):
            pass

        with self.assertRaisesRegex(NotImplementedError, "Subclasses of Dataset"):
            D()[0]

    def test_tensor_dataset(self):
        t = td.TensorDataset(tp.arange(5), tp.arange(5))
        tt = torchdata.TensorDataset(torch.arange(5), torch.arange(5))
        self.assertEqual(len(t), len(tt))
        self.assertEqual(t[2][0].tolist(), tt[2][0].tolist())
        with self.assertRaises(AssertionError):
            td.TensorDataset(tp.arange(5), tp.arange(3))
        with self.assertRaises(AssertionError):
            torchdata.TensorDataset(torch.arange(5), torch.arange(3))

    def test_concat_dataset(self):
        tp_ds = td.ConcatDataset([_DS(3), _DS(4)])
        t_ds = torchdata.ConcatDataset([_TDS(3), _TDS(4)])
        self.assertEqual(len(tp_ds), len(t_ds))
        for i in range(7):
            self.assertEqual(tp_ds[i].tolist(), t_ds[i].tolist())
        # negative indices
        self.assertEqual(tp_ds[-1].tolist(), t_ds[-1].tolist())
        with self.assertRaises(ValueError):
            tp_ds[-8]
        with self.assertRaises(ValueError):
            t_ds[-8]
        with self.assertRaisesRegex(AssertionError, "empty iterable"):
            td.ConcatDataset([])
        with self.assertRaisesRegex(AssertionError, "empty iterable"):
            torchdata.ConcatDataset([])
        # deprecated alias
        with self.assertWarns(FutureWarning):
            self.assertEqual(tp_ds.cummulative_sizes, tp_ds.cumulative_sizes)

    def test_subset(self):
        s = td.Subset(_DS(10), [2, 5, 7])
        ts = torchdata.Subset(_TDS(10), [2, 5, 7])
        self.assertEqual(len(s), len(ts))
        self.assertEqual(s[1].tolist(), ts[1].tolist())
        self.assertEqual(s[[0, 2]][0].tolist(), ts[[0, 2]][0].tolist())

    def test_random_split(self):
        for lengths in ([7, 3], [0.7, 0.3]):
            tp_parts = td.random_split(_DS(10), lengths)
            t_parts = torchdata.random_split(_TDS(10), lengths)
            self.assertEqual([len(p) for p in tp_parts], [len(p) for p in t_parts])
            tp_idx = sorted(sum([p.indices for p in tp_parts], []))
            t_idx = sorted(sum([p.indices for p in t_parts], []))
            self.assertEqual(tp_idx, t_idx)
        with self.assertRaises(ValueError):
            td.random_split(_DS(10), [5, 6])
        with self.assertRaises(ValueError):
            torchdata.random_split(_TDS(10), [5, 6])
        with self.assertRaisesRegex(ValueError, "Fraction at index"):
            td.random_split(_DS(10), [1.5, -0.5])
        with self.assertRaisesRegex(ValueError, "Fraction at index"):
            torchdata.random_split(_TDS(10), [1.5, -0.5])

    def test_stack_dataset(self):
        t = td.StackDataset(_DS(3), _DS(3))
        tt = torchdata.StackDataset(_TDS(3), _TDS(3))
        self.assertEqual(len(t), len(tt))
        self.assertEqual(t[1][1].tolist(), tt[1][1].tolist())
        d = td.StackDataset(a=_DS(3), b=_DS(3))
        dd = torchdata.StackDataset(a=_TDS(3), b=_TDS(3))
        self.assertEqual(sorted(d[0].keys()), sorted(dd[0].keys()))
        with self.assertRaises(ValueError):
            td.StackDataset()
        with self.assertRaises(ValueError):
            torchdata.StackDataset()

    def test_chain_dataset(self):
        class IDS(td.IterableDataset):
            def __iter__(self):
                return iter(range(4))

            def __len__(self):
                return 4

        class TIDS(torchdata.IterableDataset):
            def __iter__(self):
                return iter(range(4))

            def __len__(self):
                return 4

        c = td.ChainDataset([IDS(), IDS()])
        tc = torchdata.ChainDataset([TIDS(), TIDS()])
        self.assertEqual(list(c), list(tc))
        self.assertEqual(len(c), len(tc))
        # __add__ of iterables chains
        self.assertEqual(list(IDS() + IDS()), list(TIDS() + TIDS()))


class _TDS(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.tensor([i])


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestSamplerParity(unittest.TestCase):
    def test_sequential(self):
        self.assertEqual(
            list(td.SequentialSampler(_DS(7))),
            list(torchdata.SequentialSampler(_TDS(7))),
        )

    def test_random_sampler(self):
        s = td.RandomSampler(_DS(7), num_samples=20)
        ts = torchdata.RandomSampler(_TDS(7), num_samples=20)
        self.assertEqual(len(s), len(ts))
        self.assertEqual(len(list(s)), 20)
        self.assertEqual(len(list(ts)), 20)
        # replacement
        sr = td.RandomSampler(_DS(7), replacement=True, num_samples=50)
        tr = torchdata.RandomSampler(_TDS(7), replacement=True, num_samples=50)
        self.assertEqual(len(list(sr)), 50)
        self.assertEqual(len(list(tr)), 50)
        # validations
        with self.assertRaises(TypeError):
            td.RandomSampler(_DS(7), replacement="x")
        with self.assertRaises(TypeError):
            torchdata.RandomSampler(_TDS(7), replacement="x")
        with self.assertRaises(ValueError):
            td.RandomSampler(_DS(7), num_samples=0)
        with self.assertRaises(ValueError):
            torchdata.RandomSampler(_TDS(7), num_samples=0)

    def test_weighted_random_sampler(self):
        w = [0.1, 0.9, 0.4]
        s = td.WeightedRandomSampler(w, 100, True)
        ts = torchdata.WeightedRandomSampler(w, 100, True)
        self.assertEqual(len(s), len(ts))
        self.assertEqual(len(list(s)), 100)
        self.assertEqual(len(list(ts)), 100)
        with self.assertRaises(ValueError):
            td.WeightedRandomSampler(w, 0, True)
        with self.assertRaises(ValueError):
            torchdata.WeightedRandomSampler(w, 0, True)
        with self.assertRaises(ValueError):
            td.WeightedRandomSampler([[1, 2]], 5)
        with self.assertRaises(ValueError):
            torchdata.WeightedRandomSampler([[1, 2]], 5)

    def test_batch_sampler(self):
        for drop in (True, False):
            b = td.BatchSampler(td.SequentialSampler(_DS(10)), 3, drop)
            tb = torchdata.BatchSampler(torchdata.SequentialSampler(_TDS(10)), 3, drop)
            self.assertEqual(list(b), list(tb))
            self.assertEqual(len(b), len(tb))
        with self.assertRaises(ValueError):
            td.BatchSampler(td.SequentialSampler(_DS(10)), 0, False)
        with self.assertRaises(ValueError):
            torchdata.BatchSampler(torchdata.SequentialSampler(_TDS(10)), 0, False)
        with self.assertRaises(ValueError):
            td.BatchSampler(td.SequentialSampler(_DS(10)), 3, "x")
        with self.assertRaises(ValueError):
            torchdata.BatchSampler(torchdata.SequentialSampler(_TDS(10)), 3, "x")


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestCollateParity(unittest.TestCase):
    def _check(self, batch):
        out = td.default_collate(batch)
        tout = torchdata.default_collate(batch)
        if isinstance(out, dict):
            self.assertEqual(sorted(out.keys()), sorted(tout.keys()))
            for k in out:
                self.assertEqual(out[k].tolist(), tout[k].tolist())
        elif isinstance(out, (list, tuple)):
            self.assertEqual(len(out), len(tout))
            for a, b in zip(out, tout):
                self.assertEqual(a.tolist(), b.tolist())
        else:
            self.assertEqual(out.tolist(), tout.tolist())

    def test_numbers(self):
        self._check([0, 1, 2])
        self._check([0.5, 1.5, 2.5])
        self._check([True, False, True])

    def test_strings_and_bytes(self):
        self.assertEqual(td.default_collate(["a", "b"]), torchdata.default_collate(["a", "b"]))
        self.assertEqual(td.default_collate([b"a", b"b"]), torchdata.default_collate([b"a", b"b"]))

    def test_tensors(self):
        out = td.default_collate([tp.tensor([1, 2]), tp.tensor([3, 4])])
        self.assertEqual(out.tolist(), [[1, 2], [3, 4]])

    def test_numpy(self):
        self._check([np.array([1, 2]), np.array([3, 4])])
        self._check([np.float64(0.5), np.float64(1.5)])
        self._check([np.int64(1), np.int64(2)])

    def test_dict(self):
        self._check([{"A": 0, "B": 1}, {"A": 100, "B": 100}])
        self._check([{"A": np.array([1]), "B": np.array([2])}, {"A": np.array([3]), "B": np.array([4])}])
        out = td.default_collate([{"A": tp.tensor([1])}, {"A": tp.tensor([3])}])
        self.assertEqual(out["A"].tolist(), [[1], [3]])

    def test_namedtuple(self):
        import collections
        P = collections.namedtuple("Point", ["x", "y"])
        out = td.default_collate([P(0, 0), P(1, 1)])
        tout = torchdata.default_collate([P(0, 0), P(1, 1)])
        self.assertEqual(out.x.tolist(), tout.x.tolist())
        self.assertEqual(out.y.tolist(), tout.y.tolist())

    def test_sequence(self):
        self._check([(0, 1), (2, 3)])
        self._check([[0, 1], [2, 3]])

    def test_mismatched_length(self):
        with self.assertRaisesRegex(RuntimeError, "equal size"):
            td.default_collate([[0, 1], [2]])
        with self.assertRaisesRegex(RuntimeError, "equal size"):
            torchdata.default_collate([[0, 1], [2]])

    def test_unsupported(self):
        with self.assertRaisesRegex(TypeError, "default_collate: batch must contain"):
            td.default_collate([object()])
        with self.assertRaisesRegex(TypeError, "default_collate: batch must contain"):
            torchdata.default_collate([object()])

    def test_default_convert(self):
        out = td.default_convert([np.array([1, 2]), 3])
        tout = torchdata.default_convert([np.array([1, 2]), 3])
        self.assertEqual(out[0].tolist(), tout[0].tolist())
        self.assertEqual(out[1], tout[1])


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestDataLoaderParity(unittest.TestCase):
    def _check_batches(self, tp_loader, t_loader):
        # ``len()`` raises TypeError for IterableDatasets without __len__ on
        # both sides; compare lengths only when both succeed.
        try:
            tp_len = len(tp_loader)
        except TypeError:
            tp_len = None
        try:
            t_len = len(t_loader)
        except TypeError:
            t_len = None
        if tp_len is None or t_len is None:
            self.assertIsNone(tp_len, "len(dataloader) should raise TypeError like torch")
            self.assertIsNone(t_len, "len(dataloader) should raise TypeError like torch")
        else:
            self.assertEqual(tp_len, t_len)
        for b, tb in zip(tp_loader, t_loader):
            self.assertEqual(b.tolist(), tb.tolist())

    def test_basic(self):
        self._check_batches(td.DataLoader(_DS(10), batch_size=3), torchdata.DataLoader(_TDS(10), batch_size=3))
        self._check_batches(td.DataLoader(_DS(10), batch_size=3, drop_last=True), torchdata.DataLoader(_TDS(10), batch_size=3, drop_last=True))
        self._check_batches(td.DataLoader(_DS(10), batch_size=4), torchdata.DataLoader(_TDS(10), batch_size=4))

    def test_shuffle(self):
        tp_batches = [b.tolist() for b in td.DataLoader(_DS(10), batch_size=4, shuffle=True)]
        t_batches = [b.tolist() for b in torchdata.DataLoader(_TDS(10), batch_size=4, shuffle=True)]
        self.assertEqual(
            sorted(sum(tp_batches, [])),
            sorted(sum(t_batches, [])),
        )
        self.assertEqual(len(tp_batches), len(t_batches))

    def test_no_collation(self):
        tp_out = [x.tolist() for x in td.DataLoader(_DS(5), batch_size=None)]
        t_out = [x.tolist() for x in torchdata.DataLoader(_TDS(5), batch_size=None)]
        self.assertEqual(tp_out, t_out)

    def test_custom_sampler(self):
        s = td.SequentialSampler(_DS(5))
        ts = torchdata.SequentialSampler(_TDS(5))
        self._check_batches(td.DataLoader(_DS(5), sampler=s), torchdata.DataLoader(_TDS(5), sampler=ts))

    def test_custom_batch_sampler(self):
        bs = td.BatchSampler(td.SequentialSampler(_DS(6)), 2, False)
        tbs = torchdata.BatchSampler(torchdata.SequentialSampler(_TDS(6)), 2, False)
        self._check_batches(td.DataLoader(_DS(6), batch_sampler=bs), torchdata.DataLoader(_TDS(6), batch_sampler=tbs))

    def test_collate_fn(self):
        def cf(batch):
            return sum(batch)
        self._check_batches(td.DataLoader(_DS(5), batch_size=2, collate_fn=cf), torchdata.DataLoader(_TDS(5), batch_size=2, collate_fn=cf))

    def test_iterable_dataset(self):
        class IDS(td.IterableDataset):
            def __iter__(self):
                return iter(range(9))
        class TIDS(torchdata.IterableDataset):
            def __iter__(self):
                return iter(range(9))
        self._check_batches(td.DataLoader(IDS(), batch_size=3), torchdata.DataLoader(TIDS(), batch_size=3))
        self._check_batches(td.DataLoader(IDS(), batch_size=3, drop_last=True), torchdata.DataLoader(TIDS(), batch_size=3, drop_last=True))
        tp_out = [x for x in td.DataLoader(IDS(), batch_size=None)]
        t_out = [x for x in torchdata.DataLoader(TIDS(), batch_size=None)]
        self.assertEqual(tp_out, t_out)

    def _check_raises(self, fn, tf, exc, msg=None):
        with self.assertRaises(exc) as cm:
            fn()
        if msg:
            self.assertIn(msg, str(cm.exception))
        with self.assertRaises(exc) as tcm:
            tf()
        if msg:
            self.assertIn(msg, str(tcm.exception))

    def test_validation_messages(self):
        class IDS(td.IterableDataset):
            def __iter__(self):
                return iter(range(3))
        class TIDS(torchdata.IterableDataset):
            def __iter__(self):
                return iter(range(3))
        self._check_raises(
            lambda: td.DataLoader(_DS(5), num_workers=-1),
            lambda: torchdata.DataLoader(_TDS(5), num_workers=-1),
            ValueError, "num_workers option should be non-negative",
        )
        self._check_raises(
            lambda: td.DataLoader(_DS(5), timeout=-1),
            lambda: torchdata.DataLoader(_TDS(5), timeout=-1),
            ValueError, "timeout option should be non-negative",
        )
        self._check_raises(
            lambda: td.DataLoader(_DS(5), shuffle=True, sampler=td.SequentialSampler(_DS(5))),
            lambda: torchdata.DataLoader(_TDS(5), shuffle=True, sampler=torchdata.SequentialSampler(_TDS(5))),
            ValueError, "sampler option is mutually exclusive with shuffle",
        )
        self._check_raises(
            lambda: td.DataLoader(_DS(5), batch_size=4, batch_sampler=td.BatchSampler(td.SequentialSampler(_DS(5)), 2, False)),
            lambda: torchdata.DataLoader(_TDS(5), batch_size=4, batch_sampler=torchdata.BatchSampler(torchdata.SequentialSampler(_TDS(5)), 2, False)),
            ValueError, "batch_sampler option is mutually exclusive",
        )
        self._check_raises(
            lambda: td.DataLoader(_DS(5), batch_size=None, drop_last=True),
            lambda: torchdata.DataLoader(_TDS(5), batch_size=None, drop_last=True),
            ValueError, "batch_size=None option disables auto-batching",
        )
        self._check_raises(
            lambda: td.DataLoader(IDS(), shuffle=True),
            lambda: torchdata.DataLoader(TIDS(), shuffle=True),
            ValueError, "expected unspecified shuffle option",
        )
        self._check_raises(
            lambda: td.DataLoader(IDS(), sampler=td.SequentialSampler(_DS(5))),
            lambda: torchdata.DataLoader(TIDS(), sampler=torchdata.SequentialSampler(_TDS(5))),
            ValueError, "expected unspecified sampler option",
        )
        self._check_raises(
            lambda: td.DataLoader(IDS(), batch_sampler=td.BatchSampler(td.SequentialSampler(_DS(5)), 2, False)),
            lambda: torchdata.DataLoader(TIDS(), batch_sampler=torchdata.BatchSampler(torchdata.SequentialSampler(_TDS(5)), 2, False)),
            ValueError, "expected unspecified batch_sampler option",
        )

    @_ignore_unraisable
    def test_mp_option_validation(self):
        # prefetch_factor / persistent_workers / multiprocessing_context follow
        # torch's validation messages.
        self._check_raises(
            lambda: td.DataLoader(_DS(5), num_workers=0, prefetch_factor=2),
            lambda: torchdata.DataLoader(_TDS(5), num_workers=0, prefetch_factor=2),
            ValueError, "prefetch_factor option could only be specified in multiprocessing",
        )
        self._check_raises(
            lambda: td.DataLoader(_DS(5), num_workers=2, prefetch_factor=-1),
            lambda: torchdata.DataLoader(_TDS(5), num_workers=2, prefetch_factor=-1),
            ValueError, "prefetch_factor option should be non-negative",
        )
        # prefetch_factor=0 passes construction but fails at iteration (torch
        # parity). torch's own half-initialized iterator leaks an unraisable
        # from __del__, so ignore that warning for this test.
        self._check_raises(
            lambda: list(td.DataLoader(_DS(5), num_workers=2, prefetch_factor=0)),
            lambda: list(torchdata.DataLoader(_TDS(5), num_workers=2, prefetch_factor=0)),
            AssertionError, "prefetch_factor must be greater than 0",
        )
        self._check_raises(
            lambda: td.DataLoader(_DS(5), num_workers=0, persistent_workers=True),
            lambda: torchdata.DataLoader(_TDS(5), num_workers=0, persistent_workers=True),
            ValueError, "persistent_workers option needs num_workers > 0",
        )
        with self.assertRaises(ValueError) as cm:
            td.DataLoader(_DS(5), num_workers=2, multiprocessing_context="bogus")
        self.assertIn("should specify a valid start method", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            td.DataLoader(_DS(5), num_workers=0, multiprocessing_context="fork")
        self.assertIn("multiprocessing_context can only be used with", str(cm.exception))
        # default prefetch_factor is applied when num_workers > 0
        self.assertEqual(td.DataLoader(_DS(5), num_workers=2).prefetch_factor, 2)


    def test_in_order_false(self):
        # out-of-order delivery still yields every batch exactly once
        loader = td.DataLoader(_DS(12), batch_size=3, num_workers=2, in_order=False)
        batches = [tuple(sorted(x[0] for x in b)) for b in loader]
        flat = sorted(v for b in batches for v in b)
        self.assertEqual(flat, list(range(12)))
        self.assertEqual(len(batches), 4)

    def test_worker_rationality_warning(self):
        import os as _os
        cpus = len(_os.sched_getaffinity(0)) if hasattr(_os, "sched_getaffinity") \
            else (_os.cpu_count() or 1)
        with self.assertWarns(UserWarning) as cm:
            td.DataLoader(_DS(4), batch_size=2, num_workers=cpus + 1)
        self.assertIn("suggested max number of worker", str(cm.warning))

    def test_no_rationality_warning_when_reasonable(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            td.DataLoader(_DS(4), batch_size=2, num_workers=1)


class TestDataLoaderFunctional(unittest.TestCase):
    def test_multiprocess(self):
        batches = [b.tolist() for b in td.DataLoader(_DS(12), batch_size=3, num_workers=2)]
        self.assertEqual(batches, [[[0], [1], [2]], [[3], [4], [5]], [[6], [7], [8]], [[9], [10], [11]]])

    def test_worker_init_fn_and_info(self):
        import multiprocessing
        manager = multiprocessing.Manager()
        # A plain list cannot propagate appends from forked children; use a
        # managed list (same approach as torch's own DataLoader tests).
        seen = manager.list()

        def worker_init_fn(worker_id):
            info = td.get_worker_info()
            seen.append((worker_id, info.num_workers, info.seed, len(info.dataset)))

        base_seed = 12345
        gen = tp.Generator()
        gen.manual_seed(base_seed)
        list(td.DataLoader(_DS(8), batch_size=2, num_workers=2, worker_init_fn=worker_init_fn, generator=gen))
        self.assertEqual(len(seen), 2)
        self.assertEqual(sorted(w[0] for w in seen), [0, 1])
        self.assertTrue(all(w[1] == 2 for w in seen))
        # per-worker seeds are base_seed + worker_id (torch contract), where
        # base_seed is drawn from the loader generator
        seeds = sorted(w[2] for w in seen)
        self.assertEqual(seeds[1], seeds[0] + 1)
        self.assertTrue(all(w[3] == 8 for w in seen))
        self.assertIsNone(td.get_worker_info())
        manager.shutdown()

    def test_iterable_dataset_multiprocess_sharding(self):
        import multiprocessing

        class ShardIDS(td.IterableDataset):
            def __init__(self, n):
                self.n = n

            def __iter__(self):
                info = td.get_worker_info()
                if info is None:
                    return iter(range(self.n))
                # split workload across workers (torch docs pattern)
                per_worker = (self.n + info.num_workers - 1) // info.num_workers
                start = info.id * per_worker
                return iter(range(start, min(start + per_worker, self.n)))

        loader = td.DataLoader(ShardIDS(9), batch_size=2, num_workers=3)
        got = sorted(x.item() for b in loader for x in b)
        self.assertEqual(got, list(range(9)))
        # no __len__ on the dataset -> len(dataloader) raises (torch parity)
        with self.assertRaises(TypeError):
            len(loader)

    def test_persistent_workers_reuse_processes(self):
        import multiprocessing
        manager = multiprocessing.Manager()
        pids = manager.list()

        def worker_init_fn(worker_id):
            import os
            pids.append(os.getpid())

        loader = td.DataLoader(_DS(8), batch_size=4, num_workers=2,
                               persistent_workers=True, worker_init_fn=worker_init_fn)
        epochs = []
        for _ in range(2):
            batch_vals = [b.tolist() for b in loader]
            epochs.append(batch_vals)
            worker_pids = set(pids)
            epochs.append(worker_pids)
        # same data each epoch
        self.assertEqual(epochs[0], epochs[2])
        # workers were NOT re-created between epochs
        self.assertEqual(epochs[1], epochs[3])
        self.assertEqual(len(epochs[1]), 2)
        loader._iterator._shutdown_workers()
        manager.shutdown()

    def test_worker_exception_propagation(self):
        class BadDS(td.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, i):
                if i == 2:
                    raise ValueError("boom at 2")
                return tp.tensor([i])

        # torch re-raises the original exception type, wrapped with the
        # worker origin and original traceback
        with self.assertRaises(ValueError) as cm:
            list(td.DataLoader(BadDS(), batch_size=2, num_workers=2))
        self.assertIn("Caught ValueError in DataLoader worker process", str(cm.exception))
        self.assertIn("boom at 2", str(cm.exception))

    def test_timeout(self):
        import time

        class SlowDS(td.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, i):
                time.sleep(0.5)
                return tp.tensor([i])

        with self.assertRaises(RuntimeError) as cm:
            list(td.DataLoader(SlowDS(), batch_size=1, num_workers=1, timeout=0.05))
        self.assertIn("DataLoader timed out after 0.05 seconds", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
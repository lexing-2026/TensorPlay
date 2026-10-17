"""Throwaway probe: which reduction call ends the process on the macOS 14
runner, and which reduction answers wrongly.

Every case runs in its own child process. A case that kills the interpreter
would otherwise take every later case with it, and the run would show one death
instead of the shape of the problem. Parent and child evaluate the same
expression string, the child over MPS tensors and the parent over CPU tensors,
so a case reports both survival and whether the values are the ones the same
call on CPU produces.

Cases are grouped by what a fix would have to change: the single-axis form of a
reduction selector against the axes-array form, the reshape that follows a
reduction which drops a rank, keepdim, rank, storage type, and whether a
reduction that runs after another reduction in the same process behaves
differently (the graph cache keeps entries for the life of the process).
"""

import math
import platform
import subprocess
import sys
import time
import unittest

import tensorplay as tp

_CHILD_TIMEOUT = 300


def _nest(flat, shape):
    """Nest `flat` into `shape`, so a bool operand needs no sequence of bools."""
    if len(shape) == 1:
        return flat[:shape[0]]
    step = math.prod(shape[1:])
    return [_nest(flat[i * step:(i + 1) * step], shape[1:])
            for i in range(shape[0])]


def _source(shape, dtype_name, device):
    numel = math.prod(shape)
    flags = _nest([i % 3 != 0 for i in range(numel)], shape)
    return (
        "import tensorplay as tp\n"
        f"x = tp.reshape(tp.arange({numel}, dtype=tp.{dtype_name}, "
        f"device='{device}'), {shape})\n"
        f"b = tp.tensor({flags}, dtype=tp.bool, device='{device}')\n"
    )


def _run_child(code):
    """Return (exit status, stdout, stderr tail) for a child that may die."""
    try:
        done = subprocess.run([sys.executable, "-c", code],
                              capture_output=True, text=True,
                              timeout=_CHILD_TIMEOUT)
        return done.returncode, done.stdout.strip(), done.stderr[-1500:]
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or b""
        err = exc.stderr or b""
        if isinstance(out, bytes):
            out, err = out.decode(), err.decode()
        return "TIMEOUT", out.strip(), err[-1500:]


# (label, shape, dtype name, expression)
CASES = [
    # Full reductions: the nil-axes form, with no reshape behind it.
    ("r3_full_sum", [2, 3, 4], "float32", "tp.sum(x).cpu().item()"),
    ("r3_full_mean", [2, 3, 4], "float32", "tp.mean(x).cpu().item()"),
    ("r3_full_max", [2, 3, 4], "float32", "tp.max(x).cpu().item()"),
    ("r3_full_prod", [2, 2, 6], "float32", "tp.prod(x).cpu().item()"),
    # One axis of a rank-3 tensor: the reduced rank is 2, then a reshape to the
    # shape the caller asked for. This is the call reported to end the process.
    ("r3_sum_axis1", [2, 3, 4], "float32", "tp.sum(x, dim=1).cpu().tolist()"),
    ("r3_sum_axis0", [2, 3, 4], "float32", "tp.sum(x, dim=0).cpu().tolist()"),
    ("r3_sum_axis2", [2, 3, 4], "float32", "tp.sum(x, dim=2).cpu().tolist()"),
    ("r3_sum_neg1", [2, 3, 4], "float32", "tp.sum(x, dim=-1).cpu().tolist()"),
    ("r3_mean_axis1", [2, 3, 4], "float32", "tp.mean(x, dim=1).cpu().tolist()"),
    ("r3_prod_axis1", [2, 3, 4], "float32", "tp.prod(x, dim=1).cpu().tolist()"),
    ("r3_min_axis1", [2, 3, 4], "float32", "tp.min(x, dim=1).cpu().tolist()"),
    ("r3_all_axis1", [2, 3, 2], "bool", "tp.all(b, dim=1).cpu().tolist()"),
    ("r3_any_axis1", [2, 3, 2], "bool", "tp.any(b, dim=1).cpu().tolist()"),
    # Two axes: one array of them, or two selectors applied highest-first.
    ("r3_sum_02", [2, 3, 4], "float32", "tp.sum(x, dim=[0, 2]).cpu().tolist()"),
    ("r3_sum_20", [2, 3, 4], "float32", "tp.sum(x, dim=[2, 0]).cpu().tolist()"),
    ("r3_sum_01", [2, 3, 4], "float32", "tp.sum(x, dim=[0, 1]).cpu().tolist()"),
    ("r3_mean_02", [2, 3, 4], "float32", "tp.mean(x, dim=[0, 2]).cpu().tolist()"),
    # keepdim: the rank the graph produces and the rank the result is asked for
    # differ by the axes that were kept.
    ("r3_sum_axis1_keep", [2, 3, 4], "float32",
     "tp.sum(x, dim=1, keepdim=True).cpu().tolist()"),
    ("r3_sum_02_keep", [2, 3, 4], "float32",
     "tp.sum(x, dim=[0, 2], keepdim=True).cpu().tolist()"),
    ("r3_sum_full_keep", [2, 3, 4], "float32",
     "tp.sum(x, keepdim=True).cpu().tolist()"),
    # Rank 2: a partial reduction leaves rank 1, so the reshape is same-rank.
    ("r2_sum_axis1", [3, 4], "float32", "tp.sum(x, dim=1).cpu().tolist()"),
    ("r2_sum_axis0", [3, 4], "float32", "tp.sum(x, dim=0).cpu().tolist()"),
    ("r2_mean_axis1", [3, 4], "float32", "tp.mean(x, dim=1).cpu().tolist()"),
    # Higher ranks than the reported case.
    ("r4_sum_axis1", [2, 3, 2, 2], "float32",
     "tp.sum(x, dim=1).cpu().tolist()"),
    ("r4_sum_axis3", [2, 3, 2, 2], "float32",
     "tp.sum(x, dim=3).cpu().tolist()"),
    ("r4_sum_13", [2, 3, 2, 2], "float32",
     "tp.sum(x, dim=[1, 3]).cpu().tolist()"),
    ("r5_sum_axis1", [2, 2, 2, 1, 3], "float32",
     "tp.sum(x, dim=1).cpu().tolist()"),
    # Other storage types through the same path.
    ("r3_sum_axis1_i64", [2, 3, 4], "int64", "tp.sum(x, dim=1).cpu().tolist()"),
    ("r3_sum_axis1_i32", [2, 3, 4], "int32", "tp.sum(x, dim=1).cpu().tolist()"),
    ("r3_sum_axis1_f16", [2, 3, 4], "float16", "tp.sum(x, dim=1).cpu().tolist()"),
    # Value and index from one dispatch.
    ("r3_maxdim_axis1", [2, 3, 4], "float32",
     "[t.cpu().tolist() for t in tp.max(x, dim=1)]"),
    ("r3_argmax_axis1", [2, 3, 4], "float32", "tp.argmax(x, dim=1).cpu().tolist()"),
    # Views: the operand is made contiguous before the graph sees it.
    ("r3_sum_axis1_of_transpose", [2, 3, 4], "float32",
     "tp.sum(tp.transpose(x, 0, 1), dim=1).cpu().tolist()"),
    ("r3_sum_axis1_of_slice", [2, 3, 4], "float32",
     "tp.sum(x[0:2, 1:3, 1:4], dim=1).cpu().tolist()"),
    # Size-1 dimensions: the reduced axis may hold one element, and a kept axis
    # may.
    ("r3_sum_axis1_on_1_3_4", [1, 3, 4], "float32",
     "tp.sum(x, dim=1).cpu().tolist()"),
    ("r3_sum_axis2_on_2_3_1", [2, 3, 1], "float32",
     "tp.sum(x, dim=2).cpu().tolist()"),
    ("r3_sum_axis1_on_2_1_4", [2, 1, 4], "float32",
     "tp.sum(x, dim=1).cpu().tolist()"),
    # Rank 1: reducing the only axis is a full reduction.
    ("r1_sum_axis0", [6], "float32", "tp.sum(x, dim=0).cpu().tolist()"),
    ("r1_mean_axis0", [6], "float32", "tp.mean(x, dim=0).cpu().tolist()"),
    # All axes named at once rather than left to the reduce-all form.
    ("r3_sum_012", [2, 3, 4], "float32",
     "tp.sum(x, dim=[0, 1, 2]).cpu().tolist()"),
    ("r3_sum_012_keep", [2, 3, 4], "float32",
     "tp.sum(x, dim=[0, 1, 2], keepdim=True).cpu().tolist()"),
    # Variance and standard deviation: a different graph in the same file, whose
    # single-axis case uses the variance selector with an axes array and whose
    # multi-axis case reduces with per-axis sums instead.
    ("r3_var_axis1", [2, 3, 4], "float32",
     "tp.var(x, dim=1).cpu().tolist()"),
    ("r3_std_axis1", [2, 3, 4], "float32",
     "tp.std(x, dim=1).cpu().tolist()"),
    ("r3_std_02", [2, 3, 4], "float32",
     "tp.std(x, dim=[0, 2]).cpu().tolist()"),
    ("r3_std_full", [2, 3, 4], "float32", "tp.std(x).cpu().tolist()"),
    ("r2_std_axis1", [3, 4], "float32", "tp.std(x, dim=1).cpu().tolist()"),
]

# Reductions that follow another reduction inside one process: the cache is what
# makes the order visible.
SEQUENCES = [
    ("full_then_partial", ["tp.sum(x).cpu().item()",
                           "tp.sum(x, dim=1).cpu().tolist()"]),
    ("partial_then_full", ["tp.sum(x, dim=1).cpu().tolist()",
                           "tp.sum(x).cpu().item()"]),
    ("partial_then_other_partial", ["tp.sum(x, dim=1).cpu().tolist()",
                                    "tp.sum(x, dim=2).cpu().tolist()"]),
    ("partial_twice", ["tp.sum(x, dim=1).cpu().tolist()",
                       "tp.sum(x, dim=1).cpu().tolist()"]),
    ("mean_then_sum_same_dim", ["tp.mean(x, dim=1).cpu().tolist()",
                                "tp.sum(x, dim=1).cpu().tolist()"]),
    ("max_then_sum_same_dim", ["tp.max(x, dim=1).cpu().tolist()",
                               "tp.sum(x, dim=1).cpu().tolist()"]),
]


class ProbeReduce(unittest.TestCase):
    def test_01_report(self):
        print(f"host={platform.platform()} mac={platform.mac_ver()[0]} "
              f"machine={platform.machine()} mps={tp.mps.is_available()}",
              flush=True)

    def test_02_cases(self):
        dead = []
        for label, shape, dtype_name, expr in CASES:
            with self.subTest(case=label):
                started = time.time()
                status, out, err = _run_child(
                    _source(shape, dtype_name, "mps")
                    + f"print({expr}, flush=True)\n")
                reference = None
                try:
                    namespace = {}
                    exec(_source(shape, dtype_name, "cpu"), namespace)
                    reference = repr(eval(expr, namespace))  # noqa: S307
                except Exception as exc:
                    reference = f"reference failed: {exc!r}"
                print(f"{label}: exit={status} "
                      f"elapsed={time.time() - started:.1f}s got={out!r} "
                      f"want={reference}", flush=True)
                if status != 0:
                    print(f"{label}: stderr tail=\n{err}", flush=True)
                    dead.append(f"{label} (exit {status})")
        print("PROBE-CASES-DONE", flush=True)
        self.assertFalse(dead, f"cases that did not return: {dead}")

    def test_03_same_process_pairs(self):
        dead = []
        for label, exprs in SEQUENCES:
            with self.subTest(sequence=label):
                code = _source([2, 3, 4], "float32", "mps")
                for index, expr in enumerate(exprs):
                    code += f"print({index}, {expr}, flush=True)\n"
                status, out, err = _run_child(code)
                print(f"{label}: exit={status} out={out!r}", flush=True)
                if status != 0:
                    print(f"{label}: stderr tail=\n{err}", flush=True)
                    dead.append(f"{label} (exit {status})")
        print("PROBE-SEQUENCES-DONE", flush=True)
        self.assertFalse(dead, f"sequences that did not return: {dead}")

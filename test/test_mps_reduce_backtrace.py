"""Throwaway probe: the faulting stack behind a reduction that dies on the
macOS 14 runner.

The single-axis partial reductions of a rank-3 tensor end the process without
writing anything to stderr, which says only that the fault is below Python. Run
the same calls under a debugger and print the frames, so the answer says which
function faulted: a selector inside the graph framework, the encode of the
compiled graph, or our own code around it.

Cases include a reduction that survives next to ones that do not, to show that
the stack is really about the call and not about the harness.
"""

import platform
import shutil
import subprocess
import sys
import unittest

import tensorplay as tp

_X = ("import tensorplay as tp\n"
      "x = tp.reshape(tp.arange(24, dtype=tp.float32, device='mps'), [2, 3, 4])\n"
      "y = tp.reshape(tp.arange(24, dtype=tp.float32, device='mps'), [2, 3, 2, 2])\n")

CASES = [
    ("sum_axis1", "tp.sum(x, dim=1).cpu().tolist()"),
    ("mean_axis1", "tp.mean(x, dim=1).cpu().tolist()"),
    ("sum_axis1_keep", "tp.sum(x, dim=1, keepdim=True).cpu().tolist()"),
    ("sum_02", "tp.sum(x, dim=[0, 2]).cpu().tolist()"),
    ("r4_sum_axis3", "tp.sum(y, dim=3).cpu().tolist()"),
    ("r4_sum_axis1", "tp.sum(y, dim=1).cpu().tolist()"),
    ("maxdim_axis1", "[t.cpu().tolist() for t in tp.max(x, dim=1)]"),
]

_FRAME_MARKS = ("frame #", "Stop reason", "EXC_", "Signal", "libp10",
                "MetalPerformanceShadersGraph", "Metal", "tensorplay")


class ProbeBacktrace(unittest.TestCase):
    def test_01_tooling(self):
        lldb = shutil.which("lldb")
        print(f"host={platform.platform()} mac={platform.mac_ver()[0]} "
              f"machine={platform.machine()} mps={tp.mps.is_available()} "
              f"lldb={lldb}", flush=True)
        self.assertIsNotNone(lldb, "no debugger on this runner")

    def test_02_stacks(self):
        lldb = shutil.which("lldb")
        for label, expr in CASES:
            with self.subTest(case=label):
                code = _X + f"print({expr}, flush=True)\n"
                done = subprocess.run(
                    [lldb, "--batch", "-o", "run", "-o", "thread backtrace",
                     "-o", "quit", "--", sys.executable, "-c", code],
                    capture_output=True, text=True, timeout=420)
            out = (done.stdout or "") + (done.stderr or "")
            keep = [line.strip() for line in out.splitlines()
                    if any(mark in line for mark in _FRAME_MARKS)]
            print(f"--- {label}: {len(keep)} stack lines", flush=True)
            for line in keep[:26]:
                print(f"    {line[:170]}", flush=True)
            if not keep:
                print(f"    raw tail:\n{out[-800:]}", flush=True)

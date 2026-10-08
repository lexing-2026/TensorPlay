"""A failed device-side check stops the program in release builds too.

The check faults the launch, which leaves the CUDA context unusable, so it
runs in a child process.
"""

import os
import subprocess
import sys

import pytest

import tensorplay as tp

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA not available")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CHILD = """
import tensorplay as tp
check = tp.ops.tp._assert_async.msg
check(tp.ones(1, device="cuda"), "kept")
tp.cuda.synchronize()
print("held", flush=True)
check(tp.zeros(1, device="cuda"), "zero on the device")
tp.cuda.synchronize()
print("unreachable", flush=True)
"""


def test_failed_device_check_stops_the_program():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, (_REPO, os.environ.get("PYTHONPATH")))))
    result = subprocess.run(
        [sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=300, env=env
    )
    assert "held" in result.stdout, result.stderr
    assert "unreachable" not in result.stdout
    assert result.returncode != 0

"""Tests for the elastic launch stack, rendezvous, debug server, and
flight recorder."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORKER_OK = """\
import os, sys
print(f"rank {os.environ['RANK']}/{os.environ['WORLD_SIZE']}", flush=True)
"""

WORKER_FLAKY = """\
import os, sys
if os.environ.get("TORCHELASTIC_RESTART_COUNT", "0") == "0":
    sys.exit(7)
print("recovered", flush=True)
"""

WORKER_FAIL = """\
import sys
sys.exit(3)
"""


def _run_launch(args, timeout=120):
    return subprocess.run(
        [sys.executable, "-m", "tensorplay.distributed.run", *args],
        capture_output=True, text=True, timeout=timeout, cwd=REPO,
    )


class TestStoreCompareAndSwap(unittest.TestCase):
    def test_tcp_cas(self):
        from tensorplay.distributed._store import TCPStore
        server = TCPStore("127.0.0.1", 0, is_master=True)
        client = TCPStore("127.0.0.1", server.port, is_master=False)
        swapped, current = client.compare_and_swap("k", b"", b"v1")
        self.assertTrue(swapped)
        swapped, current = client.compare_and_swap("k", b"", b"v2")
        self.assertFalse(swapped)
        self.assertEqual(current, b"v1")
        swapped, _ = client.compare_and_swap("k", b"v1", b"v2")
        self.assertTrue(swapped)
        self.assertEqual(client.get("k"), b"v2")
        server.stop()

    def test_file_cas(self):
        from tensorplay.distributed._store import FileStore
        fd, path = tempfile.mkstemp(prefix="tp_cas_")
        os.close(fd)
        os.unlink(path)
        try:
            store = FileStore(path)
            swapped, current = store.compare_and_swap("k", b"", b"a")
            self.assertTrue(swapped)
            swapped, current = store.compare_and_swap("k", b"", b"b")
            self.assertFalse(swapped)
            self.assertEqual(current, b"a")
        finally:
            os.unlink(path)


class TestRendezvous(unittest.TestCase):
    def _params(self, endpoint="", backend="core", **config):
        from tensorplay.distributed.elastic.rendezvous import RendezvousParameters
        return RendezvousParameters(
            backend=backend, endpoint=endpoint, run_id=f"test_{time.time()}",
            local_world_size=1, config=config,
        )

    def test_static_next_rendezvous(self):
        from tensorplay.distributed.elastic.rendezvous.static_tcp_rendezvous import (
            StaticTCPRendezvous,
        )
        handler = StaticTCPRendezvous(self._params(endpoint=""))
        info = handler.next_rendezvous()
        self.assertEqual((info.rank, info.world_size), (0, 1))
        self.assertTrue(info.bootstrap_store_info.port > 0)

    def test_dynamic_single_node_completes(self):
        from tensorplay.distributed.elastic.rendezvous.dynamic_rendezvous import (
            DynamicRendezvousHandler, RendezvousSettings,
        )
        from datetime import timedelta
        settings = RendezvousSettings(
            join=timedelta(seconds=30), last_call=timedelta(seconds=1),
            min_nodes=1, max_nodes=2,
        )
        handler = DynamicRendezvousHandler.from_backend(
            run_id=f"dyn_{time.time()}", endpoint="localhost:0",
            settings=settings, store_type="tcp",
        )
        info = handler.next_rendezvous()
        self.assertEqual(info.world_size, 1)
        self.assertEqual(info.rank, 0)
        self.assertFalse(handler.is_closed())
        self.assertEqual(handler.num_nodes_waiting(), 0)
        self.assertTrue(handler.shutdown())
        self.assertTrue(handler.is_closed())

    def test_registry(self):
        from tensorplay.distributed.elastic.rendezvous import create_handler
        handler = create_handler(self._params(endpoint="", backend_hint="static"))
        self.assertIsNotNone(handler)

    def test_p10d_registry_and_export(self):
        from tensorplay.distributed.elastic.rendezvous import (
            P10dRendezvousBackend,
            create_handler,
        )

        fd, path = tempfile.mkstemp(prefix="tp_p10d_rdzv_")
        os.close(fd)
        os.unlink(path)
        handler = None
        try:
            params = self._params(
                endpoint=path,
                backend="p10d",
                store_type="file",
            )
            handler = create_handler(params)
            self.assertEqual(handler.get_backend(), "p10d")
            self.assertIsInstance(handler._backend, P10dRendezvousBackend)
        finally:
            if handler is not None:
                handler.shutdown()
            if os.path.exists(path):
                os.unlink(path)

    def test_p10d_state_fencing(self):
        from tensorplay.distributed.elastic.rendezvous.p10d_rendezvous_backend import (
            create_backend,
        )

        fd, path = tempfile.mkstemp(prefix="tp_p10d_state_")
        os.close(fd)
        os.unlink(path)
        try:
            backend, store = create_backend(
                self._params(
                    endpoint=path,
                    backend="p10d",
                    store_type="file",
                )
            )
            self.assertIsNone(backend.get_state())
            first = backend.set_state(b"first", 0)
            self.assertEqual(first, (b"first", 1))
            self.assertEqual(backend.set_state(b"stale", 0), first)
            self.assertEqual(backend.set_state(b"second", 1), (b"second", 2))
            self.assertEqual(backend.get_state(), (b"second", 2))
            del store
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestStoreUtils(unittest.TestCase):
    def test_barrier_and_synchronize(self):
        from tensorplay.distributed._store import FileStore
        from tensorplay.distributed.elastic.utils.store import barrier, synchronize
        fd, path = tempfile.mkstemp(prefix="tp_barrier_")
        os.close(fd)
        os.unlink(path)
        try:
            store = FileStore(path)
            store.timeout = 5
            synchronize(store, b"payload", rank=0, world_size=1, key_prefix="s/")
            self.assertEqual(store.get("s/0"), b"payload")
            with barrier(store, world_size=1, key_prefix="b/"):
                pass
            with barrier(store, world_size=1, key_prefix="b/"):
                pass
        finally:
            os.unlink(path)


class TestLaunch(unittest.TestCase):
    def _script(self, text):
        fd, path = tempfile.mkstemp(suffix=".py", prefix="tp_worker_")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self.addCleanup(os.unlink, path)
        return path

    def test_static_two_workers(self):
        script = self._script(WORKER_OK)
        result = _run_launch(["--nproc_per_node", "2", "--rdzv_backend", "static", script])
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("rank 0/2", result.stdout)
        self.assertIn("rank 1/2", result.stdout)

    def test_core_default(self):
        script = self._script(WORKER_OK)
        result = _run_launch(["--nproc_per_node", "2", script])
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("rank 1/2", result.stdout)

    def test_restart_on_failure(self):
        script = self._script(WORKER_FLAKY)
        result = _run_launch(
            ["--nproc_per_node", "1", "--max_restarts", "1", script])
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("recovered", result.stdout)

    def test_failure_raises_child_failed(self):
        script = self._script(WORKER_FAIL)
        result = _run_launch(["--nproc_per_node", "1", "--max_restarts", "0", script])
        self.assertEqual(result.returncode, 1)
        self.assertIn("ChildFailedError", result.stderr)

    def test_tee(self):
        script = self._script(WORKER_OK)
        result = _run_launch(
            ["--nproc_per_node", "1", "--rdzv_backend", "static",
             "--tee", "3", script])
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("rank 0/1", result.stdout)


class TestDebugServer(unittest.TestCase):
    def test_index_and_stacks(self):
        from tensorplay.distributed import debug
        port = debug.start_debug_server(port=0)
        try:
            s = socket.create_connection(("localhost", port), timeout=5)
            s.sendall(json.dumps({"handler": "index"}).encode() + b"\n")
            response = json.loads(s.makefile("r").readline())
            self.assertIn("index", response["handlers"])
            s.sendall(json.dumps({"handler": "stacks"}).encode() + b"\n")
            response = json.loads(s.makefile("r").readline())
            self.assertTrue(len(response["stacks"]) >= 1)
        finally:
            debug.stop_debug_server()


class TestFlightRecorder(unittest.TestCase):
    def test_record_dump_trace(self):
        from tensorplay.distributed import flight_recorder as fr
        recorder = fr.start_flight_recorder()
        recorder.record("all_reduce", "started", pg="0")
        recorder.record("all_reduce", "finished", pg="0")
        dump_dir = tempfile.mkdtemp(prefix="tp_fr_")
        path = fr.dump_flight_recorder(os.path.join(dump_dir, "dump_{rank}"))
        self.assertTrue(path and os.path.exists(path))
        result = subprocess.run(
            [sys.executable, "-m",
             "tensorplay.distributed.flight_recorder.fr_trace",
             dump_dir, "-p", "dump_", "-j"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("all_reduce", result.stderr)


class TestGatedSurfaces(unittest.TestCase):
    def test_nn_functional_reports_requirement(self):
        from tensorplay.distributed import nn as dist_nn
        try:
            import tensorplay.distributed.rpc  # noqa: F401
        except Exception:
            with self.assertRaises(RuntimeError):
                dist_nn.functional.remote_linear()
        else:
            # RPC-backed surfaces are wired by the RPC workstream.
            self.skipTest("RPC runtime available; gate is open")

    def test_nn_jit_reports_requirement(self):
        from tensorplay.distributed.nn import jit
        try:
            import tensorplay.distributed.rpc  # noqa: F401
        except Exception:
            with self.assertRaises(RuntimeError):
                jit.remote
        else:
            self.skipTest("RPC runtime available; gate is open")


if __name__ == "__main__":
    unittest.main()

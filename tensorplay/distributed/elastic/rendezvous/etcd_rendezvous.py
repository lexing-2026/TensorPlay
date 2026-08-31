from __future__ import annotations

import json
import logging
import threading
import time
import uuid

try:
    import etcd as _etcd
except ImportError:
    from . import _etcd_stub as _etcd

from .api import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStoreInfo,
    RendezvousTimeoutError,
)
from .etcd_store import EtcdStore, cas_delay
from .utils import parse_rendezvous_endpoint

__all__ = [
    "EtcdRendezvousRetryableFailure",
    "EtcdRendezvousRetryImmediately",
    "EtcdRendezvousHandler",
    "EtcdRendezvous",
    "create_rdzv_handler",
]

logger = logging.getLogger(__name__)
CONST_ETCD_SETUP_TTL = 5
CONST_ETCD_FROZEN_TTL = 10
CONST_ETCD_JOINABLE_EPHEMERAL_TTL = 10
CONST_WORKER_KEEPALIVE_TTL = 10
CONST_RUNID_SUBROOT_TTL = 7200


class EtcdRendezvousRetryableFailure(Exception):
    pass


class EtcdRendezvousRetryImmediately(Exception):
    pass


def _value(node) -> str:
    return "" if node.value is None else str(node.value)


class EtcdRendezvousHandler(RendezvousHandler):
    def __init__(self, rdzv_impl: "EtcdRendezvous", local_addr: str | None) -> None:
        self._rdzv_impl = rdzv_impl
        self._local_addr = local_addr
        self._participant_id = f"{local_addr or 'localhost'}-{uuid.uuid4().hex}"
        self._closed = False

    def get_backend(self) -> str:
        return "etcd"

    def use_agent_store(self) -> bool:
        return False

    def next_rendezvous(self) -> RendezvousInfo:
        version, rank, world_size = self._rdzv_impl.rendezvous_barrier(self._participant_id)
        store = self._rdzv_impl.setup_kv_store(version)
        client = self._rdzv_impl.client
        bootstrap = RendezvousStoreInfo(
            master_addr=self._local_addr or getattr(client, "host", "localhost"),
            port=int(getattr(client, "port", 0)),
        )
        return RendezvousInfo(store, rank, world_size, bootstrap_store_info=bootstrap)

    def is_closed(self) -> bool:
        return self._closed or self._rdzv_impl.is_closed()

    def set_closed(self) -> None:
        self._closed = True
        self._rdzv_impl.set_closed()

    def num_nodes_waiting(self) -> int:
        state = self._rdzv_impl.get_rdzv_state()[1]
        return int(state.get("num_workers_waiting", 0))

    def get_run_id(self) -> str:
        return self._rdzv_impl._run_id

    def shutdown(self) -> bool:
        try:
            self.set_closed()
            return True
        except Exception:
            logger.exception("rendezvous shutdown failed")
            return False


class EtcdRendezvous:
    def __init__(
        self,
        client,
        prefix: str,
        run_id: str,
        num_min_workers: int,
        num_max_workers: int,
        timeout: float,
        last_call_timeout: float,
    ) -> None:
        if not run_id or num_min_workers <= 0 or num_max_workers < num_min_workers:
            raise ValueError("invalid rendezvous parameters")
        self.client = client
        self._prefix = prefix.rstrip("/")
        self._run_id = run_id
        self._num_min_workers = int(num_min_workers)
        self._num_max_workers = int(num_max_workers)
        self._timeout = float(timeout)
        self._last_call_timeout = float(last_call_timeout)
        self._participant_lock = threading.Lock()
        self._rendezvous_deadline = 0.0
        self._ensure_path(self.get_path("/rdzv"))
        try:
            self.client.write(self.get_path("/rdzv/version_counter"), "0", prevExist=False)
        except _etcd.EtcdAlreadyExist:
            pass

    def _ensure_path(self, path: str) -> None:
        try:
            self.client.write(path, None, dir=True, prevExist=False)
        except _etcd.EtcdAlreadyExist:
            pass

    def rendezvous_barrier(self, participant_id: str | None = None) -> tuple[int, int, int]:
        participant_id = participant_id or f"local-{uuid.uuid4().hex}"
        self._rendezvous_deadline = time.monotonic() + self._timeout
        while time.monotonic() < self._rendezvous_deadline:
            try:
                active, state = self.get_rdzv_state()
            except _etcd.EtcdKeyNotFound:
                try:
                    active = self.try_create_rendezvous()
                    state = json.loads(_value(active))
                except _etcd.EtcdAlreadyExist:
                    continue
            if state.get("status") == "closed":
                raise RendezvousClosedError("rendezvous is closed")
            if state.get("status") == "joinable":
                return self.join_phase(int(state["version"]), participant_id)
            if state.get("status") == "frozen":
                self.wait_for_peers(int(state["version"]))
                return self.confirm_phase(int(state["version"]), participant_id)
            if state.get("status") == "final":
                self.announce_self_waiting(int(state["version"]), participant_id)
                self.wait_for_rendezvous_to_free(int(state["version"]))
            time.sleep(0.01)
        raise RendezvousTimeoutError("timed out waiting for rendezvous")

    def init_phase(self):
        active = self.try_create_rendezvous()
        state = json.loads(_value(active))
        return self.join_phase(int(state["version"]), f"local-{uuid.uuid4().hex}")

    def join_phase(self, expected_version, participant_id: str | None = None):
        participant_id = participant_id or f"local-{uuid.uuid4().hex}"
        active, this_rank = self.join_rendezvous(expected_version, participant_id)
        state = json.loads(_value(active))
        if state.get("status") == "joinable" and len(state.get("participants", [])) >= self._num_min_workers:
            self.handle_join_last_call(expected_version, time.monotonic() + self._last_call_timeout)
        self.wait_for_peers(expected_version)
        return self.confirm_phase(expected_version, participant_id, this_rank)

    def confirm_phase(self, expected_version, participant_id: str | None = None, this_rank: int | None = None):
        if this_rank is None:
            active, state = self.get_rdzv_state()
            participants = state.get("participants", [])
            if participant_id not in participants:
                raise RendezvousError("participant is not in rendezvous")
            this_rank = participants.index(participant_id)
        active, state = self.get_rdzv_state()
        if int(state.get("version", -1)) != int(expected_version):
            raise EtcdRendezvousRetryImmediately("rendezvous version changed")
        if state.get("status") == "frozen":
            state["status"] = "final"
            try:
                self.client.test_and_set(self.get_path("/rdzv/active_version"), json.dumps(state), _value(active))
            except _etcd.EtcdCompareFailed:
                active, state = self.get_rdzv_state()
        return int(state["version"]), int(this_rank), len(state.get("participants", []))

    def handle_existing_rendezvous(self, expected_version, participant_id: str | None = None):
        self.announce_self_waiting(expected_version, participant_id)
        self.wait_for_rendezvous_to_free(expected_version)

    def try_create_rendezvous(self):
        counter_key = self.get_path("/rdzv/version_counter")
        while True:
            counter = self.client.read(counter_key)
            old_value = _value(counter)
            version = int(old_value) + 1
            try:
                self.client.test_and_set(counter_key, str(version), old_value)
                break
            except _etcd.EtcdCompareFailed:
                cas_delay()
        state = {"status": "joinable", "version": version, "participants": [], "num_workers_waiting": 0}
        return self.client.write(self.get_path("/rdzv/active_version"), json.dumps(state), prevExist=False, ttl=CONST_ETCD_SETUP_TTL)

    def join_rendezvous(self, expected_version, participant_id: str):
        while True:
            active, state = self.get_rdzv_state()
            if state.get("status") != "joinable":
                raise EtcdRendezvousRetryableFailure("rendezvous is no longer joinable")
            if int(state.get("version", -1)) != int(expected_version):
                raise EtcdRendezvousRetryImmediately("rendezvous version changed")
            participants = list(state.get("participants", []))
            if participant_id in participants:
                return active, participants.index(participant_id)
            if len(participants) >= self._num_max_workers:
                state["status"] = "frozen"
            else:
                participants.append(participant_id)
                state["participants"] = participants
                if len(participants) >= self._num_max_workers:
                    state["status"] = "frozen"
            try:
                updated = self.client.test_and_set(self.get_path("/rdzv/active_version"), json.dumps(state), _value(active), ttl=CONST_ETCD_FROZEN_TTL if state["status"] == "frozen" else CONST_ETCD_JOINABLE_EPHEMERAL_TTL)
                return updated, participants.index(participant_id)
            except _etcd.EtcdCompareFailed:
                cas_delay()

    def wait_for_peers(self, expected_version):
        while time.monotonic() < self._rendezvous_deadline:
            active, state = self.get_rdzv_state()
            if int(state.get("version", -1)) != int(expected_version):
                raise EtcdRendezvousRetryImmediately("rendezvous version changed")
            if state.get("status") in {"frozen", "final"}:
                return active
            time.sleep(0.01)
        raise RendezvousTimeoutError("timed out waiting for peers")

    def confirm_membership(self, expected_version, this_rank):
        active, state = self.get_rdzv_state()
        if state.get("status") == "frozen":
            state["status"] = "final"
            try:
                return self.client.test_and_set(self.get_path("/rdzv/active_version"), json.dumps(state), _value(active))
            except _etcd.EtcdCompareFailed:
                return self.client.read(self.get_path("/rdzv/active_version"))
        return active

    def wait_for_final(self, expected_version):
        while time.monotonic() < self._rendezvous_deadline:
            active, state = self.get_rdzv_state()
            if int(state.get("version", -1)) != int(expected_version):
                raise EtcdRendezvousRetryImmediately("rendezvous version changed")
            if state.get("status") == "final":
                return active
            time.sleep(0.01)
        raise RendezvousTimeoutError("timed out waiting for final rendezvous state")

    def announce_self_waiting(self, expected_version, participant_id: str | None = None):
        active, state = self.get_rdzv_state()
        if int(state.get("version", -1)) != int(expected_version):
            raise EtcdRendezvousRetryImmediately("rendezvous version changed")
        state["num_workers_waiting"] = int(state.get("num_workers_waiting", 0)) + 1
        try:
            return self.client.test_and_set(self.get_path("/rdzv/active_version"), json.dumps(state), _value(active))
        except _etcd.EtcdCompareFailed:
            return self.client.read(self.get_path("/rdzv/active_version"))

    def wait_for_rendezvous_to_free(self, expected_version):
        while time.monotonic() < self._rendezvous_deadline:
            _, state = self.get_rdzv_state()
            if int(state.get("version", -1)) != int(expected_version) or state.get("status") != "final":
                return
            time.sleep(0.05)
        raise RendezvousTimeoutError("timed out waiting for rendezvous to reopen")

    def handle_join_last_call(self, expected_version, deadline):
        while time.monotonic() < min(deadline, self._rendezvous_deadline):
            active, state = self.get_rdzv_state()
            if int(state.get("version", -1)) != int(expected_version) or state.get("status") != "joinable":
                return
            if len(state.get("participants", [])) >= self._num_max_workers:
                state["status"] = "frozen"
                try:
                    self.client.test_and_set(self.get_path("/rdzv/active_version"), json.dumps(state), _value(active), ttl=CONST_ETCD_FROZEN_TTL)
                except _etcd.EtcdCompareFailed:
                    continue
                return
            time.sleep(0.02)
        active, state = self.get_rdzv_state()
        if state.get("status") == "joinable" and int(state.get("version", -1)) == int(expected_version) and len(state.get("participants", [])) >= self._num_min_workers:
            state["status"] = "frozen"
            try:
                self.client.test_and_set(self.get_path("/rdzv/active_version"), json.dumps(state), _value(active), ttl=CONST_ETCD_FROZEN_TTL)
            except _etcd.EtcdCompareFailed:
                pass

    def set_closed(self):
        active, state = self.get_rdzv_state()
        state["status"] = "closed"
        try:
            self.client.test_and_set(self.get_path("/rdzv/active_version"), json.dumps(state), _value(active))
        except _etcd.EtcdCompareFailed:
            pass

    def is_closed(self) -> bool:
        try:
            return self.get_rdzv_state()[1].get("status") == "closed"
        except _etcd.EtcdKeyNotFound:
            return False

    def get_rdzv_state(self):
        active = self.client.read(self.get_path("/rdzv/active_version"))
        return active, json.loads(_value(active))

    def try_wait_for_state_change(self, etcd_index, timeout=None):
        try:
            self.client.watch(self.get_path("/rdzv/active_version"), index=etcd_index, timeout=timeout or self._timeout)
        except _etcd.EtcdWatchTimedOut:
            pass
        return self.get_rdzv_state()

    def get_path(self, path):
        path = path if str(path).startswith("/") else "/" + str(path)
        return f"{self._prefix}/run_{self._run_id}{path}"

    def create_path_if_not_exists(self, full_path, ttl=None):
        self._ensure_path(full_path)

    def setup_lease_renewal(self, full_path, ttl):
        stop = threading.Event()

        def renew() -> None:
            while not stop.wait(max(0.1, ttl / 2)):
                try:
                    self.client.refresh(full_path, ttl=ttl)
                except Exception:
                    return

        threading.Thread(target=renew, daemon=True, name="tp_etcd_lease").start()
        return stop

    def store_extra_data(self, rdzv_version, key, value):
        path = self.get_path(f"/rdzv/v_{rdzv_version}/extra_data")
        while True:
            try:
                node = self.client.read(path)
                data = json.loads(_value(node))
                data[key] = value
                self.client.test_and_set(path, json.dumps(data), _value(node))
                return
            except _etcd.EtcdKeyNotFound:
                try:
                    self.client.write(path, json.dumps({key: value}), prevExist=False)
                    return
                except _etcd.EtcdAlreadyExist:
                    continue
            except _etcd.EtcdCompareFailed:
                cas_delay()

    def load_extra_data(self, rdzv_version, key, timeout=None):
        deadline = time.monotonic() + (self._timeout if timeout is None else float(timeout))
        path = self.get_path(f"/rdzv/v_{rdzv_version}/extra_data")
        while time.monotonic() < deadline:
            try:
                data = json.loads(_value(self.client.read(path)))
                if key in data:
                    return data[key]
            except _etcd.EtcdKeyNotFound:
                pass
            time.sleep(0.01)
        raise RendezvousTimeoutError(f"timed out waiting for extra data {key}")

    def setup_kv_store(self, rdzv_version):
        path = self.get_path(f"/rdzv/v_{rdzv_version}/kv")
        self._ensure_path(path)
        return EtcdStore(self.client, path)


def _create_etcd_client(params: RendezvousParameters):
    host, port = parse_rendezvous_endpoint(params.endpoint, default_port=2379)
    return _etcd.Client(host, port, read_timeout=params.get_as_int("read_timeout", 60) or 60)


def create_rdzv_handler(params: RendezvousParameters) -> EtcdRendezvousHandler:
    client = _create_etcd_client(params)
    prefix = params.get("etcd_prefix", "/tp/elastic")
    timeout = params.get_as_int("timeout", int(params.timeout.join.total_seconds())) or 600
    last_call = params.get_as_int("last_call_timeout", int(params.timeout.last_call.total_seconds())) or 30
    impl = EtcdRendezvous(client, prefix, params.run_id, params.min_nodes, params.max_nodes, timeout, last_call)
    return EtcdRendezvousHandler(impl, params.local_addr)

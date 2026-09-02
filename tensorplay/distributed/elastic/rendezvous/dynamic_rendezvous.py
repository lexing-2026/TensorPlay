"""Elastic rendezvous with node churn, re-rendezvous, and scale-up support.

Participants coordinate through an optimistic-concurrency state document
(see :mod:`.core_rendezvous_backend`). The state machine tracks the current
round, its participants with fresh heartbeats, and a wait list of nodes
queued for the next round. Heartbeats are refreshed by a background timer so
departed nodes are pruned automatically.

Round lifecycle:

* nodes join the participant set until ``min_nodes`` is reached, which arms
  the last-call window; the round completes once the window elapses, the
  participant set reaches ``max_nodes``, or waiters are queued;
* a completed round can be re-entered idempotently by its participants;
* when a participant re-enters while the wait list is non-empty (scale-up),
  a new round opens and includes the waiters;
* nodes that lose heartbeats are pruned; when all participants of a
  completed round vanish, queued nodes are promoted immediately.
"""
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta

from tensorplay.distributed import Store

from .api import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStateError,
    RendezvousStoreInfo,
    RendezvousTimeout,
    RendezvousTimeoutError,
)
from .core_rendezvous_backend import RendezvousBackend, Token, create_backend
from .utils import _delay, _PeriodicTimer

logger = logging.getLogger(__name__)


@dataclass
class RendezvousSettings:
    """Rendezvous algorithm configuration."""

    join: timedelta = timedelta(seconds=600)
    last_call: timedelta = timedelta(seconds=30)
    close: timedelta = timedelta(seconds=30)
    heartbeat: timedelta = timedelta(seconds=5)
    min_nodes: int = 1
    max_nodes: int = 1


def _node_desc(addr: str | None = None, pid: int | None = None, local_id: str | None = None) -> str:
    """Stable unique descriptor of one agent instance."""
    addr = addr or "localhost"
    pid = pid if pid is not None else os.getpid()
    local_id = local_id or uuid.uuid4().hex
    return f"{addr}@{pid}@{local_id}"


def _now_ms() -> float:
    return time.monotonic() * 1000


def _default_state() -> dict:
    return {
        "round": 0,
        "complete": False,
        "closed": False,
        "participants": {},
        "wait_list": {},
        "round_deadline": None,
        "last_call": None,
        "bootstrap": None,
        "rank_order": [],
    }


class _RendezvousState:
    """Typed view over the JSON state document."""

    def __init__(self, data: dict) -> None:
        self.data = data

    @property
    def round(self) -> int:
        return self.data["round"]

    @property
    def complete(self) -> bool:
        return self.data["complete"]

    @property
    def closed(self) -> bool:
        return self.data["closed"]

    @property
    def participants(self) -> dict[str, float]:
        return self.data["participants"]

    @property
    def wait_list(self) -> dict[str, float]:
        return self.data["wait_list"]

    def prune_dead_nodes(self, expire_ms: float) -> list[str]:
        """Drop participants and waiters whose heartbeat lapsed."""
        now = _now_ms()
        pruned = []
        for key in list(self.participants):
            if now - self.participants[key] > expire_ms:
                del self.participants[key]
                pruned.append(key)
        for key in list(self.wait_list):
            if now - self.wait_list[key] > expire_ms:
                del self.wait_list[key]
                pruned.append(key)
        if pruned:
            if len(self.participants) < self.data.get("min_nodes", 1):
                self.data["last_call"] = None
            if not self.participants and self.complete:
                self.data["complete"] = False
                self.data["last_call"] = None
        return pruned


class _RendezvousOpExecutor:
    """Read-modify-CAS loop shared by every rendezvous operation."""

    def __init__(self, backend: RendezvousBackend, settings: RendezvousSettings) -> None:
        self._backend = backend
        self._settings = settings

    def run(self, action) -> dict:
        """Apply ``action(state)`` (in-place mutation) until a CAS succeeds."""
        deadline = _now_ms() + self._settings.join.total_seconds() * 1000
        while True:
            raw = self._backend.get_state()
            if raw is None:
                data, token = _default_state(), 0
            else:
                state_bytes, token = raw
                data = json.loads(state_bytes.decode())
            action(_RendezvousState(data))
            encoded = json.dumps(data).encode()
            result = self._backend.set_state(encoded, token)
            if isinstance(result, tuple) and len(result) == 2 and result[1] == token + 1:
                return data
            if _now_ms() >= deadline:
                raise RendezvousTimeoutError("Rendezvous operation timed out")
            _delay((0.05, 0.25))


class DynamicRendezvousHandler(RendezvousHandler):
    """Rendezvous handler over a shared store with re-rendezvous support."""

    def __init__(
        self,
        backend: RendezvousBackend,
        settings: RendezvousSettings,
        local_addr: str | None,
        node_rank: int,
        run_id: str,
        store: Store | None = None,
        backend_name: str = "core",
    ) -> None:
        self._backend = backend
        self._backend_name = backend_name
        self._settings = settings
        self._node_rank = node_rank
        self._run_id = run_id
        self._store = store if store is not None else getattr(backend, "_store", None)
        self._node = _node_desc(addr=local_addr)
        self._local_addr = local_addr
        self._executor = _RendezvousOpExecutor(backend, settings)
        self._keep_alive_timer: _PeriodicTimer | None = None
        self._last_bootstrap: RendezvousStoreInfo | None = None

    @classmethod
    def from_backend(
        cls,
        run_id: str,
        endpoint: str,
        settings: RendezvousSettings,
        local_addr: str | None = None,
        **kwargs,
    ) -> "DynamicRendezvousHandler":
        """Create a handler together with its backend store.

        ``kwargs`` accepts ``store_type`` (``tcp``/``file``), ``node_rank``,
        ``local_world_size``, ``config``, and any extra
        :class:`RendezvousParameters` fields.
        """
        store_type = kwargs.pop("store_type", "tcp")
        params = RendezvousParameters(
            backend="core",
            endpoint=endpoint,
            run_id=run_id,
            local_addr=local_addr,
            node_rank=kwargs.pop("node_rank", 0),
            local_world_size=kwargs.pop("local_world_size", 1),
            config=kwargs.pop("config", {}),
            **kwargs,
        )
        params.config.setdefault("min_nodes", settings.min_nodes)
        params.config.setdefault("max_nodes", settings.max_nodes)
        backend, store = create_backend(params, store_type=store_type)
        return cls(
            backend,
            settings,
            local_addr,
            params.node_rank,
            run_id,
            store=store,
            backend_name=backend.name,
        )

    @property
    def settings(self) -> RendezvousSettings:
        return self._settings

    def get_backend(self) -> str:
        return self._backend_name

    def use_agent_store(self) -> bool:
        # The rendezvous store doubles as the worker bootstrap store and is
        # hosted inside the agent process.
        return True

    def get_run_id(self) -> str:
        return self._run_id

    def next_rendezvous(self) -> RendezvousInfo:
        """Join the current round, opening a new one when waiters queue up."""
        self._start_keep_alive()
        deadline = _now_ms() + self._settings.join.total_seconds() * 1000
        while True:
            state, token = self._read_state()
            if state.closed:
                self._stop_keep_alive()
                raise RendezvousClosedError
            state.data.setdefault("min_nodes", self._settings.min_nodes)
            state.prune_dead_nodes(self._lapse_ms())
            now = _now_ms()
            me = self._node

            if state.complete:
                if me in state.participants:
                    if state.wait_list:
                        # Scale-up: an active participant re-entering the
                        # rendezvous opens a new round with the waiters.
                        self._open_next_round(state, now)
                    else:
                        return self._finish(state)
                elif not state.participants:
                    # Every participant departed; promote whoever is waiting.
                    self._promote_waiters(state, now)
                else:
                    state.wait_list.setdefault(me, now)
            else:
                if me not in state.participants:
                    if len(state.participants) < self._settings.max_nodes:
                        state.participants[me] = now
                        state.wait_list.pop(me, None)
                        self._arm_last_call(state, now)
                    else:
                        state.wait_list.setdefault(me, now)
                if (
                    len(state.participants) >= self._settings.min_nodes
                    and (
                        state.wait_list
                        or (
                            state.data["last_call"] is not None
                            and now >= state.data["last_call"]
                        )
                    )
                ):
                    self._mark_complete(state)

            if state.closed:
                self._stop_keep_alive()
                raise RendezvousClosedError
            encoded = json.dumps(state.data).encode()
            result = self._backend.set_state(encoded, token)
            if isinstance(result, tuple) and len(result) == 2 and result[1] == token + 1:
                new_state = _RendezvousState(json.loads(result[0].decode()))
                if new_state.complete and me in new_state.participants:
                    return self._finish(new_state)
            if _now_ms() >= deadline:
                raise RendezvousTimeoutError(
                    f"Rendezvous join timed out after {self._settings.join}"
                )
            _delay((0.05, 0.25))

    def _lapse_ms(self) -> float:
        return self._settings.heartbeat.total_seconds() * 1000 * 3

    def _arm_last_call(self, state: _RendezvousState, now: float) -> None:
        window = self._settings.last_call.total_seconds() * 1000
        if len(state.participants) >= self._settings.max_nodes:
            state.data["last_call"] = now
        elif len(state.participants) >= self._settings.min_nodes:
            if state.data["last_call"] is None:
                state.data["last_call"] = now + window

    def _mark_complete(self, state: _RendezvousState) -> None:
        state.data["complete"] = True
        state.data["round_deadline"] = _now_ms() + self._settings.join.total_seconds() * 1000
        state.data["rank_order"] = sorted(state.participants)

    def _open_next_round(self, state: _RendezvousState, now: float) -> None:
        fresh = {}
        for key in sorted(state.participants):
            if len(fresh) >= self._settings.max_nodes:
                break
            fresh[key] = now
        for key in sorted(state.wait_list):
            if len(fresh) >= self._settings.max_nodes:
                break
            fresh[key] = now
        state.data["participants"] = fresh
        state.data["wait_list"] = {
            key: stamp for key, stamp in state.wait_list.items() if key not in fresh
        }
        state.data["round"] = state.round + 1
        state.data["complete"] = False
        state.data["last_call"] = None
        state.data["round_deadline"] = None

    def _promote_waiters(self, state: _RendezvousState, now: float) -> None:
        fresh = {}
        for key in sorted(state.wait_list):
            if len(fresh) >= self._settings.max_nodes:
                break
            fresh[key] = now
        state.data["participants"] = fresh
        state.data["wait_list"] = {
            key: stamp for key, stamp in state.wait_list.items() if key not in fresh
        }
        state.data["complete"] = False
        state.data["last_call"] = None

    def _read_state(self) -> tuple[_RendezvousState, Token]:
        raw = self._backend.get_state()
        if raw is None:
            return _RendezvousState(_default_state()), 0
        state_bytes, token = raw
        try:
            return _RendezvousState(json.loads(state_bytes.decode())), token
        except (ValueError, UnicodeDecodeError) as e:
            raise RendezvousStateError(f"Rendezvous state is corrupt: {e}") from e

    def _finish(self, state: _RendezvousState) -> RendezvousInfo:
        self._start_keep_alive()
        ordering = state.data.get("rank_order") or sorted(state.participants)
        rank = ordering.index(self._node)
        bootstrap = self._resolve_bootstrap(state, rank)
        return RendezvousInfo(
            store=self._store,
            rank=rank,
            world_size=len(ordering),
            bootstrap_store_info=bootstrap,
            participants=ordering,
            wait_list=sorted(state.wait_list),
        )

    def _resolve_bootstrap(self, state: _RendezvousState, rank: int) -> RendezvousStoreInfo | None:
        stored = state.data.get("bootstrap")
        if stored:
            return RendezvousStoreInfo(master_addr=stored["addr"], port=int(stored["port"]))
        if self._store is None:
            return None
        if rank == 0 and hasattr(self._store, "master_addr_port"):
            addr, port = self._store.master_addr_port
            if not addr or addr == "0.0.0.0":
                addr = self._local_addr or "127.0.0.1"
            state.data["bootstrap"] = {"addr": addr, "port": int(port)}
            raw = self._backend.get_state()
            if raw is not None:
                state_bytes, token = raw
                current = json.loads(state_bytes.decode())
                if not current.get("bootstrap"):
                    current["bootstrap"] = state.data["bootstrap"]
                    self._backend.set_state(json.dumps(current).encode(), token)
            return RendezvousStoreInfo(master_addr=addr, port=int(port))
        end = time.monotonic() + self._settings.join.total_seconds()
        while time.monotonic() < end:
            fresh, _ = self._read_state()
            if fresh.data.get("bootstrap"):
                info = fresh.data["bootstrap"]
                return RendezvousStoreInfo(master_addr=info["addr"], port=int(info["port"]))
            time.sleep(0.1)
        raise RendezvousTimeoutError("Timed out waiting for the bootstrap store address")

    def _start_keep_alive(self) -> None:
        if self._keep_alive_timer is not None:
            return
        timer = _PeriodicTimer(
            self._settings.heartbeat / 3,
            self._keep_alive,
            name=f"tp_elastic_rdzv_keepalive_{self._run_id}",
        )
        self._keep_alive_timer = timer
        timer.start()

    def _stop_keep_alive(self) -> None:
        if self._keep_alive_timer is not None:
            self._keep_alive_timer.cancel()
            self._keep_alive_timer = None

    def _keep_alive(self) -> None:
        try:
            state, token = self._read_state()
            if state.closed:
                self._stop_keep_alive()
                return
            state.data.setdefault("min_nodes", self._settings.min_nodes)
            state.prune_dead_nodes(self._lapse_ms())
            now = _now_ms()
            if self._node in state.participants:
                state.participants[self._node] = now
            if not state.data["complete"] and len(state.participants) >= self._settings.min_nodes:
                self._arm_last_call(state, now)
                if state.wait_list or (
                    state.data["last_call"] is not None and now >= state.data["last_call"]
                ):
                    self._mark_complete(state)
            encoded = json.dumps(state.data).encode()
            self._backend.set_state(encoded, token)
        except RendezvousError:
            self._stop_keep_alive()
        except Exception:
            pass

    def is_closed(self) -> bool:
        try:
            state, _ = self._read_state()
            return state.closed
        except RendezvousError:
            return False

    def set_closed(self) -> None:
        def _close(state: _RendezvousState):
            state.data["closed"] = True

        self._executor.run(_close)
        self._stop_keep_alive()

    def num_nodes_waiting(self) -> int:
        try:
            state, _ = self._read_state()
            return len(state.wait_list)
        except RendezvousError:
            return 0

    def shutdown(self) -> bool:
        """Leave the rendezvous and close it for future joiners."""
        try:
            def _leave(state: _RendezvousState):
                state.participants.pop(self._node, None)
                state.wait_list.pop(self._node, None)
                if not state.participants:
                    state.data["complete"] = False
                    state.data["last_call"] = None
                state.data["closed"] = True

            self._executor.run(_leave)
        except RendezvousError:
            return False
        finally:
            self._stop_keep_alive()
        return True

    def __del__(self):
        try:
            self._stop_keep_alive()
        except Exception:
            pass


def create_handler(params: RendezvousParameters) -> DynamicRendezvousHandler:
    """Registry entry point for the ``core`` backend."""
    settings = RendezvousSettings(
        join=params.timeout.join,
        last_call=params.timeout.last_call,
        close=params.timeout.close,
        heartbeat=params.timeout.heartbeat,
        min_nodes=params.get_as_int("min_nodes", 1),
        max_nodes=params.get_as_int("max_nodes", params.get_as_int("min_nodes", 1)),
    )
    if params.get("store_type"):
        store_type = str(params.get("store_type"))
    elif params.get_as_bool("is_file", False):
        store_type = "file"
    else:
        store_type = "tcp"
    return DynamicRendezvousHandler.from_backend(
        run_id=params.run_id,
        endpoint=params.endpoint,
        settings=settings,
        local_addr=params.local_addr,
        node_rank=params.node_rank,
        local_world_size=params.local_world_size,
        store_type=store_type,
        config=dict(params.config),
        **params.kwargs,
    )

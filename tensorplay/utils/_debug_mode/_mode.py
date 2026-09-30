"""Record what a region of code actually asks the tensor backend to do.

Running a function is the only way to find out what it will ask for, and
running it on values chosen for the purpose answers a different question from
the one that will be asked of it later.  So inside this region each operator
that reaches the backend is recorded as it is issued -- the call, what it was
given, what it returned, and optionally the inputs hashed so that two runs can
be compared without keeping the values themselves.

The recording happens where operators are dispatched rather than at the call
sites, because that is the one place every operator passes through, whether it
was written here, arrived from a library, or came out of a compiled graph.
"""

import contextlib
import functools
import json
import logging
import os
from collections.abc import Callable
from typing import Any, IO

import tensorplay as tp

from . import _calls
from . import _utils
from ._calls import (
    _AnnotateCall,
    _deserialize_debug_call,
    _OpCall,
    _OutputPlacementCall,
    _RedistributeCall,
    _serialize_debug_call,
    _TritonKernelCall,
)
from ._utils import (
    _compute_rel_diff,
    _get_op_name,
    _get_user_stack_trace,
    _stringify_dtensor_spec,
    TensorIdTracker,
    hash_tensor_fn,
    norm_hash_fn,
)
from .._dispatch import TensorPlayDispatchMode, _get_current_dispatch_mode_stack
from .._pytree import (
    keystr,
    tree_all,
    tree_map,
    tree_map_only,
    tree_map_with_path,
)

log = logging.getLogger(__name__)


def _lookup_op(namespace: str, name: str) -> Any:
    """The operator, or a value that matches no operator.

    A region that is asked to watch for a particular operator should say
    nothing when that operator does not exist here, rather than fail on the
    first call it is handed; so an absent operator resolves to a sentinel that
    no operator is identical to, and the branch simply never fires.
    """
    try:
        return getattr(getattr(tp.ops, namespace), name)
    except AttributeError:
        return _NO_SUCH_OP


class _NoSuchOp:
    """A stand-in for an operator this build does not have."""

    def __repr__(self) -> str:
        return "<no such operator>"


_NO_SUCH_OP = _NoSuchOp()

_ANNOTATE = _lookup_op("debug_mode_ops", "annotate")
_RECORD_FUNCTION_ENTER = _lookup_op("profiler", "_record_function_enter_new")
_RECORD_FUNCTION_EXIT = _lookup_op("profiler", "_record_function_exit")
_PRIM_DEVICE = _lookup_op("prim", "device")

_SERIALIZED_LOG_FORMAT = "tensorplay.utils._debug_mode.DebugMode.logs"
_SERIALIZED_LOG_VERSION = 1

# Whether any region is currently recording.  Asked first, and on its own, so
# that a program which never records does not pay for a stack walk on every
# operator.
_ACTIVE_DEBUG_MODE_COUNT = 0

# A tag asked for inside a region is an operator like any other, and so needs
# a definition; whether the region meant anything by it is decided at dispatch.
_ANNOTATE_DECORATED = False


def _ensure_annotate_decorated() -> None:
    """Give the tag operator a lowering, so it survives to where it means something.

    A tag is a request for something to be said about the code around it, and
    a request that nothing recognizes can be dropped as useless long before it
    reaches the region that would have acted on it.  So the lowering says
    plainly that it is dropped here, rather than leaving it unrecognized.
    """
    global _ANNOTATE_DECORATED
    if _ANNOTATE_DECORATED:
        return

    from ...compiler.backends.stax.op_lowerings import register_lowering

    def _annotate_lowering(tag: str) -> None:
        from .._logging import warning_once

        warning_once(log, "a tag is a no-op when the graph is lowered")
        return None

    register_lowering(_ANNOTATE)(_annotate_lowering)
    _ANNOTATE_DECORATED = True


@tp.library.custom_op("debug_mode_ops::annotate", mutates_args=())
def _annotate(tag: str) -> None:
    # What the tag does is decided where the operator is intercepted, since
    # only there is the region that asked for it known; on its own it is
    # nothing at all.
    return None


@_annotate.register_fake
def _annotate_fake(tag: str) -> None:
    return None


class DebugMode(TensorPlayDispatchMode):
    """A region in which operators are recorded as they reach the backend.

    Whether a call is recorded is a question about the call rather than about
    the region, so it is asked of each region and of the call: a region asked
    to record the values themselves can be made to, but the default records
    the shape of the values and not the values, since what a run does to the
    values is rarely what one wants to keep.
    """

    def __init__(
        self,
        *,
        record_faketensor=False,
        record_realtensor=True,
        record_tensor_attributes=None,
        record_nn_module=False,
        store_original_args=False,
        record_stack_trace=False,
        record_output=True,
        record_ids=False,
        record_profiler_context=True,
        record_localtensor=True,
    ) -> None:
        super().__init__()
        _ensure_annotate_decorated()

        # Records calls on tensors that stand in for values.
        self.record_faketensor = record_faketensor

        # Records calls on real tensors.
        self.record_realtensor = record_realtensor

        # Records calls on a local (unsharded) tensor.
        self.record_localtensor = record_localtensor

        # Optional list[str] of tensor attributes, to be annotated in the
        # string dump.
        self.record_tensor_attributes = record_tensor_attributes or []

        # Whether a module's name is recorded on the way into it, which says
        # which part of a model a call came from.
        self.record_nn_module = record_nn_module

        # If True, stores call args/kwargs in logs, without immediately
        # stringifying.  Defaults to False for memory concerns.
        self.store_original_args = store_original_args

        # Stores the stack a call was made from, so that a line in the dump
        # can be traced back to the line of source responsible for it.
        self.record_stack_trace = record_stack_trace

        # Records what each call returned.
        self.record_output: bool = record_output

        # Annotates string dumps with graph-style tensor ids, e.g.
        # op($1, $2) -> $3.
        self.record_ids: bool = record_ids

        # Annotates string dumps with the groups the profiler has named, which
        # say what the calls around them were for.
        self.record_profiler_context: bool = record_profiler_context

        self.reset()

    def reset(self) -> None:
        self.operators = []
        self.call_depth = 0
        self._tensor_memo = TensorIdTracker()
        self._output_info: dict[int, object] = {}
        self.ignored_record_functions = 0
        self.current_call_function_stack: list[str] = []
        self.current_nn_module_stack: list[str] = []
        self.fx_stack_trace = None

    def _track_op_output(self, op_index: int, result: Any) -> None:
        """Give the outputs of one call ids, and remember which ids they got.

        An output is named by where it came out at, so that a later call
        taking it as an input can be read as passing it along rather than as
        naming a number that happens to be the same.
        """
        self._output_info[op_index] = result

    def _record_call(self, call) -> None:
        if _utils._IN_TP_BENCHMARK:
            # A benchmark asks the backend for numbers as often as it can;
            # recording them would measure the recording.
            return

        if str(call).startswith("profiler::_record_function"):
            return

        if not self.store_original_args:
            call.stringify_args(
                self.record_tensor_attributes,
                self._tensor_memo if self.record_ids else None,
            )
        if self.fx_stack_trace:
            call.stack_trace = call.fwd_stack_trace = self.fx_stack_trace
        self.operators.append(call)

    def _record_call_output(self, call, output: Any) -> None:
        if not self.record_output:
            return
        call.stringify_output(
            output,
            self.record_tensor_attributes,
            self._tensor_memo if self.record_ids else None,
        )

    def _maybe_record_function(self, tag: str) -> None:
        """Open a group in the dump for whatever the profiler has just named.

        A name the profiler gave to a span of work is worth keeping next to the
        calls made inside it; without the group the calls are there but the
        span they belong to is not.  Groups belonging to a measurement are
        dropped instead, since a measurement launches kernels as a matter of
        course and those say nothing about what the run was asked to do.
        """
        if any(
            tag.startswith(prefix)
            for prefix in [
                "CachingAutotuner.",
                "Benchmarker.",
                "compile_fx.<locals>.",
            ]
        ):
            self.ignored_record_functions += 1
            return

        call = _AnnotateCall(
            tag, "record function", self.call_depth, stack=self.record_stack_trace
        )
        self.operators.append(call)
        self.call_depth += 1

    def _maybe_exit_record_function(self) -> None:
        if self.ignored_record_functions < 0:
            raise AssertionError(
                f"ignored_record_functions is negative: {self.ignored_record_functions}"
            )
        if self.ignored_record_functions > 0:
            self.ignored_record_functions -= 1
        else:
            self.call_depth -= 1

    def __tensorplay_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # The groups the profiler has just opened and closed, which say what
        # the calls around them were for.
        if self.record_profiler_context:
            if func is _RECORD_FUNCTION_ENTER:
                if len(args) != 1:
                    raise AssertionError(f"expected 1 arg, got {len(args)}")
                self._maybe_record_function(args[0])
            elif func is _RECORD_FUNCTION_EXIT:
                self._maybe_exit_record_function()

        # A tag asked to be remembered, rather than a computation.
        if func is _ANNOTATE:
            if len(args) != 1:
                raise AssertionError(f"expected 1 arg, got {len(args)}")
            self._handle_annotate(args[0])
            return

        from ...distributed._functional_collectives import AsyncCollectiveTensor
        from ...distributed._local_tensor import LocalTensor
        from ...distributed.tensor import DTensor

        # Record the operation with its call depth
        call = None
        if DTensor in types:
            call = _OpCall(
                func, args, kwargs, self.call_depth, stack=self.record_stack_trace
            )
            self._record_call(call)
            return NotImplemented
        elif _utils.is_fake_tensor(types[0]) if len(types) == 1 else any(
            _utils.is_fake_tensor(t) for t in types
        ):
            if self.record_faketensor:
                if func != tp.ops.prim.device:
                    call = _OpCall(
                        func,
                        args,
                        kwargs,
                        self.call_depth + 1,
                        stack=self.record_stack_trace,
                    )
                    self._record_call(call)
        elif LocalTensor in types:
            if self.record_localtensor:
                call = _OpCall(
                    func,
                    args,
                    kwargs,
                    self.call_depth + 1,
                    stack=self.record_stack_trace,
                )
                self._record_call(call)
        elif AsyncCollectiveTensor in types:
            # Recorded so that tracing tools can see what a collective did.
            if self.record_realtensor:
                call = _OpCall(
                    func,
                    args,
                    kwargs,
                    self.call_depth + 1,
                    stack=self.record_stack_trace,
                )
                self._record_call(call)
        elif len(types) == 0:
            if self.record_realtensor:
                call = _OpCall(
                    func,
                    args,
                    kwargs,
                    self.call_depth + 1,
                    stack=self.record_stack_trace,
                )
                self._record_call(call)

        # Run pre-hooks before executing the operation to hash inputs.  This
        # has to happen before the call, in case the operation writes into one
        # of its inputs.
        if call:
            _utils._run_dispatch_pre_log_hooks(call, func, types, args, kwargs)

        result = func(*args, **kwargs)
        if call:
            self._record_call_output(call, result)
            _utils._run_dispatch_hooks(call, func, types, args, kwargs, result)

        return result

    def __enter__(self):
        global _ACTIVE_DEBUG_MODE_COUNT
        _ACTIVE_DEBUG_MODE_COUNT += 1

        super().__enter__()
        if self.record_nn_module:
            self.module_tracker.__enter__()

        if self.record_stack_trace:
            from ...autograd.anomaly_mode import set_detect_anomaly

            self.anomaly_for_traces = set_detect_anomaly(True, check_nan=False)
            self.anomaly_for_traces.__enter__()
        return self

    def __exit__(self, *args):
        global _ACTIVE_DEBUG_MODE_COUNT
        _ACTIVE_DEBUG_MODE_COUNT -= 1
        super().__exit__(*args)
        if self.record_nn_module:
            self.module_tracker.__exit__(*args)
        if self.record_stack_trace:
            self.anomaly_for_traces.__exit__(*args)

    @contextlib.contextmanager
    def set_fx_stack_trace(self, stack_trace):
        self.fx_stack_trace = stack_trace
        try:
            yield
        finally:
            self.fx_stack_trace = None

    def _enter_nn_module_call(self, fqn, header) -> None:
        call = _AnnotateCall(
            fqn, header, self.call_depth + 1, stack=self.record_stack_trace
        )
        self.operators.append(call)
        self.current_nn_module_stack.append(fqn)
        self.call_depth += 1

    def _exit_nn_module_call(self) -> None:
        self.call_depth -= 1
        self.current_nn_module_stack.pop()

    def module_tracker_setup(self) -> None:
        from ...distributed._tools.mod_tracker import ModTracker

        self.module_tracker = ModTracker()

        # module pre-fw hook: record module call
        def pre_fw_hook(module, input) -> None:
            fqn = self.module_tracker._get_mod_name(module)
            self._enter_nn_module_call(fqn, "nn.Mod")

        # module post-fw hook: decrement call depth
        def post_fw_hook(module, input, output) -> None:
            self._exit_nn_module_call()

        self.module_tracker.register_user_hooks(pre_fw_hook, post_fw_hook)

    def _handle_fx_nn_module_stack(
        self,
        base_stack: list[str],
        nn_module_stack: dict[str, tuple[str, Any]] | None,
        fwd_nn_module_stack: dict[str, tuple[str, Any]] | None,
    ) -> None:
        """Reconcile the module stack observed from a compiled graph.

        The modules actually entered can differ from the ones tracked here --
        a compiled graph names the modules it was traced through, and running
        it may not go through the same ones.  So the depth is adjusted to what
        was really entered, rather than the dump claiming a nesting that never
        happened.
        """
        nn_module_stack = nn_module_stack or {}
        fwd_nn_module_stack = fwd_nn_module_stack or {}
        if nn_module_stack and fwd_nn_module_stack:
            raise AssertionError(
                "Expecting at most one of nn_module_stack and fwd_nn_module_stack."
            )

        is_fwd = nn_module_stack
        stack = nn_module_stack if is_fwd else fwd_nn_module_stack

        # forward stack
        current_stack = self.current_nn_module_stack
        new_stack = base_stack + [v[0] for v in stack.values()]

        entered = set(new_stack) - set(current_stack)
        exited = set(current_stack) - set(new_stack)

        # Decrement depth for exited modules
        for _ in exited:
            self._exit_nn_module_call()
        if self.call_depth < 0:
            raise AssertionError("Unexpectedly, DebugMode call_depth is negative")

        # Add [nn.Module] entries for newly entered modules
        for fqn in sorted(entered):
            self._enter_nn_module_call(
                fqn, "nn.Mod (compile)" if is_fwd else "nn.Mod (compile bwd)"
            )

        self.current_nn_module_stack = new_stack

    @contextlib.contextmanager
    def record_redistribute_calls(
        self,
        arg,
        src_placement,
        dst_placement,
        transform_info_str: str | None = None,
        is_explicit: bool = False,
    ):
        try:
            self._record_call(
                _RedistributeCall(
                    arg,
                    src_placement=src_placement,
                    dst_placement=dst_placement,
                    transform_info_str=transform_info_str,
                    call_depth=self.call_depth + 1,
                    stack=self.record_stack_trace,
                    is_explicit=is_explicit,
                )
            )
            self.call_depth += 1
            yield
        finally:
            self.call_depth -= 1

    def record_output_placements(self, output_spec) -> None:
        """Record where a sharded result ended up, as a line of its own.

        The placements are not an argument to anything, so recording them with
        the call that produced them would put them where nothing reads them.
        """
        if not self.record_output:
            return
        from ...distributed.tensor._dtensor_spec import DTensorSpec

        placements_str = str(
            tree_map_only(
                DTensorSpec, _stringify_dtensor_spec, output_spec
            )
        )
        call = _OutputPlacementCall(placements_str, self.call_depth + 1)
        self._record_call(call)

    def record_triton_kernel(
        self, kernel_name: str, kwargs: dict[str, Any]
    ) -> _TritonKernelCall:
        """Record a kernel about to be launched, and hand it back.

        The call is returned rather than only kept so that the caller can fill
        in the outputs once it has them, which is the only point at which
        those values are known.
        """
        call = _TritonKernelCall(kernel_name, kwargs, self.call_depth + 1)
        call.stringify_args(self.record_tensor_attributes)
        self.operators.append(call)
        return call

    def debug_string(self, show_stack_trace: bool | None = None) -> str:
        """The recorded calls as text, one per line, indented by nesting."""
        show_stack_trace = (
            self.record_stack_trace if show_stack_trace is None else show_stack_trace
        )

        if not show_stack_trace:
            return "\n".join(
                "  "
                + "  " * op.call_depth
                + op.render(self.record_tensor_attributes)
                for op in self.operators
            )

        # Group operations by stack trace
        lines = []
        prev_stack_summary = None

        for op in self.operators:
            # Get the stack trace: prefer fwd_stack_trace, fallback to stack_trace
            stack_trace = None
            if hasattr(op, "fwd_stack_trace") and op.fwd_stack_trace:
                stack_trace = op.fwd_stack_trace
            elif hasattr(op, "stack_trace") and op.stack_trace:
                stack_trace = op.stack_trace

            stack_summary = None
            if stack_trace:
                stack_summary = _get_user_stack_trace(stack_trace)

            if stack_summary and stack_summary != prev_stack_summary:
                # add blank line before stack trace comment for readability
                if lines:  # don't add blank line at the very start
                    lines.append("")
                indent = "  " * (op.call_depth + 1)
                lines.append(indent + "# " + stack_summary)
                prev_stack_summary = stack_summary

            # Add the operation line
            lines.append(
                "  " + "  " * op.call_depth + op.render(self.record_tensor_attributes)
            )

        return "\n".join(lines)

    @staticmethod
    @contextlib.contextmanager
    def dispatch_hooks(
        record_hook: Callable | None = None,
        log_hook: Callable | None = None,
        pre_log_hook: Callable | None = None,
    ):
        """Install hooks on the calls that are intercepted.

        Hook signatures are the dispatch arguments plus the return value, i.e.
        ``(func, types, args, kwargs, result)``; the pre-log hook is
        ``(func, types, args, kwargs, call)`` and runs before the operation, so
        that state can be read before an in-place write changes it.  Hook
        outputs are dictionaries.
        """
        if record_hook:
            _utils._DISPATCH_RECORD_HOOKS.append(record_hook)
        if log_hook:
            _utils._DISPATCH_LOG_HOOKS.append(log_hook)
        if pre_log_hook:
            _utils._DISPATCH_PRE_LOG_HOOKS.append(pre_log_hook)
        try:
            yield
        finally:
            if record_hook:
                _utils._DISPATCH_RECORD_HOOKS.pop()
            if log_hook:
                _utils._DISPATCH_LOG_HOOKS.pop()
            if pre_log_hook:
                _utils._DISPATCH_PRE_LOG_HOOKS.pop()

    @staticmethod
    @contextlib.contextmanager
    def record_outputs():
        """Keep a copy of what each call returned.

        A copy rather than the value itself, because the value is about to be
        written through and what the dump should show is what it was.
        """

        def dispatch_hook(func, types, args, kwargs, result):
            out = tree_map(
                lambda x: x.clone() if isinstance(x, tp.Tensor) else x, result
            )
            return {"output": out}

        try:
            _old_record_triton = _utils._RECORD_TRITON_OUTPUTS
            _utils._RECORD_TRITON_OUTPUTS = True
            with DebugMode.dispatch_hooks(record_hook=dispatch_hook):
                yield
        finally:
            _utils._RECORD_TRITON_OUTPUTS = _old_record_triton

    @staticmethod
    @contextlib.contextmanager
    def log_tensor_hashes(
        hash_fn: Callable | str | list[str] = "norm", hash_inputs: bool = False
    ):
        """Record a fingerprint of each value, so two runs can be compared.

        A run is only comparable to another if the two did the same things, and
        a fingerprint of the values is what says whether they did.  Hashing the
        inputs before the call catches a difference that a later write has
        already hidden.
        """

        def hash_fn_option(hash_type):
            if not isinstance(hash_type, str) or hash_type not in [
                "norm",
                "hash_tensor",
            ]:
                raise AssertionError(
                    f"hash_type must be 'norm' or 'hash_tensor', got {hash_type!r}"
                )
            return functools.partial(
                norm_hash_fn if hash_type == "norm" else hash_tensor_fn, use_scalar=True
            )

        if callable(hash_fn):
            fn = hash_fn
        elif isinstance(hash_fn, str):
            fn = hash_fn_option(hash_fn)
        elif isinstance(hash_fn, list):
            fns = [hash_fn_option(fn) for fn in hash_fn]
            fn = lambda x: tuple(fn(x) for fn in fns)  # noqa: E731
        else:
            raise NotImplementedError(
                f"log_tensor_hashes() expected hash_fn to be callable, str, or list[str], but found {type(hash_fn)}"
            )

        def _tree_hash(obj):
            return tree_map(
                lambda x: fn(x) if isinstance(x, tp.Tensor) else None, obj
            )

        def _dispatch_pre_log_hook(func, types, args, kwargs, call):
            """Pre-hook to capture input hashes before operation executes"""
            if "empty" in str(func) or "profiler" in str(func):
                return None

            if hash_inputs:
                # Capture input hashes before the operation
                input_hash = _tree_hash((args, kwargs))
                if not tree_all(lambda x: x is None, input_hash):
                    return {"input_hash": input_hash}
            return None

        def _dispatch_post_hook(func, types, args, kwargs, result):
            """Post-hook to capture output hashes after operation executes"""
            if "empty" in str(func) or "profiler" in str(func):
                return None

            out = {}
            out["hash"] = _tree_hash(result)

            if tree_all(lambda x: x is None, out.values()):
                return None
            return out

        try:
            if hash_inputs:
                _old_input_hfn = _utils._TRITON_INPUT_HASH_FN
                _utils._TRITON_INPUT_HASH_FN = fn
            _old_output_hfn = _utils._TRITON_OUTPUT_HASH_FN
            _utils._TRITON_OUTPUT_HASH_FN = fn
            with DebugMode.dispatch_hooks(
                log_hook=_dispatch_post_hook,
                pre_log_hook=_dispatch_pre_log_hook if hash_inputs else None,
            ):
                yield
        finally:
            if hash_inputs:
                _utils._TRITON_INPUT_HASH_FN = _old_input_hfn
            _utils._TRITON_OUTPUT_HASH_FN = _old_output_hfn

    @staticmethod
    @contextlib.contextmanager
    def _benchmarking_tp():
        """Turn recording off for the duration of a measurement.

        Autotuning launches kernels as often as it can while looking for the
        fastest; recording those would fill the dump with kernels that were
        tried and thrown away, and measure the recording rather than the work.
        """
        try:
            _utils._IN_TP_BENCHMARK = True
            yield
        finally:
            _utils._IN_TP_BENCHMARK = False

    @property
    def logs(self):
        return list(self.operators)

    @staticmethod
    def save_logs(logs: list, destination: str | os.PathLike | IO[str]) -> None:
        """Write recorded calls out, so a run can be read back in another process.

        Written as structured records rather than as text, because the point
        of keeping them is to compare them, and a comparison needs the values
        a line of text has thrown away.
        """
        data = {
            "format": _SERIALIZED_LOG_FORMAT,
            "version": _SERIALIZED_LOG_VERSION,
            "logs": [_serialize_debug_call(call) for call in logs],
        }
        if isinstance(destination, (str, os.PathLike)):
            with open(destination, "w", encoding="utf-8") as f:
                json.dump(data, f, allow_nan=False)
                f.write("\n")
        else:
            json.dump(data, destination, allow_nan=False)
            destination.write("\n")

    @staticmethod
    def load_logs(source: str | os.PathLike | IO[str]) -> list:
        """Read back what :meth:`save_logs` wrote."""
        if isinstance(source, (str, os.PathLike)):
            with open(source, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.load(source)

        if not isinstance(data, dict):
            raise ValueError(
                f"Serialized DebugMode logs must be a dict, but found {type(data).__name__}"
            )
        if data.get("format") != _SERIALIZED_LOG_FORMAT:
            raise ValueError(
                "Serialized DebugMode logs have an unrecognized format: "
                f"{data.get('format')!r}"
            )
        if data.get("version") != _SERIALIZED_LOG_VERSION:
            raise ValueError(
                "Serialized DebugMode logs have an unsupported version: "
                f"{data.get('version')!r}"
            )
        logs = data.get("logs")
        if not isinstance(logs, list):
            raise ValueError("Serialized DebugMode logs must contain a logs list")
        return [_deserialize_debug_call(call) for call in logs]

    def _handle_annotate(self, tag) -> None:
        """Record a tag, so that what follows it can be read in context."""
        call = _AnnotateCall(tag, "annotate", self.call_depth, self.record_stack_trace)
        self.operators.append(call)

    @staticmethod
    def _annotate(tag: Any) -> None:
        """Say something about the code around here, so the dump says where.

        An operator rather than a plain function, because the point is to mark
        a place in a graph that is about to be handed to a compiler, and only
        an operator survives that.  Note that an operator defined here is
        resolved and run by its own definition rather than by the dispatch
        stack, so a tag does not reach :meth:`_handle_annotate`; the tag is
        still carried in the graph, which is what a compiler needs.
        """
        _annotate(tag)

    @staticmethod
    def check_hash_mismatches(
        logs1: list, logs2: list, compare_inputs: bool = False
    ) -> list[dict]:
        """Find where two runs first disagreed about the values they saw.

        The two runs are first required to have done the same things in the
        same order, because a difference between runs that did different things
        says nothing about the values; only once that holds is any difference
        a difference about the values.  A pair where a hash is present on one
        side only is refused rather than skipped, since that means the two runs
        were not measured the same way.

        Raises:
            ValueError: If the logs differ in length, call type, name, or depth.
        """
        if len(logs1) != len(logs2):
            raise ValueError(f"Log lengths don't match: {len(logs1)} vs {len(logs2)}")

        difference_info = []
        for i, (log1, log2) in enumerate(zip(logs1, logs2)):
            # check call type
            call1_type = type(log1).__name__
            call2_type = type(log2).__name__
            if call1_type != call2_type:
                raise ValueError(
                    f"Call types don't match at index {i}: {call1_type} vs {call2_type}"
                )
            call_type = call1_type

            # check call name
            op1_name, op2_name = _get_call_name(log1), _get_call_name(log2)
            if op1_name != op2_name:
                raise ValueError(
                    f"Operators don't match at index {i}: {call_type}[{op1_name}] vs {call_type}[{op2_name}]"
                )
            op_name = op1_name

            # check call depth
            if log1.call_depth != log2.call_depth:
                raise ValueError(
                    f"Call depths for {call_type}[{op_name}] don't match at index {i}: {log1.call_depth} vs {log2.call_depth}"
                )

            # Redistribute: call args should be the same
            if isinstance(log1, _RedistributeCall):
                if log1.render([]) != log2.render([]):
                    raise ValueError(
                        f"Redistribute calls don't match at index {i}: {log1} vs {log2}"
                    )

            # Triton kernel: same arg names, arg types
            elif isinstance(log1, _TritonKernelCall):
                if log1.kwargs_str != log2.kwargs_str:
                    raise ValueError(
                        f"Triton kernel call args don't match for {log1.kernel_name} at index {i}:"
                        f"\n\nlog1: {log1.kwargs_str}\n\nlog2: {log2.kwargs_str}"
                    )

                def compare_triton_hashes(hashes1, hashes2, is_input):
                    if set(hashes1.keys()) != set(hashes2.keys()):
                        raise AssertionError(
                            f"hash key mismatch: {set(hashes1.keys())} vs {set(hashes2.keys())}"
                        )
                    for key in hashes1:
                        if hashes1[key] != hashes2[key]:
                            difference_info.append(
                                {
                                    "call_type": "triton kernel",
                                    "call": op_name,
                                    "arg_name": key,
                                    "pytree_path": None,
                                    "hash1": hashes1[key],
                                    "hash2": hashes2[key],
                                    "rel_diff": _compute_rel_diff(
                                        hashes1[key], hashes2[key]
                                    ),
                                    "is_input_hash": is_input,
                                }
                            )

                # check output hashes
                has_post_1, has_post_2 = (
                    log1.post_hashes is not None,
                    log2.post_hashes is not None,
                )
                if has_post_1 != has_post_2:
                    raise ValueError(
                        f"Triton kernel post-hash presence inconsistent for {log1.kernel_name} "
                        f"at index {i}: log1 has post_hashes={has_post_1}, log2 has post_hashes={has_post_2}"
                    )

                if has_post_1:
                    compare_triton_hashes(
                        log1.post_hashes, log2.post_hashes, is_input=False
                    )

                # maybe check input hashes
                if compare_inputs:
                    has_pre_1, has_pre_2 = (
                        log1.pre_hashes is not None,
                        log2.pre_hashes is not None,
                    )
                    if has_pre_1 != has_pre_2:
                        raise ValueError(
                            f"Triton kernel pre-hash presence inconsistent for {log1.kernel_name} "
                            f"at index {i}: log1 has pre_hashes={has_pre_1}, log2 has pre_hashes={has_pre_2}"
                        )

                    if has_pre_1:
                        compare_triton_hashes(
                            log1.pre_hashes, log2.pre_hashes, is_input=True
                        )

            # regular log calls
            elif isinstance(log1, _OpCall):

                def compare_op_hashes(hashes1, hashes2, is_input):
                    def _helper(keypath, hash1, hash2):
                        if hash1 != hash2:
                            difference_info.append(
                                {
                                    "call_type": "op",
                                    "call": op_name,
                                    "arg_name": None,
                                    "pytree_path": keystr(keypath),
                                    "hash1": hash1,
                                    "hash2": hash2,
                                    "rel_diff": _compute_rel_diff(hash1, hash2),
                                    "is_input_hash": is_input,
                                }
                            )

                    tree_map_with_path(_helper, hashes1, hashes2)

                # check output hashes
                has_hash1 = log1.log is not None and "hash" in log1.log
                has_hash2 = log2.log is not None and "hash" in log2.log
                if has_hash1 != has_hash2:
                    raise ValueError(
                        f"Output hash presence inconsistent for {call_type}[{op_name}] "
                        f"at index {i}: log1 has hash={has_hash1}, log2 has hash={has_hash2}"
                    )

                if has_hash1:
                    compare_op_hashes(
                        log1.log["hash"],
                        log2.log["hash"],
                        is_input=False,
                    )

                # maybe check input hashes
                if compare_inputs:
                    has_hash1 = log1.log is not None and "input_hash" in log1.log
                    has_hash2 = log2.log is not None and "input_hash" in log2.log
                    if has_hash1 != has_hash2:
                        raise ValueError(
                            f"Input hash presence inconsistent for {call_type}[{op_name}] "
                            f"at index {i}: log1 has input_hash={has_hash1}, log2 has input_hash={has_hash2}"
                        )

                    if has_hash1:
                        compare_op_hashes(
                            log1.log["input_hash"],
                            log2.log["input_hash"],
                            is_input=True,
                        )

        return difference_info


def _get_call_name(log) -> str:
    if isinstance(log, _OpCall):
        return _get_op_name(log.op)
    if isinstance(log, _TritonKernelCall):
        return log.kernel_name
    if isinstance(log, _AnnotateCall):
        return f"{log.header}:{log.tag}"
    if isinstance(log, _OutputPlacementCall):
        return "output_placements"
    if isinstance(log, _RedistributeCall):
        return "redistribute"
    return log.__class__.__name__


def get_active_debug_mode() -> DebugMode | None:
    """The region currently recording, if there is one.

    Asked of the stack rather than of a global, because regions nest: the
    innermost is the one whose calls are being recorded, and an outer one
    would be recording calls that are really the inner one's.
    """
    # Fast path: if no region is active, skip the stack walk
    if _ACTIVE_DEBUG_MODE_COUNT == 0:
        return None
    debug_mode = None
    for mode in _get_current_dispatch_mode_stack():
        if isinstance(mode, DebugMode):
            debug_mode = mode
            break
    return debug_mode

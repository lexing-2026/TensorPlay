"""What a backend's operators print as, one class per language.

A lowered region is arithmetic and comparisons, and each language spells those
differently.  Where a backend keeps that spelling is a class that inherits
this one: the base owns what is the same everywhere -- how a parenthesised
operand is written, how a literal is written, and what an operator nobody
registered does -- and a subclass owns one method per operator it can print.

So the questions "how does this language spell ``sigmoid``" and "can it spell
this at all" have one answer per language, and the kernel emitters ask that
answer instead of carrying a table each.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import operator
import threading
import atexit
import functools
import math
import itertools
import logging
import os
import tempfile

import re
from enum import Enum, auto
from abc import ABC, abstractmethod
from itertools import chain
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, NamedTuple, TypeVar

import sympy
from sympy.printing.python import PythonPrinter as _PythonPrinter
from sympy.printing.precedence import PRECEDENCE
import tensorplay as tp
from tensorplay.graph import Graph, Node

from tensorplay.graph.experimental.sympy_functions import (
    Max,
    OrderedSet,
    SymT,
    free_symbol_is_type,
    int_oo,
    symbol_is_type,
)
from tensorplay.graph.experimental.symbolic_shapes import ValueRanges
from .....primitives.common import ELEMENTWISE_TYPE_PROMOTION_KIND
from .. import config, metrics

log = logging.getLogger(__name__)


class DeviceOpOverrides:
    """The parts of a wrapper that are written in a device's own words.

    A wrapper has to name the device it is running on -- say which device to
    make current, which stream to run on, how to wait for that stream -- and
    each device spells those differently.  So a device is asked for them rather
    than the wrapper naming them itself, and a device that cannot answer one
    says so rather than the wrapper writing a call that would not be there.
    """

    def import_get_raw_stream_as(self, name: str) -> str:
        raise NotImplementedError

    def set_device(self, device_idx) -> str:
        raise NotImplementedError

    def synchronize(self) -> str:
        raise NotImplementedError

    def device_guard(self, device_idx) -> str:
        raise NotImplementedError

    def current_device_idx_expr(self) -> str:
        # A wrapper compiled once and run on any device has to read the device
        # it is on when it runs, not the one it was compiled for.  A device that
        # cannot say how to read that cannot have such a wrapper, and saying so
        # is better than writing one that reads a number fixed at compile time.
        raise RuntimeError(
            f"a device given as a parameter is not supported on "
            f"{type(self).__name__}: it cannot say which device it is on, so the "
            f"generated wrapper would be tied to the device it was written for."
        )

    def current_stream(self) -> str:
        raise NotImplementedError

    def cpp_stream_guard(self) -> str:
        raise NotImplementedError

    def cpp_device_guard(self) -> str:
        raise NotImplementedError

    def stream_handle(self, stream_name: str) -> str:
        return f"{stream_name}.native_handle"

    def kernel_header(self) -> str:
        raise NotImplementedError

    def kernel_driver(self) -> str:
        raise NotImplementedError


class CpuDeviceOpOverrides(DeviceOpOverrides):
    """The device that is whatever the process is running on.

    A wrapper for this device names no device, waits on nothing, and hands the
    stream the runtime already has -- there is one of those, and it is the one
    the wrapper is already using.
    """

    def import_get_raw_stream_as(self, name: str) -> str:
        return ""

    def set_device(self, device_idx) -> str:
        return ""

    def synchronize(self) -> str:
        return ""

    def device_guard(self, device_idx) -> str:
        return ""

    def current_device_idx_expr(self) -> str:
        return "0"

    def current_stream(self) -> str:
        return "0"

    def kernel_header(self) -> str:
        return ""

    def kernel_driver(self) -> str:
        return ""


class CudaDeviceOpOverrides(DeviceOpOverrides):
    """The device whose streams and devices are named by handle."""

    def import_get_raw_stream_as(self, name: str) -> str:
        return f"from tensorplay._C import _cuda_getCurrentRawStream as {name}"

    def set_device(self, device_idx) -> str:
        return f"tp.cuda.set_device({device_idx})"

    def synchronize(self) -> str:
        return "tp.cuda.synchronize()"

    def device_guard(self, device_idx) -> str:
        return f"tp.cuda.device({device_idx})"

    def current_device_idx_expr(self) -> str:
        return "tp.cuda.current_device()"

    def current_stream(self) -> str:
        return "tp.cuda.current_stream()"

    def kernel_header(self) -> str:
        return "import triton"

    def kernel_driver(self) -> str:
        return "import triton"


#: Which device's words a wrapper is written in, by device name.  Filled in by
#: :func:`register_device_op_overrides`, and read by
#: :func:`get_device_op_overrides`.
device_op_overrides_lock = threading.RLock()
device_op_overrides_dict: dict[str, DeviceOpOverrides] = {}
_device_op_overrides_initialized = False


def register_device_op_overrides(
    device: str, device_op_overrides: DeviceOpOverrides
) -> None:
    """Say which words a device's wrappers are written in."""

    with device_op_overrides_lock:
        device_op_overrides_dict[device] = device_op_overrides


def _initialize_device_op_overrides() -> None:
    """Equip the devices that answer for themselves, once.

    A flag rather than an emptiness test, because a caller may have equipped
    one device before this runs and should not have it taken away.
    """

    global _device_op_overrides_initialized
    if _device_op_overrides_initialized:
        return

    with device_op_overrides_lock:
        if _device_op_overrides_initialized:
            return

        register_device_op_overrides("cpu", CpuDeviceOpOverrides())
        register_device_op_overrides("cuda", CudaDeviceOpOverrides())
        register_device_op_overrides("xpu", CudaDeviceOpOverrides())

        _device_op_overrides_initialized = True


def get_device_op_overrides(device: str) -> DeviceOpOverrides:
    """The words one device's wrappers are written in."""

    if not isinstance(device, str):
        raise AssertionError(type(device))
    _initialize_device_op_overrides()
    return device_op_overrides_dict[device]


class PythonPrinter(_PythonPrinter):
    """Writes an expression as this project's own code rather than as sympy's.

    A size is simplified first, because an unsimplified formula is one a reader has to
    simplify to check, and a remainder is always parenthesised, because ``a % b *
    c`` means something different from ``a % (b * c)`` and a printer that guessed
    would be guessing which was meant.
    """

    def _print_str(self, expr: str) -> str:
        return expr

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        left, right = (self._print(arg) for arg in expr.args)
        return f"(({left}) // ({right}))"

    def _print_Max(self, expr: sympy.Expr) -> str:
        return f"max({', '.join(map(self._print, expr.args))})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        return f"min({', '.join(map(self._print, expr.args))})"

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        x, div, mod = (
            self.parenthesize(arg, PRECEDENCE["Atom"] - 0.5)
            for arg in expr.args
        )
        if div != "1":
            x = f"({x} // {div})"
        return f"({x} % {mod})"

    def _print_Mod(self, expr: sympy.Expr) -> str:
        # The remainder of a value the expression keeps nonnegative, spelled
        # with the language's own remainder operator.
        return self.stringify(expr.args, " % ", PRECEDENCE["Atom"] - 0.5)

    def doprint(self, expr: sympy.Expr, *, simplify: bool = True, p: bool = True) -> str:
        if simplify and isinstance(expr, sympy.Expr) and hasattr(V.graph, "sizevars"):
            expr = V.graph.sizevars.simplify(expr)
        return super().doprint(expr)

    def parenthesize(self, item: sympy.Expr, level: int, strict: bool = False) -> str:
        if isinstance(item, sympy.Mod):
            return f"({self._print(item)})"
        else:
            return super().parenthesize(item, level, strict)

from ..ops_handler import DefaultHandler, OpsHandler
from ..loops import NullKernel
from ..virtualized import V


# A decomposition writes one operation as others while a kernel is being
# printed, and those others are spelled by whichever handler is current then --
# the kernel's own, so each piece is written, named and reused like any other
# value of the kernel.
from ..virtualized import OpsValue, ops
from ..utils import (
    DeferredLineBase,
    boolean_ops,
    IndentedBuffer,
    ScopedDict,
    free_symbol_is_type,
    generate_assert,
    get_current_backend,
    sympy_subs,
    unique,
)

if TYPE_CHECKING:
    from ..loop_body import LoopBody
    from ..templates.select_algorithm import ChoiceCaller


class WorkspaceZeroMode(enum.Enum):
    """How a scratch buffer has to be left.

    The three answers are three different obligations on whoever writes into
    the buffer.  A buffer nobody zeroes is free, and one that has to be zero
    on every call has to be written before it is read, while one that has to be
    zero once for the whole graph is zeroed by the graph and left alone by
    every kernel that uses it -- which is the one a semaphore buffer wants,
    since it is a counter that must read zero when the graph starts and be
    zero again when the last user is done.
    """

    UNINITIALIZED = 0
    ZERO_ON_CALL = 1
    ZERO_PER_GRAPH = 2

    @staticmethod
    def combine(a: "WorkspaceZeroMode", b: "WorkspaceZeroMode") -> "WorkspaceZeroMode":
        """The stricter of two obligations, so joining two buffers keeps both."""

        if a == b or b == WorkspaceZeroMode.UNINITIALIZED:
            return a
        if a == WorkspaceZeroMode.UNINITIALIZED:
            return b
        raise NotImplementedError(f"WorkspaceZeroMode.combine({a!r}, {b!r})")

    @staticmethod
    def from_bool(zero_fill: bool) -> "WorkspaceZeroMode":
        return (
            WorkspaceZeroMode.ZERO_ON_CALL if zero_fill else WorkspaceZeroMode.UNINITIALIZED
        )


class CodegenSymbol(ABC):
    """A thing the code generator names, which may or may not be a buffer.

    A scratch buffer has no users, so it is not one of the region's buffers,
    yet the code that allocates and frees memory needs to treat it like one.
    This is the part of a buffer that the allocation code actually uses.
    """

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_example(self):
        pass


@dataclass
class WorkspaceArg(CodegenSymbol):
    """Memory a kernel needs while it runs and that nobody else refers to.

    It is not a buffer of the region because nothing reads it afterwards, which
    is exactly why it has to be asked for: a buffer nothing reads would be
    removed before the kernel ran.
    """

    count: Any
    zero_mode: WorkspaceZeroMode
    device: Any
    outer_name: str
    inner_name: str = "ws_ptr"
    dtype: Any = tp.uint8

    @staticmethod
    def unique_name(prefix: str = "workspace_") -> str:
        return f"{prefix}{next(V.graph.workspace_id)}"

    @staticmethod
    def can_join(a: "WorkspaceArg", b: "WorkspaceArg") -> bool:
        """Whether two requests can be served by one allocation.

        They can when the allocation would be the same one: the same name
        inside the kernel, the same element type, on the same device.
        """

        return (
            a.inner_name == b.inner_name and a.dtype == b.dtype and a.device == b.device
        )

    @staticmethod
    def join(a: "WorkspaceArg", b: "WorkspaceArg") -> "WorkspaceArg":
        """One allocation serving both requests."""

        return WorkspaceArg(
            count=a.count + b.count,
            zero_mode=WorkspaceZeroMode.combine(a.zero_mode, b.zero_mode),
            dtype=a.dtype,
            device=a.device,
            inner_name=a.inner_name,
            outer_name=a.outer_name,
        )

    @staticmethod
    def maximum(a: "WorkspaceArg", b: "WorkspaceArg") -> "WorkspaceArg":
        """One allocation big enough for either request, which may be the larger."""

        if not (
            a.dtype == b.dtype and a.device == b.device and a.inner_name == b.inner_name
        ):
            raise AssertionError(
                "WorkspaceArg.maximum requires matching dtype, device, and inner_name"
            )
        return WorkspaceArg(
            count=Max(a.count, b.count),
            zero_mode=WorkspaceZeroMode.combine(a.zero_mode, b.zero_mode),
            dtype=a.dtype,
            device=a.device,
            inner_name=a.inner_name,
            outer_name=a.outer_name,
        )

    # What follows lets a scratch buffer be handed to the code that allocates
    # and reuses buffers, which only ever asks these questions of one.
    def get_device(self) -> Any:
        return self.device

    get_device_or_error = get_device

    def get_dtype(self) -> Any:
        return self.dtype

    def get_example(self):
        return self.get_layout().get_example()

    def get_layout(self):
        from ..ir import FixedLayout

        return FixedLayout(
            device=self.device,
            dtype=self.dtype,
            size=[self.count],
            stride=[1],
        )

    @property
    def layout(self):
        return self.get_layout()

    get_output_spec = get_layout
    maybe_get_output_spec = get_layout
    maybe_get_layout = get_layout

    def get_offset(self):
        return sympy.S.Zero

    def get_size(self) -> list:
        return [self.count]

    def get_stride(self) -> list:
        return [sympy.S.One]

    def get_name(self) -> str:
        return self.outer_name

    def get_is_pinned(self) -> bool:
        return False

    def get_inputs_that_alias_output(self) -> list:
        return []


class TritonScratchWorkspace:
    """Scratch space sized in bytes, whose element type the target names itself.

    A backend spells its scratch type differently, so the spelling is left to
    the backend and only the size is stated here.
    """

    def __init__(self, size: int, generate_dtype_str: Callable):
        self.size = size
        self._generate_dtype_str = generate_dtype_str

    def generate_dtype_str(self) -> str:
        return self._generate_dtype_str()


@dataclass
class TensorArg:
    """A tensor argument, as the precompiling side needs to see it."""

    name: str
    buffer: str
    dtype: Any
    offset: Any = sympy.S.Zero
    alias_of: str | None = None


@dataclass
class SizeArg:
    """A symbolic size argument, as the precompiling side needs to see it."""

    name: str
    expr: Any

    @property
    def alias_of(self) -> str | None:
        return None


@dataclass
class ConstexprArg:
    """An argument that is part of the kernel's identity rather than its input."""

    name: str


class RemovedArg:
    """An argument the region asked for and then decided it does not have."""

    def __str__(self) -> str:
        return "REMOVED"


#: What a lookup returns for a name the region asked for and then gave up on,
#: so that a caller can tell "not an argument" from "an argument with no name
#: yet" -- the first is a mistake, the second is a name to hand out.
REMOVED = RemovedArg()


class InplacedBuffer(NamedTuple):
    """One allocation standing for several names that alias it."""

    inner_name: str
    other_names: list


@dataclass
@dataclass
class ArgName:
    """The name an argument is declared under, and whether it is a constant.

    A constant argument is not a runtime input, so it is written into the
    declaration rather than passed to the call, which is what the annotation
    in the name says.
    """

    name: str
    is_constexpr: bool = False

    def full_name(self) -> str:
        return f"{self.name}{' : tl.constexpr' if self.is_constexpr else ''}"


class CSEVariable:
    """A name for an expression, which a backend may annotate however it likes.

    The value of the class is the hook rather than the name: a backend that
    needs to know more about an expression than its name -- its shape, its
    element type, whether it is a block of a larger tensor -- overrides
    creation and the annotation hook, and every other part of the emitter
    treats it as a name.
    """

    def __init__(self, name: str, bounds, dtype=None, shape=None):
        super().__init__()
        self.name = name
        self.bounds = bounds
        self.use_count = 1
        self.dtype = dtype
        self.shape = shape

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CSEVariable) and other.name == self.name

    def update_on_args(self, name: str, args: Any, kwargs: Any) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


class CodeGen:
    """Anything that emits code, and can therefore be used as a context.

    Emitting a kernel often needs something set up around it -- a device scope,
    a buffer of temporaries to be freed at the end -- and the context is what
    guarantees the teardown happens even when the emission fails.
    """

    def __init__(self) -> None:
        super().__init__()
        self.exit_stack = contextlib.ExitStack()

    def __enter__(self):
        self.exit_stack.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)


class BracesBuffer(IndentedBuffer):
    """A buffer whose blocks are delimited by braces rather than by keywords.

    Opening a block writes an opening brace and indents; closing it writes a
    closing brace and unindents, which is the shape a C block has and the
    shape a Python ``with`` does not.
    """

    def indent(self, offset: int = 1):
        @contextlib.contextmanager
        def ctx():
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(-offset):
                self._indent -= 1
                self.writeline("}")
            yield
            for _ in range(-offset):
                self.writeline("{")
                self._indent += 1
            for _ in range(offset):
                self._indent -= 1
                self.writeline("}")

        return ctx()


def is_buffer_removed(name: str) -> bool:
    """Whether a name the region used is one the region no longer has.

    A name is looked up in each of the four places a buffer can be dropped --
    removed outright, or overwritten in place by something whose buffer is then
    dropped -- because which of them applies is not visible from the name.
    """

    return any(
        name in x
        for x in (
            V.graph.removed_buffers,
            V.kernel.removed_buffers,
            V.graph.inplaced_to_remove,
            V.kernel.inplaced_to_remove,
        )
    )


@dataclasses.dataclass
class DeviceCodegen:
    """What a device is written through, and where its results are handed back.

    Which of these a device has depends on what it can do: a device whose work
    is all done by a host program has no separate C++ wrapper, and one whose
    scheduling is not a decision at compile time may have none either.
    """

    scheduling: Any
    wrapper_codegen: Any
    cpp_wrapper_codegen: Any = None
    fx_wrapper_codegen: Any = None


#: What each device is written through, filled in by the registration below.
device_codegens: dict[str, DeviceCodegen] = {}

#: A pass a device wants run over a region before it is lowered, where the
#: device needs one and the common path does not.
custom_backend_passes: dict[str, Any] = {}

#: A device's own settings, where the device has settings of its own.
custom_backend_codegen_configs: dict[str, Any] = {}


def register_backend_for_device(
    device: str,
    device_scheduling: Any,
    device_wrapper_codegen: Any,
    device_cpp_wrapper_codegen: Any = None,
    device_fx_wrapper_codegen: Any = None,
    device_custom_pass: Any = None,
    device_custom_config: Any = None,
) -> None:
    """Equip a device with what its work is written through.

    Called once per device, and again for a device that asks to be equipped
    differently.  A device's own settings cannot be the common ones -- a device
    that was handed the common settings would be indistinguishable from not
    having any, which is why that is refused rather than ignored.
    """

    device_codegens[device] = DeviceCodegen(
        device_scheduling,
        device_wrapper_codegen,
        device_cpp_wrapper_codegen,
        device_fx_wrapper_codegen,
    )
    custom_backend_passes[device] = device_custom_pass
    if device_custom_config:
        from . import config as _config

        if not (isinstance(device_custom_config, type(_config)) and _config):
            raise AssertionError(
                f"device_custom_config={device_custom_config} cannot be the "
                f"same as the default config"
            )
    custom_backend_codegen_configs[device] = device_custom_config


def get_backend_features(device) -> "OrderedSet":
    """What a program emitter on this device can express.

    A property of the emitter rather than of a region, so it is asked of the
    emitter: the scheduling is asked what it can do rather than told.
    """

    if device is None:
        return OrderedSet()
    init_backend_registration()
    device_type = device.type if hasattr(device, "type") else device
    scheduling_ctor = get_scheduling_for_device(device_type)
    if not scheduling_ctor:
        raise AssertionError(f"no scheduling registered for device {device_type}")
    scheduling = scheduling_ctor(None)
    return scheduling.get_backend_features(device)


def init_backend_registration() -> None:
    """Equip each device this compiler writes for, once.

    A device already equipped is left alone, so that a caller which equipped it
    differently keeps what it chose.
    """

    if get_scheduling_for_device("cpu") is not None:
        return
    from .cpp import CppScheduling
    from .cpp_wrapper import CppWrapperCodegen
    from .cuda_combined_scheduling import CUDACombinedScheduling
    from .wrapper import PythonWrapperCodegen

    #: An accelerator's work may be written by more than one printer, and which
    #: one is named rather than fixed, so that the choice is a setting and not a
    #: branch in every method that writes a kernel.
    cuda_backends = {
        "triton": CUDACombinedScheduling,
    }

    register_backend_for_device(
        "cpu",
        lambda scheduling: CppScheduling(scheduling),
        PythonWrapperCodegen,
        device_cpp_wrapper_codegen=CppWrapperCodegen,
    )
    register_backend_for_device(
        "cuda",
        lambda scheduling: cuda_backends[config.cuda_backend](scheduling),
        PythonWrapperCodegen,
        device_cpp_wrapper_codegen=CppWrapperCodegen,
    )


def get_scheduling_for_device(device: str) -> Any:
    """What a device's work is scheduled by, where one is registered."""

    return device_codegens[device].scheduling if device in device_codegens else None


def get_wrapper_codegen_for_device(
    device: str, cpp_wrapper: bool = False, fx_wrapper: bool = False
) -> Any:
    """What a device's results are handed back through.

    Which of the three answers is wanted is said by which flag is set, because
    a device may have more than one way of handing results back and the caller
    is the one that knows which it needs.
    """

    if device in device_codegens:
        entry = device_codegens[device]
        if fx_wrapper:
            return entry.fx_wrapper_codegen
        if cpp_wrapper:
            return entry.cpp_wrapper_codegen
        return entry.wrapper_codegen
    return None


def get_custom_backend_pass_for_device(device: str) -> Any:
    """The pass a device wants over a region, where it wants one."""

    return custom_backend_passes.get(device)


def get_custom_backend_config_for_device(device: str) -> Any:
    """A device's own settings, where it has settings of its own."""

    return custom_backend_codegen_configs.get(device)


class FileBackedGraphModule:
    """What a printed region is: a callable with the source it was printed from.

    Exposes what a module exposes, and maps back to the graph rather than to
    Python source: the region is printed as text and run as text, so the text
    is what there is, and a debugging tool that wants to read it wants a file.
    The file goes away with the process.
    """

    gm: Any
    compiled_fn: Callable[..., Any]

    def __post_init__(self) -> None:
        self.tempfile = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".py", delete=False
        )
        atexit.register(os.remove, self.tempfile.name)
        with self.tempfile as f:
            f.write(self.value)

    @property
    def __file__(self) -> str:
        return self.tempfile.name

    def call(self, args: list) -> Any:
        return self.compiled_fn(*args)

    @property
    def value(self) -> str:
        return self.gm.code


class BackendFeature(Enum):
    """What a program emitter can express.

    A value in this vocabulary is a property of an emitter, not of a region:
    the emitters differ in what a program may ask of them, so the difference
    is declared once here and read by the lowering that picks one.
    """

    #: Takes a whole list of tensors in one program, so that a lowering which
    #: received a list did not have to be unrolled into one program per tensor.
    FOREACH = auto()
    #: Puts values into buckets by a computed position, rather than only
    #: writing each result where its index already says.
    BUCKETIZE = auto()
    #: Reads and writes one buffer in place, where nothing else wants it, so a
    #: value need never be copied to a second buffer before being overwritten.
    INPLACE_BUFFERS = auto()
    #: Scatters where the positions to write are given alongside the values,
    #: rather than being a function of the value's own index.
    MASKED_SCATTER_WITH_INDEX = auto()
    #: Carries a running total across a program's positions, where each one
    #: depends on the one before it rather than on the input alone.
    SCAN = auto()
    #: Orders the program's positions by a computed key, so a body may read
    #: them in an order the input did not arrive in.
    SORT = auto()
    #: Reduces several values at one position to a value that is itself several
    #: numbers, rather than to a single one.
    TUPLE_REDUCTION = auto()
    #: Writes in the order the program's loop already runs rather than in the
    #: order the body produced, where the two differ and the first is right.
    PREFER_STORE_LOOP_ORDER = auto()
    #: A kernel is offered as a template, written from the geometry rather than
    #: fixed, so that the same call can be measured over several shapes of it
    #: and the one that runs fastest on this device is the one kept.
    TRITON_TEMPLATES = auto()
    #: A reduction down to a single element may be written as one pass that
    #: starts from an arbitrary value, rather than needing a separate value to
    #: start from.  Without this the starting value has to be materialized.
    REDUCE_TO_SINGLE_ELEMENT = auto()


#: An operand already fully wrapped, or a single atom: no parentheses needed.
_ATOM = re.compile(r"\A[-+]?[A-Za-z_][A-Za-z_0-9]*\Z|\A\(.*\)\Z|\A-?[0-9.]+[fF]?\Z")


class BasicMathOpsMixin:
    """The operators this repository's languages all spell the same way.

    A value here is text, not an expression tree, so an operator's text is
    fully parenthesised and needs no parentheses added around it later; that
    is the one thing these spellings do differently from a tree-based
    renderer, and it is why they are written out rather than derived.  What
    belongs here is what every language spells alike -- anything a language
    spells its own way (a power, a comparison, a maximum) belongs to that
    language's class.
    """

    @staticmethod
    def add(a, b):
        return f"({a} + {b})"

    @staticmethod
    def sub(a, b):
        return f"({a} - {b})"

    @staticmethod
    def mul(a, b):
        return f"({a} * {b})"

    @staticmethod
    def truediv(a, b):
        return f"({a} / {b})"

    @staticmethod
    def and_(a, b):
        return f"({a} & {b})"

    @staticmethod
    def eq(a, b):
        return f"{a} == {b}"

    @staticmethod
    def ne(a, b):
        return f"{a} != {b}"

    @staticmethod
    def lt(a, b):
        return f"{a} < {b}"

    @staticmethod
    def gt(a, b):
        return f"{a} > {b}"

    @staticmethod
    def le(a, b):
        return f"{a} <= {b}"

    @staticmethod
    def ge(a, b):
        return f"{a} >= {b}"


class OperatorNotSupported(NotImplementedError):
    """This language has no spelling for that operator.

    A form, not a failure: the lowering that met it declines the region and
    the framework runs the region itself, which is what a fall back to the
    framework is for.
    """


class OpDecompositions:
    """How an operator is written when the language has none of its own.

    Most languages have no reciprocal, no exponential less one, no sigmoid
    and no fused multiply-add, while all of them have multiplication,
    subtraction, exponentiation and a comparison.  An operator is therefore
    decomposed here into the ones that do exist, once, so that every backend
    inherits the same decomposition rather than writing its own -- and so
    that a backend which does have the operator overrides that one and keeps
    the rest.
    """

    @staticmethod
    def identity(value):
        """The value itself, asked for so the common-subexpression pass sees it."""

        # used to trigger cse
        return value

    @staticmethod
    def reciprocal(x):
        """One over the value, with the one held at single precision.

        The constant is single precision so that a backend whose division has a
        rounding mode can be asked for the rounding that is wanted rather than
        the rounding the constant happens to have.
        """

        # Use float32 constant so that div_rn can be applied when
        # eager_numerics.division_rounding is enabled
        return ops.truediv(ops.constant(1.0, tp.float32), x)

    @staticmethod
    def square(x):
        return ops.mul(x, x)

    @staticmethod
    def erfc(x):
        return ops.sub(ops.constant(1, tp.float32), ops.erf(x))

    @staticmethod
    def erfcx(x):
        """The complement of the error function, scaled by its exponential."""

        return ops.mul(ops.exp(ops.square(x)), ops.erfc(x))

    @staticmethod
    def expm1(x):
        """The exponential less one, which loses no precision for a small argument."""

        return ops.sub(ops.exp(x), ops.constant(1, tp.float32))

    @staticmethod
    def log10(x):
        return ops.mul(ops.log(x), ops.constant(1 / math.log(10), tp.float32))

    @staticmethod
    def log2(x):
        return ops.mul(ops.log(x), ops.constant(1 / math.log(2), tp.float32))

    @staticmethod
    def exp2(x):
        return ops.exp(ops.mul(x, ops.constant(math.log(2), tp.float32)))

    @staticmethod
    def log1p(x):
        return ops.log(ops.add(x, ops.constant(1, tp.int32)))

    @staticmethod
    def sigmoid(x):
        one = ops.constant(1, tp.int32)
        return ops.truediv(one, ops.add(one, ops.exp(ops.neg(x))))

    @staticmethod
    def relu(x):
        return ops.maximum(x, ops.constant(0, tp.int32))

    @staticmethod
    def fma(x, y, z):
        """A multiply and an add that round once rather than twice.

        Where the language has no fused multiply-add, the two operations are
        written separately and the sum rounds twice.
        """

        # for backends that don't override this (halide)
        return ops.add(ops.mul(x, y), z)

    @staticmethod
    def mul_rn(x, y):
        """A multiplication rounded as the language rounds a multiplication."""

        # for backends that don't override this, just use regular mul
        return ops.mul(x, y)

    @staticmethod
    def div_rn(x, y):
        """A division rounded as the language rounds a division."""

        # for backends that don't override this, just use regular div
        return ops.truediv(x, y)

    @staticmethod
    def floor_to_int(a, dtype):
        return ops.to_dtype(ops.floor(a), dtype)

    @staticmethod
    def ceil_to_int(a, dtype):
        return ops.to_dtype(ops.ceil(a), dtype)

    @staticmethod
    def trunc_to_int(a, dtype):
        return ops.to_dtype(ops.trunc(a), dtype)

    @staticmethod
    def remainder(a, b):
        """The remainder whose sign follows the divisor rather than the dividend.

        A language's own remainder takes the sign of the dividend, so when the
        two signs differ the answer is the remainder that came out plus the
        divisor, and when they agree it is the remainder that came out.
        """

        r = ops.mod(a, b)
        cond = ops.and_(
            ops.ne(r, ops.constant(0, tp.int32)),
            ops.ne(ops.signbit(r), ops.signbit(b)),
        )
        return ops.where(cond, ops.add(r, b), r)

    @staticmethod
    def round_to_int(a, dtype):
        return ops.to_dtype(ops.round(a), dtype)
class OpOverrides(BasicMathOpsMixin, OpDecompositions, OpsHandler):
    """The operators a language spells, and the spelling of the rest.

    A subclass adds one method per operator it prints.  A method takes the
    operands its caller already spelled and returns the expression, so an
    operator is written once and every kernel in that language prints it the
    same way.  An operator with no method is not spelled here, and asking for
    it says so rather than printing something approximate.
    """

    #: The name a method owns: an operator is spelled the same way in the
    #: graph as it is here, so the mapping is mechanical.
    @staticmethod
    def _unimplemented(name: str):
        """The handler installed for an operation this target cannot print.

        A named failure beats a silent fallback: the operation is one the
        target has no form for, and saying so at the point of use is what
        tells the caller which one it was.
        """

        def unimplemented(self, *args, **kwargs):
            raise NotImplementedError(
                f"{type(self).__name__} does not implement ops.{name}"
            )

        unimplemented.__name__ = name
        unimplemented.is_unimplemented = True
        return unimplemented

    @classmethod
    def _is_unimplemented(cls, name: str) -> bool:
        """Whether an operation on this target is still the generic handler.

        Three things count as unimplemented: nothing defined it, what is
        defined is what the generic handler defines, or what is defined is
        one of the named failures above.
        """

        fn = getattr(cls, name, None)
        default_fn = getattr(OpsHandler, name, None)
        return not fn or fn == default_fn or getattr(fn, "is_unimplemented", False)

    @classmethod
    def _initialize_pointwise_overrides(cls, target: str) -> None:
        """Install this target's form of every element-wise operation.

        Each operation's data names one implementation per target, or none.
        Where a target has none, the operation is left as a named failure
        rather than bound to something that would print the wrong thing.
        """

        if target not in ("triton", "cpp", "cppvec", "halide", "mps"):
            raise AssertionError(target)

        for funcname, data in pointwise_overrides_data.items():
            impl = getattr(data, target, None)
            if impl is None:
                if cls._is_unimplemented(funcname):
                    setattr(cls, funcname, cls._unimplemented(funcname))
            else:
                if funcname in cls.__dict__:
                    raise AssertionError(
                        f"multiple definitions of {funcname} on {cls.__name__}"
                    )
                impl.__name__ = funcname
                setattr(cls, funcname, staticmethod(impl))

    @staticmethod
    def logical_not(a: str) -> str:
        return f"{OpOverrides.paren(a)} == 0"

    @staticmethod
    def bitwise_and(x: str, y: str) -> str:
        return f"{OpOverrides.paren(x)} & {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_or(x: str, y: str) -> str:
        return f"{OpOverrides.paren(x)} | {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_xor(x: str, y: str) -> str:
        return f"{OpOverrides.paren(x)} ^ {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_not(x: str) -> str:
        return f"~{OpOverrides.paren(x)}"

    @staticmethod
    def bitwise_left_shift(x: str, y: str) -> str:
        return f"{OpOverrides.paren(x)} << {OpOverrides.paren(y)}"

    @staticmethod
    def bitwise_right_shift(x: str, y: str) -> str:
        return f"{OpOverrides.paren(x)} >> {OpOverrides.paren(y)}"

    @staticmethod
    def int_truediv(a: str, b: str) -> str:
        from ..loops import ops

        # A whole number divided by a whole number is not the same as the
        # quotient of the division, and the caller asked for the former.
        return ops.truediv(a, b)

    @staticmethod
    def load_seed(name: str, offset) -> str:
        from ..loops import ops

        return ops.load(name, sympy.Integer(offset))

    def indirect_indexing(
        self,
        var: str,
        size,
        check: bool = True,
        wrap_neg: bool = True,
    ):
        from ..utils import sympy_index_symbol

        return sympy_index_symbol(str(var))

    def check_bounds(self, expr, size, lower: bool, upper: bool) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: check_bounds should be handled by CSEProxy"
        )

    def load(self, name: str, index) -> str:
        raise NotImplementedError(
            f"{type(self).__name__}: load should be handled by CSEProxy"
        )

    def store(self, name: str, index, value: str, mode=None) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: store should be handled by CSEProxy"
        )

    def store_reduction(self, name: str, index, value: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: store_reduction should be handled by CSEProxy"
        )

    def reduction(
        self,
        dtype,
        src_dtype,
        reduction_type,
        value,
    ):
        raise NotImplementedError(
            f"{type(self).__name__}: reduction should be handled by CSEProxy"
        )

    def scan(
        self,
        dtypes,
        combine_fn,
        values,
    ):
        raise NotImplementedError(
            f"{type(self).__name__}: scan should be handled by CSEProxy"
        )

    def sort(
        self,
        dtypes,
        values,
        stable: bool,
        descending: bool,
    ):
        raise NotImplementedError(
            f"{type(self).__name__}: sort should be handled by CSEProxy"
        )

    def bucketize(
        self,
        values: str,
        boundaries,
        boundary_indices: str,
        indexing_dtype,
        right: bool,
        sorter=None,
        sorter_indices=None,
    ) -> str:
        raise NotImplementedError(
            f"{type(self).__name__}: bucketize should be handled by CSEProxy"
        )

    def device_assert_async(self, cond, msg: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: device_assert_async should be handled by CSEProxy"
        )

    def halide_clamp(self, value: str, size, check: bool) -> str:
        raise NotImplementedError(
            f"{type(self).__name__}: halide_clamp only implemented for Halide backend"
        )

    def dot(self, x: str, y: str) -> str:
        raise NotImplementedError(
            f"{type(self).__name__}: dot only implemented for Triton backend"
        )

    def inline_asm_elementwise(
        self,
        *inputs: str,
        asm: str,
        constraints: str | None = None,
        dtype=None,
        is_pure: bool = True,
        pack: int = 1,
        input_dtypes=None,
    ) -> str:
        raise NotImplementedError(
            f"{type(self).__name__}: inline_asm_elementwise only implemented for Triton backend"
        )

    def output(self, *args: str) -> None:
        raise AssertionError(
            f"{type(self).__name__}: ops.output should not appear at codegen time"
        )

    def placeholder(self, index: int) -> str:
        raise AssertionError(
            f"{type(self).__name__}: ops.placeholder should not appear at codegen time"
        )

    @staticmethod
    def method_name(op: str) -> str:
        return re.sub(r"\W", "_", op)

    @staticmethod
    def paren(string: str) -> str:
        """``string`` wrapped unless it is already one atom or a whole group."""

        return string if _ATOM.match(string) else f"({string})"

    @staticmethod
    def constant(value: Any) -> str:
        """A literal as this language spells one."""

        return repr(value)

    def knows(self, op: str) -> bool:
        return callable(getattr(self, self.method_name(op), None))

    def fallback(self, op: str, *operands: str) -> str:
        """What an operator with no method does: it is not spelled here."""

        raise OperatorNotSupported(f"no spelling for {op!r}")

    def expression(self, op: str, *operands: str) -> str:
        """The expression for ``op`` over already-spelled operands."""

        method = getattr(self, self.method_name(op), None)
        if method is None:
            return self.fallback(op, *operands)
        arity = self.arity(op)
        return method(*operands[:arity])

    def arity(self, op: str) -> int:
        """How many operands ``op`` takes; one unless the class says."""

        return 2 if op in self.BINARY else 1

    #: Operators that take two operands.  A subclass that prints any sets this.
    BINARY: frozenset = frozenset()

    @staticmethod
    def _unimplemented(name: str) -> Callable[..., Any]:
        """The spelling of an operation this language has none for.

        Asking for it is a mistake worth reporting at the point of the ask,
        where the operation is named, rather than as a missing attribute.
        """

        def unimplemented(self: "OpOverrides", *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement ops.{name}"
            )

        unimplemented.__name__ = name
        unimplemented.is_unimplemented = True
        return unimplemented

    @classmethod
    def _is_unimplemented(cls, name: str) -> bool:
        """Whether ``name`` is absent, inherited unchanged, or a placeholder.

        All three mean the same thing here: the class in question has no
        spelling of that operation of its own.
        """

        fn = getattr(cls, name, None)
        default_fn = getattr(OpsHandler, name, None)
        return not fn or fn == default_fn or getattr(fn, "is_unimplemented", False)


@dataclass
class OptimizationContext:
    """What type an operation is carried in, and which operation it is.

    The type is decided where the operation is scheduled, and read back where
    the code for it is written, so that a narrow value is not widened and
    narrowed again around every operation that could have stayed in it.
    """

    key: ClassVar[str] = "opt_ctx"

    dtype: Any = None
    ops_name: str = ""


def index_prevent_reordering(
    index: Sequence[Expr],
    index_vars: Sequence[Expr],
    sizes: Sequence[Expr],
) -> list[Expr]:
    """The accesses, plus one more that pins the loop order.

    Simplifying loops is free to reorder them, which would change what the
    accesses mean.  An extra access that walks the loops in their current order
    has a stride ordering of its own that nothing else has, so the reordering
    cannot get past it.
    """

    from ..ir import FlexibleLayout
    from ..utils import sympy_dot

    # added contiguous index prevents reordering
    return [*index, sympy_dot(index_vars, FlexibleLayout.contiguous_strides(sizes))]


def jinja2_env() -> Any:
    """The environment a template source is compiled in, or ``None`` without it.

    Strict undefined is the point: a name a template uses that the caller did
    not supply is an error rather than an empty string, because a kernel that
    silently renders a hole is worse than one that refuses to render.
    """

    try:
        import jinja2

        return jinja2.Environment(undefined=jinja2.StrictUndefined)
    except ImportError:
        return None


def check_dtype(buffer: "IndentedBuffer", var: "CSEVariable", dtype) -> None:
    """Write a check of the type the emitter believes a value has.

    The check goes into the generated code rather than into this process,
    because the belief being checked is the emitter's, and only the code that
    was emitted can say whether it turned out to be right.
    """

    backend = get_current_backend()
    if config.test_configs.runtime_triton_dtype_assert:
        buffer.writeline(f"static_assert({var}.dtype == {dtype})")
    elif config.test_configs.static_cpp_dtype_assert and backend == "cpp":
        from .cpp_utils import CppCSEVariable, DTYPE_TO_CPP

        if not isinstance(var, CppCSEVariable):
            raise AssertionError(type(var))
        if dtype == tp.bool:
            if var.is_vec:
                is_same_dt = f"IsVecMaskType<decltype({var})>::value"
            else:
                # In the host language the bitwise and of two booleans is an
                # integer, and a value used as a condition may be one or the
                # other, so both count as the boolean it was written as.
                is_same_dt = (
                    f"std::is_same_v<decltype({var}), bool> || "
                    f"std::is_same_v<decltype({var}), int>"
                )
        else:
            c_var_type = f"decltype({var})"
            if var.is_vec:
                c_var_type = f"tensorplay::vec::Vectorized<{c_var_type}>"
            is_same_dt = f"std::is_same_v<{c_var_type}, {DTYPE_TO_CPP[dtype]}>"
        buffer.writeline(f"static_assert({is_same_dt});")


def check_shape(buffer: "IndentedBuffer", var: "CSEVariable", shape) -> None:
    """Write a check of the shape the emitter believes a value has.

    The check goes into the generated code, because the belief being checked is
    the emitter's and only the emitted code can say whether it was right.
    """

    if shape is None:
        raise AssertionError("expected shape to be not None")
    if config.test_configs.runtime_triton_shape_assert:
        shape_str = (
            ", ".join(str(d) for d in shape) if len(shape) != 1 else f"{shape[0]},"
        )
        buffer.writeline(f"static_assert({var}.shape == ({shape_str}))")


def check_nan(buffer: "IndentedBuffer", var: "CSEVariable") -> None:
    """Write a check that a value is neither a not-a-number nor an infinity.

    A kernel that reads a value nobody wrote sees whatever was in memory, and
    on the accelerator that is often a not-a-number rather than a small
    number, so the check is the difference between a wrong answer and an
    answer that says the memory was wrong.
    """

    backend = get_current_backend()
    if backend == "triton":
        msg = "NaN or Inf found"
        buffer.writeline(
            f"static_assert(({var} == {var}) & ({var} != inf) & ({var} != -inf), '{msg}')"
        )


class KernelArgs:
    """The arguments one kernel is called with, and the names they are given.

    A name the region uses -- a buffer, an extent, a scratch buffer -- is not
    the name the kernel is called with, and the mapping from one to the other
    is what a kernel's declaration and its call site are generated from.  The
    mapping is built as the kernel is emitted, one name at a time, and every
    name the region asked for gets exactly one argument name, so that two
    different buffers can never be handed to the kernel under one name.

    An argument that has been written in place appears under one name for the
    input and the output, since the kernel is handed one pointer for both and
    the distinction is the region's rather than the kernel's.
    """

    @staticmethod
    def _lookup(prefix, odict, name):
        """The name for one value, handing out a new one the first time.

        A name that is already mapped is used as it is, which is what makes
        the same buffer arriving twice resolve to the same argument; a name
        that is not is given the next name under the prefix of its kind.
        """

        result = odict.get(name, REMOVED)
        if isinstance(result, RemovedArg):
            odict[name] = new_result = f"{prefix}{len(odict)}"
            return new_result
        return result

    def __init__(self) -> None:
        self.input_buffers: dict = {}
        self.output_buffers: dict = {}
        self.inplace_buffers: dict = {}
        self.sizevars: dict = {}
        self.workspace_args: list = []

    def __repr__(self) -> str:
        return "KernelArgs({})".format(
            ", ".join(
                map(
                    repr,
                    [
                        self.input_buffers,
                        self.output_buffers,
                        self.inplace_buffers,
                        self.sizevars,
                    ],
                )
            )
        )

    @staticmethod
    def _buffer_is_marked_removed(name: object) -> bool:
        return isinstance(name, RemovedArg)

    def input(self, name: str) -> str:
        """The argument name for a value the kernel reads."""

        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        if name in V.graph.removed_buffers:
            raise AssertionError(name)
        if name in self.output_buffers:
            return self.output_buffers[name]
        if name in self.inplace_buffers:
            return self.inplace_buffers[name].inner_name
        if name.startswith("seed"):
            return self._lookup("seed", self.input_buffers, name)
        return self._lookup("in_ptr", self.input_buffers, name)

    def output(self, name: str) -> str:
        """The argument name for a value the kernel writes."""

        if V.graph.scheduler:
            name = V.graph.scheduler.mutation_real_name.get(name, name)
        if name in V.graph.removed_buffers:
            raise AssertionError(name)
        if name in self.inplace_buffers:
            return self.inplace_buffers[name].inner_name
        return self._lookup("out_ptr", self.output_buffers, name)

    def make_inplace(self, input_name: str, output_name: str) -> None:
        """Hand one argument for two names, because the kernel writes in place."""

        if input_name in V.graph.unaligned_buffers:
            V.graph.unaligned_buffers.add(output_name)
        if output_name in self.inplace_buffers:
            raise AssertionError(output_name)
        if input_name in self.inplace_buffers:
            buf = self.inplace_buffers[input_name]
            if isinstance(buf, RemovedArg):
                raise AssertionError("buf must not be a RemovedArg")
            buf.other_names.append(output_name)
            self.inplace_buffers[output_name] = buf
        else:
            alive_buffers = [
                val
                for val in self.inplace_buffers.values()
                if not isinstance(val, RemovedArg)
            ]
            removed_buffers = [
                val
                for val in self.inplace_buffers.values()
                if isinstance(val, RemovedArg)
            ]
            inplace_buffer_idx = len(unique(alive_buffers)) + len(removed_buffers)
            buf = InplacedBuffer(
                f"in_out_ptr{inplace_buffer_idx}",
                [input_name, output_name],
            )
            self.inplace_buffers[input_name] = buf
            self.inplace_buffers[output_name] = buf

    def workspace(self, nelem, zero_fill: bool, dtype=None):
        """Ask for scratch memory of a given size, or for a share of some.

        Two requests can share one allocation when the allocation would be the
        same one -- the same name inside the kernel, the same element type, on
        the same device -- and a shared allocation is bigger than either
        request, so the second request is answered with the first one's
        beginning and the buffer grows.  A caller that asked for a region of
        this buffer therefore has to be told which region, which is the offset
        this returns.
        """

        if dtype is None:
            dtype = tp.uint8
        arg = WorkspaceArg(
            count=nelem,
            zero_mode=WorkspaceZeroMode.from_bool(zero_fill),
            device=V.graph.get_current_device_or_throw(),
            outer_name=WorkspaceArg.unique_name(),
            dtype=dtype,
        )
        for i, existing_arg in enumerate(self.workspace_args):
            if WorkspaceArg.can_join(existing_arg, arg):
                offset = existing_arg.count
                self.workspace_args[i] = WorkspaceArg.join(existing_arg, arg)
                return existing_arg.inner_name, existing_arg.outer_name, offset
            if not (
                existing_arg.inner_name != arg.inner_name
                and existing_arg.outer_name != arg.outer_name
            ):
                raise AssertionError(existing_arg)
        self.workspace_args.append(arg)
        return arg.inner_name, arg.outer_name, 0

    def semaphores(self, min_size) -> str:
        """A counter buffer shared by every kernel, zeroed once for the graph.

        A kernel that counts leaves the buffer zero when it is done, so the
        next kernel can count from zero; the graph zeroes it once at the start.
        The name is emitted into the wrapper, so it carries the device's index
        unless the run is one device, where the name alone is enough.
        """

        current_device = V.graph.get_current_device_or_throw()
        suffix = f"_{current_device.index}"
        arg = WorkspaceArg(
            count=min_size,
            zero_mode=WorkspaceZeroMode.ZERO_PER_GRAPH,
            dtype=tp.uint32,
            inner_name="sem_ptr",
            outer_name=f"semaphores_{current_device.type}{suffix}",
            device=current_device,
        )
        for existing_arg in self.workspace_args:
            if existing_arg.inner_name == arg.inner_name:
                if arg != existing_arg:
                    raise AssertionError((arg, existing_arg))
        self.workspace_args.append(arg)
        return arg.inner_name

    def seed_offset(self, name: str, value: int) -> str:
        """A constant extent as an argument, so that a stored decision can hit.

        An extent that is a constant is still handed to the kernel as an
        argument rather than written into it, because a kernel that takes its
        extent as an argument is the same kernel for every extent, and a stored
        decision about it applies to the next call rather than only to this one.
        """

        if not isinstance(value, int):
            raise AssertionError((type(value), value))
        value = sympy.Integer(value)
        if value in self.sizevars:
            return self.sizevars[value]
        if name in self.sizevars.values():
            name = (
                f"{name}{sum(1 for v in self.sizevars.values() if v.startswith(name))}"
            )
        self.sizevars[value] = name
        return name

    def size(self, name) -> str:
        """The argument name for an extent."""

        if not isinstance(name, sympy.Symbol):
            raise AssertionError((type(name), name))
        if name.name == "seed":
            self.sizevars[name] = "seed"
            return "seed"
        return self._lookup("ks", self.sizevars, name)

    def call_names(self):
        """Every name the kernel is called with, in the order they are passed."""

        return chain(
            self.input_buffers.keys(), self.output_buffers.keys(), self.sizevars.keys()
        )

    def arg_name(self, name: str):
        """The argument name a region's name was given, if it has one."""

        inplaced = self.inplace_buffers.get(name, None)
        if inplaced is not None and not isinstance(inplaced, RemovedArg):
            return inplaced.inner_name
        output_name = self.output_buffers.get(name, None)
        if output_name is not None and not isinstance(output_name, RemovedArg):
            return output_name
        return self.input_buffers.get(name, None)

    def wrap_ptr_arg(self, buf: str, dtype):
        return buf

    def wrap_size_arg(self, size):
        return str(size)

    def cpp_argdefs(self, dtype_to_cpp_type=None):
        """The declaration, the call arguments, and the types, as host text.

        The three come back together because they have to agree: the kernel is
        declared with the first, called with the second, and a mismatch between
        them is a kernel that compiles and reads the wrong pointer.
        """

        from .cpp_utils import INDEX_TYPE
        from .cpp_utils import DTYPE_TO_CPP

        if dtype_to_cpp_type is None:
            dtype_to_cpp_type = DTYPE_TO_CPP

        call_args = []
        arg_defs = []
        arg_types = []
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            outer = inplaced.other_names[-1]
            inner = inplaced.inner_name
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = dtype_to_cpp_type[dtype]
            arg_defs.append(f"{cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"{cpp_dtype}*")
        for outer, inner in self.input_buffers.items():
            if outer in self.inplace_buffers:
                continue
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = dtype_to_cpp_type[dtype]
            arg_defs.append(f"const {cpp_dtype}* {inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"const {cpp_dtype}*")
        for outer, maybe_inner in self.output_buffers.items():
            if outer in self.inplace_buffers or isinstance(maybe_inner, RemovedArg):
                continue
            dtype = V.graph.get_dtype(outer)
            cpp_dtype = dtype_to_cpp_type[dtype]
            arg_defs.append(f"{cpp_dtype}* {maybe_inner}")
            call_args.append(self.wrap_ptr_arg(outer, dtype))
            arg_types.append(f"{cpp_dtype}*")
        for outer, inner in self.sizevars.items():
            if isinstance(outer, sympy.Symbol) and symbol_is_type(
                outer, (SymT.UNBACKED_FLOAT,)
            ):
                arg_defs.append(f"const float {inner}")
                arg_types.append("const float")
            else:
                arg_defs.append(f"const {INDEX_TYPE} {inner}")
                arg_types.append(f"const {INDEX_TYPE}")
            call_args.append(self.wrap_size_arg(outer))
            if V.graph.wrapper_code:
                V.graph.wrapper_code.ensure_size_computed(outer)
        if self.workspace_args:
            raise AssertionError("Workspace not supported on CPU ")
        return arg_defs, call_args, arg_types

    def python_argdefs(self):
        """The arguments as the side that calls the kernel has to name them.

        This is the other end of the same mapping: a declaration for the text
        of the kernel, and a list of what to pass for the call.  The types come
        back as well, because the side that builds the arguments has to build
        them with those types.
        """

        arg_defs: list = []
        call_args: list[str] = []
        arg_types: list = []
        precompile_args: list = []
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            arg_defs.append(ArgName(inplaced.inner_name))
            call_args.append(inplaced.other_names[-1])
            arg_types.append(V.graph.get_dtype(inplaced.other_names[-1]))
            precompile_args.append(
                TensorArg(
                    name=inplaced.inner_name,
                    buffer=inplaced.other_names[-1],
                    dtype=V.graph.get_dtype(inplaced.other_names[-1]),
                )
            )
        for outer, inner in chain(
            self.input_buffers.items(),
            self.output_buffers.items(),
        ):
            if outer in self.inplace_buffers or isinstance(inner, RemovedArg):
                continue
            arg_defs.append(ArgName(inner))
            call_args.append(outer)
            arg_types.append(V.graph.get_dtype(outer))
            precompile_args.append(
                TensorArg(
                    name=inner,
                    buffer=outer,
                    dtype=V.graph.get_dtype(outer),
                )
            )
        for outer, inner in self.sizevars.items():
            arg_defs.append(ArgName(inner))
            call_args.append(outer)
            arg_types.append(type(outer))
            precompile_args.append(SizeArg(inner, outer))
            if V.graph.wrapper_code:
                V.graph.wrapper_code.ensure_size_computed(outer)
        for arg in self.workspace_args:
            arg_defs.append(ArgName(arg.inner_name))
            call_args.append(arg.outer_name)
            precompile_args.append(arg)
            arg_types.append(arg.dtype)
        return arg_defs, call_args, precompile_args, arg_types

    def aliases(self):
        """The pairs of argument names that are the same pointer.

        A name that was dropped after being written in place is not reported,
        because the buffer it stood for is no longer passed to anything.
        """

        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            for other in inplaced.other_names:
                if (
                    other in V.graph.inplaced_to_remove
                    or other in V.kernel.inplaced_to_remove
                ):
                    continue
                if other in self.input_buffers:
                    yield self.input_buffers[other], inplaced.inner_name
                if other in self.output_buffers:
                    yield self.output_buffers[other], inplaced.inner_name

    def is_removed(self, name: str) -> bool:
        return isinstance(
            self.output_buffers.get(name, REMOVED), RemovedArg
        ) and isinstance(self.inplace_buffers.get(name, REMOVED), RemovedArg)

    def live_output_buffers(self):
        """The names of the buffers that hold new data once the kernel has run.

        This is the question a caller has to answer before it frees anything: a
        buffer the kernel wrote in place holds new data under two names, and
        only the last of them is the one that is still the buffer.
        """

        live_outs: OrderedSet = OrderedSet()
        for inplaced in unique(self.inplace_buffers.values()):
            if isinstance(inplaced, RemovedArg):
                continue
            live_outs.add(inplaced.other_names[-1])
        for outer, inner in self.output_buffers.items():
            if outer in self.inplace_buffers or isinstance(inner, RemovedArg):
                continue
            live_outs.add(outer)
        return live_outs


class CSE:
    """Common subexpression elimination, over the names a kernel writes.

    An expression that has been written once is written again rather than
    recomputed, and the name it was written under is what the rest of the
    kernel refers to it by.  The key is the expression's own text, which is
    what makes two expressions the same when they are written the same way, and
    a backend that needs to distinguish two expressions that print alike widens
    the key rather than reimplementing the cache.
    """

    def __init__(
        self,
        prefix: str = "",
        suffix: str = "",
        name_prefix: str = "tmp",
        iter_buffers=None,
        store_cache=None,
        reduction_cache=None,
        varname_map=None,
    ):
        self.prefix = prefix
        self.suffix = suffix
        self._cache: dict = {}
        self.name_prefix = name_prefix
        self.store_cache: dict = store_cache if store_cache is not None else {}
        self.reduction_cache: dict = (
            reduction_cache if reduction_cache is not None else {}
        )
        self.iter_buffer_ids = iter_buffers if iter_buffers is not None else itertools.count()
        self.invalidated_stores: OrderedSet = OrderedSet()
        self.varname_map: dict = varname_map if varname_map is not None else {}

    def invalidate(self, keep_vars) -> None:
        """Drop everything that is not still needed, and say what was dropped.

        A store that is dropped is a buffer the kernel was going to write and
        will not, so the caller has to be able to find out which ones those
        were.
        """

        for name, tmp in [*self.store_cache.items()]:
            if tmp not in keep_vars:
                del self.store_cache[name]
                self.invalidated_stores.add(name)
        if keep_vars:
            self._cache = {k: v for k, v in self._cache.items() if v in keep_vars}
        else:
            self._cache = {}

    def clone(self):
        """A copy that shares the names already handed out and the stores."""

        return type(self)(
            prefix=self.prefix,
            suffix=self.suffix,
            name_prefix=self.name_prefix,
            iter_buffers=self.iter_buffer_ids,
            store_cache=self.store_cache,
            varname_map=self.varname_map,
            reduction_cache=self.reduction_cache,
        )

    def scoped_copy(self):
        """A copy whose additions do not become visible in this one."""

        new_cse = self.clone()
        new_cse._cache = ScopedDict(self._cache)
        new_cse.reduction_cache = ScopedDict(self.reduction_cache)
        new_cse.store_cache = ScopedDict(self.store_cache)
        return new_cse

    def augment_key(self, cache_key: str):
        """The key an expression is cached under, widened by the backend.

        Two expressions that print the same are the same expression unless the
        backend knows of a reason to treat them differently, and a backend with
        such a reason widens the key here.
        """

        return cache_key

    def put(self, cache_key: str, val) -> None:
        self._cache[self.augment_key(cache_key)] = val

    def contains(self, cache_key: str) -> bool:
        return self.augment_key(cache_key) in self._cache

    def try_get(self, cache_key: str):
        return self._cache.get(self.augment_key(cache_key), None)

    def get(self, cache_key: str):
        return self._cache[self.augment_key(cache_key)]

    def contains_value(self, value) -> bool:
        """Whether a name is already in use, in any of the three tables.

        A name in the expression table has been written, one in the store table
        is a store the kernel performs, and one in the reduction table is the
        accumulator of a reduction.  A caller that has a name in hand needs to
        know it is not already taken by any of them.
        """

        return (
            value in self._cache.values()
            or value in self.store_cache.values()
            or value in self.reduction_cache.values()
        )

    def generate(
        self,
        buffer: "IndentedBuffer",
        expr,
        *,
        bounds=None,
        write: bool = True,
        assignment: bool = True,
        dtype=None,
        shape=None,
    ):
        """Write the expression if it is new, and return the name to use for it.

        An expression that is already written is not written again; it is
        referred to by the name it was written under, with its range of
        possible values narrowed by whatever is now known about it.  A value
        that is not a name is written out, and the way it is written depends on
        what it is: a block of text is spliced in, a deferred line is rewritten
        with the name in front, and a plain string becomes the right-hand side
        of an assignment.
        """

        if bounds is None:
            bounds = ValueRanges.unknown()
        if isinstance(expr, OpsValue):
            # A value spelled through the handler arrives wrapped so that it
            # can be written with operators; what it holds is the value.
            expr = expr.value

        if not (write or assignment):
            raise AssertionError("expected write or assignment to be set")
        if isinstance(expr, CSEVariable):
            # The bounds a value was created with may be wider than what is
            # known now, so they are narrowed rather than replaced; replacing
            # them would throw away a fact the caller had established.
            expr.bounds = expr.bounds.tighten(bounds)
            expr.use_count += 1
            return expr
        elif isinstance(expr, IndentedBuffer):
            cache_key = expr.getvalue()
        elif isinstance(expr, DeferredLineBase):
            cache_key = expr.line
        else:
            if not isinstance(expr, str):
                raise AssertionError(f"expected str, got {type(expr)}")
            cache_key = expr
        var = self.try_get(cache_key)
        if shape is None and not assignment:
            # There is no assignment to hang a shape on, so any shape but none
            # will do: an unknown shape is a failure, a wrong one is not.
            shape = ()
        if not var:
            var = self.newvar(bounds, dtype, shape)
            self.put(cache_key, var)
            if write:
                if isinstance(expr, IndentedBuffer):
                    if assignment:
                        buffer.writeline(f"{self.prefix}{var} =")
                    buffer.splice(expr)
                    buffer.writeline(self.suffix)
                elif isinstance(expr, DeferredLineBase):
                    if not assignment:
                        raise AssertionError("expected assignment to be set")
                    buffer.writeline(
                        expr._new_line(f"{self.prefix}{var} = {expr.line}{self.suffix}")
                    )
                else:
                    if assignment:
                        line = f"{self.prefix}{var} = {expr}{self.suffix}"
                    else:
                        line = f"{expr}{self.suffix}"
                    buffer.writeline(line)

                    # The host emitter cannot tell a vector from a scalar at
                    # this point, so the check is only written where the answer
                    # is already known.
                    if (
                        assignment
                        and (
                            config.test_configs.runtime_triton_dtype_assert
                            or config.test_configs.static_cpp_dtype_assert
                        )
                        and dtype is not None
                        and get_current_backend() != "cpp"
                    ):
                        check_dtype(buffer, var, dtype)

        else:
            var.bounds = var.bounds.tighten(bounds)
            var.use_count += 1

        return var

    def newvar(self, bounds=None, dtype=None, shape=None):
        """A name that has not been used, for an expression about to be written."""

        if bounds is None:
            bounds = ValueRanges.unknown()
        var_name = f"{self.name_prefix}{next(self.iter_buffer_ids)}"
        var = V.kernel.create_cse_var(var_name, bounds, dtype, shape)
        self.varname_map[var_name] = var
        return var

    def namedvar(self, name: str, bounds=None, dtype=None, shape=None):
        """A name the caller chose, which must not already be in use.

        A name that is already taken would make two different expressions
        referable by one name, so the collision is refused here rather than
        producing a kernel that reads one value as another.
        """

        if bounds is None:
            bounds = ValueRanges.unknown()
        if name in self.varname_map:
            raise AssertionError(f"duplicate name: {name}")
        var = V.kernel.create_cse_var(name, bounds, dtype, shape)
        self.varname_map[name] = var
        return var


class CSEProxy(DefaultHandler):
    """The operator handler a kernel is emitted through.

    An operation is not written where it is written: it is asked for, the
    backend says what it looks like there, and the result is given a name the
    rest of the kernel refers to.  This sits between the two, so that the
    backend's answer is turned into a name exactly once and every operation
    goes through the same bookkeeping -- the store cache, the counts, the
    trace -- no matter which backend answered.
    """

    name = "CSEProxy"

    def __init__(self, kernel, parent_handler):
        super().__init__()
        self.vr_analysis = None
        self.kernel = kernel
        self.parent_handler = parent_handler

    def _default(self, name, args, kwargs):
        """Answer one operation, and give the answer a name."""

        bounds = self._bound_variable(name, *args, **kwargs)

        value = getattr(self.parent_handler, name)(*args, **kwargs)
        from ..dtype_propagation import DtypePropagationOpsHandler
        from ..shape_propagation import ShapePropagationOpsHandler

        dtype_handler = DtypePropagationOpsHandler()
        shape_handler = ShapePropagationOpsHandler()

        backend = get_current_backend()

        shape_op = getattr(shape_handler, name, None)
        output_dtype = None
        output_shape = None

        if name == "masked" and backend == "triton":
            output_dtype = getattr(value, "dtype", None)
            output_shape = getattr(value, "shape", None)
        elif name == "masked" and backend == "cpp":
            # The masked part is a block of the body, not a value: its type is
            # the one the dtype pass settled for the node, and its shape is not
            # known here.
            opt_ctx = V.interpreter.current_node.meta.get(OptimizationContext.key, None)
            output_dtype = getattr(opt_ctx, "dtype", None)
        elif backend in ("triton", "cpp", "mps"):
            dtype_op = getattr(dtype_handler, name, None)
            if dtype_op is not None:
                output_dtype = dtype_op(*args, **kwargs)
            if shape_op is not None:
                output_shape = shape_op(*args, **kwargs)

        if backend in ("triton", "cpp"):
            # Some backends' answers carry no type of their own, and a name
            # with no type is one the rest of the kernel cannot check, so it is
            # required here rather than left to fail later.
            if output_dtype is None:
                output_dtype = getattr(value, "dtype", None)

        output_idx = 0

        def do_cse(v):
            """Name one value of the answer, which may be several."""

            nonlocal output_idx
            var_dtype = (
                output_dtype[output_idx]
                if isinstance(output_dtype, (list, tuple))
                else output_dtype
            )
            var_shape = (
                output_shape[output_idx]
                if isinstance(output_shape, (list, tuple))
                and len(output_shape) > 0
                and isinstance(output_shape[0], (list, tuple))
                else output_shape
            )
            output_idx += 1

            # A backend's answer may be a name already, and may be a name with
            # no type or no shape of its own; filling those in here means the
            # rest of the kernel can rely on them being there.
            if isinstance(v, CSEVariable):
                if backend == "cpp" and v.dtype is None:
                    v.dtype = var_dtype
                if v.shape is None:
                    v.shape = var_shape

            csevar = V.kernel.cse.generate(
                V.kernel.compute,
                v,
                bounds=bounds,
                dtype=output_dtype,
                shape=output_shape,
            )

            csevar.update_on_args(name, args, kwargs)

            if (
                config.test_configs.runtime_triton_dtype_assert
                or config.test_configs.static_cpp_dtype_assert
            ) and var_dtype is not None:
                check_dtype(V.kernel.compute, csevar, var_dtype)

            if config.test_configs.runtime_triton_shape_assert:
                shape_to_check = csevar.shape if csevar.shape is not None else var_shape
                if shape_to_check is not None:
                    check_shape(V.kernel.compute, csevar, shape_to_check)

            if config.runtime_triton_nan_asserts:
                check_nan(V.kernel.compute, csevar)

            return csevar

        from tensorplay.utils._pytree import tree_map

        result = tree_map(do_cse, value)
        self.kernel.record_op_trace(name, args, kwargs, result)
        return result

    def _bound_variable(self, name, *args, **kwargs):
        """The range of values a name produced by this operation can take.

        A bound that has already been computed for this node is used as it is;
        one that has not is computed, but only where an analysis is installed
        and only for the operations it knows, because computing a bound for an
        operation nobody asked about is work whose result is thrown away.
        """

        if isinstance(V.kernel, (NullKernel,)):
            return ValueRanges.unknown()

        if V.interpreter is None or getattr(V.interpreter, "current_node", None) is None:
            return ValueRanges.unknown()

        fx_node = V.interpreter.current_node
        if fx_node.target == name and self.kernel.node_to_bounds is not None:
            if not isinstance(self.kernel.node_to_bounds, dict):
                raise AssertionError(type(self.kernel.node_to_bounds))
            return self.kernel.node_to_bounds.get(fx_node, ValueRanges.unknown())
        elif config.compute_all_bounds and self.vr_analysis is not None and hasattr(
            self.vr_analysis, name
        ):
            # These produce a great many intermediate strings, and a bound for
            # them is of little use, so they are left unbounded.
            if any(s in fx_node.target for s in ("set_indirect", "reduction", "scan")):
                return ValueRanges.unknown()

            if kwargs:
                raise AssertionError("expected no kwargs")

            def arg_to_bound(x):
                if isinstance(x, CSEVariable):
                    return x.bounds
                return x

            arg_bounds = list(map(arg_to_bound, args))
            return getattr(self.vr_analysis, name)(*arg_bounds)
        return ValueRanges.unknown()

    def indirect_indexing(self, var, size, check: bool = True, wrap_neg: bool = True):
        """Turn an index that may be negative into one that is inside the buffer.

        A negative index counts from the end, which is arithmetic the generated
        code has to do explicitly; doing it here rather than in every load
        means one place decides what a negative index becomes, and one place
        knows what the resulting value can be.
        """

        if isinstance(size, int):
            size = sympy.Integer(size)
        if not isinstance(size, sympy.Expr):
            raise AssertionError((type(size), size))
        # No name is produced here, so there is nothing to look up.

        if var.bounds.lower < 0:
            if wrap_neg:
                stm = ops.add(var, ops.index_expr(size, tp.int64))
                # Where the index may be either sign, the ones that are
                # negative have been shifted and the others have not, so the
                # two are chosen between.
                if var.bounds.upper >= 0:
                    lt = ops.lt(var, 0)
                    stm = ops.where(lt, stm, var)
            else:
                stm = var

            # A bound is propagated only where it can be computed: with the
            # shift in place the negative part moves by the extent, and with
            # wrap_neg off a negative index stays negative and has to keep the
            # bound that says so.
            new_bounds = var.bounds if not wrap_neg else ValueRanges.unknown()
            if (
                wrap_neg
                and var.bounds != ValueRanges.unknown()
                and isinstance(size, sympy.Number)
            ):
                # Take the negative part of the bound, shift it by the extent,
                # and join it with the positive part.  That is tighter than
                # what a plain conditional would give, because the condition
                # says which of the two parts each value came from.
                neg_bounds = var.bounds & ValueRanges(-int_oo, -1)
                new_bounds = ValueRanges(
                    neg_bounds.lower + size, neg_bounds.upper + size
                )
                if var.bounds.upper >= 0:
                    pos = var.bounds & ValueRanges(0, int_oo)
                    new_bounds = new_bounds | pos

            var = self.kernel.cse.generate(
                self.kernel.compute,
                stm,
                bounds=new_bounds,
                dtype=var.dtype,
                shape=var.shape,
            )

        sympy_var = self.parent_handler.indirect_indexing(var, size, check)
        if generate_assert(check):
            assert_lower = not (var.bounds.lower >= 0)
            # A bound cannot compare a symbol against a symbol, so the upper
            # check is only made when the extent is a number.
            assert_upper = not isinstance(size, sympy.Number) or not (
                var.bounds.upper < size
            )
            self.kernel.check_bounds(sympy_var, size, assert_lower, assert_upper)
        return sympy_var

    def check_bounds(self, expr, size, lower: bool, upper: bool) -> None:
        return self.kernel.check_bounds(expr, size, lower, upper)

    def load(self, name: str, index):
        if name in self.kernel.cse.invalidated_stores:
            # A read of a store that was dropped still needs the buffer it was
            # dropped from, so the drop is undone for this kernel.
            V.kernel.must_keep_buffers.add(name)
        if free_symbol_is_type(index, SymT.TMP):
            return self.kernel.indirect_load(name, index)
        store_cache = self.kernel.cse.store_cache
        if name in store_cache:
            return store_cache[name]
        out = self.kernel.load(name, index)
        # Only a read that was not already in hand is counted, since a read
        # served from the store cache costs nothing at run time.
        if out.use_count == 1:
            self.kernel.num_load += 1
        self.kernel.record_op_trace("load", (name, index), {}, out)
        return out

    def _update_store_cache(self, name: str, value) -> None:
        """Remember what a buffer holds, under every name that means that buffer.

        A write to a buffer that others alias is a write to all of them, and a
        later read of any of the names has to find the value.
        """

        self.kernel.cse.store_cache[name] = value
        if self.kernel.current_node and name in V.graph.name_to_buffer:
            buf = self.kernel.current_node.get_output(name)
            for other_name in buf.get_mutations():
                self.kernel.cse.store_cache[other_name] = value

    def store(self, name: str, index, value, mode=None) -> None:
        self.kernel.store_buffer_names.add(name)
        # An accumulating write leaves the buffer holding more than was written
        # to it, so it does not go into the store cache.
        if mode != "atomic_add":
            self._update_store_cache(name, value)
        if name not in V.graph.removed_buffers:
            self.kernel.store(name, index, value, mode=mode)
            self.kernel.num_store += 1
        self.kernel.record_op_trace("store", (name, index, value, mode), {})

    def device_assert_async(self, cond, msg: str) -> None:
        self.kernel.device_assert_async(cond, msg)
        self.kernel.record_op_trace("device_assert_async", (cond, msg), {})

    def partial_accumulate(self, *args) -> None:
        self.kernel.partial_accumulate(*args)

    def store_reduction(self, name: str, index, value) -> None:
        self.kernel.store_buffer_names.add(name)
        self._update_store_cache(name, value)

        if name not in V.graph.removed_buffers:
            self.kernel.num_store += 1
            return self.kernel.store_reduction(name, index, value)

    def reduction(self, dtype, src_dtype, reduction_type, value):
        self.kernel.num_reduction += 1
        return self.kernel.reduction(dtype, src_dtype, reduction_type, value)

    def scan(self, dtypes, combine_fn, values):
        return self.kernel.scan(dtypes, combine_fn, values)

    def sort(self, dtypes, values, stable: bool, descending: bool):
        return self.kernel.sort(dtypes, values, stable, descending)

    def bucketize(
        self,
        values,
        boundaries,
        boundary_indices,
        indexing_dtype,
        right: bool,
        sorter=None,
        sorter_indices=None,
    ):
        """The bucket each value falls in, among a set of boundaries.

        The boundaries arrive described rather than given: a name, how long one
        set of them is, how long the whole buffer is, and the stride of one
        set.  The indices say which set each value is measured against.  Which
        side of a boundary belongs to which bucket is what the flag says, and
        the boundaries have to be in order unless a sort order is given, since
        an unordered set of boundaries does not divide anything.
        """

        return self.kernel.bucketize(
            values,
            boundaries,
            boundary_indices,
            indexing_dtype,
            right,
            sorter,
            sorter_indices,
        )


#: What a kernel is written in terms of: the value a common subexpression
#: is named by.  A kernel that writes for one backend narrows this, and one
#: that does not leaves it at the value every backend starts from.
CSEVariableType = TypeVar(
    "CSEVariableType", bound=CSEVariable, default=CSEVariable
)


class Kernel(CodeGen, Generic[CSEVariableType]):
    """A kernel being written, and the three places its text goes.

    A kernel is read, computed on, and written, and the three are separate
    buffers because a backend often needs them apart: a load that depends on
    something just computed has to go after it, and a store has to go after
    both.  Everything else here is the discipline around that -- the names the
    arguments are given, the expressions that were written once, the buffers
    this kernel turned out not to need.
    """

    newvar_prefix: str = ""
    suffix: str = ""
    overrides = None

    def __init__(self, args=None, increase_kernel_count: bool = True) -> None:
        super().__init__()
        if increase_kernel_count:
            metrics.generated_kernel_count += 1
        self.args = args or KernelArgs()
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()

        self.atomic_add_found = False
        self.num_load = 0
        self.num_store = 0
        self.num_reduction = 0

        self.cse: CSE = CSE(self.newvar_prefix, self.suffix)
        self.must_keep_buffers: OrderedSet = OrderedSet()
        self.store_buffer_names: OrderedSet = OrderedSet()
        self._load_mask = None
        self._load_other = None
        self.current_node = None
        self.node_to_bounds: dict | None = None

        self.removed_buffers: OrderedSet = OrderedSet()
        self.inplaced_to_remove: OrderedSet = OrderedSet()

        # Which buffer's memory a written buffer may reuse, and which buffer it
        # has to read in order to do so.
        self.inplace_update_buffers: dict = {}
        # The fewest elements one thread is asked to process, which a backend
        # raises when its threads are too wide to be given one element each.
        self.min_elem_per_thread = 1
        self.kernel_name: str | None = None

    @contextlib.contextmanager
    def set_current_node(self, node):
        """The node being written, and the bounds already computed for it."""

        prior = self.current_node
        self.current_node = node
        bounds = getattr(getattr(node, "body", None), "bounds", None)
        if callable(bounds):
            self.node_to_bounds = bounds().get_bounds()
        else:
            self.node_to_bounds = None
        try:
            yield
        finally:
            self.current_node = prior

    @contextlib.contextmanager
    def swap_buffers(self, lb, cb=None, sb=None):
        """Write into other buffers for the duration, and put them back after.

        A backend that keeps its loads in a different buffer while it works --
        while it walks a tile, for instance -- swaps them in here, and the
        common-subpression table is swapped with them so that an expression
        written into the temporary buffers is not mistaken for one that belongs
        to the kernel.
        """

        if cb is None:
            cb = lb
        if disallow_stores := sb is None:
            sb = IndentedBuffer()
        loads = self.loads
        compute = self.compute
        stores = self.stores
        cse = self.cse
        self.loads = lb
        self.compute = cb
        self.stores = sb
        self.cse = cse.scoped_copy()
        try:
            yield
        finally:
            self.loads = loads
            self.compute = compute
            self.stores = stores
            self.cse = cse
            if disallow_stores:
                if sb:
                    raise AssertionError("unexpected store inside swap_buffers")

    def emit_kernel_override(
        self,
        wrapper,
        src_code: str,
        kernel_name: str,
        node_schedule,
        kernel_path: str,
        get_kernel_metadata,
    ) -> bool:
        """Whether this kernel is written by something other than this class.

        A handler outside the compiler can take over the writing of a kernel;
        it says so by returning true here, and returning false means the
        writing goes on as usual.
        """

        return False

    def load(self, name: str, index):
        raise NotImplementedError

    def indirect_load(self, name: str, index):
        """A read whose index was itself read, so it belongs after the compute."""

        prior = self.loads
        try:
            # The read goes into the compute section, because the index it
            # reads by may be something this section has just computed.
            self.loads = self.compute
            return self.load(name, index)
        finally:
            self.loads = prior

    def store_reduction(self, name: str, index, value) -> None:
        raise NotImplementedError

    def store(self, name: str, index, value, mode=None) -> None:
        raise NotImplementedError

    def device_assert_async(self, cond, msg: str) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}: device_assert_async should be handled by CSEProxy"
        )

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError

    def partial_accumulate(self, name, reduction_type, value, extra_meta) -> None:
        raise NotImplementedError

    def scan(self, dtypes, combine_fn, values):
        raise NotImplementedError

    def sort(self, dtypes, values, stable: bool, descending: bool):
        raise NotImplementedError

    def var_ranges(self):
        raise NotImplementedError

    def bucketize(
        self,
        values,
        boundaries,
        boundary_indices,
        indexing_dtype,
        right: bool,
        sorter=None,
        sorter_indices=None,
    ):
        raise NotImplementedError

    @property
    def assert_function(self) -> str:
        raise NotImplementedError

    def indirect_assert(self, var, lower, upper, mask=None) -> str:
        """A check that an index is inside the extent it indexes.

        Both bounds are written as two comparisons rather than as the chained
        form, because the chained form means something else in the language the
        generated code is written in; the message says what a reader would have
        written, because that is what the failure means.
        """

        if isinstance(var, CSEVariable):
            var = str(var)
        if not isinstance(var, str):
            raise AssertionError(type(var))
        if not (lower is None or isinstance(lower, str)):
            raise AssertionError(f"expected lower to be None or str, got {type(lower)}")
        if not (upper is None or isinstance(upper, str)):
            raise AssertionError(f"expected upper to be None or str, got {type(upper)}")
        if lower and upper:
            cond = f"({lower} <= {var}) & ({var} < {upper})"
            cond_print = f"{lower} <= {var} < {upper}"
        elif lower:
            cond = f"{lower} <= {var}"
            cond_print = cond
        else:
            if not upper:
                raise AssertionError("expected upper to be set")
            cond = f"{var} < {upper}"
            cond_print = cond

        if mask:
            cond = f"({cond}) | ~({mask})"

        return f'{self.assert_function}({cond}, "index out of bounds: {cond_print}")'

    def check_bounds(self, expr, size, lower: bool, upper: bool) -> None:
        raise NotImplementedError

    def index_to_str(self, index) -> str:
        raise NotImplementedError

    def __enter__(self):
        super().__enter__()
        if not self.overrides:
            raise AssertionError("expected overrides to be set")
        self.exit_stack.enter_context(
            V.set_ops_handler(CSEProxy(self, self.overrides()))
        )
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_kernel_local_buffers()
        super().__exit__(exc_type, exc_val, exc_tb)

    def remove_kernel_local_buffers(self) -> None:
        """Drop the buffers this kernel wrote and read and nothing else wanted.

        A buffer written here whose last use is also here was never needed
        outside this kernel, so passing it around costs memory for nothing.  A
        buffer that was written in place is only dropped when every name that
        stood for it is being dropped too, since one surviving name is a buffer
        somebody else still has to be able to read.
        """

        scheduler = V.graph.scheduler
        if not scheduler:
            return
        fused_node_names = OrderedSet(
            scheduler.name_to_buf[buf].defining_op_name()
            for buf in self.store_buffer_names
            if buf in scheduler.name_to_buf
        )
        names_to_remove: OrderedSet = OrderedSet()
        for name in self.store_buffer_names:
            if (
                name not in self.must_keep_buffers
                and name not in self.args.input_buffers
                and scheduler.can_buffer_be_removed_through_fusion(
                    name, fused_node_names
                )
            ):
                self.num_store -= 1
                names_to_remove.add(name)

        for name in names_to_remove:
            if name in self.args.inplace_buffers:
                buf = self.args.inplace_buffers[name]
                if isinstance(buf, RemovedArg):
                    continue
                remove = all(n in names_to_remove for n in buf.other_names)
                if remove:
                    self.remove_inplace_buffer(name)
                self.inplaced_to_remove.add(name)
            else:
                self.remove_buffer(name)

    def remove_buffer(self, name: str) -> None:
        # The entry is marked rather than deleted, because the number of
        # output buffers is what the next argument's name is counted from, and
        # deleting one would hand out a name that is already in use.
        log.debug("remove_buffer(%r)", name)
        self.args.output_buffers[name] = REMOVED
        self.removed_buffers.add(name)

    def remove_inplace_buffer(self, name: str) -> None:
        log.debug("removing_inplace_buffer(%r)", name)
        self.args.inplace_buffers[name] = REMOVED
        self.removed_buffers.add(name)

    def rename_indexing(self, index):
        """Rewrite an index in terms of the names this kernel is called with.

        An index is written over the region's symbols; the kernel is handed
        extents as arguments and knows them by the names the arguments were
        given.  Every symbol of a kind the kernel was handed has to be replaced
        by that name, or the generated code would refer to something the caller
        never passes.
        """

        if isinstance(index, (list, tuple)):
            return [self.rename_indexing(x) for x in index]
        index = V.graph.sizevars.simplify(index)
        sorted_symbols = sorted(index.free_symbols, key=lambda s: s.name)
        replacements = {
            x: self.args.size(x)
            for x in sorted_symbols
            if symbol_is_type(
                x,
                (
                    SymT.UNBACKED_INT,
                    SymT.SIZE,
                    SymT.PRECOMPUTED_SIZE,
                    SymT.UNBACKED_FLOAT,
                ),
            )
        }
        return sympy_subs(index, replacements)

    def create_cse_var(self, *args, **kwargs) -> CSEVariable:
        return CSEVariable(*args, **kwargs)

    def arg_name(self, node):
        """The argument name a node was given, or none if it is not an argument."""

        if node is None:
            return None
        return self.args.arg_name(node.get_name())

    def record_op_trace(self, name, args, kwargs, result=None) -> None:
        pass


#: The type each type is computed in, which is not always the type it is held
#: in.  A half-precision value is held in half precision and computed in
#: single, because computing it in half loses digits the result is meant to
#: keep; everything else is computed in the type it is held in.
DTYPE_TO_COMPUTATION_DTYPE: dict = {
    tp.bfloat16: tp.float32,
    tp.float16: tp.float32,
    **{
        dtype: dtype
        for dtype in [
            tp.bool,
            tp.float32,
            tp.float64,
            tp.int8,
            tp.int16,
            tp.int32,
            tp.int64,
            tp.uint8,
            tp.uint16,
            tp.uint32,
            tp.uint64,
        ]
    },
}


@dataclass
class OverridesData:
    """What one operation looks like in each language, and how it promotes.

    An operation is spelled once per language that has an implementation of it,
    and a language that has none says so by leaving its entry empty rather than
    by being absent from the table: the difference between "not implemented
    here" and "not an operation" is the difference between a fallback and a
    hole.  The promotion kind is here too, because the type an operation
    produces is a property of the operation rather than of the language it is
    written in.
    """

    name: str
    cpp: Callable
    triton: Optional[Callable] = None
    cppvec: Optional[Callable] = None
    type_promotion_kind: Any = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    halide: Optional[Callable] = None
    mps: Optional[Callable] = None


def _bessel_at_infinity(order: int, kind: str, x: str) -> str:
    """A special function written so that an infinite argument gives a not-a-number.

    The mathematical limit at an infinity is a number, but the eager result is
    a not-a-number, and the code generated for a kernel has to agree with the
    eager one rather than with the mathematics.  Subtracting the argument from
    itself gives a not-a-number in the argument's own type, which is the type
    the result has to be in anyway.
    """

    return (
        f"tl.where(tl_math.abs({x}) == float('inf'), "
        f"{x} - {x}, "
        f"libdevice.{kind}{order}({x}))"
    )


def _cyl_bessel_i_at_infinity(order: int, x: str) -> str:
    """The same treatment for the modified Bessel function of the first kind."""

    return (
        f"tl.where(tl.abs({x}) == float('inf'), "
        f"{x} - {x}, "
        f"libdevice.cyl_bessel_i{order}({x}))"
    )


pointwise_overrides_data: dict = dict(
    airy_ai=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"airy_ai_forward({x})",
        name="special_airy_ai",
    ),
    bessel_j0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_j0_forward({x})",
        triton=lambda x: _bessel_at_infinity(0, "j", x),
        name="special_bessel_j0",
    ),
    bessel_j1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_j1_forward({x})",
        triton=lambda x: _bessel_at_infinity(1, "j", x),
        name="special_bessel_j1",
    ),
    bessel_y0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_y0_forward({x})",
        triton=lambda x: _bessel_at_infinity(0, "y", x),
        name="special_bessel_y0",
    ),
    bessel_y1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"bessel_y1_forward({x})",
        triton=lambda x: _bessel_at_infinity(1, "y", x),
        name="special_bessel_y1",
    ),
    digamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_digamma({x})",
        cppvec=lambda x: f"{x}.digamma()",
        name="digamma",
    ),
    # no cpp nor triton implementation for entr, it is defined as decomposition
    # erf, erfc
    erfcx=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_erfcx({x})",
        triton=lambda x: f"libdevice.erfcx({x})",
        name="special_erfcx",
    ),
    fma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y, z: f"std::fma({x}, {y}, {z})",
        cppvec=lambda x, y, z: f"fmadd({x}, {y}, {z})",
        # The fused form is the one to use here: the unfused product and sum
        # round twice, and a result that rounds twice differs from the eager
        # one by more than the operation being computed.
        triton=lambda x, y, z: f"tl.fma({x}, {y}, {z})",
        name="fma",
    ),
    # mul_rn: Multiplication with round-to-nearest. This prevents Triton's
    # compiler from fusing the multiplication with subsequent operations,
    # which is needed to match eager's rounding behavior in operations like
    # addcmul where the product must be rounded before use.
        # One device has no correctly-rounded product, so there the ordinary
        # product is used and the difference is a rounding rather than a wrong
        # answer.
    mul_rn=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        cpp=lambda x, y: f"({x}) * ({y})",  # C++ doesn't need special handling
        triton=lambda x, y: f"({x}) * ({y})"
        if tp.version.hip
        else f"libdevice.mul_rn({x}, {y})",
        name="mul_rn",
    ),
    # erfinv, exp2, expit, gammaln
    igamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igamma({x}, {y})",
        name="igamma",
    ),
    igammac=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igammac({x}, {y})",
        name="igammac",
    ),
    gammainc=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igamma({x}, {y})",
        name="special_gammainc",
    ),
    gammaincc=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"calc_igammac({x}, {y})",
        name="special_gammaincc",
    ),
    i0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i0({x})",
        triton=lambda x: _cyl_bessel_i_at_infinity(0, x),
        name="i0",
    ),
    i0e=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i0e({x})",
        name="special_i0e",
    ),
    i1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i1({x})",
        triton=lambda x: _cyl_bessel_i_at_infinity(1, x),
        name="special_i1",
    ),
    i1e=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_i1e({x})",
        name="special_i1e",
    ),
    log_ndtr=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_log_ndtr({x})",
        name="special_log_ndtr",
    ),
    # logit
    modified_bessel_i0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_i0_forward({x})",
        triton=lambda x: _cyl_bessel_i_at_infinity(0, x),
        name="special_modified_bessel_i0",
    ),
    modified_bessel_i1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_i1_forward({x})",
        triton=lambda x: _cyl_bessel_i_at_infinity(1, x),
        name="special_modified_bessel_i1",
    ),
    modified_bessel_k0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_k0_forward({x})",
        name="special_modified_bessel_k0",
    ),
    modified_bessel_k1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"modified_bessel_k1_forward({x})",
        name="special_modified_bessel_k1",
    ),
    # multigamma
    ndtr=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_ndtr({x})",
        name="special_ndtr",
    ),
    ndtri=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"calc_ndtri({x})",
        name="special_ndtri",
    ),
    polygamma=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x,
        y: f"{x} == 0 ? calc_digamma({y}) : ({x} == 1 ? trigamma({y}) : calc_polygamma({y}, {x}))",
        name="polygamma",
    ),
    # psi - alias to digamma
    # round
    scaled_modified_bessel_k0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"scaled_modified_bessel_k0_forward({x})",
        name="special_scaled_modified_bessel_k0",
    ),
    scaled_modified_bessel_k1=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"scaled_modified_bessel_k1_forward({x})",
        name="special_scaled_modified_bessel_k1",
    ),
    # sinc
    spherical_bessel_j0=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x: f"spherical_bessel_j0_forward({x})",
        name="special_spherical_bessel_j0",
    ),
    zeta=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"zeta({x}, {y})",
        name="special_zeta",
    ),
    chebyshev_polynomial_t=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_t_forward({x}, {y})",
        name="special_chebyshev_polynomial_t",
    ),
    chebyshev_polynomial_u=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_u_forward({x}, {y})",
        name="special_chebyshev_polynomial_u",
    ),
    chebyshev_polynomial_v=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_v_forward({x}, {y})",
        name="special_chebyshev_polynomial_v",
    ),
    chebyshev_polynomial_w=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"chebyshev_polynomial_w_forward({x}, {y})",
        name="special_chebyshev_polynomial_w",
    ),
    legendre_polynomial_p=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"legendre_polynomial_p_forward({x}, {y})",
        name="special_legendre_polynomial_p",
    ),
    shifted_chebyshev_polynomial_t=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_t_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_t",
    ),
    shifted_chebyshev_polynomial_u=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_u_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_u",
    ),
    shifted_chebyshev_polynomial_v=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_v_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_v",
    ),
    shifted_chebyshev_polynomial_w=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"shifted_chebyshev_polynomial_w_forward({x}, {y})",
        name="special_shifted_chebyshev_polynomial_w",
    ),
    hermite_polynomial_h=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"hermite_polynomial_h_forward({x}, {y})",
        name="special_hermite_polynomial_h",
    ),
    hermite_polynomial_he=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"hermite_polynomial_he_forward({x}, {y})",
        name="special_hermite_polynomial_he",
    ),
    laguerre_polynomial_l=OverridesData(
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        cpp=lambda x, y: f"laguerre_polynomial_l_forward({x}, {y})",
        name="special_laguerre_polynomial_l",
    ),
)


class KernelTemplate:
    """One operation, implemented by kernels chosen from a configuration space.

    A subclass says what to do with one configuration in ``generate``, and
    everything else here is the discipline around it: a refusal is caught and
    reported rather than propagated, an identity is stable across processes so
    a stored decision can be recognised, and a source digest lets a stored
    decision be thrown away when the kernel it named has changed.
    """

    @staticmethod
    def indent_except_first(
        source: str, num_indents: int, indents_spacing: int = 4
    ) -> str:
        """Indent every line but the first, so a block sits under its header.

        A rendered body that keeps its own leading indentation has to be
        shifted by whatever the surrounding text already emitted, and the
        first line is not shifted because it is the one that follows the
        header directly.
        """

        lines = source.splitlines(True)
        if len(lines) > 1:
            lines[1:] = [
                (" " * indents_spacing * num_indents) + line for line in lines[1:]
            ]
        return "".join(lines)

    @classmethod
    def _template_from_string(cls, source: str) -> Any:
        """Compile a template source, or say which line of it is wrong.

        A template that fails to compile is reported with the lines around the
        offending one and a caret under the column, because the source being
        rendered is text that was assembled somewhere else and the bare
        message does not say where in it the problem is.
        """

        env = jinja2_env()
        if env is None:
            return None
        env.filters["indent_except_first"] = KernelTemplate.indent_except_first
        from jinja2 import TemplateSyntaxError

        try:
            return env.from_string(source)
        except TemplateSyntaxError as e:

            class DetailedTemplateSyntaxError(TemplateSyntaxError):
                def __init__(self, original_error: TemplateSyntaxError) -> None:
                    super().__init__(
                        original_error.message,
                        original_error.lineno,
                        original_error.name,
                        original_error.filename,
                    )
                    self.original_error = original_error

                def __str__(self) -> str:
                    error_info = f"Error in template at line {self.lineno}\n"
                    error_info += f"Error message: {self.message}\n"
                    if hasattr(self.original_error, "source"):
                        lines = self.original_error.source.split("\n")
                        error_info += "Context:\n"
                        start = max(0, self.lineno - 2)
                        end = min(len(lines), self.lineno + 2)
                        for i in range(start, end):
                            if i == self.lineno - 1:
                                error_info += f"{i + 1}: --> {lines[i]}\n"
                                if hasattr(self.original_error, "column"):
                                    error_info += (
                                        "     "
                                        + " " * (self.original_error.column - 1)
                                        + "^\n"
                                    )
                            else:
                                error_info += f"{i + 1}:     {lines[i]}\n"
                    return error_info

            raise DetailedTemplateSyntaxError(e) from e

    @classmethod
    def _fake_get_dtype(cls, fake_outs):
        """Answer type queries from the results being faked, then from the region.

        While a template is being rendered the outputs do not exist yet, so
        the names it asks about are answered from the layouts it was handed;
        anything else still comes from the region, which does know.
        """

        from ..loops import V

        _get_dtype_real = V.graph.get_dtype
        if isinstance(fake_outs, (list, tuple)):
            lookup = {buf.get_name(): buf.get_dtype() for buf in fake_outs}
        else:
            lookup = {fake_outs.get_name(): fake_outs.get_dtype()}

        def get_dtype(name):
            result = lookup.get(name)
            if result is not None:
                return result
            return _get_dtype_real(name)

        return get_dtype

    def __init__(self, name: str, hash: str | None = None):
        self.name = name
        self._hash = hash

    @property
    def uid(self) -> str:
        """A stable identity, so a stored decision can be found again.

        Every template is unique in the system, and the identity has to
        survive a restart, so nothing about it may depend on where it was
        defined.  A template that emits a source folds a digest of it in, so
        that editing the kernel changes the identity rather than leaving a
        stored decision pointing at something that no longer exists.
        """
        return self.name

    @property
    def src_hash(self) -> str | None:
        """A digest of the source this template emits, when it has one.

        A template that emits a kernel can say what the kernel was, so a
        stored decision is not reused once the kernel it names has moved.
        """

        return self._hash

    def choice_or_none(self, **kwargs: Any) -> "ChoiceCaller | None":
        """The choice for one configuration, or ``None`` when it does not fit."""

        collected: list[ChoiceCaller] = []
        error = self.maybe_append_choice(collected, **kwargs)
        if error is None and len(collected) == 1:
            return collected[0]
        return None

    def maybe_append_choice(
        self, choices: "list[ChoiceCaller]", **kwargs: Any
    ) -> NotImplementedError | None:
        """Add the choice for one configuration, or report why there is none.

        Refusing is a normal answer and comes back as the refusal rather than
        as an exception, so a caller can offer a whole table and keep the
        ones that apply.
        """

        try:
            choices.append(self.generate(**kwargs))
        except NotImplementedError as error:
            return error
        return None

    def generate(self, **kwargs: Any) -> "ChoiceCaller":
        """The choice one configuration describes."""

        raise NotImplementedError


__all__ = [
    "BACKEND_FEATURES", "BACKEND_ORDER", "BACKEND_ORDER_BY_DEVICE",
    "BackendFeature", "BasicMathOpsMixin", "DTYPE_TO_COMPUTATION_DTYPE", "CSE", "CSEProxy", "Kernel",
    "KernelArgs",
    "KernelTemplate", "OpOverrides",
    "OperatorNotSupported", "backend_supports", "required_features",
    "select_backend", "jinja2_env",
]


def deduce_output_dtype_by_name(
    op_name: str,
    *args: Any,
    **kwargs: Any,
) -> tp.dtype | None:
    """What element type this operation produces, from its name and arguments alone.

    Most operations do not have to be looked at to know their result type: a
    comparison is a truth value whatever it was given, a conversion produces the
    type it was asked for, a load produces whatever the buffer holds, and a
    reduction produces whatever the reduction was asked to produce.  Only the
    operations whose result is not fixed this way are left to the caller to work
    out from their inputs, which is why nothing is returned for them.
    """

    if op_name in boolean_ops():
        return tp.bool
    elif op_name in (
        "to_dtype",
        "index_expr",
        "value_expr",
    ):
        return kwargs["dtype"] if "dtype" in kwargs else args[-1]
    elif op_name in (
        "rand",
        "randn",
    ):
        return tp.float
    elif op_name in (
        "get_index",
        "randint64",
        "load_seed",
    ):
        return tp.int64
    elif op_name == "reduction":
        return kwargs["dtype"] if "dtype" in kwargs else args[1]
    elif op_name == "constant":
        return kwargs["dtype"] if "dtype" in kwargs else args[-1]
    elif op_name in (
        "load",
        "store",
        "store_reduction",
    ):
        buf_name = args[1]
        return V.graph.get_dtype(buf_name)  # type: ignore[arg-type]
    elif op_name == "to_dtype_bitcast":
        return kwargs["dtype"] if "dtype" in kwargs else args[-2]
    return None


class DataTypePropagation:
    """Work out, for every operation in a body, the element type it produced.

    Narrow floating point cannot be computed on directly by most of the
    operations, so a body that contains it is computed on in the wider type and
    narrowed back at the end.  Which operations those are can only be said
    after knowing what each one produced, and an operation's result is often
    settled by what it was given -- so the types are settled in order, each read
    off the ones already settled.  Where a subgraph is reached, the subgraph is
    settled first and its result stands for the subgraph as a whole.

    A type of None means "not narrow": either the operation genuinely has no
    single element type, or what it was given is not yet known.
    """

    def __init__(self, body: "LoopBody") -> None:
        self.body = body
        self.graphs: dict[Callable[..., Any] | str, Any] = {
            "root": body.root_block.graph
        }
        for k, v in body.subblocks.items():
            self.graphs[k] = v.graph

    def deduce_node_dtype_by_inputs(self, node: Node) -> tp.dtype | None:
        """The type this operation produced, promoted from everything it was given.

        Where even one input has no settled type, nothing can be said: the
        promoted type would be missing that input's contribution.
        """

        inputs = node.all_input_nodes
        input_nodes = [
            n for n in inputs if isinstance(n, Node) and n.op != "placeholder"
        ]
        if len(input_nodes) == 0:
            return None

        all_input_nodes_propagated = all(
            OptimizationContext.key in n.meta
            and n.meta[OptimizationContext.key].dtype is not None
            for n in input_nodes
        )
        if not all_input_nodes_propagated:
            return None

        return functools.reduce(
            tp.promote_types,
            [n.meta[OptimizationContext.key].dtype for n in input_nodes],
        )

    def deduce_node_dtype_by_subgraph(self, node: Node) -> tp.dtype:
        """The type of a whole subgraph, which is whatever its output settled to."""

        sub_graph = self.graphs[node.target]
        dtype = self.propagate_graph(sub_graph)
        if not dtype:
            raise AssertionError("expected subgraph to propagate a dtype")
        return dtype

    def deduce_node_dtype(self, node: Node) -> tp.dtype | None:
        if node.op == "placeholder":
            return None

        if node.target == "output" and len(node.args) != 1:
            # An output taking more than one value is a tuple, which has no
            # single element type; one taking a single value is that value's.
            return None

        if node.target is operator.getitem:
            node_arg = node.args[0]
            if not isinstance(node_arg, Node):
                raise AssertionError(type(node_arg))
            return self.deduce_node_dtype(node_arg)

        if not isinstance(node.target, str):
            raise AssertionError(type(node.target))

        if node.target.startswith("masked_subblock"):
            return self.deduce_node_dtype_by_subgraph(node)

        if (
            output_dtype := deduce_output_dtype_by_name(
                node.target,
                *node.args,
                **node.kwargs,
            )
        ) is not None:
            return output_dtype

        return self.deduce_node_dtype_by_inputs(node)

    def propagate_graph(self, graph: Graph) -> tp.dtype | None:
        if not graph.nodes:
            raise AssertionError("expected graph to have nodes")
        graph_dtype: tp.dtype | None = None
        # For masked_subblock, the output's dtype stands for the dtype of the
        # subgraph as a whole.  For anything else there may be no single type,
        # and graph_dtype stays None.
        for node in graph.nodes:
            if OptimizationContext.key in node.meta:
                opt_ctx = node.meta[OptimizationContext.key]
            else:
                opt_ctx = OptimizationContext()

            opt_ctx.dtype = self.deduce_node_dtype(node)
            node.meta[OptimizationContext.key] = opt_ctx
            if node.target == "output":
                graph_dtype = opt_ctx.dtype
        return graph_dtype

    def propagate(self) -> tp.dtype | None:
        return self.propagate_graph(self.graphs["root"])

    @classmethod
    def propagate_loopbody(cls, body: "LoopBody") -> tp.dtype | None:
        return cls(body).propagate()

    @classmethod
    def propagate_scheduler_node(cls, node: "SchedulerNode") -> tp.dtype | None:
        from ..scheduler import SchedulerNode
        from ..loop_body import LoopBody

        if not isinstance(node, SchedulerNode):
            raise AssertionError(type(node))
        if not isinstance(node._body, LoopBody):
            raise AssertionError(type(node._body))
        return DataTypePropagation.propagate_loopbody(node._body)


class TMADescriptorArg:
    name: str
    api_type: str  # "experimental" or "stable"
    block_shape: list[sympy.Expr] | None  # only needed for "stable"
    dtype: tp.dtype | None  # only needed for "stable"


#: What a launch may be handed.  Every case is a value the launch's own
#: signature has a slot for; a launch that wanted anything else would have to
#: be handed a buffer for it instead.
KernelArgType = (
    WorkspaceArg | TensorArg | SizeArg | TMADescriptorArg | ConstexprArg
)


class DeferredLine(DeferredLineBase):
    """A line that can be taken back out of the code it was written into.

    A write to a buffer that the region turned out not to need is decided after
    the line naming it was written, so the line is written with the name of the
    buffer it writes and removed if that buffer is dropped.  A line that is not
    about a named buffer cannot be removed and says so here.
    """

    def __init__(self, name: str, line: str):
        super().__init__(line)
        self.name = name
        if isinstance(line, DeferredLineBase):
            raise AssertionError("line must not be a DeferredLineBase")

    def __call__(self) -> str | None:
        if not is_buffer_removed(self.name):
            return self.line
        return None

    def _new_line(self, line: str) -> "DeferredLine":
        return DeferredLine(self.name, line)

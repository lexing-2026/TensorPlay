from __future__ import annotations

"""Hints about how a kernel is to be shaped, asked for rather than measured.

These are the choices a kernel could make many ways -- how wide a tile is,
which loop a reduction is carried out over -- written down so that two kernels
that should be shaped alike are.  They are hints and not measurements: nothing
about what is computed rests on them.
"""

import enum
import functools
import typing


class ReductionHint(enum.Enum):
    """Which loop a reduction is to be carried out over, when that is a choice.

    A reduction can be done inside one iteration or spread over several and
    combined afterwards, and which is better depends on how much there is of
    each: a short reduction is cheapest done whole, a long one is cheapest
    spread out.  ``INNER`` is the first and ``DEFAULT`` is to let the extents
    decide.
    """

    INNER = 0
    OUTER = 1
    OUTER_TINY = 1
    DEFAULT = 3


class TileHint(enum.Enum):
    """Whether a tile is to be square, where that is a choice."""

    SQUARE = 0
    DEFAULT = 1


class DeviceProperties(typing.NamedTuple):
    """Copy device properties into a data structure not requiring torch to be imported"""

    type: str  # type: ignore[assignment]
    index: int  # type: ignore[assignment]
    multi_processor_count: int
    cc: int
    major: int | None = None
    regs_per_multiprocessor: int | None = None
    max_threads_per_multi_processor: int | None = None
    max_threads_per_block: int | None = None
    warp_size: int | None = None

    @property
    def warp_size_or_default(self) -> int:
        if self.warp_size is not None:
            return self.warp_size
        if self.type in ("cuda", "hip"):
            raise RuntimeError(f"{self.type} device properties must report warp_size")
        return 32

    @classmethod
    @functools.cache
    def create(cls, device) -> DeviceProperties:
        import torch
        from torch._dynamo.device_interface import get_interface_for_device

        device_type = device.type

        if torch.version.hip and device_type == "cuda":
            device_type = "hip"

        device_interface = get_interface_for_device(device)
        props = device_interface.get_device_properties(device)
        try:
            multi_processor_count = props.multi_processor_count
        except AttributeError:
            if device_type == "xpu":
                multi_processor_count = props.gpu_subslice_count
            elif device_type == "mtia":
                multi_processor_count = 64
            else:
                raise
        return cls(
            type=device_type,
            index=device.index,
            multi_processor_count=multi_processor_count,
            cc=device_interface.get_compute_capability(device),
            major=getattr(props, "major", None),
            regs_per_multiprocessor=getattr(props, "regs_per_multiprocessor", None),
            max_threads_per_multi_processor=getattr(
                props, "max_threads_per_multi_processor", None
            ),
            max_threads_per_block=getattr(props, "max_threads_per_block", 1024),
            warp_size=getattr(props, "warp_size", None),
        )


class TritonMeta(typing.TypedDict, total=False):
    """Metadata bag threaded from Triton codegen into the runtime launcher.

    total=False because the key set is populated incrementally across codegen
    (signature/device/constants/configs first, then backend/ROCm/tlx extras)
    and the whole bag is forwarded verbatim to external Triton APIs, which
    tolerate and ignore keys they do not recognize. `device` is typed as the
    codegen-time DeviceProperties; CachingAutotuner rewrites it to the integer
    device index before reaching Triton, and the runtime read sites that expect
    that int narrow it with an explicit cast.
    """

    signature: dict[str, typing.Any]
    device: DeviceProperties
    device_type: str
    constants: dict[str, typing.Any]
    configs: list[typing.Any]
    native_matmul: bool
    launch_cooperative_grid: bool
    enable_fp_fusion: bool
    launch_pdl: bool
    disable_ftz: bool
    matrix_instr_nonkdim: int
    waves_per_eu: int
    kpack: int
    restore_value: tuple[str, ...]
    reset_to_zero: tuple[str, ...]
    backend_options: dict[str, typing.Any]


# ---------------------------------------------------------------------------
# 把「哪些参数有哪种性质」写成发射器能读的形式
# ---------------------------------------------------------------------------
#
# 三种写法都把同一件事说出来：某个参数是 16 的倍数、某个恒为 1、某个指针
# 只碰 4G 以内。区别只在它们放在参数列表里的哪一层。装着的发射器认哪一种，
# 决定了这里该返回哪一种；返回另一种，参数的性质就会被读成没有。

try:
    import triton
    import triton.backends.compiler
    import triton.compiler.compiler

    if hasattr(triton.backends.compiler, "AttrsDescriptor"):
        from triton.backends.compiler import AttrsDescriptor

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
            pointer_range_32=None,
        ):
            """每一层的性质，逐个参数写出来。"""

            d = AttrsDescriptor()
            if divisible_by_16 is not None:
                d.divisibility_16 = divisible_by_16
            if equal_to_1 is not None:
                d.equal_to_1 = equal_to_1
            if pointer_range_32 is not None:
                d.pointer_range_32 = pointer_range_32
            return d

    elif hasattr(triton.compiler.compiler, "AttrsDescriptor"):
        from triton.compiler.compiler import AttrsDescriptor

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
            pointer_range_32=None,
        ):
            """同一件事，写成属性字典。"""

            kwargs = {
                "tt.divisibility": divisible_by_16,
                "tt.equal_to": equal_to_1,
                "tt.pointer_range": pointer_range_32,
            }
            return AttrsDescriptor(tuple(kwargs.items()))

    else:

        def AttrsDescriptorWrapper(
            divisible_by_16=None,
            equal_to_1=None,
            pointer_range_32=None,
        ):
            """同一件事，写成按参数下标归拢的属性表。

            一个参数可以同时有两种性质，所以两种是分开归拢再并到一起的，
            而不是各写一张表。
            """

            result = {(x,): [["tt.divisibility", 16]] for x in (divisible_by_16 or ())}
            for x in pointer_range_32 or ():
                key = (x,)
                if key in result:
                    result[key].append(["tt.pointer_range", 32])
                else:
                    result[key] = [["tt.pointer_range", 32]]
            return result

except ImportError:

    class AttrsDescriptorWrapper:
        """没有发射器可问时，性质就只留在调用它的地方，不外传。

        返回一个空属性表，而不是报错：调用方问的是参数有什么性质，答案是
        「没有可用的性质」——这与「性质为空」在发射器那里是同一件事。
        """

        def __init__(
            self,
            divisible_by_16=None,
            equal_to_1=None,
            pointer_range_32=None,
        ):
            self.divisible_by_16 = divisible_by_16
            self.equal_to_1 = equal_to_1
            self.pointer_range_32 = pointer_range_32

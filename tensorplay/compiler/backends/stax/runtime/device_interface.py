"""One way of asking a device the same questions, whatever kind of device it is.

Every kind of device does the same handful of things -- run on the device it is
currently on, wait for it, hand back the stream it is using, say what it can do
-- and spells each of them under its own name.  Code that needs one of these
does not care which kind of device it is on: a measurement, a launch and a
question about a device all want the same answers and would otherwise each have
to know every spelling.  So the questions are asked of this, and what is behind
it is the module for the kind of device in hand.

Two of the answers are not a spelling but a shape.  Which device is current is a
question every kind answers, but making it current is a matter of exchanging it
with what was current and getting the old one back, and a device with no index
at all has nothing to exchange.  And the stream a device is on is a number a
launch is given, which is not the same thing as the stream object the device
hands out and is cheaper to get.
"""

from __future__ import annotations

from typing import Any

import tensorplay as tp


class DeviceInterface:
    """The questions that can be asked of a device, asked of this kind of one.

    What a particular kind of device can answer is what the subclass says; what
    is asked here is what is asked of any of them.  A question a kind of device
    has no answer to is not quietly answered differently -- it is not answered,
    and saying so is better than a value that means something else.
    """

    @staticmethod
    def is_available() -> bool:
        raise NotImplementedError

    @staticmethod
    def current_device() -> int:
        raise NotImplementedError

    @staticmethod
    def get_raw_stream(device_idx: Any) -> int:
        raise NotImplementedError

    @staticmethod
    def exchange_device(device_idx: int) -> int:
        raise NotImplementedError

    @staticmethod
    def maybe_exchange_device(device_idx: int) -> int:
        raise NotImplementedError

    @staticmethod
    def get_compute_capability(device: Any = None) -> Any:
        raise NotImplementedError

    @staticmethod
    def synchronize(device: Any = None) -> None:
        raise NotImplementedError

    @classmethod
    def get_device_properties(cls, device: Any = None) -> Any:
        raise NotImplementedError


class CudaInterface(DeviceInterface):
    """A device that is a processor with a device index and a stream.

    The module for the kind of device answers nearly everything, so this is
    mostly a name for it.  What it does not answer is making one device current
    in place of another -- which is an exchange, and the device that was current
    comes back so it can be made current again afterwards -- and the number of
    the stream a launch is given.
    """

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        # Everything not answered here is the module's own business, and a
        # module is a module: this is a question about how it is asked, not
        # about what it can do.
        return getattr(self._module, name)

    @staticmethod
    def is_available() -> bool:
        return tp.cuda.is_available()

    @staticmethod
    def current_device() -> int:
        return tp.cuda.current_device()

    @staticmethod
    def get_raw_stream(device_idx: Any) -> int:
        return tp._C._cuda_getCurrentRawStream(device_idx)

    @staticmethod
    def exchange_device(device_idx: int) -> int:
        return tp.cuda._exchange_device(device_idx)

    @staticmethod
    def maybe_exchange_device(device_idx: int) -> int:
        return tp.cuda._maybe_exchange_device(device_idx)

    @staticmethod
    def get_compute_capability(device: Any = None) -> Any:
        # A device says what it is as two numbers, and what a kernel is compiled
        # for is one: the two are written as one number here so that a caller
        # asking what a device is gets something it can hand on rather than
        # something it has to take apart first.  Where the two numbers are not
        # the answer at all -- a device named by what it is built for rather than
        # by numbers -- the name is the answer.
        if not tp.version.hip:
            major, minor = tp.cuda.get_device_capability(device)
            return major * 10 + minor
        return tp.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]

    @staticmethod
    def synchronize(device: Any = None) -> None:
        tp.cuda.synchronize(device)

    @classmethod
    def get_device_properties(cls, device: Any = None) -> Any:
        return tp.cuda.get_device_properties(device)


class XpuInterface(CudaInterface):
    """The same shape of device, spelled the other way."""

    @staticmethod
    def is_available() -> bool:
        return tp.xpu.is_available()

    @staticmethod
    def current_device() -> int:
        return tp.xpu.current_device()

    @staticmethod
    def get_raw_stream(device_idx: Any) -> int:
        return tp._C._xpu_getCurrentRawStream(device_idx)

    @staticmethod
    def exchange_device(device_idx: int) -> int:
        return tp.xpu._exchange_device(device_idx)

    @staticmethod
    def maybe_exchange_device(device_idx: int) -> int:
        return tp.xpu._maybe_exchange_device(device_idx)

    @staticmethod
    def get_compute_capability(device: Any = None) -> str:
        return tp.xpu.get_device_capability(device)

    @staticmethod
    def synchronize(device: Any = None) -> None:
        tp.xpu.synchronize(device)

    @classmethod
    def get_device_properties(cls, device: Any = None) -> Any:
        return tp.xpu.get_device_properties(device)


class MtiaInterface(DeviceInterface):
    """A device that is a processor with a device index and a stream."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    @staticmethod
    def is_available() -> bool:
        return tp.mtia.is_available()

    @staticmethod
    def current_device() -> int:
        return tp.mtia.current_device()

    @staticmethod
    def get_raw_stream(device_idx: Any) -> int:
        return tp._C._mtia_getCurrentRawStream(device_idx)

    @staticmethod
    def exchange_device(device_idx: int) -> int:
        return tp.mtia._exchange_device(device_idx)

    @staticmethod
    def maybe_exchange_device(device_idx: int) -> int:
        return tp.mtia._maybe_exchange_device(device_idx)

    @staticmethod
    def get_compute_capability(device: Any = None) -> Any:
        return ""

    @staticmethod
    def synchronize(device: Any = None) -> None:
        tp.mtia.synchronize(device)

    @classmethod
    def get_device_properties(cls, device: Any = None) -> Any:
        return tp.mtia.get_device_properties(device)


class CpuInterface(DeviceInterface):
    """A device with no index and no stream, so nothing to exchange and nowhere to run.

    What a launch would be handed for a stream is the same for every such
    device, which is a number that means no particular stream -- so a launch
    here is a launch and not a wait on anything.
    """

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return -1

    @staticmethod
    def get_raw_stream(device_idx: Any) -> int:
        return 0

    @staticmethod
    def exchange_device(device_idx: int) -> int:
        return -1

    @staticmethod
    def maybe_exchange_device(device_idx: int) -> int:
        return -1

    @staticmethod
    def get_compute_capability(device: Any = None) -> Any:
        return ""

    @staticmethod
    def synchronize(device: Any = None) -> None:
        pass

    @classmethod
    def get_device_properties(cls, device: Any = None) -> Any:
        return cls._module.get_device_properties(device)


#: Which of these answers for which kind of device.  A kind that is not named
#: here has no interface of its own, and asking for one says so rather than
#: handing back something that would answer some of the questions.
INTERFACES: dict[str, type[DeviceInterface]] = {
    "cuda": CudaInterface,
    "hip": CudaInterface,
    "xpu": XpuInterface,
    "mtia": MtiaInterface,
    "cpu": CpuInterface,
}


def get_device_interface(device_type: str) -> DeviceInterface:
    """The interface for this kind of device, over the module that talks to it.

    The module is what actually does it; the interface is what the questions are
    called and which of them this kind of device answers itself.
    """

    device_type = device_type.replace("hip", "cuda")
    cls = INTERFACES.get(device_type)
    if cls is None:
        raise NotImplementedError(f"there is no device interface for {device_type}")
    return cls(tp.get_device_module(device_type))

from typing import TypeAlias

import tensorplay as tp
from tensorplay import Tensor
from tensorplay.autograd.grad_mode import no_grad


def _get_foreach_kernels_supported_devices() -> list[str]:
    r"""Return the device type list that supports foreach kernels."""
    return ["cuda"]


def _get_fused_kernels_supported_devices() -> list[str]:
    r"""Return the device type list that supports fused kernels in optimizer."""
    return ["cuda", "cpu"]


TensorListList: TypeAlias = list[list[Tensor | None]]
Indices: TypeAlias = list[int]
_foreach_supported_types = [tp.Tensor]


# This util function splits tensors into groups by device and dtype, which is
# useful before sending tensors off to a foreach implementation, which requires
# tensors to be on one device and dtype. The first tensor list is required to
# contain a defined tensor at every index; other lists may be empty or contain
# None at an index.
@no_grad()
def _group_tensors_by_device_and_dtype(
    tensorlistlist: TensorListList,
    with_indices: bool = False,
) -> dict[tuple[tp.Device, tp.dtype], tuple[TensorListList, Indices]]:
    # the Python implementation below as an ABI/backward-compatible fallback
    # for an extension built before the native helper was introduced.
    native_group = getattr(getattr(tp, "_C", None),
                           "_group_tensors_by_device_and_dtype", None)
    if native_group is not None:
        return native_group(tensorlistlist, with_indices)

    if not tensorlistlist or not tensorlistlist[0]:
        raise RuntimeError(
            "Expected the first nested tensor list to be non-empty"
        )

    num_tensors = len(tensorlistlist[0])
    for tensorlist in tensorlistlist[1:]:
        if len(tensorlist) not in (0, num_tensors):
            raise RuntimeError(
                "Expected every nested tensor list to have the same length "
                "as the first list or to be empty"
            )

    grouped: dict[tuple[tp.Device, tp.dtype], tuple[TensorListList, Indices]] = {}
    for index in range(num_tensors):
        first_tensor = tensorlistlist[0][index]
        if first_tensor is None:
            raise RuntimeError(
                "Tensors of the first list of nested Tensor lists are "
                "supposed to be defined"
            )

        key = (first_tensor.device, first_tensor.dtype)
        if key not in grouped:
            grouped[key] = ([[] for _ in tensorlistlist], [])

        grouped_lists, indices = grouped[key]
        for list_index, tensorlist in enumerate(tensorlistlist):
            if tensorlist:
                grouped_lists[list_index].append(tensorlist[index])
        if with_indices:
            indices.append(index)

    return grouped


def _device_has_foreach_support(device: tp.Device) -> bool:
    return device.type in (_get_foreach_kernels_supported_devices() + ["cpu"])


def _has_foreach_support(tensors: list[Tensor], device: tp.Device) -> bool:
    return _device_has_foreach_support(device) and all(
        tensor is None or type(tensor) in _foreach_supported_types
        for tensor in tensors
    )

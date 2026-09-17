from .core import MaskedTensor


__all__ = [
    "as_masked_tensor",
    "masked_tensor",
]


# Two factory helpers with distinct construction guarantees:
#     masked_tensor - builds the value directly from the given data and mask
#     as_masked_tensor - constructor that can participate in autograd by
#         keeping the construction differentiable with respect to data


def masked_tensor(
    data: object, mask: object, requires_grad: bool = False
) -> MaskedTensor:
    return MaskedTensor(data, mask, requires_grad)


def as_masked_tensor(data: object, mask: object) -> MaskedTensor:
    return MaskedTensor._from_values(data, mask)

from .. import _C

def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    This function checks if all input and other are close to each other.
    input: Tensor
    other: Tensor
    rtol: float
    atol: float
    equal_nan: bool (not supported yet)
    """
    # Ensure tensors are on the same device for comparison
    # If devices differ, move other to input's device
    if input.device != other.device:
        other = other.to(input.device)
        
    diff = (input - other).abs()
    # We need to handle element-wise comparison.
    # Note: <= might return a Bool Tensor. .all() reduces it.
    return (diff <= (atol + rtol * other.abs())).all().item()

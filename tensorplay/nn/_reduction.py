# NB: Keep this file in sync with enums in p10/src/Reduction.h


def get_enum(reduction: str) -> int:
    """Converts a string-based reduction mode to its corresponding integer enumeration value.

    This function maps standard reduction string identifiers (e.g., "none", "mean", "sum")
    to integer constants for internal enumeration use.

    Args:
        reduction: A string specifying the reduction mode. Valid options are:
            - "none": No reduction is applied (returns raw values)
            - "mean": Reduce by computing the arithmetic mean of the values
            - "sum": Reduce by computing the sum of the values

    Returns:
        Integer enumeration value corresponding to the reduction mode:
            - 0 for "none"
            - 1 for "mean"
            - 2 for "sum"
    """
    if reduction == "none":
        ret = 0
    elif reduction == "mean":
        ret = 1
    elif reduction == "sum":
        ret = 2
    else:
        raise ValueError(f"{reduction} is not a valid value for reduction")
    return ret
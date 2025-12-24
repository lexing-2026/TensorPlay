import tensorplay


def show():
    """
    Return a human-readable string with descriptions of the
    configuration of TensorPlay.
    """
    return tensorplay._C._show_config()


def get_build_info():
    """
    Return a dictionary containing detailed build information.
    """
    return tensorplay._C._get_build_info()


def _cxx_flags() -> str:
    """Returns the CXX_FLAGS used when building TensorPlay."""
    return tensorplay._C._cxx_flags()


def parallel_info() -> str:
    r"""Returns detailed string with parallelization settings"""
    return tensorplay._C._parallel_info()
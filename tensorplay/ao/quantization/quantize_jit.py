"""JIT-based quantization entry points.

The scripting runtime is not part of this project, so these APIs exist for
interface compatibility and raise ``NotImplementedError`` when invoked.
"""

__all__ = [
    "script_qconfig",
    "script_qconfig_dict",
    "fuse_conv_bn_jit",
    "prepare_jit",
    "prepare_dynamic_jit",
    "convert_jit",
    "convert_dynamic_jit",
    "quantize_jit",
    "quantize_dynamic_jit",
]


def _no_jit(*args, **kwargs):
    raise NotImplementedError(
        "JIT quantization requires a scripting runtime, which is not available"
    )


script_qconfig = _no_jit
script_qconfig_dict = _no_jit
fuse_conv_bn_jit = _no_jit
prepare_jit = _no_jit
prepare_dynamic_jit = _no_jit
convert_jit = _no_jit
convert_dynamic_jit = _no_jit
quantize_jit = _no_jit
quantize_dynamic_jit = _no_jit
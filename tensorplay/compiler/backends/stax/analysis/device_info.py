"""What each device can do, as the numbers a schedule has to be judged against.

Two numbers matter for deciding an order: how much arithmetic the device
does in a second, and how much data it moves in a second.  A region that
reads more than it computes is bound by the second and one that computes
more than it reads by the first, and an order chosen without asking which
is which will move the wrong thing.

The numbers are the vendor's, for the devices named here, and they are a
property of the hardware rather than of anything decided here.  A device
not named is looked up and not found, which is a different answer from a
device that is slow: the caller falls back to an estimate rather than
refusing.
"""

import logging
from dataclasses import dataclass

import tensorplay as tp


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """
    Theoretical numbers from data sheet.  When a data sheet reports both
    Tensor/Matrix-Core and non-Tensor-Core numbers, the higher (Tensor Core)
    number is used.

    NVIDIA data sheets since Hopper (H100) only publish Tensor-Core TFLOPS
    with 2:4 structured sparsity (marked ``*With sparsity``).  For devices
    whose ``tops`` include sparsity, set ``tops_sparsity_factor`` to 2 so
    that callers can recover the dense (non-sparse) throughput used by
    cuBLAS / cuDNN in normal (non-sparse) workloads.

    Bandwidth numbers are tricky, because there are platform differences
    that may not show up in the profiler trace.
    """

    tops: dict[tp.dtype | str, float]
    dram_bw_gbs: float
    dram_gb: float
    tops_sparsity_factor: int = 1


# Indexing is based on the name the device reports for itself, normalized to upper-case.
# TODO investigate profiler support for tf32 and allow device to report correct number when it's turned on.
_device_mapping: dict[str, DeviceInfo] = {
    # Source: NVIDIA Blackwell datasheet, "Individual Blackwell GPU Specifications",
    # HGX B200 column. Tensor Core rows there are SPARSE; dense is 1/2. FP32/FP64 are
    # already dense. Values below are all DENSE, so no sparsity factor.
    # @lint-ignore https://www.nvidia.com/en-us/data-center/hgx/
    "NVIDIA B200": DeviceInfo(
        tops={
            tp.float64: 37.0,
            tp.float32: 75.0,
            "tp.tf32": 1125.0,
            tp.bfloat16: 2250.0,
            tp.float16: 2250.0,
            tp.float8_e4m3fn: 4500.0,
            tp.float8_e4m3fnuz: 4500.0,
            tp.float8_e5m2: 4500.0,
            tp.float8_e5m2fnuz: 4500.0,
            tp.float8_e8m0fnu: 4500.0,
            tp.int8: 4500.0,
        },
        dram_bw_gbs=7700.0,
        dram_gb=180.0,
    ),
    # Source:
    # @lint-ignore https://www.nvidia.com/en-us/data-center/h100/
    # Tensor Core values are *with sparsity* per the datasheet.
    "NVIDIA H100": DeviceInfo(
        tops={
            tp.float64: 67.0,
            tp.float32: 67.0,
            "tp.tf32": 989.0,
            tp.bfloat16: 1979.0,
            tp.float16: 1979.0,
            tp.float8_e8m0fnu: 3958.0,
            tp.float8_e8m0fnu: 3958.0,
            tp.float8_e4m3fnuz: 3958.0,
            tp.float8_e5m2: 3958.0,
            tp.float8_e5m2fnuz: 3958.0,
            tp.float8_e8m0fnu: 3958.0,
            tp.int8: 3958.0,
        },
        dram_bw_gbs=3350,
        dram_gb=80,
        tops_sparsity_factor=2,
    ),
    # Source:
    # @lint-ignore https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/
    # nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    "NVIDIA A100": DeviceInfo(
        tops={
            tp.float64: 19.5,
            tp.float32: 19.5,
            tp.bfloat16: 312.5,
            tp.float16: 312.5,
            # Not in datasheet: float8
            tp.int8: 624.0,
            "tp.tf32": 156.0,
        },
        dram_bw_gbs=2039.0,
        dram_gb=80.0,
    ),
    # Source:
    # @lint-ignore https://resources.nvidia.com/en-us-gpu-resources/l4-tensor-datasheet
    "NVIDIA L4": DeviceInfo(
        tops={
            # This is a guess, not in datasheet
            tp.float64: 15.1,
            tp.float32: 30.3,
            "tp.tf32": 120.0,
            tp.bfloat16: 242.0,
            tp.float16: 242.0,
            tp.float8_e8m0fnu: 485.0,
            tp.float8_e8m0fnu: 485.0,
            tp.float8_e4m3fnuz: 485.0,
            tp.float8_e5m2: 485.0,
            tp.float8_e5m2fnuz: 485.0,
            tp.float8_e8m0fnu: 485.0,
            tp.int8: 485.0,
        },
        dram_bw_gbs=3350,
        dram_gb=24,
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/en/documents\
    # /instinct-tech-docs/product-briefs/amd-instinct-mi350x-gpu-brochure.pdf
    "AMD MI350X": DeviceInfo(
        tops={
            tp.float64: 72.1,
            tp.float32: 144.2,
            # not specified, fall back to float32 numbers
            "tp.tf32": 144.2,
            tp.bfloat16: 2309.6,
            tp.float16: 2309.6,
            tp.float8_e8m0fnu: 4614.0,
            tp.float8_e8m0fnu: 4614.0,
            tp.float8_e4m3fnuz: 4614.0,
            tp.float8_e5m2: 4614.0,
            tp.float8_e5m2fnuz: 4614.0,
            tp.float8_e8m0fnu: 4614.0,
            tp.int8: 4614.0,
        },
        dram_bw_gbs=8000.0,
        dram_gb=288.0,
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/en/documents\
    # /instinct-tech-docs/data-sheets/amd-instinct-mi300a-data-sheet.pdf
    "AMD MI300A": DeviceInfo(
        tops={
            tp.float64: 122.6,
            tp.float32: 122.6,
            "tp.tf32": 490.3,
            tp.bfloat16: 980.6,
            tp.float16: 980.6,
            tp.float8_e8m0fnu: 1961.2,
            tp.float8_e8m0fnu: 1961.2,
            tp.float8_e4m3fnuz: 1961.2,
            tp.float8_e5m2: 1961.2,
            tp.float8_e5m2fnuz: 1961.2,
            tp.float8_e8m0fnu: 1961.2,
            tp.int8: 1961.2,
        },
        dram_bw_gbs=5300.0,
        dram_gb=128.0,
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/en/documents/\
    # instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
    "AMD MI300X": DeviceInfo(
        tops={
            tp.float64: 163.4,
            tp.float32: 163.4,
            "tp.tf32": 653.7,
            tp.bfloat16: 1307.4,
            tp.float16: 1307.4,
            tp.float8_e8m0fnu: 2614.9,
            tp.float8_e8m0fnu: 2614.9,
            tp.float8_e4m3fnuz: 2614.9,
            tp.float8_e5m2: 2614.9,
            tp.float8_e5m2fnuz: 2614.9,
            tp.float8_e8m0fnu: 2614.9,
            tp.int8: 2614.9,
        },
        dram_bw_gbs=5300.0,
        dram_gb=192.0,
    ),
    # Source:
    # @lint-ignore https://www.amd.com/content/dam/amd/\
    # en/documents/instinct-business-docs/product-briefs/instinct-mi210-brochure.pdf
    "AMD MI210X": DeviceInfo(
        tops={
            tp.float64: 45.3,
            tp.float32: 45.3,
            # not specified, fall back to float32 numbers
            "tp.tf32": 45.3,
            tp.bfloat16: 181.0,
            tp.float16: 181.0,
            # not specified, fall back to float16 numbers
            tp.float8_e8m0fnu: 181.0,
            tp.float8_e8m0fnu: 181.0,
            tp.float8_e4m3fnuz: 181.0,
            tp.float8_e5m2: 181.0,
            tp.float8_e5m2fnuz: 181.0,
            tp.float8_e8m0fnu: 181.0,
            tp.int8: 181.0,
        },
        # pcie4.0x16
        dram_bw_gbs=1600.0,
        dram_gb=64.0,
    ),
    # Source:
    # @lint-ignore https://www.intel.com/content/www/us/en/products/sku/241598/
    # intel-arc-b580-graphics/specifications.html
    "INTEL B580": DeviceInfo(
        tops={
            # Estimated from published single-precision throughput.
            tp.float64: 6.83,
            tp.float32: 13.67,
            "tp.tf32": 116.5,
            tp.bfloat16: 116.5,
            tp.float16: 116.5,
            # not specified, fall back to fp16 matrix throughput
            tp.float8_e8m0fnu: 116.5,
            tp.float8_e4m3fnuz: 116.5,
            tp.float8_e5m2: 116.5,
            tp.float8_e5m2fnuz: 116.5,
            tp.int8: 233,
        },
        dram_bw_gbs=456.0,
        dram_gb=12.0,
    ),
    # Source:
    # @lint-ignore https://www.intel.com/content/www/us/en/products/sku/245797/
    # intel-arc-pro-b70-graphics/specifications.html
    "INTEL B70": DeviceInfo(
        tops={
            tp.float64: 11.47,
            tp.float32: 22.94,
            "tp.tf32": 183.5,
            tp.bfloat16: 183.5,
            tp.float16: 183.5,
            tp.float8_e8m0fnu: 183.5,
            tp.float8_e4m3fnuz: 183.5,
            tp.float8_e5m2: 183.5,
            tp.float8_e5m2fnuz: 183.5,
            tp.int8: 367,
        },
        dram_bw_gbs=608.0,
        dram_gb=32.0,
    ),
    # Source:
    # @lint-ignore https://www.intel.com/content/www/us/en/products/sku/232876/\
    # intel-data-center-gpu-max-1100/specifications.html
    "Intel(R) Data Center GPU Max 1100": DeviceInfo(
        # XeCore count: 56,
        # vector/matrix engines per XeCore: 8
        # freq: 1.55GHz
        tops={
            # Vector engine
            tp.float64: 11.1,
            tp.float32: 22.2,
            # not specified, fall back to float32 numbers
            "tp.tf32": 22.2,
            # Matrix engine
            tp.float16: 355.5,
            tp.bfloat16: 355.5,
            # not supported, fall back to float32 numbers
            tp.float8_e8m0fnu: 22.2,
            tp.float8_e4m3fnuz: 22.2,
            tp.float8_e5m2: 22.2,
            tp.float8_e5m2fnuz: 22.2,
            tp.int8: 711.1,
        },
        dram_bw_gbs=1228.8,
        dram_gb=48,
    ),
}
_device_mapping["AMD INSTINCT MI350X"] = _device_mapping["AMD MI350X"]
_device_mapping["AMD INSTINCT MI300X"] = _device_mapping["AMD MI300X"]
_device_mapping["AMD INSTINCT MI210X"] = _device_mapping["AMD MI210X"]
_device_mapping["Intel(R) Arc(TM) B580 Graphics"] = _device_mapping["INTEL B580"]
_device_mapping["Intel(R) Arc(TM) Pro B70 Graphics"] = _device_mapping["INTEL B70"]

# Enforce the upper-case-key invariant so entries cannot silently miss
# `lookup_device_info` (which upper-cases the query before lookup).
_device_mapping = {k.upper(): v for k, v in _device_mapping.items()}


def lookup_device_info(name: str) -> DeviceInfo | None:
    """
    Problem: when diffing profiles between amd and nvidia, we don't have access to the device information
    of the other one. Also, since the analysis is static, we should be able to do it on another device unrelated
    to the recorded device. Therefore, _device_mapping statically contains the information for lots of devices.
    If one is missing, please run DeviceInfo.get_device_info() and add it to _device_mapping.
      name (str): the device's own name, as the device reports it.
      Will be upper-cased before lookup.
    """
    return _device_mapping.get(name.upper())


def datasheet_tops(
    dtype: tp.dtype, is_tf32: bool = False, device_name: str | None = None
) -> float | None:
    """
    Get the theoretical *dense* TFLOPS of the device for a given dtype.

    If the datasheet values include 2:4 structured sparsity
    (``tops_sparsity_factor > 1``), they are divided by that factor so
    callers always receive the throughput achievable by cuBLAS/cuDNN on
    non-sparse data.

    If ``device_name`` is given, the datasheet is looked up for that named
    device instead of querying the current device. This lets callers pin a
    device spec for deterministic, hardware-independent estimates.
    """
    if device_name is not None:
        name: str | None = device_name
    elif tp.cuda.is_available():
        name = tp.cuda.get_device_name()
    elif tp.xpu.is_available():
        name = tp.xpu.get_device_name()
    else:
        log.info("No supported device available, skipping datasheet lookup")
        return None

    if name is None:
        log.info("No device found, returning None")
        return None
    device_info = lookup_device_info(name)
    if device_info is None:
        log_str = f"Device {name} not in datasheet, returning None"
        log.info(log_str)
        return None
    if dtype not in device_info.tops:
        log.info(
            "Device %s does not have a datasheet entry for %s, returning None",
            name,
            dtype,
        )
        return None

    tops = device_info.tops[
        "tp.tf32" if dtype == tp.float32 and is_tf32 else dtype
    ]
    return tops / device_info.tops_sparsity_factor


def datasheet_dram_bw_gbs(device_name: str | None = None) -> float | None:
    """
    Get the theoretical DRAM bandwidth (GB/s) of a device from the datasheet,
    or None if the device is not present.

    If ``device_name`` is given, that named device is looked up; otherwise the
    current CUDA/XPU device is queried.
    """
    if device_name is None:
        if tp.cuda.is_available():
            device_name = tp.cuda.get_device_name()
        elif tp.xpu.is_available():
            device_name = tp.xpu.get_device_name()
        else:
            log.info("No supported device available, skipping datasheet lookup")
            return None

    if device_name is None:
        log.info("No device found, returning None")
        return None

    device_info = lookup_device_info(device_name)
    if device_info is None:
        log.info("Device %s not in datasheet, returning None", device_name)
        return None
    return device_info.dram_bw_gbs

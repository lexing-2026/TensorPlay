from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

CPU_SOURCES = (
    "p10/src/backend/cpu/ReduceKernels.cpp",
    "p10/src/backend/cpu/ShapeOpsKernels.cpp",
    "p10/src/backend/cpu/TensorAdvancedIndexingKernels.cpp",
    "p10/src/backend/cpu/SortingKernels.cpp",
    "p10/src/backend/cpu/TensorCompareKernels.cpp",
    "p10/src/backend/cpu/OpsKernels.cpp",
)

CUDA_SOURCES = (
    "p10/src/backend/cuda/ReduceKernels.cu",
    "p10/src/backend/cuda/ShapeOpsKernels.cu",
    "p10/src/backend/cuda/TensorAdvancedIndexingKernels.cu",
    "p10/src/backend/cuda/SortingKernels.cu",
    "p10/src/backend/cuda/TensorCompareKernels.cu",
    "p10/src/backend/cuda/OpsKernels.cu",
)


def _read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _registrations(relative_paths):
    pattern = re.compile(r'\bm\.impl\("([^"]+)"')
    return [name for path in relative_paths for name in pattern.findall(_read(path))]


def test_operator_split_is_in_build_manifest():
    manifest = _read("p10/CMakeLists.txt")
    for path in CPU_SOURCES + CUDA_SOURCES:
        assert path.removeprefix("p10/") in manifest
        assert (ROOT / path).is_file()


def test_operator_split_files_have_unique_dispatch_entries():
    for paths, backend in ((CPU_SOURCES, "CPU"), (CUDA_SOURCES, "CUDA")):
        registrations = _registrations(paths)
        assert registrations
        assert len(registrations) == len(set(registrations)), backend
        for path in paths:
            source = _read(path)
            assert f"TENSORPLAY_LIBRARY_IMPL({backend}," in source

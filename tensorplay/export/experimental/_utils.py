"""Small source generators for packaged native model runners."""

from __future__ import annotations

from typing import Any

__all__ = ["_get_main_cpp_file", "_get_make_file"]


def _check_names(model_names: list[str]) -> None:
    if not model_names or any(not isinstance(name, str) or not name for name in model_names):
        raise ValueError("model_names must contain non-empty strings")


def _get_main_cpp_file(
    package_name: str,
    model_names: list[str],
    example_inputs_map: dict[str, int] | None,
    device_type: str,
) -> str:
    _check_names(model_names)
    if not package_name or not device_type:
        raise ValueError("package_name and device_type are required")
    lines = [
        "#include <iostream>",
        "#include <memory>",
        "",
        "int main() {",
        f'    std::cout << "package={package_name} device={device_type}" << std::endl;',
    ]
    for name in model_names:
        count = None if example_inputs_map is None else example_inputs_map.get(name)
        suffix = "" if count is None else f" inputs={count}"
        lines.append(f'    std::cout << "model={name}{suffix}" << std::endl;')
    lines.extend(["    return 0;", "}", ""])
    return "\n".join(lines)


def _get_make_file(package_name: str, model_names: list[str], device_type: str) -> str:
    _check_names(model_names)
    if not package_name or not device_type:
        raise ValueError("package_name and device_type are required")
    lines = [
        "cmake_minimum_required(VERSION 3.18)",
        "project(TensorPlayPackage LANGUAGES CXX)",
        "set(CMAKE_CXX_STANDARD 20)",
        "add_executable(main main.cpp)",
        f"target_compile_definitions(main PRIVATE TP_DEVICE_{device_type.upper()})",
    ]
    for name in model_names:
        lines.append(f"add_subdirectory({package_name}/data/{name})")
    return "\n".join(lines) + "\n"

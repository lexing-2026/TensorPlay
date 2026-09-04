#!/usr/bin/env python3
"""Compile the Vulkan compute shaders into an embedded SPIR-V registry.

For every GLSL template and its declared variants (see glsl/shader_params.yaml):
  1. preprocess the template with the xngen-style mini language:
       - `${PARAM}` substitutions,
       - `$if <python expr>:` / `$elif` / `$else` blocks evaluated against the
         variant's parameters,
  2. run the GLSL compiler (glslangValidator) targeting Vulkan 1.0,
  3. parse the set-0 binding declarations to recover the descriptor
     signature of the shader,
  4. embed the SPIR-V words into a generated C++ registry source that
     registers one api::ShaderInfo per compiled variant at load time.
"""

from __future__ import annotations

import argparse
import array
import copy
import glob
import io
import os
import re
import subprocess
import sys
from itertools import product
from pathlib import Path

CPP_H_NAME = "spv.h"
CPP_SRC_NAME = "spv.cpp"

DEFAULT_ENV: dict[str, object] = {
    "PRECISION": "highp",
    "FLOAT_IMAGE_FORMAT": "rgba32f",
    "INT_IMAGE_FORMAT": "rgba32i",
    "UINT_IMAGE_FORMAT": "rgba32ui",
}

try:
    import yaml
except ImportError:
    yaml = None

# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def extract_leading_whitespace(line: str) -> str:
    match = re.match(r"\s*", line)
    return match.group(0) if match else ""


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def escape(line: str) -> str:
    output_parts = []
    while "${" in line:
        start_pos = line.index("${")
        end_pos = line.index("}", start_pos + 2)
        if start_pos != 0:
            output_parts.append('"' + line[:start_pos].replace('"', '\\"') + '"')
        output_parts.append("str(" + line[start_pos + 2 : end_pos] + ")")
        line = line[end_pos + 1 :]
    if line:
        output_parts.append('"' + line.replace('"', '\\"') + '"')
    return " + ".join(output_parts)


# https://github.com/google/XNNPACK/blob/master/tools/xngen.py
def preprocess(
    input_text: str, variables: dict[str, object], input_path: str = "codegen"
) -> str:
    input_lines = input_text.splitlines()
    python_lines = []

    blank_lines = 0

    last_indent = ""

    # List of tuples (total_index, python_indent)
    indent_stack = [("", "")]

    # Indicates whether this is the first line inside Python
    # code block (i.e. for, while, if, elif, else)
    python_block_start = True
    for input_line in input_lines:
        if input_line == "":
            blank_lines += 1
            continue
        # Skip lint markers.
        if "LINT" in input_line:
            continue

        input_indent = extract_leading_whitespace(input_line)
        if python_block_start:
            if not input_indent.startswith(last_indent):
                raise AssertionError("input_indent must start with last_indent")
            extra_python_indent = input_indent[len(last_indent) :]
            python_indent = indent_stack[-1][1] + extra_python_indent
            indent_stack.append((input_indent, python_indent))
            if not input_indent.startswith(indent_stack[-1][0]):
                raise AssertionError("input_indent must start with indent_stack top")
        else:
            while not input_indent.startswith(indent_stack[-1][0]):
                del indent_stack[-1]
        python_block_start = False

        python_indent = indent_stack[-1][1]
        stripped_input_line = input_line.strip()
        if stripped_input_line.startswith("$") and not stripped_input_line.startswith(
            "${"
        ):
            if stripped_input_line.endswith(":"):
                python_block_start = True
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(python_indent + stripped_input_line.replace("$", ""))
        else:
            if not input_line.startswith(python_indent):
                raise AssertionError("input_line must start with python_indent")
            while blank_lines != 0:
                python_lines.append(python_indent + "print(file=OUT_STREAM)")
                blank_lines -= 1
            python_lines.append(
                python_indent
                + f"print({escape(input_line[len(python_indent):])}, file=OUT_STREAM)"
            )
        last_indent = input_indent

    while blank_lines != 0:
        # pyrefly: ignore [unbound-name]
        python_lines.append(python_indent + "print(file=OUT_STREAM)")
        blank_lines -= 1

    exec_globals = dict(variables)
    output_stream = io.StringIO()
    exec_globals["OUT_STREAM"] = output_stream

    python_bytecode = compile("\n".join(python_lines), input_path, "exec")
    exec(python_bytecode, exec_globals)

    return output_stream.getvalue()




_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"')


def inline_includes(text: str, shader_dir: Path, depth: int = 0) -> str:
    """Inline #include "header" directives; glslc does this implicitly but
    glslangValidator requires GL_GOOGLE_include_directive, so the generator
    resolves includes itself."""
    if depth > 4:
        raise RuntimeError("shader #include nesting too deep")
    out = []
    for line in text.splitlines(keepends=True):
        m = _INCLUDE_RE.match(line)
        if m:
            header = shader_dir / m.group(1)
            out.append(inline_includes(
                header.read_text(encoding="utf-8"), shader_dir, depth + 1))
        else:
            out.append(line)
    return "".join(out)


def parse_bindings_text(expanded: str) -> list[str]:
    """Recover the ordered descriptor signature from expanded shader text."""
    bindings: dict[int, str] = {}
    pattern = re.compile(r"layout\(set\s*=\s*0,\s*binding\s*=\s*(\d+)[^)]*\)(.*)")
    for line in expanded.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        binding_idx = int(match.group(1))
        rest = match.group(2)
        # Typed image and sampler declarations carry a one-letter prefix
        # (iimage3D, usampler3D, ...); the prefix is part of the same
        # identifier token, so the word-boundary match must allow it.
        if re.search(r"\b[iu]?image[123]D\b", rest):
            descriptor_type = "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE"
        elif re.search(r"\b[iu]?sampler[123]D\b", rest):
            descriptor_type = "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
        elif re.search(r"\buniform\b", rest):
            descriptor_type = "VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER"
        elif re.search(r"\bbuffer\b", rest):
            descriptor_type = "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"
        else:
            raise AssertionError(
                f"No matching descriptor type for line: {line.strip()!r}")
        bindings[binding_idx] = descriptor_type

    if not bindings:
        raise AssertionError(f"No set-0 bindings found in {src_path}")

    return [bindings[i] for i in sorted(bindings)]


def create_shader_params(
    env: dict[str, object], variant_params: dict[str, object] | None = None
) -> dict[str, object]:
    if variant_params is None:
        variant_params = {}
    shader_params = copy.deepcopy(env)
    for key, value in variant_params.items():
        shader_params[key] = value

    shader_dtype = shader_params.get("DTYPE", "float")

    if shader_dtype == "int":
        shader_params["FORMAT"] = env["INT_IMAGE_FORMAT"]
    elif shader_dtype == "uint":
        shader_params["FORMAT"] = env["UINT_IMAGE_FORMAT"]
    elif shader_dtype == "int32":
        shader_params["FORMAT"] = "rgba32i"
    elif shader_dtype == "uint32":
        shader_params["FORMAT"] = "rgba32ui"
    elif shader_dtype == "int8":
        shader_params["FORMAT"] = "rgba8i"
    elif shader_dtype == "uint8":
        shader_params["FORMAT"] = "rgba8ui"
    elif shader_dtype == "float32":
        shader_params["FORMAT"] = "rgba32f"
    # Assume float by default
    else:
        shader_params["FORMAT"] = env["FLOAT_IMAGE_FORMAT"]

    return shader_params


def generate_variant_combinations(
    iterated_params: dict[str, list[dict]],
    exclude_params: set[str] | None = None,
) -> list[list[tuple[str, str, object]]]:
    if exclude_params is None:
        exclude_params = set()
    all_iterated_params = []
    for param_name, value_list in iterated_params.items():
        if param_name not in exclude_params:
            param_values = []
            for value in value_list:
                suffix = value.get("SUFFIX", value["VALUE"])
                param_values.append((param_name, suffix, value["VALUE"]))
            all_iterated_params.append(param_values)

    return list(product(*all_iterated_params))


def generate(shader_dir: Path, out_dir: Path, glslang: Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for Vulkan shader generation")

    templates: dict[str, Path] = {
        p.stem: p for p in sorted(shader_dir.glob("*.glsl"))
    }
    spec_path = shader_dir / "shader_params.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}

    out_dir.mkdir(parents=True, exist_ok=True)

    compiled: dict[str, tuple[Path, list[str]]] = {}
    covered_templates: set[str] = set()
    # Optional per-shader output tile sizes declared as
    # `output_tile_sizes: {name: [x, y, z]}` inside a template spec.  The
    # tile is metadata for the dispatch loop, which divides the global
    # workgroup by it; shaders default to a (1, 1, 1) tile.
    tiles: dict[str, tuple[int, int, int]] = {}

    for template_name, template_spec in spec.items():
        defaults = dict(template_spec.get("parameter_names_with_default_values", {}))
        for tile_name, tile_value in (
            template_spec.get("output_tile_sizes", {}) or {}
        ).items():
            tiles[tile_name] = tuple(int(v) for v in tile_value)
        forall = template_spec.get("generate_variant_forall", {}) or {}
        for variant in template_spec["shader_variants"]:
            variant_keys = set(variant.keys()) - {"NAME"}
            iter_params = {
                k: v for k, v in forall.items() if k not in variant_keys
            }
            combos = generate_variant_combinations(
                iter_params, variant_keys) or [[]]

            for combo in combos:
                params = {**DEFAULT_ENV, **defaults}
                suffix = ""
                for param, param_suffix, value in combo:
                    params[param] = value
                    suffix += str(param_suffix)
                for key, value in variant.items():
                    if key != "NAME":
                        params[key] = value

                shader_name = variant["NAME"] + suffix
                base = template_name
                if not params.get("INPLACE", False):
                    base = template_name
                elif template_name + "_inplace" in templates:
                    base = template_name + "_inplace"
                else:
                    # One template with $if INPLACE handling.
                    base = template_name

                src_path = templates.get(base)
                if src_path is None:
                    raise RuntimeError(
                        f"GLSL template not found for '{template_name}'")
                covered_templates.add(base)

                expanded = preprocess(
                    inline_includes(
                        src_path.read_text(encoding="utf-8"), shader_dir),
                    create_shader_params(DEFAULT_ENV, params))
                out_glsl = out_dir / f"{shader_name}.glsl"
                out_glsl.write_text(expanded, encoding="utf-8")

                out_spv = out_dir / f"{shader_name}.spv"
                subprocess.check_call([
                    str(glslang),
                    "-V", "-S", "comp",
                    "--target-env", "vulkan1.0",
                    "-o", str(out_spv),
                    str(out_glsl),
                ])

                compiled[shader_name] = (out_spv, parse_bindings_text(expanded))

    # Shaders without template entries are compiled verbatim.
    for stem, src_path in templates.items():
        if stem in covered_templates:
            continue
        expanded = preprocess(
            inline_includes(
                src_path.read_text(encoding="utf-8"), shader_dir),
            create_shader_params(DEFAULT_ENV))
        out_glsl = out_dir / f"{stem}.glsl"
        out_glsl.write_text(expanded, encoding="utf-8")
        out_spv = out_dir / f"{stem}.spv"
        subprocess.check_call([
            str(glslang),
            "-V", "-S", "comp",
            "--target-env", "vulkan1.0",
            "-o", str(out_spv),
            str(out_glsl),
        ])
        compiled[stem] = (out_spv, parse_bindings_text(expanded))

    # C++ file generation
    spv_bin_arrays = []
    register_shader_infos = []
    for shader_name, (spv_path, layout) in sorted(compiled.items()):
        words = array.array("I", spv_path.read_bytes())
        # The consumer multiplies this count by sizeof(uint32_t) to obtain the
        # SPIR-V byte size, so it must be the word count, not the byte count.
        num_words = len(words)
        bin_str = ",\n".join(str(w) for w in words)
        spv_bin_arrays.append(
            f"const uint32_t {shader_name}_bin[] = {{\n{bin_str}\n}};")

        layout_str = "{" + ",\n ".join(layout) + "}"
        tile = tiles.get(shader_name, (1, 1, 1))
        tile_str = f"{{{tile[0]}u, {tile[1]}u, {tile[2]}u}}"
        register_shader_infos.append(
            "  api::shader_registry().register_shader(api::ShaderInfo(\n"
            f'      "{shader_name}",\n'
            f"      {shader_name}_bin,\n"
            f"      {num_words},\n"
            f"      {layout_str},\n"
            f"      {tile_str}));")

    header = f"""// Generated by tools/gen_vulkan_spv.py -- DO NOT EDIT
#pragma once

#ifdef USE_VULKAN

#endif /* USE_VULKAN */
"""

    source = f"""// Generated by tools/gen_vulkan_spv.py -- DO NOT EDIT
#ifdef USE_VULKAN

#include "Shader.h"
#include "ShaderRegistry.h"

#include <cstdint>

namespace tensorplay {{
namespace vulkan {{
namespace api {{

namespace {{

{chr(10).join(spv_bin_arrays)}

void register_fn() {{
{chr(10).join(register_shader_infos)}
}}

// Registration runs while this translation unit loads, before any dispatch
// touches the registry; shader_registry() itself never triggers registration.
const ShaderRegisterInit register_shaders(&register_fn);

}} // namespace

}} // namespace api
}} // namespace vulkan
}} // namespace tensorplay

#endif /* USE_VULKAN */
"""

    (out_dir / CPP_H_NAME).write_text(header, encoding="utf-8")
    (out_dir / CPP_SRC_NAME).write_text(source, encoding="utf-8")
    print(f"gen_vulkan_spv: compiled {len(compiled)} shaders -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glsl-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--glslang-path", required=True)
    args = parser.parse_args()

    generate(
        Path(args.glsl_path).resolve(),
        Path(args.output_path).resolve(),
        Path(args.glslang_path).resolve())


if __name__ == "__main__":
    main()

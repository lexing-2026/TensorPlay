#!/usr/bin/env python3
"""Embed a compiled .metallib into a C++ header as a byte array.

Used by the MPS backend CMake rules: `xcrun metal` compiles the .metal shader
sources into a .metallib, then this script turns the binary into an unsigned
char array that MetalShaderLibrary loads with `newLibraryWithData:`.
"""

import sys


def embed_metallib(metallib_filename: str, output_filename: str) -> None:
    with open(metallib_filename, "rb") as f:
        data = f.read()
    with open(output_filename, "w") as out:
        out.write("#pragma once\n")
        out.write("// Generated from %s\n" % metallib_filename)
        out.write("namespace tensorplay {\nnamespace mps {\n")
        out.write("static const unsigned char kQuantizedMetallib[] = {\n")
        for i in range(0, len(data), 16):
            chunk = data[i : i + 16]
            out.write("  " + ", ".join("0x%02x" % b for b in chunk) + ",\n")
        out.write("};\n")
        out.write("static const unsigned long kQuantizedMetallibSize = sizeof(kQuantizedMetallib);\n")
        out.write("} // namespace mps\n} // namespace tensorplay\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage:\n {sys.argv[0]} metallib_filename output_filename")
        sys.exit(1)
    embed_metallib(sys.argv[1], sys.argv[2])
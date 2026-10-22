#!/usr/bin/env python3
"""Carry the destination variants the runtime derives.

A schema declares its destination in one of two ways: it names one
outright, or it declares the in-place form and lets the runtime derive the
destination form from it.  The derived spelling is a destination variant
the runtime carries, and it is the same map with a destination in place of
the tensor the in-place form would have written into.

This reads the schema files, works out which derived spellings the project
does not carry, and writes:

  * the schema entries, into the marked section of the project's schema
    file, so the bindings and the dispatcher carry the variant;
  * the composite wrappers, into a source file of their own, which answer
    the map and lay the answer on the destination the caller holds.

The wrappers hold no arithmetic of their own: each one answers the map
the variant is derived from and hands the answer to the destination.

Usage:
    python tools/codegen/gen_out_variants.py --reference PATH [--project PATH]

The reference schema file is named on the command line; this tool does not
look for it.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from tools.codegen.api_types import stub_arg_type  # noqa: E402
from tools.codegen.model import parse_native_yaml  # noqa: E402

DEFAULT_PROJECT = os.path.join(ROOT, "config", "native_functions.yaml")
COMPOSITE_DIR = os.path.join(ROOT, "p10", "src", "backend", "composite")
COMPOSITE_OUT = os.path.join(
    COMPOSITE_DIR, "VariantOutCompositesGenerated.cpp")
REFUSED_FILE = os.path.join(HERE, "out_variants_refused.txt")

BEGIN_MARK = "# --- the destination variants the runtime derives ---"
END_MARK = "# --- end of the destination variants the runtime derives ---"


# ---------------------------------------------------------------------------
# The reference's list of derived spellings
# ---------------------------------------------------------------------------

def read_autogen(path):
    """The spellings the reference derives, with the entry each comes from.

    The reference writes the derived spellings in an `autogen:` line on the
    entry they are derived from: the in-place form, or the map itself when
    the destination replaces one of the answers it hands back.
    """
    import yaml

    entries = []
    with open(path, encoding="utf-8") as handle:
        document = yaml.compose(handle)
    for node in document.value:
        entry = {}
        for key, value in node.value:
            if isinstance(value, yaml.ScalarNode):
                entry[key.value] = value.value
            elif isinstance(value, yaml.SequenceNode):
                if all(isinstance(v, yaml.ScalarNode) for v in value.value):
                    entry[key.value] = [v.value for v in value.value]
        entries.append(entry)

    derived = []
    for entry in entries:
        autogen = entry.get("autogen")
        if not autogen:
            continue
        names = autogen if isinstance(autogen, list) else [autogen]
        for full in names:
            for piece in str(full).split(","):
                piece = piece.strip()
                if "." not in piece:
                    continue
                operation, spelling = piece.rsplit(".", 1)
                if spelling == "out" or spelling.endswith("_out"):
                    derived.append((operation, spelling, entry.get("func", "")))
    return derived


# ---------------------------------------------------------------------------
# The project's own entries
# ---------------------------------------------------------------------------

def read_registrations(directory):
    """The spellings the composites already answer.

    A variant that is carried in the schema file but has no answer is
    answered here: the schema says the variant exists, so it should.
    """
    import glob
    import re

    names = set()
    for path in sorted(glob.glob(os.path.join(directory, "*.cpp"))):
        if os.path.abspath(path) == os.path.abspath(COMPOSITE_OUT):
            # The file this tool writes is not an answer that is
            # already there; it is the one being written.
            continue
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        for match in re.finditer(r'm\.impl\("([^"]+)"', text):
            names.add(match.group(1))
    return names


def read_spellings(path):
    """The spellings the schema file carries, as they are written.

    The section this tool writes is left out: it is written afresh on
    every run, and the spellings in it are the ones being decided.
    """
    import re

    import yaml

    names = []
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    text = re.sub(re.escape(BEGIN_MARK) + r".*?" + re.escape(END_MARK),
                  "", text, flags=re.S)
    document = yaml.compose(text)
    for node in document.value:
        for key, value in node.value:
            if key.value == "func" and isinstance(value, yaml.ScalarNode):
                names.append(value.value.split("(", 1)[0])
    return names


def inplace_name(source):
    """The in-place form of the map a derived spelling comes from."""
    name = source.split("(", 1)[0]
    head, sep, tail = name.rpartition(".")
    if head and head.endswith("_"):
        return f"{head[:-1]}_.{tail}"
    if name.endswith("_"):
        return name
    return None


def base_name(source):
    """The map a derived spelling comes from.

    The in-place form is named after the map with an underscore in front
    of the overload: `relu_`, `add_.Scalar`.  The map it stands for is the
    name without that underscore.
    """
    name = source.split("(", 1)[0]
    head, sep, tail = name.rpartition(".")
    if head and head.endswith("_"):
        return f"{head[:-1]}{sep}{tail}"
    if name.endswith("_"):
        return name[:-1]
    return name


def alias_letter(index):
    """The alias the index-th destination carries.

    The first destination is written `a!`, the second `b!`, and so on.
    """
    return chr(ord("a") + index)


def spelling_of(name):
    """The overload a name carries, if any."""
    head, sep, tail = name.rpartition(".")
    if sep and head and tail:
        return tail
    return ""


def cxx_name(name):
    """The C++ name a map carries.

    The generated operations are overloaded on their arguments, so the map
    is named by its operation alone.
    """
    return spelling_of(name) and name.rsplit(".", 1)[0] or name


def arguments_of(schema_text):
    """The argument list of a schema, as it is written."""
    head = schema_text.split("->", 1)[0]
    inside = head[head.index("(") + 1:head.rindex(")")]
    parts = []
    depth = 0
    current = ""
    for char in inside:
        if char in "[({":
            depth += 1
        elif char in "])}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
            continue
        current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def argument_name(argument):
    """The name of an argument, without its type or its default."""
    name = argument.split("=", 1)[0].strip()
    return name.split()[-1].lstrip("*").strip()


def is_keyword_only(arguments):
    """Whether the argument list already names a keyword-only separator."""
    return any(argument.strip() == "*" for argument in arguments)


# ---------------------------------------------------------------------------
# The pieces that are written
# ---------------------------------------------------------------------------

def _without_alias(text):
    """An argument spelling with its write alias removed."""
    for alias in ("(a!)", "(b!)", "(c!)", "(d!)", "(e!)"):
        text = text.replace(alias, "")
    return text


def plain_arguments(base):
    """The arguments of a map, with the tensor it writes into made plain.

    The in-place form writes into the tensor it is given; the variant
    that carries a destination takes that tensor as an ordinary input
    and gives the answer to the destination instead.
    """
    texts = [argument for argument in arguments_of(base.schema)
             if argument.strip() != "*"]
    if len(texts) != len(base.args):
        return None
    written = base.self_arg()
    pairs = []
    for text, argument in zip(texts, base.args):
        if written is not None and argument.name == written.name:
            text = _without_alias(text)
            argument = _PlainArgument(argument)
        pairs.append((text, argument))
    return pairs


class _PlainArgument:
    """An argument that is written into no longer."""

    def __init__(self, argument):
        self.name = argument.name
        self.type = argument.type


def destinations_of(base, in_place=False):
    """The destinations a derived spelling takes, and what it answers.

    A map that answers once takes one destination.  A map that answers
    several times takes one destination for each answer.  A map that
    answers with a list takes the list of destinations.
    """
    kind = base.cpp_return_kind
    if in_place:
        if base.self_arg().type.is_list:
            return ["Tensor(a!)[] out"], "()"
        return ["Tensor(a!) out"], "Tensor(a!)"
    if kind == "list":
        return ["Tensor(a!)[] out"], "()"
    if kind == "tuple":
        answers = [answer for answer in base.returns
                   if answer.type.is_tensor_like]
        names = [
            f"Tensor({alias_letter(index)}!)[] out{index}"
            if answer.type.is_list
            else f"Tensor({alias_letter(index)}!) out{index}"
            for index, answer in enumerate(answers)]
        returns = "(" + ', '.join(
            f"Tensor({alias_letter(index)}!)[]"
            if answer.type.is_list
            else f"Tensor({alias_letter(index)}!)"
            for index, answer in enumerate(answers)) + ")"
        return names, returns
    return ["Tensor(a!) out"], "Tensor(a!)"


def schema_entry(operation, spelling, base, in_place=False):
    """The schema entry a derived spelling needs.

    The arguments are the ones the map takes; the destinations are added
    after them, and the answer the variant gives is the destination.
    """
    arguments = arguments_of(base.schema)
    if in_place:
        # The tensor the in-place form writes into is an ordinary input
        # of the variant that carries a destination.  The other tensors
        # the in-place form writes into keep writing into the caller's
        # tensors; only the answer of the map moves to the destination.
        written = base.self_arg()
        arguments = [
            _without_alias(argument)
            if (argument.strip() != "*" and written is not None
                and argument_name(argument) == written.name)
            else argument
            for argument in arguments]
    # A destination variant answers into a tensor the caller holds; the
    # marking flag describes how a fresh result is built, and no
    # destination variant takes it.
    arguments = [argument for argument in arguments
                 if argument_name(argument) != "requires_grad"]
    keyword_only = is_keyword_only(arguments)
    if not keyword_only:
        arguments = arguments + ["*"]
    destinations, returns = destinations_of(base, in_place)
    arguments = arguments + destinations
    kinds = ", ".join(base.variants) if base.variants else "function"
    return (f"- func: {operation}.{spelling}("
            f"{', '.join(arguments)}) -> {returns}\n"
            f"  variants: {kinds}")


def wrapper_name(operation, spelling):
    """The C++ name of the wrapper a derived spelling needs."""
    return "out_wrap_" + re.sub(r"[^0-9A-Za-z_]", "_", f"{operation}_{spelling}")


def arguments_with_types(base):
    """The arguments of a map, with their types, in the order they are given.

    The keyword-only separator is not an argument and is left out.
    """
    texts = [argument for argument in arguments_of(base.schema)
             if argument.strip() != "*"]
    if len(texts) != len(base.args):
        return None
    return list(zip(texts, base.args))


def wrapper(operation, spelling, base, in_place=False):
    """The composite that answers a derived spelling.

    The map is answered and the answer is laid on the destination the
    caller holds.
    """
    pairs = arguments_with_types(base)
    if pairs is None:
        return None
    if in_place:
        pairs = plain_arguments(base)
        if pairs is None:
            return None
    parameters = []
    values = []
    written = base.self_arg().name if (in_place and base.self_arg()) else None
    for _, argument in pairs:
        name = argument.name
        if name == "requires_grad":
            # The dispatcher does not carry the marking flag; the map is
            # answered without marking the fresh result it builds.
            values.append("false")
            continue
        # A composite answers through the dispatcher, whose ABI carries
        # owning non-trivial values by const reference.
        declared = stub_arg_type(argument.type)
        if name == written:
            # The in-place form writes into the tensor it is given, so
            # it is handed a copy of the caller's tensor.
            declared = "const std::vector<Tensor>&" if (
                argument.type.is_list) else "const Tensor&"
        parameters.append(f"{declared} {name}")
        # Every parameter the wrapper declares is forwarded: a schema
        # default describes what a caller may omit, not what the wrapper
        # substitutes for the argument it was handed.
        values.append("answer" if name == written else name)
    call = f"ops::{cxx_name(base.func_name)}({', '.join(values)})"
    kind = base.cpp_return_kind
    if in_place:
        self_is_list = base.self_arg().type.is_list
        if self_is_list:
            prepare = (
                "    std::vector<Tensor> answer;\n"
                "    for (const Tensor& item : self) {\n"
                "        answer.push_back(ops::clone(item));\n"
                "    }\n")
        else:
            prepare = "    Tensor answer = ops::clone(self);\n"
        if self_is_list or kind == "list":
            # A map that writes into a list of tensors is answered with
            # the list of the tensors it wrote into.
            parameters.append("std::vector<Tensor> out")
            if self_is_list:
                written_answers = "    {call};\n"
                source = "answer"
            else:
                written_answers = "    const auto answers = {call};\n"
                source = "answers"
            written_answers = written_answers.format(call=call)
            return (f"void {wrapper_name(operation, spelling)}("
                    f"{', '.join(parameters)}) {{\n"
                    f"{prepare}"
                    f"{written_answers}"
                    f"    for (size_t index = 0;\n"
                    f"         index < {source}.size() && index < out.size();\n"
                    f"         ++index) {{\n"
                    f"        write_out(out[index], {source}[index]);\n"
                    f"    }}\n"
                    f"}}\n")
        parameters.append("Tensor& out")
        return (f"Tensor& {wrapper_name(operation, spelling)}("
                f"{', '.join(parameters)}) {{\n"
                f"{prepare}"
                f"    {call};\n"
                f"    write_out(out, answer);\n"
                f"    return out;\n"
                f"}}\n")
    if kind == "list":
        parameters.append("std::vector<Tensor> out")
        return (f"void {wrapper_name(operation, spelling)}("
                f"{', '.join(parameters)}) {{\n"
                f"    const auto answers = {call};\n"
                f"    for (size_t index = 0;\n"
                f"         index < answers.size() && index < out.size();\n"
                f"         ++index) {{\n"
                f"        write_out(out[index], answers[index]);\n"
                f"    }}\n"
                f"}}\n")
    if kind == "tuple":
        answers = [answer for answer in base.returns
                   if answer.type.is_tensor_like]
        for index, answer in enumerate(answers):
            if answer.type.is_list:
                parameters.append(f"std::vector<Tensor> out{index}")
            else:
                parameters.append(f"Tensor& out{index}")
        lays = []
        for index, answer in enumerate(answers):
            if answer.type.is_list:
                lays.append(
                    f"    for (size_t item = 0;\n"
                    f"         item < std::get<{index}>(answers).size()"
                    f" && item < out{index}.size();\n"
                    f"         ++item) {{\n"
                    f"        write_out(out{index}[item], "
                    f"std::get<{index}>(answers)[item]);\n"
                    f"    }}")
            else:
                lays.append(
                    f"    write_out(out{index}, "
                    f"std::get<{index}>(answers));")
        lays = "\n".join(lays)
        gives = ', '.join(f"out{index}" for index in range(len(answers)))
        tensors = ', '.join(
            "std::vector<Tensor>" if answer.type.is_list else "Tensor"
            for answer in answers)
        return (f"std::tuple<{tensors}> "
                f"{wrapper_name(operation, spelling)}("
                f"{', '.join(parameters)}) {{\n"
                f"    const auto answers = {call};\n"
                f"{lays}\n"
                f"    return {{{gives}}};\n"
                f"}}\n")
    parameters.append("Tensor& out")
    return (f"Tensor& {wrapper_name(operation, spelling)}("
            f"{', '.join(parameters)}) {{\n"
            f"    write_out(out, {call});\n"
            f"    return out;\n"
            f"}}\n")


# ---------------------------------------------------------------------------
# The writing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--write", action="store_true",
                        help="write the entries and the wrappers")
    args = parser.parse_args()

    derived = read_autogen(args.reference)
    project = parse_native_yaml(args.project)
    by_name = {f.func_name: f for f in project}
    # The spellings the project carries are read from the schema file
    # itself: the parser renames some of the entries it reads, and the
    # spelling a caller writes is the one the file carries.
    carried = set()
    for name in read_spellings(args.project):
        head, sep, tail = name.rpartition(".")
        carried.add((head, tail) if sep and head else (name, ""))

    entries = []
    wrappers = []
    written = []
    skipped = []
    mismatched = []
    refused_missing = []
    refused = set()
    if os.path.exists(REFUSED_FILE):
        with open(REFUSED_FILE, encoding="utf-8") as handle:
            refused = {line.strip() for line in handle if line.strip()}
    registered = read_registrations(COMPOSITE_DIR)
    for operation, spelling, source in derived:
        if (operation, spelling) in carried and \
                f"{operation}.{spelling}" in registered:
            continue
        base = by_name.get(base_name(source))
        in_place = False
        if base is None:
            # The project may carry only the in-place form of the map;
            # the variant is answered by that form.
            inplace = inplace_name(source)
            base = by_name.get(inplace) if inplace else None
            in_place = base is not None
        if base is None:
            skipped.append((operation, spelling, source))
            continue
        if f"{operation}.{spelling}" in refused:
            refused_missing.append((operation, spelling, base.func_name))
            continue
        text = wrapper(operation, spelling, base, in_place)
        if text is None:
            mismatched.append((operation, spelling, base.func_name))
            continue
        if (operation, spelling) not in carried:
            entries.append(schema_entry(operation, spelling, base,
                                      in_place))
        if f"{operation}.{spelling}" not in registered:
            wrappers.append(text)
            written.append((operation, spelling,
                            cxx_name(base.func_name)))

    print(f"derived spellings the reference declares: {len(derived)}")
    carried_count = sum(1 for operation, spelling, _ in derived
                        if (operation, spelling) in carried)
    print(f"already carried: {carried_count}")
    print(f"written: {len(entries)}")
    print(f"skipped for want of the map: {len(skipped)}")
    for operation, spelling, source in skipped:
        print(f"   {operation}.{spelling}   <- {source}")
    print(f"skipped for want of the arguments: {len(mismatched)}")
    for operation, spelling, source in mismatched:
        print(f"   {operation}.{spelling}   <- {source}")
    print(f"left out on purpose: {len(refused_missing)}")
    for operation, spelling, source in refused_missing:
        print(f"   {operation}.{spelling}   <- {source}")
    print(f"total: {carried_count + len(entries) + len(skipped) + len(mismatched) + len(refused_missing)} of {len(derived)}")

    if not args.write:
        print("\n(dry run; pass --write to write the entries and the wrappers)")
        return 0

    with open(args.project, encoding="utf-8") as handle:
        text = handle.read()
    block = BEGIN_MARK + "\n" + "\n".join(entries) + "\n" + END_MARK
    if BEGIN_MARK in text:
        text = re.sub(re.escape(BEGIN_MARK) + r".*?" + re.escape(END_MARK),
                      block, text, flags=re.S)
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    with open(args.project, "w", encoding="utf-8") as handle:
        handle.write(text)

    registrations = [
        f'    m.impl("{operation}.{spelling}", '
        f'composite::{wrapper_name(operation, spelling)});'
        for operation, spelling, _ in written]

    with open(COMPOSITE_OUT, "w", encoding="utf-8") as handle:
        handle.write(
            '#include "Tensor.h"\n'
            '#include "Dispatcher.h"\n'
            '#include "tensorplay/ops/TPXOpsGenerated.h"\n\n'
            '#include <optional>\n'
            '#include <string>\n'
            '#include <tuple>\n'
            '#include <vector>\n'
            '#include "OutWrite.h"\n\n'
            'namespace tensorplay {\n'
            'namespace composite {\n\n'
            'namespace ops = tensorplay::tpx::ops;\n\n'
            + "\n".join(wrappers)
            + "\n} // namespace composite\n"
              "} // namespace tensorplay\n\n"
              "namespace tensorplay {\n"
              "TENSORPLAY_LIBRARY_IMPL(Composite, VariantOutGenerated) {\n"
            + "\n".join(registrations)
            + "\n}\n"
              "} // namespace tensorplay\n")

    print(f"the schema entries went to {args.project}")
    print(f"the wrappers went to {COMPOSITE_OUT}")
    index = os.path.join(HERE, "out_variants_index.txt")
    with open(index, "w", encoding="utf-8") as handle:
        for operation, spelling, map_name in written:
            handle.write(f"{operation}.{spelling}\t"
                         f"{wrapper_name(operation, spelling)}\t"
                         f"{map_name}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Populate local vision models with import rewriting.

Generated model modules use local package imports and the local runtime.

Relative imports (..transforms._presets, ..ops.misc, ..utils, ._api, ._meta,
._utils) are kept: matching modules exist under tensorplay/vision.
"""
import re
import sys
from pathlib import Path

TV = Path("/home/bluemoon/miniconda3/lib/python3.13/site-packages/torchvision")
DST = Path("tensorplay/vision")

MODELS = [
    "alexnet.py", "convnext.py", "densenet.py", "efficientnet.py",
    "googlenet.py", "inception.py", "maxvit.py", "mnasnet.py",
    "mobilenetv2.py", "mobilenetv3.py", "regnet.py", "resnet.py",
    "shufflenetv2.py", "squeezenet.py", "swin_transformer.py",
    "vision_transformer.py", "vgg.py",
]

SUPPORT = {
    "_api.py": TV / "models/_api.py",
    "_meta.py": TV / "models/_meta.py",
    "_utils.py": TV / "models/_utils.py",
}

REWRITES = [
    # exact-match first
    (re.compile(r"^import torch$"), "import tensorplay as torch"),
    (re.compile(r"^import torch\.nn\.functional as F$"), "import tensorplay.nn.functional as F"),
    (re.compile(r"^import torch\.nn as nn$"), "import tensorplay.nn as nn"),
    (re.compile(r"^import torch\.nn\.init as init$"), "import tensorplay.nn.init as init"),
    (re.compile(r"^from torch import nn$"), "from tensorplay import nn"),
    (re.compile(r"^from torch import Tensor$"), "from tensorplay import Tensor"),
    (re.compile(r"^import torch\.utils\.checkpoint as cp$"), "import tensorplay.utils.checkpoint as cp"),
    (re.compile(r"^import torch\.utils\.model_zoo as model_zoo$"), "import tensorplay.hub as model_zoo"),
]


def rewrite(text: str) -> str:
    out_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        new_line = line
        for pat, rep in REWRITES:
            if pat.match(stripped):
                indent = line[: len(line) - len(line.lstrip())]
                new_line = indent + rep
                break
        else:
            if stripped == "import torch":
                pass  # unreachable; pattern above handles it
        out_lines.append(new_line)
    text = "\n".join(out_lines)
    return text


def main() -> None:
    (DST / "models").mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        src = TV / "models" / name
        text = rewrite(src.read_text())
        header = "# Imports rewritten for the local package.\n"
        # insert after module docstring if present, else at top
        m = re.match(r'^("""[\s\S]*?"""\n)', text)
        if m:
            text = m.group(1) + header + text[m.end():]
        else:
            text = header + text
        (DST / "models" / name).write_text(text)
        print(f"wrote models/{name}")

    for name, src in SUPPORT.items():
        text = rewrite(src.read_text())
        header = "# Imports rewritten for the local package.\n"
        (DST / "models" / name).write_text(header + text)
        print(f"wrote models/{name}")


if __name__ == "__main__":
    main()

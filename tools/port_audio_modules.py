#!/usr/bin/env python3
"""Copy torchaudio sources into tensorplay/audio with import rewriting.

Mirrors tools/port_vision_models.py: the model/dataset/functional definitions
are copied verbatim from the vendored torchaudio checkout
(third_party/audio, 2.11.0a0); only module references are rewritten:

    torch       -> tensorplay            (aliased import keeps call sites verbatim)
    torchaudio  -> tensorplay.audio
    _internal   -> local hub-backed shims
    _extension  -> local availability shims

Relative imports are preserved; matching subpackages exist under
tensorplay/audio.
"""
import re
import shutil
from pathlib import Path

TA = Path("third_party/audio/src/torchaudio")
DST = Path("tensorplay/audio")

# (relative source path under TA, destination relative path under DST)
FILES = {
    # functional
    "functional/functional.py": "functional/functional.py",
    "functional/filtering.py": "functional/filtering.py",
    "functional/_alignment.py": "functional/_alignment.py",
    # transforms
    **{
        f"transforms/{n}": f"transforms/{n}"
        for n in [
            "__init__.py", "_transforms.py", "_multi_channel.py",
        ]
    },
    # compliance
    "compliance/kaldi.py": "compliance/kaldi.py",
    # datasets
    **{
        f"datasets/{n}": f"datasets/{n}"
        for n in sorted(p.name for p in (TA / "datasets").glob("*.py"))
    },
    # models
    **{
        str(p.relative_to(TA)): str(p.relative_to(TA))
        for p in (TA / "models").rglob("*.py")
        if "_cuda_ctc_decoder" not in p.name
    },
    # utils
    **{
        f"utils/{n}": f"utils/{n}"
        for n in ["global_s normalization.py".replace(" ", ""), ] if False
    },
}

REWRITES = [
    (re.compile(r"^import torch$"), "import tensorplay as torch"),
    (re.compile(r"^import torch\.nn\.functional as F$"), "import tensorplay.nn.functional as F"),
    (re.compile(r"^import torch\.nn\.init as init$"), "import tensorplay.nn.init as init"),
    (re.compile(r"^from torch import Tensor$"), "from tensorplay import Tensor"),
    (re.compile(r"^from torch\.nn\.functional import (\w+)$"), r"from tensorplay.nn.functional import \1"),
    (re.compile(r"^import torchaudio$"), "import tensorplay.audio as torchaudio"),
    (re.compile(r"^import torchaudio\.functional as F$"), "import tensorplay.audio.functional as F"),
    (re.compile(r"^from torchaudio import transforms"), "from tensorplay.audio import transforms"),
    (re.compile(r"^from torchaudio\._internal import download_url_to_file$"), "from tensorplay.hub import download_url_to_file"),
    (re.compile(r"^from torchaudio\._internal import load_state_dict_from_url$"), "from tensorplay.hub import load_state_dict_from_url"),
    (re.compile(r"^from torchaudio\._internal import download_url_to_file, module_utils as _mod_utils$"),
     "from tensorplay.hub import download_url_to_file\nfrom . import _module_utils as _mod_utils"),
    (re.compile(r"^from torchaudio\._internal import module_utils$"), "from . import _module_utils"),
    (re.compile(r"^from torchaudio\._internal\.module_utils import fail_with_message, no_op$"),
     "from ._module_utils import fail_with_message, no_op"),
    (re.compile(r"^from torchaudio\._extension import _IS_AUDIO_EXT_AVAILABLE$"),
     "from ._extension import _IS_AUDIO_EXT_AVAILABLE"),
    (re.compile(r"^from torchaudio\._extension import fail_if_no_align$"),
     "from ._extension import fail_if_no_align"),
]


def rewrite(text: str) -> str:
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        new_line = None
        for pat, rep in REWRITES:
            m = pat.match(stripped)
            if m:
                indent = line[: len(line) - len(line.lstrip())]
                new_line = indent + rep
                break
        out.append(new_line if new_line is not None else line)
    return "\n".join(out)


def main():
    copied = 0
    for src_rel, dst_rel in FILES.items():
        src = TA / src_rel
        dst = DST / dst_rel
        if not src.exists():
            print(f"SKIP missing {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = rewrite(src.read_text())
        dst.write_text(text)
        copied += 1
    print(f"ported {copied} files into {DST}")


if __name__ == "__main__":
    main()

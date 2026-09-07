"""Helpers shared between the Windows wheel-build scripts.

The setup/dependency phases each hand their environment back to the parent
bash wrapper through a --env-out file, and both stream toolchain pieces from
network URLs. Keeping those primitives here means a fix lands once for every
writer (e.g. the cygpath -up PATH conversion or the bash-identifier filter).
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
import urllib.request
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


_BASH_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def shell_quote(value: str) -> str:
    if value and all(c.isalnum() or c in "_-./:=" for c in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def _to_posix_path_list(windows_path_list: str) -> str:
    """Convert a Windows `;`-separated path list to POSIX `:`-separated.

    cmd-side setup scripts emit PATH in Windows form (`;` separators,
    backslash directory separators). The parent bash needs `:` separators
    and POSIX-style paths to resolve executables; sourcing PATH unmodified
    leaves the shell with one bogus entry and the next `python` lookup dies
    with exit 127. `cygpath -up` is the canonical translator.
    """
    if not windows_path_list:
        return windows_path_list
    result = subprocess.run(
        ["cygpath", "-up", windows_path_list],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def write_env_exports(env: dict[str, str], path: Path | None) -> None:
    """Write `export KEY=VALUE` lines for the parent bash wrapper to source.

    PATH is converted from Windows form to POSIX form when it carries the
    `;` separator; other path-like variables (INCLUDE, LIB, LIBPATH, ...)
    stay in Windows form because the MSVC tools consuming them expect that.

    Keys that are not valid bash identifiers are skipped: a shell function
    serialized into the environment (e.g. `BASH_FUNC_retry%%=() { ... }`)
    cannot be re-exported under an identifier containing `%`, and sourcing
    such a line would abort with `not a valid identifier`.
    """
    if path is None:
        return
    lines = []
    for k, v in env.items():
        if not _BASH_IDENT_RE.match(k):
            continue
        if k.upper() == "PATH" and ";" in v:
            # A `;` separator marks the value as Windows-form; a value
            # already in POSIX form (no `;`) is left untouched.
            v = _to_posix_path_list(v)
        lines.append(f"export {k}={shell_quote(v)}")
    path.write_text("\n".join(lines) + "\n")


def download(url: str, dest: Path, attempts: int = 5) -> None:
    """Stream `url` to `dest`, retrying with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading {url} -> {dest} (attempt {attempt}/{attempts})")
            with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            return
        except Exception as exc:
            if attempt == attempts:
                sys.exit(f"Failed to download {url}: {exc}")
            time.sleep(2**attempt)

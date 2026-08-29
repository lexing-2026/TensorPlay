"""Weight download plumbing for the vision package.

``load_state_dict_from_url`` follows tensorplay.hub.load_state_dict_from_url's
contract: download (or reuse) the cached file, verify the hash when the file
name embeds one, and load it as a state dict.
"""

import hashlib
import os
import re
from pathlib import Path

import tensorplay
from tensorplay import hub

__all__ = ["load_state_dict_from_url"]

_HOME = Path.home() / ".cache" / "tensorplay" / "datasets" / "vision"


def _get_tensorplay_home() -> str:
    return os.environ.get("TENSORPLAY_HOME", str(_HOME))


def _download_url_to_file(url, dst, hash_prefix=None, progress=True):
    """tensorplay.hub.download_url_to_file with SHA256 check semantics of tensorplay."""
    hub.download_url_to_file(url, str(dst), hash_prefix=hash_prefix, progress=progress)


def load_state_dict_from_url(
    url: str,
    model_dir: str | None = None,
    map_location=None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: str | None = None,
    weights_only: bool = False,
) -> dict:
    r"""Loads the Tensor serialized at ``url`` into a state dict.

    Same contract as tensorplay.hub.load_state_dict_from_url.  If ``check_hash``
    is True and the file name embeds a ``<sha256>-<filename>`` prefix, the
    downloaded object is verified against that sha256 prefix.
    """
    if model_dir is None:
        model_dir = _get_tensorplay_home()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    parts = url.rstrip("/").split("/")
    filename = file_name or parts[-1]
    cached_file = model_dir / filename

    if not cached_file.exists():
        hash_prefix = None
        if check_hash:
            HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")
            match = HASH_REGEX.search(filename)
            hash_prefix = match.group(1) if match else None
        _download_url_to_file(url, cached_file, hash_prefix=hash_prefix, progress=progress)

    state_dict = tensorplay.load(str(cached_file), map_location=map_location)
    return state_dict

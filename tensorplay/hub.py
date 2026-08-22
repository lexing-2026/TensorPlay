import hashlib
import re
from types import ModuleType
import logging
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

__all__ = ["get_dir", "set_dir", "download_url_to_file"]

logger = logging.getLogger(__name__)

# Cache Directory Management
DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "tensorplay"
if not DEFAULT_CACHE_DIR.exists():
    try:
        DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Fallback to a local directory if home is not writable
        DEFAULT_CACHE_DIR = Path("utils").resolve() / ".tensorplay_cache"
        DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_hub_dir: Optional[Path] = None

def get_dir() -> Path:
    """Get the TensorPlay Hub cache directory used for storing downloaded models & weights."""
    if _hub_dir is not None:
        return _hub_dir
    return DEFAULT_CACHE_DIR / "hub"

def set_dir(d: Union[str, Path]) -> None:
    r"""
    Optionally set the TensorPlay Hub directory used to save downloaded models & weights.

    Args:
        d (str): path to a local folder to save downloaded models & weights.
    """
    if not isinstance(d, (str, Path)):
        raise TypeError(f"Expected directory path to be str or Path, but got {type(d).__name__}.")
    global _hub_dir
    _hub_dir = Path(d).expanduser().resolve()
    _hub_dir.mkdir(parents=True, exist_ok=True)

# Download Utility
DEFAULT_RETRY_DELAY : float = 1.0
MAX_RETRY_DELAY : float = 10.0
READ_DATA_CHUNK: int = 128 * 1024
USER_AGENT = "TensorPlay/1.0 (Python/{}.{}; {})".format(
    sys.version_info.major,
    sys.version_info.minor,
    sys.platform
)

def _human_readable_size(size: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"

def download_url_to_file(
        url: str,
        dst: Union[str, Path],
        hash_prefix: Optional[str] = None,
        progress: bool = True,
        timeout: float = 10.0,
        max_retries: int = 3,
        overwrite: bool = False,
        allow_resume: bool = True,
        user_agent: str = USER_AGENT,
) -> None:
    r"""
    Download a URL to a local file(Safe download: temp file + hash check + progress feedback).

    Features:
    - First download to a temp file, then move to the destination path to avoid corrupting the destination file.
    - Support SHA256 hash prefix check to ensure file integrity.
    - Show download progress bar (disable with progress=False).
    - Support network timeout and retry mechanism for stability.
    - Automatically create parent directories for the destination path.

    Args:
        url (str): URL address to download (supports HTTP/HTTPS).
        dst (str | Path): Destination path (including filename) to save the file.
        hash_prefix (Optional[str]): SHA256 hash prefix for integrity check, default None.
        progress (bool): Whether to show download progress bar, default True.
        timeout (float): Network request timeout in seconds, default 10.0.
        max_retries (int): Maximum number of retry attempts for network errors, default 3.
        overwrite (bool): Whether to overwrite existing file, default False.
        allow_resume (bool): Whether to support resuming interrupted downloads, default True.
        user_agent (str): Custom User-Agent header for HTTP requests, default USER_AGENT.
    """
    if not url.strip():
        raise ValueError("Download URL cannot be an empty string")
    if hash_prefix is not None:
        if not isinstance(hash_prefix, str) or len(hash_prefix) < 4:
            raise TypeError(
                f"hash_prefix must be a string with length ≥ 4, but got {type(hash_prefix).__name__} "
                f"(length: {len(hash_prefix) if isinstance(hash_prefix, str) else 'N/A'})"
            )

    # Detect hash algorithm based on length if possible, or default to sha256
    # MD5: 32 chars, SHA256: 64 chars
    hash_algo = hashlib.sha256
    if hash_prefix and len(hash_prefix) == 32:
        hash_algo = hashlib.md5

    dst_path = Path(dst).resolve()
    dst_parent = dst_path.parent

    try:
        dst_parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise RuntimeError(f"Permission denied: cannot create parent directory {dst_parent} - {e}") from e

    if dst_path.exists():
        if overwrite:
            dst_path.unlink(missing_ok=True)
        else:
            # If hash check is required and file exists, verify it
            if hash_prefix:
                hasher = hash_algo()
                with open(dst_path, "rb") as f:
                    while chunk := f.read(READ_DATA_CHUNK):
                        hasher.update(chunk)
                if hasher.hexdigest().startswith(hash_prefix.lower()):
                    logger.info(f"Target file {dst_path} already exists and hash matches, skip downloading.")
                    return
                else:
                    logger.warning(f"Target file {dst_path} exists but hash mismatch. Redownloading.")
                    dst_path.unlink()
            else:
                logger.info(f"Target file {dst_path} already exists, skip downloading.")
                return

    tmp_suffix = f".partial.{uuid.uuid4().hex}"
    tmp_dst = dst_path.with_suffix(f"{dst_path.suffix}{tmp_suffix}")
    downloaded_size = 0
    hasher = hash_algo() if hash_prefix else None

    if allow_resume and tmp_dst.exists():
        try:
            downloaded_size = tmp_dst.stat().st_size
            if downloaded_size > 0:
                if hasher:
                    with open(tmp_dst, "rb") as f:
                        while chunk := f.read(READ_DATA_CHUNK):
                            hasher.update(chunk)
            else:
                tmp_dst.unlink(missing_ok=True)
        except Exception as e:
            tmp_dst.unlink(missing_ok=True)
            downloaded_size = 0

    retry_count = 0
    while retry_count < max_retries:
        retry_delay = min(DEFAULT_RETRY_DELAY * (2 ** retry_count), MAX_RETRY_DELAY)
        try:
            headers = {"User-Agent": user_agent, "Accept": "*/*"}
            if allow_resume and downloaded_size > 0:
                headers["Range"] = f"bytes={downloaded_size}-"

            req = Request(url, headers=headers)

            with urlopen(req, timeout=timeout) as u:
                status_code = u.status
                if status_code == 200:
                    total_size = int(u.headers.get("Content-Length", 0)) if u.headers.get("Content-Length", "").isdigit() else None
                    if allow_resume and downloaded_size > 0:
                        # Server ignored Range header, restart download
                        downloaded_size = 0
                        tmp_dst.unlink(missing_ok=True)
                        if hasher:
                             hasher = hash_algo()
                elif status_code == 206 and allow_resume and downloaded_size > 0:
                    remaining_size = int(u.headers.get("Content-Length", 0)) if u.headers.get("Content-Length", "").isdigit() else None
                    total_size = downloaded_size + remaining_size if remaining_size is not None else None
                elif status_code == 404:
                    raise HTTPError(url, status_code, "File not found", u.headers, None)
                elif status_code >= 500:
                    raise HTTPError(url, status_code, "Server internal error", u.headers, None)
                else:
                    raise RuntimeError(f"Unsupported HTTP status code: {status_code} (URL: {url})")

                mode = "ab" if downloaded_size > 0 else "wb"
                
                pbar = None
                if progress:
                    # Print header: Downloading URL (SIZE)
                    readable_size = "Unknown size"
                    if total_size is not None:
                        readable_size = _human_readable_size(float(total_size))
                    
                    # Use standard print for the header message
                    print(f"Downloading {url} ({readable_size})")

                    if tqdm is not None:
                        # Configure tqdm to look like pip's bar
                        # Format: Indentation + Colored Bar + Stats
                        # Use Magenta (\033[95m) for the bar to match typical pip style
                        # Characters: '━' for fill, '╸' for tip
                        bar_fmt = "    \033[95m{bar:40}\033[0m {n_fmt}/{total_fmt} {rate_fmt} eta {remaining}"
                        
                        pbar = tqdm(
                            total=total_size,
                            initial=downloaded_size,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            bar_format=bar_fmt,
                            ascii=" ╸━",
                            file=sys.stderr,
                            leave=True
                        )
                
                try:
                    with open(tmp_dst, mode) as f:
                        while True:
                            buffer = u.read(READ_DATA_CHUNK)
                            if not buffer:
                                break
                            f.write(buffer)
                            if hasher:
                                hasher.update(buffer)
                            if pbar:
                                pbar.update(len(buffer))
                finally:
                    if pbar:
                        pbar.close()

                if hash_prefix:
                    assert hasher is not None, "Hash checker is not initialized"
                    digest = hasher.hexdigest()
                    if not digest.startswith(hash_prefix.lower()):
                        tmp_dst.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"Hash check failed!\n"
                            f"File path: {dst_path}\n"
                            f"Expected prefix: {hash_prefix}\n"
                            f"Actual hash: {digest}\n"
                        )

                try:
                    shutil.move(str(tmp_dst), str(dst_path))
                except PermissionError as e:
                    time.sleep(1)
                    shutil.move(str(tmp_dst), str(dst_path))
                except Exception as e:
                    raise RuntimeError(f"Failed to move temp file to final destination: {e}") from e

                return

        except HTTPError as e:
            retry_count += 1
            status_code = e.code
            if status_code == 404:
                tmp_dst.unlink(missing_ok=True)
                raise RuntimeError(f"File not found: {url}") from e
            
            logger.warning(f"Download failed (Retry {retry_count}/{max_retries}): HTTP {status_code}")
            time.sleep(retry_delay)

            if retry_count >= max_retries:
                tmp_dst.unlink(missing_ok=True)
                raise RuntimeError(f"Network error: Retried {max_retries} times, still failed. URL: {url}") from e

        except URLError as e:
            retry_count += 1
            logger.warning(f"Download failed (Retry {retry_count}/{max_retries}): {e}")
            time.sleep(retry_delay)
            if retry_count >= max_retries:
                tmp_dst.unlink(missing_ok=True)
                raise RuntimeError(f"Network error: Retried {max_retries} times, still failed. URL: {url}") from e




# ---------------------------------------------------------------------------
# Unified remote-model hub: MEGA (default) + GitHub/torch.hub compatibility.
#
#   tp.hub.load_state_dict("org/model", "weights.mst")                 # mega
#   tp.hub.load_model("org/model", model_class="...resnet50")          # mega
#   tp.hub.load("pytorch/vision", "resnet50")                          # github
#   sd = tp.hub.load_state_dict(
#       "https://download.pytorch.org/models/resnet50.pth",
#       source="github")                                               # url
#
# The GitHub path mirrors torch.hub.load: shallow-clone the repo into the hub
# cache, execute its ``hubconf.py`` under a torch->tensorplay module aliasing,
# then invoke the requested entrypoint.
# ---------------------------------------------------------------------------

__all__ += [
    "load_state_dict",
    "load_model",
    "snapshot_download",
    "load",
    "list_entrypoints",
    "load_state_dict_from_url",
]

_WEIGHT_URL_RE = re.compile(r"^https?://.*\.(pth|pt|ckpt|safetensors|mst)([?#].*)?$", re.I)


# ---------------------------------------------------------------------------
# torch -> tensorplay module aliasing (for executing foreign hubconf.py)
# ---------------------------------------------------------------------------

class _LazyAlias(ModuleType):
    """Module placeholder whose attribute access proxies to ``target``."""

    def __init__(self, name: str, target):
        super().__init__(name)
        self._target = target

    def __getattr__(self, item):
        return getattr(self._target, item)


class torch_hub_alias:
    """Temporarily aliases ``torch*`` modules onto tensorplay so that foreign
    hubconf.py files (written against PyTorch) import our implementation.

    Restores the previous ``sys.modules`` entries on exit.
    """

    def __init__(self):
        self._saved = {}

    def _aliases(self) -> dict:
        import tensorplay
        import tensorplay.nn

        hub_shim = ModuleType("tensorplay.hub.shim")
        hub_shim.        hub_shim.download_url_to_file = download_url_to_file
        hub_shim.get_dir = get_dir
        hub_shim.set_dir = set_dir
        hub_shim._get_torch_home = lambda: str(get_dir())

        utils_shim = ModuleType("tensorplay.utils.shim")
        from tensorplay.utils import checkpoint as _cp  # noqa: F401
        utils_shim.checkpoint = _cp

        jit_mod = _import_optional("tensorplay.jit")
        fx_mod = _import_optional("tensorplay.fx")

        return {
            "torch": tensorplay,
            "torch.nn": tensorplay.nn,
            "torch.nn.functional": tensorplay.nn.functional,
            "torch.nn.init": tensorplay.nn.init,
            "torch.Tensor": type(tensorplay.Tensor([0.0])) if False else None,
            "torch.hub": hub_shim,
            "torch.utils": utils_shim,
            "torch.utils.checkpoint": _cp,
            "torch.jit": jit_mod,
            "torch.fx": fx_mod,
            "torchvision": _import_optional("tensorplay.vision"),
            "torchvision.models": _import_optional("tensorplay.vision.models"),
            "torchvision.transforms": _import_optional("tensorplay.vision.transforms"),
            "torchvision.datasets": _import_optional("tensorplay.vision.datasets"),
            "torchvision.io": _import_optional("tensorplay.vision.io"),
        }

    def __enter__(self):
        for name, target in self._aliases().items():
            if target is None:
                continue
            self._saved[name] = sys.modules.get(name)
            if isinstance(target, ModuleType):
                sys.modules[name] = target
            else:
                sys.modules[name] = _LazyAlias(name, target)
        return self

    def __exit__(self, *exc):
        for name, prev in self._saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        return False


def _import_optional(name: str):
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError:
        mod = ModuleType(name)
        return mod


# ---------------------------------------------------------------------------
# GitHub backend
# ---------------------------------------------------------------------------

def _github_repo_dir(repo_id: str, ref: str | None = None) -> Path:
    name = repo_id.split("/")[-1]
    base = get_dir() / "github" / repo_id.replace("/", "_")
    if base.exists():
        return base
    base.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"]
    url = f"https://github.com/{repo_id}.git"
    if ref:
        cmd += ["--branch", ref]
    cmd += [url, str(base)]
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git clone failed for {url}: {proc.stderr.strip()}")
    return base


def _exec_hubconf(repo_dir: Path):
    import importlib.util

    conf = repo_dir / "hubconf.py"
    if not conf.exists():
        raise RuntimeError(f"{repo_dir} has no hubconf.py")
    spec = importlib.util.spec_from_file_location("_tensorplay_foreign_hubconf", conf)
    module = importlib.util.module_from_spec(spec)
    with torch_hub_alias():
        spec.loader.exec_module(module)
    return module


def list_entrypoints(repo_id: str, ref: str | None = None) -> list[str]:
    """Entrypoints declared by a GitHub repo's hubconf.py (torch.hub.list)."""
    repo_dir = _github_repo_dir(repo_id, ref)
    module = _exec_hubconf(repo_dir)
    deps = {"dependencies", "verbose"}
    return sorted(n for n in dir(module)
                  if not n.startswith("_") and n not in deps and callable(getattr(module, n)))


def load(repo_id: str, model: str | None = None, *args, ref: str | None = None, **kwargs):
    """Unified model loader.

    mega:    tp.hub.load("org/repo", filename="weights.mst", model_class=...)
    github:  tp.hub.load("pytorch/vision", "resnet50")  # torch.hub.load equivalent

    Dispatches on ``source`` exactly like :func:`load_model`; the two-argument
    form is the torch.hub.load shape.
    """
    repo_dir = _github_repo_dir(repo_id, ref)
    module = _exec_hubconf(repo_dir)
    fn = getattr(module, model, None)
    if fn is None:
        raise RuntimeError(f"'{model}' is not an entrypoint of {repo_id}; "
                           f"available: {list_entrypoints(repo_id, ref)}")
    with torch_hub_alias():
        return fn(*args, **kwargs)


def load_state_dict_from_url(url: str, progress: bool = True, check_hash: bool = False,
                             map_location=None, **kwargs) -> dict:
    """Downloads a raw checkpoint URL and loads it (torch semantics)."""
    parts = url.rstrip("/").split("/")
    cached = get_dir() / "checkpoints" / parts[-1]
    cached.parent.mkdir(parents=True, exist_ok=True)
    hash_prefix = None
    if check_hash:
        m = re.search(r"-([a-f0-9]{8,})\.", parts[-1])
        hash_prefix = m.group(1) if m else None
    if not cached.exists():
        download_url_to_file(url, str(cached), hash_prefix=hash_prefix, progress=progress)

    # Delegate deserialization to the torch-compat serialization layer.
    try:
        from tensorplay import _serialization_torch as ser_torch
    except ImportError:
        ser_torch = None
    import tensorplay as tp

    if ser_torch is not None:
        return ser_torch.load(str(cached), map_location=map_location)
    return tp.load(str(cached), map_location=map_location)


# ---------------------------------------------------------------------------
# MEGA backend (megatensors SDK, imported lazily)
# ---------------------------------------------------------------------------

_WEIGHT_EXTS = (".safetensors", ".mst", ".pt", ".pth", ".bin")


def _mega_client(endpoint=None, token=None):
    try:
        from megatensors.hub import MegaHubClient
    except ImportError as exc:
        raise ImportError(
            "MEGA hub support requires the 'megatensors' package. "
            "Install it with `pip install megatensors`."
        ) from exc
    return MegaHubClient(endpoint=endpoint, token=token)


def _mega_load_state_dict(paths, device, load_kwargs):
    import megatensors

    kwargs = {"framework": "tensorplay", "device": device}
    kwargs.update(load_kwargs)
    return megatensors.load_state_dict([str(p) for p in paths], **kwargs)


def snapshot_download(
    repo_id: str,
    *,
    source: str = "mega",
    revision: str = "main",
    include=None,
    exclude=None,
    endpoint=None,
    token=None,
    ref: str | None = None,
) -> Path:
    """Downloads a full repository snapshot (mega or github) into the cache."""
    if source == "mega":
        local_dir = get_dir() / "mega" / repo_id
        client = _mega_client(endpoint, token)
        return client.snapshot_download(
            repo_id, local_dir=local_dir, revision=revision, include=include, exclude=exclude
        )
    elif source == "github":
        return _github_repo_dir(repo_id, ref or revision)
    raise ValueError(f"unknown source '{source}' (expected 'mega' or 'github')")


def load_state_dict(
    repo_or_url: str,
    filename: str | None = None,
    *,
    source: str = "auto",
    device: str = "cpu",
    revision: str = "main",
    endpoint=None,
    token=None,
    ref: str | None = None,
    **load_kwargs,
) -> dict:
    """Loads weights from MEGA, a GitHub checkpoint URL, or a github repo.

    ``source='auto'`` inspects the argument: an http(s) URL uses the github
    checkpoint path, anything else is treated as a MEGA ``repo_id``.
    """
    src = source
    if src == "auto":
        src = "github" if _WEIGHT_URL_RE.match(repo_or_url) else "mega"

    if src == "github":
        if _WEIGHT_URL_RE.match(repo_or_url):
            return load_state_dict_from_url(repo_or_url)
        # github repo holding a bare state-dict entrypoint is rare; route
        # through the entrypoint loader when a filename/entrypoint is given.
        if filename is not None:
            obj = load(repo_or_url, filename, ref=ref, **load_kwargs)
            return obj
        raise ValueError("github source needs a full checkpoint URL or an entrypoint name")

    client = _mega_client(endpoint, token)
    cache_root = get_dir() / "mega" / repo_or_url
    if filename is not None:
        path = client.download_file(repo_or_url, filename, local_dir=cache_root, revision=revision)
        return _mega_load_state_dict([path], device, load_kwargs)

    local_dir = client.snapshot_download(repo_or_url, local_dir=cache_root, revision=revision)
    paths = sorted(p for p in local_dir.rglob("*") if p.is_file() and p.suffix.lower() in _WEIGHT_EXTS)
    if not paths:
        raise RuntimeError(f"No weight files found in MEGA repo '{repo_or_url}' (revision={revision})")
    return _mega_load_state_dict(paths, device, load_kwargs)


def load_model(
    repo_or_url: str,
    filename: str | None = None,
    *,
    source: str = "auto",
    device: str = "cpu",
    revision: str = "main",
    endpoint=None,
    token=None,
    ref: str | None = None,
    model=None,
    model_class=None,
    model_kwargs: dict | None = None,
    strict: bool = True,
    assign: bool = False,
    **load_kwargs,
):
    """Loads weights and returns a ready-to-run model (mega or github).

    Architecture resolution (mega backend): ``model`` instance >
    ``model_class`` callable/dotted-path > repository metadata
    (``model.class`` / ``model.init.*`` via megatensors).
    For the github backend this is exactly ``torch.hub.load``.
    """
    src = source
    if src == "auto":
        src = "github" if (_WEIGHT_URL_RE.match(repo_or_url) or "/" in repo_or_url and filename is None) else "mega"

    if src == "github":
        if model is not None or model_class is not None or model_kwargs is not None:
            raise ValueError("github source resolves architecture via the repo entrypoint")
        entry = filename if filename is not None else repo_or_url.rsplit("/", 1)[-1]
        return load(repo_or_url, entry, ref=ref, **load_kwargs)

    client = _mega_client(endpoint, token)
    cache_root = get_dir() / "mega" / repo_or_url
    import megatensors

    if model is not None:
        sd = load_state_dict(
            repo_or_url, filename, source="mega", device=device, revision=revision,
            endpoint=endpoint, token=token, **load_kwargs,
        )
        try:
            model.load_state_dict(sd, strict=strict, assign=assign)
        except TypeError:
            model.load_state_dict(sd, strict=strict)
        if hasattr(model, "to") and device != "cpu":
            model = model.to(device)
        return model

    if filename is not None:
        paths = [client.download_file(repo_or_url, filename, local_dir=cache_root, revision=revision)]
    else:
        local_dir = client.snapshot_download(repo_or_url, local_dir=cache_root, revision=revision)
        index = local_dir / ".mega.index.json"
        paths = sorted(p for p in local_dir.rglob("*") if p.is_file() and p.suffix.lower() in _WEIGHT_EXTS)
        if index.exists():
            paths = [index] + paths
    if not paths:
        raise RuntimeError(f"No weight files found in MEGA repo '{repo_or_url}' (revision={revision})")

    return megatensors.load_model(
        [str(p) for p in paths],
        device=device,
        framework="tensorplay",
        model_class=model_class,
        model_kwargs=model_kwargs,
        strict=strict,
        assign=assign,
        **load_kwargs,
    )

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import logging
import shutil
import sys
import time
import uuid
import warnings
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

__all__ = [
    "download_url_to_file",
    "get_dir",
    "help",
    "list",
    "load",
    "load_state_dict",
    "load_state_dict_from_url",
    "load_model",
    "set_dir",
    "snapshot_download",
    "list_entrypoints",
]

logger = logging.getLogger(__name__)

# Cache directory resolution: the hub tree lives under $TENSORPLAY_HOME, or
# under $XDG_CACHE_HOME/tensorplay, or ~/.cache/tensorplay when neither is set.
ENV_HOME = "TENSORPLAY_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
ENV_GITHUB_TOKEN = "GITHUB_TOKEN"

VAR_DEPENDENCY = "dependencies"
MODULE_HUBCONF = "hubconf.py"
HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")

# Matches checkpoint URLs pointing directly at a weight file.
_WEIGHT_URL_RE = re.compile(r"^https?://.*\.(pth|pt|ckpt|safetensors|mst)([?#].*)?$", re.I)

# Repositories owned by these accounts are trusted without prompting. Any other
# owner needs explicit acknowledgement on first download.
_TRUSTED_REPO_OWNERS: tuple[str, ...] = ()

_hub_dir: Optional[Path] = None


def get_dir() -> Path:
    """Get the TensorPlay Hub cache directory used for storing downloaded models & weights."""
    if _hub_dir is not None:
        return _hub_dir
    env_home = os.getenv(ENV_HOME)
    if env_home:
        base = Path(env_home).expanduser()
    else:
        xdg = os.getenv(ENV_XDG_CACHE_HOME)
        base = (
            Path(xdg).expanduser() / "tensorplay"
            if xdg
            else Path.home() / ".cache" / "tensorplay"
        )
    return base / "hub"


def set_dir(d: Union[str, Path]) -> None:
    r"""
    Optionally set the TensorPlay Hub directory used to save downloaded models & weights.

    Args:
        d (str): path to a local folder to save downloaded models & weights.
    """
    if not isinstance(d, (str, Path)):
        raise TypeError(f"Expected directory path to be str or Path, but got {type(d).__name__}.")
    global _hub_dir
    _hub_dir = Path(d).expanduser()

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
# Repository resolution, caching, trust and hubconf loading
# ---------------------------------------------------------------------------


def _parse_repo_info(github: str) -> tuple[str, str, str | None]:
    """Split 'owner/name[:ref]' into its parts, defaulting the ref to main/master."""
    if ":" in github:
        repo_info, ref = github.split(":", 1)
    else:
        repo_info, ref = github, None
    if "/" not in repo_info:
        raise ValueError(f"Invalid repo identifier {github!r}; expected 'owner/name[:ref]'")
    repo_owner, repo_name = repo_info.split("/", 1)

    if ref is None:
        # Default branch: main when it exists, otherwise master.
        try:
            with urlopen(f"https://github.com/{repo_owner}/{repo_name}/tree/main/"):
                ref = "main"
        except HTTPError as e:
            if e.code != 404:
                raise
            ref = "master"
        except URLError as e:
            # Offline: fall back to whatever branch is already in the cache.
            for possible_ref in ("main", "master"):
                if (get_dir() / f"{repo_owner}_{repo_name}_{possible_ref}").exists():
                    ref = possible_ref
                    break
            if ref is None:
                raise RuntimeError(
                    "No internet connection and the repo was not found in the "
                    f"cache ({get_dir()})"
                ) from e
    return repo_owner, repo_name, ref


def _read_url(request: Request) -> str:
    with urlopen(request) as r:
        return r.read().decode(r.headers.get_content_charset("utf-8"))


def _ref_exists_in_repo(repo_owner: str, repo_name: str, ref: str) -> bool:
    # The ref must belong to the repository owner; a fork could otherwise smuggle
    # code in under a trusted owner's ref.
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get(ENV_GITHUB_TOKEN)
    if token is not None:
        headers["Authorization"] = f"token {token}"
    for url_prefix in (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/branches",
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/tags",
    ):
        page = 0
        while True:
            page += 1
            url = Request(f"{url_prefix}?per_page=100&page={page}", headers=headers)
            try:
                response = json.loads(_read_url(url))
            except HTTPError:
                # The token may simply lack permissions; retry without it.
                headers.pop("Authorization", None)
                response = json.loads(_read_url(url))
            if not response:
                break
            for br in response:
                if br["name"] == ref or br["commit"]["sha"].startswith(ref):
                    return True
    return False


def _validate_ref(repo_owner: str, repo_name: str, ref: str) -> None:
    if not _ref_exists_in_repo(repo_owner, repo_name, ref):
        raise ValueError(
            f"Cannot find {ref} in https://github.com/{repo_owner}/{repo_name}. "
            "If it is a commit from a forked repo, call load() with the fork directly "
            "and pass skip_validation=True."
        )


def _check_repo_is_trusted(repo_owner: str, repo_name: str, owner_name_branch: str,
                           trust_repo) -> None:
    hub_dir = get_dir()
    filepath = hub_dir / "trusted_list"

    if not filepath.exists():
        filepath.touch()
    trusted_repos = tuple(line.strip() for line in filepath.read_text().splitlines())

    # Repos already present in the cache count as trusted: they were downloaded
    # on purpose previously.
    trusted_legacy = {d.name for d in hub_dir.iterdir() if d.is_dir()}

    owner_name = f"{repo_owner}_{repo_name}"
    is_trusted = (
        owner_name in trusted_repos
        or owner_name_branch in trusted_legacy
        or repo_owner in _TRUSTED_REPO_OWNERS
    )

    if (trust_repo is False) or (trust_repo == "check" and not is_trusted):
        response = input(
            f"The repository {owner_name} is not on the trusted list and cannot be "
            "downloaded. Do you trust this repository and wish to add it to the "
            "trusted list of repositories (y/N)?"
        )
        if response.lower() not in ("y", "yes"):
            raise RuntimeError("Untrusted repository.")
        is_trusted = True

    if trust_repo is True and not is_trusted:
        with open(filepath, "a") as f:
            f.write(owner_name + "\n")


def _git_archive_link(repo_owner: str, repo_name: str, ref: str) -> str:
    return f"https://github.com/{repo_owner}/{repo_name}/zipball/{ref}"


def _safe_extract_zip(zip_file, extract_to):
    """Extract an archive, rejecting entries that escape the target directory."""
    extract_to = Path(extract_to).resolve(strict=False)

    for member in zip_file.infolist():
        filename = os.path.normpath(member.filename)

        if filename.startswith(("/", "\\")):
            raise ValueError(f"Archive entry has absolute path: {member.filename}")
        if len(filename) >= 2 and filename[1] == ":" and filename[0].isalpha():
            raise ValueError(f"Archive entry has absolute path: {member.filename}")
        if ".." in re.split(r"[/\\]", filename):
            raise ValueError(f"Archive entry contains directory traversal: {member.filename}")

        out = (extract_to / filename).resolve(strict=False)
        if not out.is_relative_to(extract_to):
            raise ValueError(f"Archive entry escapes target directory: {member.filename}")

        zip_file.extract(member, extract_to)


def _get_cache_or_reload(
    github: str,
    force_reload: bool,
    trust_repo,
    verbose: bool = True,
    skip_validation: bool = False,
) -> Path:
    """Download (or reuse) the repo under the hub cache and return its directory."""
    hub_dir = get_dir()
    hub_dir.mkdir(parents=True, exist_ok=True)
    repo_owner, repo_name, ref = _parse_repo_info(github)
    normalized_br = ref.replace("/", "_")
    repo_dir = hub_dir / "_".join([repo_owner, repo_name, normalized_br])

    _check_repo_is_trusted(repo_owner, repo_name, repo_dir.name, trust_repo)

    if (not force_reload) and repo_dir.exists():
        if verbose:
            print(f"Using cache found in {repo_dir}", file=sys.stderr)
        return repo_dir

    if not skip_validation:
        _validate_ref(repo_owner, repo_name, ref)

    cached_file = hub_dir / f"{normalized_br}.zip"
    cached_file.unlink(missing_ok=True)

    url = _git_archive_link(repo_owner, repo_name, ref)
    try:
        print(f'Downloading: "{url}" to {cached_file}')
        download_url_to_file(url, str(cached_file), progress=False)
    except HTTPError as err:
        if err.code != 300:
            raise
        # A 300 (Multiple Choices) usually means the ref names both a tag and a
        # branch. Follow git's convention: assume the branch.
        warnings.warn(
            f"The ref {ref} is ambiguous: it may be both a tag and a branch. "
            "Assuming it is a branch; pass refs/heads/<ref> or refs/tags/<ref> "
            "to be explicit (may require skip_validation=True).",
            stacklevel=2,
        )
        url = _git_archive_link(repo_owner, repo_name, f"refs/heads/{ref}")
        download_url_to_file(url, str(cached_file), progress=False)

    with zipfile.ZipFile(cached_file) as zf:
        # The archive's first entry is the top-level directory (typically
        # 'repo-<sha>/'); derive it from the first path component so archives
        # without an explicit directory entry also work.
        extracted_name = zf.infolist()[0].filename.split("/", 1)[0]
        _safe_extract_zip(zf, hub_dir)
    cached_file.unlink(missing_ok=True)

    extracted_repo = hub_dir / extracted_name
    if extracted_repo != repo_dir:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        shutil.move(str(extracted_repo), str(repo_dir))
    return repo_dir


# ---------------------------------------------------------------------------
# hubconf discovery, dependency checks and entrypoint loading
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _add_to_sys_path(path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path.remove(str(path))


def _import_module(name: str, path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise AssertionError(f"failed to load spec from {path}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise AssertionError(f"no loader for {path}")
    spec.loader.exec_module(module)
    return module


def _check_module_exists(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _check_dependencies(m) -> None:
    dependencies = getattr(m, VAR_DEPENDENCY, None)
    if dependencies is not None:
        missing = [pkg for pkg in dependencies if not _check_module_exists(pkg)]
        if missing:
            raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")


def _load_entry_from_hubconf(m, model):
    if not isinstance(model, str):
        raise ValueError("Invalid input: model should be a string of function name")
    _check_dependencies(m)
    fn = getattr(m, model, None)
    if fn is None or not callable(fn):
        raise RuntimeError(f"Cannot find callable {model} in hubconf")
    return fn


def _load_hub_module(repo_dir: Path):
    with _add_to_sys_path(repo_dir):
        return _import_module(MODULE_HUBCONF, str(repo_dir / MODULE_HUBCONF))


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def list(
    github,
    force_reload=False,
    skip_validation=False,
    trust_repo="check",
    verbose=True,
):
    """List all callable entrypoints available in the repo specified by ``github``.

    Args:
        github (str): 'owner/name[:ref]' with an optional ref (tag or branch).
            If the ref is omitted, the default branch is used (``main`` if it
            exists, otherwise ``master``).
        force_reload (bool): discard the cache and force a fresh download.
        skip_validation (bool): skip the check that the ref belongs to the
            repository owner.
        trust_repo (bool or str): ``"check"``, ``True`` or ``False``; how to
            treat repositories that are not on the trusted list.
        verbose (bool): mute messages about hitting local caches.
    """
    repo_dir = _get_cache_or_reload(
        github, force_reload, trust_repo, verbose=verbose, skip_validation=skip_validation
    )
    hub_module = _load_hub_module(repo_dir)
    return [
        f
        for f in dir(hub_module)
        if callable(getattr(hub_module, f)) and not f.startswith("_")
    ]


def help(github, model, force_reload=False, skip_validation=False, trust_repo="check"):
    """Show the docstring of entrypoint ``model``."""
    repo_dir = _get_cache_or_reload(
        github, force_reload, trust_repo, verbose=True, skip_validation=skip_validation
    )
    hub_module = _load_hub_module(repo_dir)
    entry = _load_entry_from_hubconf(hub_module, model)
    return entry.__doc__


def load(
    repo_or_dir,
    model,
    *args,
    source="github",
    trust_repo="check",
    force_reload=False,
    verbose=True,
    skip_validation=False,
    ref=None,
    **kwargs,
):
    """Load an entrypoint from a GitHub repo or a local directory.

    Args:
        repo_or_dir (str): with ``source='github'``, 'owner/name[:ref]'; with
            ``source='local'``, a path to a directory containing ``hubconf.py``.
        model (str): name of a callable (entrypoint) defined in ``hubconf.py``.
        source (str): 'github' or 'local'.
        trust_repo (bool or str): ``"check"``, ``True`` or ``False``; how to
            treat repositories that are not on the trusted list.
        force_reload (bool): discard the cache and force a fresh download
            (no effect with ``source='local'``).
        verbose (bool): mute messages about hitting local caches.
        skip_validation (bool): skip the check that the ref belongs to the
            repository owner.
        ref (str): optional tag or branch; equivalent to appending ``:ref`` to
            ``repo_or_dir``.
    """
    source = source.lower()
    if source not in ("github", "local"):
        raise ValueError(f'Unknown source: "{source}". Allowed values: "github" | "local".')

    if ref is not None:
        if ":" in repo_or_dir:
            raise ValueError("ref must not be given when repo_or_dir already contains ':'")
        repo_or_dir = f"{repo_or_dir}:{ref}"

    if source == "github":
        repo_or_dir = _get_cache_or_reload(
            repo_or_dir,
            force_reload,
            trust_repo,
            verbose=verbose,
            skip_validation=skip_validation,
        )
    return _load_local(repo_or_dir, model, *args, **kwargs)


def _load_local(hubconf_dir, model, *args, **kwargs):
    hub_module = _load_hub_module(Path(hubconf_dir))
    entry = _load_entry_from_hubconf(hub_module, model)
    return entry(*args, **kwargs)


def list_entrypoints(repo_id: str, ref: str | None = None, force_reload: bool = False,
                     **kwargs) -> list[str]:
    """Compatibility variant of :func:`list` taking the ref as a separate argument."""
    github = f"{repo_id}:{ref}" if ref else repo_id
    return list(github, force_reload=force_reload, **kwargs)


def load_state_dict_from_url(
    url: str,
    model_dir: Union[str, Path, None] = None,
    map_location=None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: Optional[str] = None,
    weights_only: bool = True,
) -> dict:
    """Load a serialized object from the given URL, caching it under ``model_dir``.

    Args:
        url (str): URL of the object to download.
        model_dir (str | Path | None): directory to cache in; defaults to
            ``<get_dir()>/checkpoints``.
        map_location: storage remapping passed to the loader.
        progress (bool): whether to display a progress bar.
        check_hash (bool): require the filename to carry a ``-<sha256-prefix>``
            suffix and verify the file against it.
        file_name (str | None): destination file name; taken from the URL when
            not given.
        weights_only (bool): restrict loading to registered data types; safer
            for untrusted sources (see ``tensorplay.load``).
    """
    if model_dir is None:
        model_dir = get_dir() / "checkpoints"
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    filename = file_name or os.path.basename(urlparse(url).path)
    cached_file = model_dir / filename
    if not cached_file.exists():
        print(f'Downloading: "{url}" to {cached_file}')
        hash_prefix = None
        if check_hash:
            m = HASH_REGEX.search(filename)
            hash_prefix = m.group(1) if m else None
        download_url_to_file(url, str(cached_file), hash_prefix=hash_prefix, progress=progress)

    import tensorplay as tp
    return tp.load(str(cached_file), map_location=map_location, weights_only=weights_only)


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
        github = f"{repo_id}:{ref or revision}" if (ref or revision) else repo_id
        return _get_cache_or_reload(
            github, force_reload=False, trust_repo=True, verbose=False, skip_validation=True
        )
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
            return load(repo_or_url, filename, ref=ref, trust_repo=True, **load_kwargs)
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
    """
    src = source
    if src == "auto":
        src = "github" if (_WEIGHT_URL_RE.match(repo_or_url) or "/" in repo_or_url and filename is None) else "mega"

    if src == "github":
        if model is not None or model_class is not None or model_kwargs is not None:
            raise ValueError("github source resolves architecture via the repo entrypoint")
        entry = filename if filename is not None else repo_or_url.rsplit("/", 1)[-1]
        return load(repo_or_url, entry, ref=ref, trust_repo=True, **load_kwargs)

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

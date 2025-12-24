import hashlib
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

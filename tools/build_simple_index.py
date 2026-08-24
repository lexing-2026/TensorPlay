#!/usr/bin/env python3
"""Build a PEP 503-compatible static index for CUDA wheel releases."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import zipfile
from email.parser import Parser
from pathlib import Path
from urllib.parse import quote

from packaging.utils import canonicalize_name, parse_wheel_filename


def _read_release_records(path: Path) -> list[dict]:
    """Read GitHub release objects written as JSON or JSON Lines."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        records: list[dict] = []
        for item in value:
            if isinstance(item, list):
                records.extend(item)
            elif isinstance(item, dict):
                records.append(item)
        return records
    raise ValueError(f"Unsupported release metadata in {path}")


def _wheel_metadata(path: Path) -> tuple[str, str | None]:
    name, _version, _build, _tags = parse_wheel_filename(path.name)
    requires_python: str | None = None
    with zipfile.ZipFile(path) as wheel:
        metadata_files = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if metadata_files:
            metadata = Parser().parsestr(wheel.read(metadata_files[0]).decode("utf-8"))
            requires_python = metadata.get("Requires-Python")
    return canonicalize_name(str(name)), requires_python


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fallback_asset_url(base_url: str, tag: str, filename: str) -> str:
    return f"{base_url.rstrip('/')}/{quote(tag, safe='')}/{quote(filename, safe='')}"


def _matches_variant(filename: str, variant: str) -> bool:
    """Match CUDA-local wheel versions, with a legacy cu124 fallback."""
    if f"+{variant}" in filename:
        return True
    return variant == "cu124" and not re.search(r"\+cu[0-9]+(?:[-_.]|$)", filename)


def _release_wheels(
    records: list[dict],
    release_base_url: str,
    variant: str,
) -> dict[str, list[dict[str, str | None]]]:
    packages: dict[str, list[dict[str, str | None]]] = {}
    for release in records:
        if release.get("draft"):
            continue
        tag = str(release.get("tag_name", "")).strip()
        if not tag:
            continue
        for asset in release.get("assets", []):
            filename = str(asset.get("name", ""))
            if not filename.lower().endswith(".whl"):
                continue
            if not _matches_variant(filename, variant):
                continue
            try:
                package_name, _version, _build, _tags = parse_wheel_filename(filename)
            except Exception as error:
                raise ValueError(f"Invalid wheel asset name: {filename}") from error
            project = canonicalize_name(str(package_name))
            url = asset.get("browser_download_url") or _fallback_asset_url(release_base_url, tag, filename)
            packages.setdefault(project, []).append(
                {"filename": filename, "url": str(url), "sha256": None, "requires_python": None}
            )
    return packages


def _add_current_wheels(
    packages: dict[str, list[dict[str, str | None]]],
    dist_dir: Path,
    release_base_url: str,
    release_tag: str,
    variant: str,
) -> None:
    for path in sorted(dist_dir.glob("*.whl")):
        if not _matches_variant(path.name, variant):
            continue
        project, requires_python = _wheel_metadata(path)
        filename = path.name
        entry = {
            "filename": filename,
            "url": _fallback_asset_url(release_base_url, release_tag, filename),
            "sha256": _sha256(path),
            "requires_python": requires_python,
        }
        existing = [item for item in packages.setdefault(project, []) if item["filename"] != filename]
        existing.append(entry)
        packages[project] = existing


def _package_index(entries: list[dict[str, str | None]]) -> str:
    links = []
    for entry in sorted(entries, key=lambda item: item["filename"] or ""):
        filename = entry["filename"] or ""
        fragment = f"#sha256={entry['sha256']}" if entry.get("sha256") else ""
        requires_python = entry.get("requires_python")
        requires_attr = (
            f' data-requires-python="{html.escape(requires_python, quote=True)}"' if requires_python else ""
        )
        links.append(
            f'<a href="{html.escape(str(entry["url"]), quote=True)}{fragment}"{requires_attr}>'
            f"{html.escape(filename)}</a>"
        )
    return "<!doctype html>\n<html><body>\n" + "\n".join(links) + "\n</body></html>\n"


def _root_index(projects: list[str], variant: str) -> str:
    links = "\n".join(f'<a href="{html.escape(project)}/">{html.escape(project)}</a>' for project in projects)
    return (
        "<!doctype html>\n<html><body>\n"
        f"<h1>TensorPlay {html.escape(variant)} wheel index</h1>\n"
        f"{links}\n</body></html>\n"
    )


def _landing_index(variants: list[str]) -> str:
    links = "\n".join(
        f'<p><a href="whl/{html.escape(variant)}/">{html.escape(variant)} index</a></p>'
        for variant in variants
    )
    return (
        "<!doctype html>\n<html><body>\n"
        "<h1>TensorPlay CUDA wheels</h1>\n"
        f"{links}\n"
        "</body></html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True, help="Directory containing the current CUDA wheels")
    parser.add_argument("--output-dir", type=Path, required=True, help="Pages output directory")
    parser.add_argument("--release-tag", required=True, help="Current Git tag, for example v1.0.0")
    parser.add_argument(
        "--release-base-url",
        required=True,
        help="GitHub release download base, for example https://github.com/org/repo/releases/download",
    )
    parser.add_argument("--releases-json", type=Path, required=True, help="GitHub Releases JSON or JSONL metadata")
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        default=None,
        help="CUDA index variant, repeat for multiple variants, for example cu124",
    )
    args = parser.parse_args()

    variants = args.variants or ["cu124"]
    invalid = [variant for variant in variants if not re.fullmatch(r"cu[0-9]+", variant)]
    if invalid:
        raise SystemExit(f"Invalid CUDA variant(s): {invalid!r}")
    if not args.dist_dir.is_dir():
        raise SystemExit(f"Wheel directory does not exist: {args.dist_dir}")

    release_records = _read_release_records(args.releases_json)

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    for variant in variants:
        packages = _release_wheels(release_records, args.release_base_url, variant)
        _add_current_wheels(packages, args.dist_dir, args.release_base_url, args.release_tag, variant)
        if not packages:
            raise SystemExit(f"No CUDA wheel assets found for {variant}")

        index_root = args.output_dir / "whl" / variant
        index_root.mkdir(parents=True)
        (index_root / "index.html").write_text(_root_index(sorted(packages), variant), encoding="utf-8")
        for project, entries in sorted(packages.items()):
            project_dir = index_root / project
            project_dir.mkdir()
            (project_dir / "index.html").write_text(_package_index(entries), encoding="utf-8")

    (args.output_dir / "index.html").write_text(_landing_index(variants), encoding="utf-8")
    (args.output_dir / "_headers").write_text(
        "/whl/*\n"
        "  Cache-Control: public, max-age=300, must-revalidate\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

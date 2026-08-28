#!/usr/bin/env python3
"""Aggregate a release-notes draft for TensorPlay, pytorch-style.

Two sources are merged:

  1. Merged PRs between two refs, bucketed by their "release notes: *" label
     (the same family the labeler applies from paths).
  2. Conventional commits in the same range, bucketed by scope (catches
     direct pushes that never went through a PR).

Commits carrying a "BREAKING CHANGE:" footer are additionally collected into
the Backwards Incompatible Changes section. The output follows
docs/release-notes/TEMPLATE.md and is a *draft*: curate it by hand, then
commit it as docs/release-notes/vX.Y.0.md so publish.yml picks it up via
`gh release create --notes-file`.

Usage:
    python tools/collect_release_notes.py                       # latest tag..HEAD
    python tools/collect_release_notes.py --from v0.1.2 --to main
    python tools/collect_release_notes.py --output draft.md

Requires git; requires an authenticated `gh` for the PR source (the commit
source still works without it).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.commit_schema import HEADER_RE, SCOPE_ENUM  # noqa: E402

RELEASE_NOTES_LABEL_PREFIX = "release notes: "

# commit scope -> release-notes template section
SECTION_BY_TYPE = {
    "feat": "New Features",
    "fix": "Bug fixes",
    "perf": "Performance",
    "refactor": "Improvements",
    "style": "Improvements",
    "docs": "Documentation",
    "build": "Developers",
    "ci": "Developers",
    "chore": "Developers",
    "revert": "Developers",
    "test": "Developers",
}

SECTIONS = [
    "Highlights",
    "Backwards Incompatible Changes",
    "Deprecations",
    "New Features",
    "Improvements",
    "Bug fixes",
    "Performance",
    "Documentation",
    "Developers",
]

PR_NUMBER_RE = re.compile(r"\(#(\d+)\)\s*$")
MERGE_PR_RE = re.compile(r"^Merge pull request #(\d+)")


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(command)}\n{result.stderr.strip()}")
    return result.stdout


def default_repo() -> str:
    url = run(["git", "remote", "get-url", "origin"]).strip()
    match = re.search(r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$", url)
    if not match:
        raise SystemExit(f"cannot derive owner/repo from remote URL: {url}")
    return match.group(1)


def default_from_rev() -> str:
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        print("warning: no tag found; aggregating the full history", file=sys.stderr)
        return ""
    return result.stdout.strip()


def commit_date(rev: str) -> str:
    return run(["git", "log", "-1", "--format=%cI", rev]).strip()


def collect_commits(rev_range: str) -> tuple[dict[str, list[str]], list[tuple[str, str, str]], set[str]]:
    """Return (entries by scope, breaking changes, PR numbers seen in subjects)."""
    entries: dict[str, list[str]] = {scope: [] for scope in SCOPE_ENUM}
    breaking: list[tuple[str, str, str]] = []
    pr_numbers: set[str] = set()

    log = run(["git", "log", "--no-merges", "--format=%H%x00%s%x00%b%x01", rev_range])
    for record in log.split("\x01"):
        record = record.strip("\n")
        if not record.strip():
            continue
        sha, _, rest = record.partition("\x00")
        subject, _, body = rest.partition("\x00")
        subject = subject.strip()

        for matcher in (PR_NUMBER_RE, MERGE_PR_RE):
            match = matcher.search(subject)
            if match:
                pr_numbers.add(match.group(1))

        header = HEADER_RE.match(subject)
        if not header:
            continue
        scope = header.group("scope")
        if scope not in SCOPE_ENUM:
            continue
        entries[scope].append(f"- {subject} (`{sha[:8]}`)")

        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("BREAKING CHANGE:") or stripped.startswith("BREAKING-CHANGE:"):
                detail = stripped.split(":", 1)[1].strip()
                breaking.append((sha[:8], subject, detail))

    return entries, breaking, pr_numbers


def collect_prs(repo: str, from_date: str, to_date: str) -> dict[str, list[str]]:
    """Return merged PR entries bucketed by their 'release notes: *' label.

    Uses the pulls list endpoint (not the search API, which fine-grained PATs
    may not query) and filters by merged_at client-side.
    """
    buckets: dict[str, list[str]] = {}
    page = 1
    while page <= 50:
        result = subprocess.run(
            [
                "gh", "api", f"repos/{repo}/pulls",
                "-f", "state=closed",
                "-f", "sort=created",
                "-f", "direction=desc",
                "-F", f"page={page}",
                "-F", "per_page=100",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"warning: gh pulls list failed ({result.stderr.strip()}); "
                "continuing with the commit source only",
                file=sys.stderr,
            )
            return buckets
        items = json.loads(result.stdout)
        if not items:
            break
        for pr in items:
            merged_at = pr.get("merged_at")
            if not merged_at:
                continue
            merged_day = merged_at[:10]
            if merged_day < from_date or merged_day > to_date:
                continue
            labels = [label["name"] for label in pr.get("labels", [])]
            for label in labels:
                if label.startswith(RELEASE_NOTES_LABEL_PREFIX):
                    buckets.setdefault(label, []).append(f"- {pr['title']} (#{pr['number']})")
        if len(items) < 100:
            break
        page += 1
    for label in buckets:
        buckets[label].sort()
    return buckets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=None, help="owner/repo (default: derived from the origin remote)")
    parser.add_argument("--from", dest="from_rev", default=None, help="start rev exclusive (default: latest tag)")
    parser.add_argument("--to", dest="to_rev", default="HEAD", help="end rev inclusive (default: HEAD)")
    parser.add_argument("--output", default=None, help="write the draft to this file instead of stdout")
    args = parser.parse_args()

    repo = args.repo or default_repo()
    from_rev = args.from_rev if args.from_rev is not None else default_from_rev()
    rev_range = f"{from_rev}..{args.to_rev}" if from_rev else args.to_rev

    entries, breaking, _pr_numbers = collect_commits(rev_range)

    pr_buckets: dict[str, list[str]] = {}
    try:
        from_date = commit_date(from_rev) if from_rev else "1970-01-01T00:00:00Z"
        to_date = commit_date(args.to_rev)
        pr_buckets = collect_prs(repo, from_date[:10], to_date[:10])
    except SystemExit as error:
        print(f"warning: {error}", file=sys.stderr)

    lines: list[str] = []
    lines.append(f"<!-- DRAFT generated by tools/collect_release_notes.py --from {from_rev or '<root>'} --to {args.to_rev} -->")
    lines.append("<!-- Curate by hand into docs/release-notes/vX.Y.0.md (see TEMPLATE.md). -->")
    lines.append("")

    lines.append("# Highlights")
    lines.append("")
    lines.append("<!-- Curate 3-10 headline items of the release, one line each. -->")
    lines.append("")

    lines.append("# Backwards Incompatible Changes")
    lines.append("")
    if breaking:
        for sha, subject, detail in breaking:
            lines.append(f"- {subject} (`{sha}`)" + (f" — {detail}" if detail else ""))
    else:
        lines.append("None.")
    lines.append("")

    for section in SECTIONS:
        if section in ("Highlights", "Backwards Incompatible Changes"):
            continue
        lines.append(f"# {section}")
        lines.append("")
        if section == "Deprecations":
            lines.append("None.")
            lines.append("")
            continue
        section_scopes = [scope for scope, typ in SECTION_BY_TYPE.items() if typ == section and scope in entries]
        empty = True
        for scope in sorted(set(section_scopes)):
            if entries[scope]:
                empty = False
                lines.append(f"## {scope}")
                lines.append("")
                lines.extend(entries[scope])
                lines.append("")
        if empty:
            lines.append("None.")
            lines.append("")

    lines.append("# Merged pull requests by release-notes label")
    lines.append("")
    lines.append("<!-- Raw bucket list for curation; fold into the sections above. -->")
    lines.append("")
    if pr_buckets:
        for label in sorted(pr_buckets):
            lines.append(f"## {label}")
            lines.append("")
            lines.extend(pr_buckets[label])
            lines.append("")
    else:
        lines.append("No merged PRs with release-notes labels found in this range.")
        lines.append("")

    text = "\n".join(lines)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"draft written to {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

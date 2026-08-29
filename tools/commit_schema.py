#!/usr/bin/env python3
"""Conventional Commits schema validator for TensorPlay.

Single source of truth for the commit message convention, consumed by:

  - the commit-msg pre-commit hook        (.pre-commit-config.yaml)
  - the PR title check                    (.github/workflows/pr-title.yml)
  - the advisory main-branch commit check (.github/workflows/commit-lint.yml)

The scope-enum maps 1:1 to the "release notes: *" label family
(.github/labels.yml), so commit scopes, PR labels and release-notes sections
stay consistent.

Versioning is deliberately NOT handled here or by `cz bump`: versions follow
version.txt and tools/generate_tensorplay_version.py.

Usage:
    python tools/commit_schema.py --commit-msg-file .git/COMMIT_EDITMSG
    python tools/commit_schema.py --message "feat(compiler): add ..."
    python tools/commit_schema.py --range origin/main..HEAD
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

TYPE_ENUM = (
    "build",
    "chore",
    "ci",
    "docs",
    "feat",
    "fix",
    "perf",
    "refactor",
    "revert",
    "style",
    "test",
)

# 1:1 with the "release notes: *" label family in .github/labels.yml.
SCOPE_ENUM = (
    "frontend",
    "autograd",
    "compiler",
    "kernels",
    "cuda",
    "build",
    "docs",
)

# Types that must carry a scope so release-notes bucketing never loses them.
SCOPED_TYPES = ("feat", "fix")

SUBJECT_MAX = 100

HEADER_RE = re.compile(
    r"^(?P<type>[a-zA-Z]+)"
    r"(?:\((?P<scope>[a-zA-Z0-9_/-]+)\))?"
    r"(?P<breaking>!)?"
    r":\s(?P<subject>\S.*)$"
)

# Commit messages that never go through the convention.
EXEMPT_PREFIXES = (
    "Merge ",
    "Revert ",
    "fixup!",
    "squash!",
    "amend!",
)

SCHEMA_HELP = (
    "Expected: type(scope): subject\n"
    f"  type  := {' | '.join(TYPE_ENUM)}\n"
    f"  scope := {' | '.join(SCOPE_ENUM)}  (required for feat/fix)\n"
    "  Breaking changes: append '!' before ':' and add a "
    "'BREAKING CHANGE: ...' footer.\n"
    "  Examples:\n"
    "    feat(compiler): add triton fallback for stax autotune\n"
    "    fix(cuda): guard stream capture reentry\n"
    "    feat(frontend)!: rename tp.dim to tp.axis\n\n"
    "    BREAKING CHANGE: tp.dim removed, use tp.axis"
)


def validate_header(header: str) -> list[str]:
    """Validate one commit subject line; return a list of problems."""
    if any(header.startswith(prefix) for prefix in EXEMPT_PREFIXES):
        return []

    match = HEADER_RE.match(header)
    if not match:
        return [f"header does not match 'type(scope): subject'\n{SCHEMA_HELP}"]

    errors: list[str] = []
    commit_type = match.group("type").lower()
    scope = match.group("scope")
    subject = match.group("subject")

    if commit_type not in TYPE_ENUM:
        errors.append(f"unknown type '{commit_type}' (allowed: {', '.join(TYPE_ENUM)})")

    if scope is not None and scope not in SCOPE_ENUM:
        errors.append(
            f"unknown scope '{scope}' (allowed: {', '.join(SCOPE_ENUM)}; "
            "scopes map 1:1 to the 'release notes: *' labels)"
        )

    if commit_type in SCOPED_TYPES and scope is None:
        errors.append(f"type '{commit_type}' requires a scope, e.g. {commit_type}(kernels): ...")

    if len(header) > SUBJECT_MAX:
        errors.append(f"header is {len(header)} chars, keep it under {SUBJECT_MAX}")

    if subject.rstrip().endswith("."):
        errors.append("subject must not end with a period")

    return errors


def validate_message(message: str) -> list[str]:
    """Validate a full commit message (header is the first meaningful line)."""
    lines = [line for line in message.splitlines() if not line.startswith("#")]
    header = next((line.strip() for line in lines if line.strip()), "")
    if not header:
        return ["empty commit message"]
    return validate_header(header)


def read_commit_msg_file(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def list_range_commits(rev_range: str) -> list[tuple[str, str]]:
    """Return (sha, subject) pairs for a git revision range."""
    result = subprocess.run(
        ["git", "log", "--format=%H%x00%s", rev_range],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"git log failed for range '{rev_range}': {result.stderr.strip()}")
    pairs = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        sha, _, subject = line.partition("\x00")
        pairs.append((sha, subject))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--commit-msg-file", help="path to a commit message file (commit-msg hook)")
    group.add_argument("--message", help="a commit message / PR title to validate")
    group.add_argument("--range", dest="rev_range", help="git revision range to validate, e.g. origin/main..HEAD")
    args = parser.parse_args()

    if args.commit_msg_file:
        errors = validate_message(read_commit_msg_file(args.commit_msg_file))
        failures = [(args.commit_msg_file, errors)] if errors else []
    elif args.message is not None:
        errors = validate_message(args.message)
        failures = [(args.message, errors)] if errors else []
    else:
        failures = []
        for sha, subject in list_range_commits(args.rev_range):
            errors = validate_header(subject)
            if errors:
                failures.append((f"{sha[:8]} {subject}", errors))

    if not failures:
        print("commit schema: ok")
        return 0

    for where, errors in failures:
        print(f"commit schema: FAIL -> {where}", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
    print(f"\n{len(failures)} message(s) violate the Conventional Commits schema", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

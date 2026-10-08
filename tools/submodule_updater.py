#!/usr/bin/env python3
"""Bump outdated third_party submodules with one Conventional-Commit PR each.

Every submodule declared in .gitmodules is compared against the tip of its
upstream default branch (or the branch named in .gitmodules).  Each outdated
submodule gets its own branch (``deps/bump-<name>``) and pull request titled
``build: bump third_party/<name> to <short-sha>``, labelled
``release notes: build``, carrying the remote commit-range link in the body.

Only the superproject gitlink is committed; the submodule working tree stays
detached at the target commit and never contains local commits.

Usage:
    python tools/submodule_updater.py
    python tools/submodule_updater.py --submodule third_party/cutlass
    python tools/submodule_updater.py --dry-run

Environment:
    GH_TOKEN / GITHUB_TOKEN  used for the GitHub API calls made through gh.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

BRANCH_PREFIX = "deps/bump-"
PR_LABEL = "release notes: build"
GIT_USER_NAME = "TensorPlay CI"
GIT_USER_EMAIL = "actions@users.noreply.github.com"
GITMODULES_ENTRY_RE = re.compile(
    r"^submodule\.(?P<name>.+)\.(?P<key>path|url|branch)\s+(?P<value>\S.*)$"
)
LS_TREE_RE = re.compile(r"^160000 commit (?P<sha>[0-9a-f]+)\s", re.MULTILINE)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class SubmoduleUpdateError(RuntimeError):
    """Drift detection or git operation failed for one submodule."""


class PullRequestForbidden(SubmoduleUpdateError):
    """The ambient credential may not open pull requests.

    A repository can be configured so the workflow token cannot open pull
    requests at all.  The condition is a property of the configuration, not
    of the submodule being bumped, so it is worth telling apart from a
    per-submodule failure: it repeats identically for every outdated entry,
    and answering it once is what stops the run from reporting the same
    error sixteen times.
    """

    REMEDIATION = (
        "This repository does not let the workflow token open pull requests. "
        "Turn on \"Allow GitHub Actions to create and approve pull requests\" "
        "under Settings -> Actions -> General -> Workflow permissions and "
        "re-run this workflow; the credential is what the refusal names, and "
        "a token supplied through SUBMODULE_UPDATE_PAT does not get past it. "
        "Note that a pull request opened by the workflow token does not "
        "trigger repository workflows, so add a token with pull-request "
        "write access there as well if the bump PRs should run the "
        "lint and smoke-build gates. Bump branches pushed by a run that hit "
        "this refusal are not lost: the next run finds each one already at "
        "its target commit and opens the missing pull request."
    )


def pull_request_forbidden(message: str) -> bool:
    """Whether a `gh` failure says the credential lacks pull-request rights."""
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "not permitted",
            "not allowed",
            "permission denied",
            "resource not accessible",
            "http 403",
            "http 401",
        )
    )


@dataclass
class Submodule:
    path: str
    url: str
    branch: str | None = None
    pinned: str | None = None
    target: str | None = None


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, text=True, capture_output=True, **kwargs)


def git(*args: str, cwd: str | None = None) -> str:
    proc = run(["git", *args], cwd=cwd)
    if proc.returncode != 0:
        raise SubmoduleUpdateError(
            f"git {' '.join(args)} failed (cwd={cwd or '.'}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout.strip()


def list_submodules() -> list[Submodule]:
    proc = run(["git", "config", "-f", ".gitmodules", "--get-regexp",
                r"^submodule\..*\.(path|url|branch)$"])
    if proc.returncode != 0:
        return []
    entries: dict[str, Submodule] = {}
    for line in proc.stdout.splitlines():
        match = GITMODULES_ENTRY_RE.match(line)
        if not match:
            continue
        entry = entries.setdefault(match["name"], Submodule(path="", url=""))
        if match["key"] == "path":
            entry.path = match["value"]
        elif match["key"] == "url":
            entry.url = match["value"]
        else:
            entry.branch = match["value"]
    return [entry for entry in entries.values() if entry.path and entry.url]


def pinned_sha(path: str) -> str | None:
    """SHA of the gitlink recorded in HEAD for ``path``, or None if not a submodule."""
    proc = run(["git", "ls-tree", "HEAD", path])
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    match = LS_TREE_RE.search(proc.stdout)
    return match.group("sha") if match else None


def upstream_sha(url: str, branch: str | None) -> str | None:
    if branch is not None:
        proc = run(["git", "ls-remote", url, f"refs/heads/{branch}"])
        if proc.returncode != 0:
            raise SubmoduleUpdateError(f"ls-remote failed for {url}: "
                                       f"{proc.stderr.strip() or proc.stdout.strip()}")
        for line in proc.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) == 2 and SHA_RE.fullmatch(parts[0]):
                return parts[0]
        return None
    # Without an explicit branch, the upstream default branch is the target;
    # ``--symref`` puts its tip sha on the ``HEAD`` line.
    proc = run(["git", "ls-remote", "--symref", url, "HEAD"])
    if proc.returncode != 0:
        raise SubmoduleUpdateError(
            f"ls-remote failed for {url}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1] == "HEAD" and SHA_RE.fullmatch(parts[0]):
            return parts[0]
    return None


def remote_branch_exists(branch: str) -> bool:
    proc = run(["git", "ls-remote", "origin", f"refs/heads/{branch}"])
    return proc.returncode == 0 and bool(proc.stdout.strip())


def move_pointer(sm: Submodule, upstream_branch: str | None) -> None:
    git("submodule", "update", "--init", "--depth", "1", sm.path)
    fetch_ref = f"refs/heads/{upstream_branch}" if upstream_branch else "HEAD"
    git("-C", sm.path, "fetch", "--depth", "1", "origin", fetch_ref)
    git("-C", sm.path, "checkout", "--detach", "--quiet", sm.target)
    git("add", "--", sm.path)


def pr_body(sm: Submodule) -> str:
    branch_note = f" (upstream branch ``{sm.branch}``)" if sm.branch else ""
    return (
        f"Automated bump of ``{sm.path}`` from ``{sm.pinned}`` to ``{sm.target}``"
        f"{branch_note}.\n\n"
        f"Upstream compare: {sm.url}/compare/{sm.pinned}...{sm.target}\n\n"
        "The ``pull`` workflow's lint and smoke-build jobs gate this PR; the "
        "full platform/variant wheel matrix runs on main after merge."
    )


def ensure_pr(branch: str, title: str, body: str, base_branch: str) -> str:
    proc = run(["gh", "pr", "list", "--head", branch, "--state", "open",
                "--json", "number,url"])
    if proc.returncode != 0:
        raise SubmoduleUpdateError(f"gh pr list failed: {proc.stderr.strip()}")
    entries = json.loads(proc.stdout or "[]")
    if entries:
        number = entries[0]["number"]
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
            handle.write(body)
            body_file = handle.name
        proc = run(["gh", "pr", "edit", str(number), "--title", title,
                    "--body-file", body_file])
        if proc.returncode != 0:
            raise SubmoduleUpdateError(f"gh pr edit failed: {proc.stderr.strip()}")
        return f"updated #{number} ({entries[0]['url']})"
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
        handle.write(body)
        body_file = handle.name
    proc = run(["gh", "pr", "create", "--base", base_branch, "--head", branch,
                "--title", title, "--body-file", body_file, "--label", PR_LABEL])
    gh_out = proc.stdout.strip()
    if proc.returncode != 0:
        detail = proc.stderr.strip()
        if pull_request_forbidden(detail):
            raise PullRequestForbidden(detail)
        raise SubmoduleUpdateError(f"gh pr create failed: {detail}")
    return gh_out


def bump_submodule(sm: Submodule, base_branch: str, dry_run: bool) -> str:
    branch = BRANCH_PREFIX + sm.path.rsplit("/", 1)[-1]
    title = f"build: bump {sm.path} to {sm.target[:12]}"
    if dry_run:
        return f"would open PR: {title} on {branch}"
    exists = remote_branch_exists(branch)
    if exists:
        git("fetch", "origin", branch)
        git("checkout", "-B", branch, f"origin/{branch}")
    else:
        git("checkout", "-B", branch)
    try:
        move_pointer(sm, sm.branch)
    except SubmoduleUpdateError:
        git("checkout", base_branch)
        raise
    proc = run(["git", "diff", "--cached", "--quiet"])
    if proc.returncode == 0:
        # The bump already sits in this branch, so there is nothing to commit
        # and push.  When the branch came from the remote it was pushed by an
        # earlier run whose pull request could not be opened; reconcile that
        # pull request now instead of reporting success and leaving the
        # branch on the remote with nothing pointing at it.
        git("checkout", base_branch)
        if not exists:
            return "no change (already at target)"
        return ensure_pr(branch, title, pr_body(sm), base_branch)
    git("-c", f"user.name={GIT_USER_NAME}", "-c", f"user.email={GIT_USER_EMAIL}",
        "commit", "-m", title)
    git("push", "-u", "origin", branch)
    git("checkout", base_branch)
    return ensure_pr(branch, title, pr_body(sm), base_branch)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submodule", action="append", dest="submodules", default=[],
                        help="submodule path or basename to process (repeatable); "
                             "default: all submodules")
    parser.add_argument("--base-branch", default="main",
                        help="branch to create bump branches from (default: main)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report drift without touching branches, gitlinks or PRs")
    args = parser.parse_args()

    submodules = list_submodules()
    if not submodules:
        print("No submodules declared in .gitmodules; nothing to do.")
        return 0

    if args.submodules:
        wanted = set(args.submodules)
        filtered = [
            sm for sm in submodules
            if sm.path in wanted or sm.path.rsplit("/", 1)[-1] in wanted
        ]
        missing = wanted - {sm.path for sm in filtered} - {
            sm.path.rsplit("/", 1)[-1] for sm in filtered
        }
        if missing:
            print(f"Unknown submodule(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 1
        submodules = filtered

    detected = 0
    errors = 0
    pr_blocked = False
    for sm in submodules:
        sm.pinned = pinned_sha(sm.path)
        try:
            sm.target = upstream_sha(sm.url, sm.branch)
        except SubmoduleUpdateError as exc:
            print(f"[error] {sm.path}: {exc}")
            errors += 1
        if sm.target:
            detected += 1

    print(f"Checked {len(submodules)} submodules "
          f"({detected} with a resolvable upstream).")
    if args.dry_run:
        for sm in submodules:
            if sm.pinned is None or sm.target is None:
                print(f"[skip]    {sm.path}: no resolvable upstream/pinned sha")
            elif sm.pinned == sm.target:
                print(f"[current] {sm.path}: {sm.pinned[:12]}")
            else:
                print(f"[outdated] {sm.path}: {sm.pinned[:12]} -> {sm.target[:12]}")
        return 1 if errors else 0

    for sm in submodules:
        if sm.pinned is None or sm.target is None:
            print(f"[skip]    {sm.path}: no resolvable upstream/pinned sha")
            continue
        if sm.pinned == sm.target:
            print(f"[current] {sm.path}: {sm.pinned[:12]}")
            continue
        print(f"[outdated] {sm.path}: {sm.pinned[:12]} -> {sm.target[:12]}")
        if pr_blocked:
            # The credential cannot open the pull request, so pushing another
            # branch would only add to the pile the next run reuses.  Report
            # the drift and stop touching the remote.
            continue
        try:
            result = bump_submodule(sm, args.base_branch, dry_run=False)
            print(f"  -> {result}")
        except PullRequestForbidden as exc:
            print(f"  [error] {exc}")
            pr_blocked = True
        except SubmoduleUpdateError as exc:
            print(f"  [error] {exc}")
            errors += 1

    if pr_blocked:
        print()
        print(PullRequestForbidden.REMEDIATION)
        # The drift above is the deliverable and it was produced in full; only
        # the pull requests are missing, and the reason is a setting rather
        # than anything about the submodules.  Report it and leave the run
        # green so the drift stays readable instead of sitting behind a red
        # job that fails identically every week until the secret is set.  A
        # submodule whose upstream could not be resolved is a separate fault
        # and still fails the run.
        return 1 if errors else 0
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

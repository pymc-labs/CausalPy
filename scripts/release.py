"""Open a version-bump pull request, or tag the merged bump.

The only version literal is ``causalpy/version.py``. This command does not
write a static version into ``pyproject.toml`` and does not hand-edit
``uv.lock``.

``bump`` creates a branch, commits, pushes that branch, and opens a pull
request. It does not commit on ``main``, push ``main``, or create a tag.

``publish`` tags the merged bump and pushes that tag with the caller's git
credentials. It strips ``GITHUB_TOKEN`` and ``GH_TOKEN`` from the tag push.
It does not commit.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VERSION_PATH = Path("causalpy/version.py")
VERSION_LINE = re.compile(r'^__version__ = "([^"]+)"\s*$', re.M)
RELEASE_VERSION = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
PARTS = ("patch", "minor", "major")
TOKEN_ENV = ("GITHUB_TOKEN", "GH_TOKEN")


class ReleaseError(Exception):
    """A refused release action. The message names the fix."""


class CommandError(Exception):
    def __init__(self, cmd: list[str], result: subprocess.CompletedProcess[str]):
        self.cmd = cmd
        self.result = result
        detail = (result.stderr or result.stdout or "").strip()
        super().__init__(detail)


def run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise CommandError(cmd, result)
    return result


def git_env(*, strip_tokens: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    # Parsed git output must not contain color codes, even when the admin's
    # config sets color.ui or color.diff to always.
    env["GIT_CONFIG_COUNT"] = "2"
    env["GIT_CONFIG_KEY_0"] = "color.ui"
    env["GIT_CONFIG_VALUE_0"] = "false"
    env["GIT_CONFIG_KEY_1"] = "color.diff"
    env["GIT_CONFIG_VALUE_1"] = "false"
    if strip_tokens:
        for key in TOKEN_ENV:
            env.pop(key, None)
    return env


def git(
    root: Path,
    args: list[str],
    *,
    check: bool = True,
    strip_tokens: bool = False,
) -> subprocess.CompletedProcess[str]:
    return run(
        ["git", *args],
        cwd=root,
        env=git_env(strip_tokens=strip_tokens),
        check=check,
    )


def git_text(root: Path, args: list[str]) -> str:
    return git(root, args).stdout.strip()


def repo_root() -> Path:
    try:
        completed = run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path.cwd(),
            env=git_env(),
        )
    except CommandError as exc:
        raise ReleaseError(
            "Refusing to continue: this directory is not a git checkout.\n"
            "Fix: run from a CausalPy checkout."
        ) from exc
    return Path(completed.stdout.strip())


def current_branch(root: Path) -> str:
    return git_text(root, ["branch", "--show-current"])


def porcelain(root: Path) -> str:
    return git(root, ["status", "--porcelain"]).stdout


def read_version(text: str) -> str:
    found = VERSION_LINE.findall(text)
    if len(found) != 1:
        raise ReleaseError(
            "Refusing to continue: causalpy/version.py must contain exactly one "
            '__version__ = "..." literal.\n'
            "Fix: set a release version in causalpy/version.py first."
        )
    return found[0]


def read_version_at(root: Path, spec: str) -> str:
    try:
        text = git(root, ["show", spec]).stdout
    except CommandError as exc:
        raise ReleaseError(
            "Refusing to continue: could not read causalpy/version.py from "
            f"{spec}.\n"
            "Fix: set a release version in causalpy/version.py first."
        ) from exc
    return read_version(text)


def require_release_version(version: str, *, action: str) -> tuple[int, int, int]:
    match = RELEASE_VERSION.fullmatch(version)
    if match is None:
        if action == "bump":
            raise ReleaseError(
                f"Refusing to bump {version}: version is not MAJOR.MINOR.PATCH.\n"
                "Fix: set a release version in causalpy/version.py first. "
                "Do not guess a bump off a pre-release."
            )
        raise ReleaseError(
            f"Refusing to publish: tag {version} would not equal a "
            "MAJOR.MINOR.PATCH package version.\n"
            "Fix: set a release version in causalpy/version.py first. "
            "Do not guess a bump off a pre-release."
        )
    return tuple(int(part) for part in match.groups())


def next_version(version: str, part: str) -> str:
    major, minor, patch = require_release_version(version, action="bump")
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "major":
        return f"{major + 1}.0.0"
    raise ReleaseError(
        f"Refusing to bump: unknown part {part!r}.\n"
        "Fix: run make release-patch, make release-minor, or make release-major."
    )


def replace_version(text: str, new: str) -> str:
    if len(VERSION_LINE.findall(text)) != 1:
        raise ReleaseError(
            "Refusing to bump: causalpy/version.py must contain exactly one "
            "__version__ literal.\n"
            "Fix: set a release version in causalpy/version.py first."
        )

    def _replace(match: re.Match[str]) -> str:
        return f'__version__ = "{new}"'

    updated, count = VERSION_LINE.subn(_replace, text, count=1)
    if count != 1 or updated == text:
        raise ReleaseError(
            "Refusing to bump: the version literal did not change.\n"
            "Fix: set a release version in causalpy/version.py first."
        )
    return updated


def require_on_main(root: Path, *, action: str) -> None:
    branch = current_branch(root)
    if branch == "main":
        return
    shown = branch or "detached HEAD"
    if action == "bump":
        raise ReleaseError(
            f"Refusing to bump: current branch is {shown!r}, not main.\n"
            "Fix: check out a clean main that matches origin/main, then rerun. "
            "This target does not commit on main."
        )
    raise ReleaseError(
        f"Refusing to publish: current branch is {shown!r}, not main.\n"
        "Fix: check out main at the merged bump commit. This target does not commit."
    )


def require_clean(root: Path, *, action: str) -> None:
    if porcelain(root).strip() == "":
        return
    if action == "bump":
        raise ReleaseError(
            "Refusing to bump: working tree is not clean.\n"
            "Fix: commit or remove unrelated changes outside this command. "
            "This target does not commit on main."
        )
    raise ReleaseError(
        "Refusing to publish: working tree is not clean.\n"
        "Fix: put unrelated changes aside. This target does not commit."
    )


def require_origin(root: Path) -> None:
    remotes = git_text(root, ["remote"]).split()
    if "origin" not in remotes:
        raise ReleaseError(
            "Refusing to continue: no origin remote.\n"
            "Fix: add an origin remote that tracks main, then rerun."
        )


def fetch_main(root: Path, *, action: str) -> None:
    try:
        git(root, ["fetch", "origin", "+refs/heads/main:refs/remotes/origin/main"])
    except CommandError as exc:
        raise ReleaseError(
            "Refusing to continue: fetching origin main failed.\n"
            "Fix: authenticate and fetch origin, then rerun. "
            "Do not continue from a stale checkout.\n"
            f"{exc}"
        ) from exc
    local = git_text(root, ["rev-parse", "HEAD"])
    remote = git_text(root, ["rev-parse", "origin/main"])
    if local == remote:
        return
    if action == "bump":
        raise ReleaseError(
            f"Refusing to bump: local main {local} does not match "
            f"origin/main {remote}.\n"
            "Fix: fetch origin and update local main to origin/main. "
            "Do not bump from a stale checkout."
        )
    raise ReleaseError(
        f"Refusing to publish: local main {local} does not match "
        f"origin/main {remote}.\n"
        "Fix: fetch origin and check out the merged bump so local main matches "
        "origin/main. This target does not commit."
    )


def require_tool(name: str, *, fix: str) -> None:
    if shutil.which(name) is None:
        raise ReleaseError(fix)


def branch_exists(root: Path, branch: str) -> bool:
    local = git(
        root,
        ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        check=False,
    )
    if local.returncode == 0:
        return True
    remote = git(root, ["ls-remote", "--heads", "origin", branch])
    return remote.stdout.strip() != ""


def require_branch_available(root: Path, branch: str) -> None:
    if not branch_exists(root, branch):
        return
    raise ReleaseError(
        f"Refusing to bump: branch {branch} already exists.\n"
        "Fix: merge the existing pull request, or delete that abandoned branch, "
        "then rerun. Do not push main and do not create a tag."
    )


def changed_paths(status: str) -> set[str]:
    paths: set[str] = set()
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path)
    return paths


def run_uv_lock(root: Path) -> bool:
    lock = root / "uv.lock"
    before = lock.read_bytes() if lock.exists() else None
    before_status = porcelain(root)
    try:
        run(["uv", "lock"], cwd=root, env=git_env())
    except CommandError as exc:
        raise ReleaseError(
            "Refusing to bump: uv lock failed.\n"
            "Fix: install uv, run from the repository root, and rerun. "
            "Do not hand-edit uv.lock.\n"
            f"{exc}"
        ) from exc
    after = lock.read_bytes() if lock.exists() else None
    unexpected = changed_paths(porcelain(root)) - changed_paths(before_status)
    unexpected.discard("uv.lock")
    unexpected.discard("causalpy/version.py")
    if unexpected:
        raise ReleaseError(
            "Refusing to bump: uv lock changed files other than uv.lock: "
            f"{', '.join(sorted(unexpected))}.\n"
            "Fix: inspect that diff. This command only stages "
            "causalpy/version.py and, if changed, uv.lock. "
            "Do not hand-edit uv.lock."
        )
    return after != before


def commit_message(old: str, new: str) -> str:
    return f"Bump version {old} -> {new}"


def pr_body(old: str, new: str, *, lock_changed: bool) -> str:
    if lock_changed:
        lock_note = (
            "`uv.lock` changed when `uv lock` ran. That diff is in this pull "
            "request. It was not hand-edited, and this command did not add a "
            "version field to the `causalpy` lock entry."
        )
    else:
        lock_note = (
            "`uv.lock` did not change. The `causalpy` lock entry has no version "
            "field, and this command did not add one."
        )
    return (
        f"Bumps the package version from {old} to {new}.\n\n"
        f"{lock_note}\n\n"
        "After this pull request merges, check out `main`, make sure it matches "
        "`origin/main`, and run:\n\n"
        "```bash\n"
        "make release-publish\n"
        "```\n\n"
        f"That tags the merged bump commit with `{new}` (no `v` prefix) and "
        "pushes only that tag with your credentials, not `GITHUB_TOKEN`. "
        "The tag push creates the GitHub Release with generated notes and starts "
        "the PyPI jobs. Test PyPI runs first. The real upload waits for approval "
        "in the `release` environment.\n\n"
        "Do not tag before this pull request merges. The bump targets do not "
        "push `main` and do not create a tag.\n"
    )


def abort_branch(root: Path, branch: str, start: str, *, lock_existed: bool) -> None:
    head = git(root, ["rev-parse", "HEAD"], check=False)
    if head.returncode != 0 or head.stdout.strip() != start:
        return
    git(root, ["checkout", "-f", "main"], check=False)
    if current_branch(root) == "main":
        git(root, ["branch", "-D", branch], check=False)
    lock = root / "uv.lock"
    if not lock_existed and lock.exists():
        tracked = git(root, ["ls-files", "--error-unmatch", "uv.lock"], check=False)
        if tracked.returncode != 0:
            lock.unlink()


def create_pr(
    root: Path, old: str, new: str, branch: str, *, lock_changed: bool
) -> str:
    title = commit_message(old, new)
    body = pr_body(old, new, lock_changed=lock_changed)
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".md",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(body)
        body_path = handle.name
    try:
        completed = run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                "main",
                "--head",
                branch,
                "--title",
                title,
                "--body-file",
                body_path,
            ],
            cwd=root,
            env=git_env(),
        )
    except CommandError as exc:
        raise ReleaseError(
            f"Refusing to continue: opening the pull request for {branch} failed.\n"
            "Fix: the branch is already pushed. Open the pull request against "
            "main yourself. Do not push main. Do not create a tag.\n"
            f"{exc}"
        ) from exc
    finally:
        Path(body_path).unlink(missing_ok=True)
    url = completed.stdout.strip()
    if url:
        print(url)
    return url


def cmd_bump(root: Path, part: str) -> None:
    if part not in PARTS:
        raise ReleaseError(
            f"Refusing to bump: unknown part {part!r}.\n"
            "Fix: run make release-patch, make release-minor, or "
            "make release-major."
        )
    require_on_main(root, action="bump")
    require_clean(root, action="bump")
    require_origin(root)
    fetch_main(root, action="bump")
    current = read_version_at(root, "HEAD:causalpy/version.py")
    new = next_version(current, part)
    branch = f"release-{new}"
    require_tool(
        "uv",
        fix=(
            "Refusing to bump: uv is not on PATH.\n"
            "Fix: install uv and rerun. Do not hand-edit uv.lock."
        ),
    )
    require_tool(
        "gh",
        fix=(
            "Refusing to bump: gh is not on PATH.\n"
            "Fix: install the GitHub CLI, authenticate, and rerun. "
            "Do not push main and do not create a tag."
        ),
    )
    require_branch_available(root, branch)
    start = git_text(root, ["rev-parse", "HEAD"])
    lock_existed = (root / "uv.lock").exists()
    git(root, ["checkout", "-b", branch])
    try:
        path = root / VERSION_PATH
        updated = replace_version(path.read_text(encoding="utf-8"), new)
        path.write_text(updated, encoding="utf-8")
        lock_changed = run_uv_lock(root)
        git(root, ["add", "--", "causalpy/version.py"])
        if lock_changed:
            git(root, ["add", "--", "uv.lock"])
        git(root, ["commit", "-m", commit_message(current, new)])
    except ReleaseError:
        abort_branch(root, branch, start, lock_existed=lock_existed)
        raise
    except (CommandError, OSError) as exc:
        abort_branch(root, branch, start, lock_existed=lock_existed)
        raise ReleaseError(
            "Refusing to bump: the version commit failed.\n"
            "Fix: resolve the git error and rerun from a clean main. "
            "This target does not push main and does not create a tag.\n"
            f"{exc}"
        ) from exc
    if current_branch(root) == "main" or git_text(root, ["rev-parse", "main"]) != start:
        raise ReleaseError(
            "Refusing to continue: main moved during the bump.\n"
            "Fix: reset the bump branch aside and rerun from a clean main. "
            "Do not push main."
        )
    try:
        git(root, ["push", "--no-follow-tags", "-u", "origin", branch])
    except CommandError as exc:
        raise ReleaseError(
            f"Refusing to continue: pushing branch {branch} failed.\n"
            "Fix: authenticate and push that branch, then open the pull request. "
            "Do not push main. Do not create a tag.\n"
            f"{exc}"
        ) from exc
    create_pr(root, current, new, branch, lock_changed=lock_changed)


def version_lines(diff: str) -> tuple[list[str], list[str]]:
    removed: list[str] = []
    added: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("-") and "__version__" in line:
            removed.append(line[1:])
        elif line.startswith("+") and "__version__" in line:
            added.append(line[1:])
    return removed, added


def is_merged_bump(root: Path, version: str) -> bool:
    parent = git(root, ["rev-parse", "--verify", "HEAD^"], check=False)
    if parent.returncode != 0:
        return False
    diff = git(
        root,
        ["diff", "--no-color", "HEAD^", "HEAD", "--", "causalpy/version.py"],
    ).stdout
    removed, added = version_lines(diff)
    if len(removed) != 1 or len(added) != 1:
        return False
    try:
        old = read_version(removed[0] + "\n")
        new = read_version(added[0] + "\n")
    except ReleaseError:
        return False
    return new == version and old != new


def tag_exists(root: Path, tag: str) -> bool:
    local = git(
        root,
        ["rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"],
        check=False,
    )
    if local.returncode == 0:
        return True
    remote = git(root, ["ls-remote", "--tags", "origin", f"refs/tags/{tag}"])
    return remote.stdout.strip() != ""


def refuse_tag_mismatch(root: Path) -> str:
    head_version = read_version_at(root, "HEAD:causalpy/version.py")
    work_text = (root / VERSION_PATH).read_text(encoding="utf-8")
    work_version = read_version(work_text)
    if work_version != head_version:
        raise ReleaseError(
            "Refusing to publish: the tag would not equal the package version "
            f"(HEAD has {head_version}, working tree has {work_version}).\n"
            "Fix: do not tag. Restore causalpy/version.py to the committed "
            "version, or merge a bump pull request so HEAD's version is the one "
            "to tag. This target does not commit."
        )
    return head_version


def cmd_publish(root: Path) -> None:
    require_on_main(root, action="publish")
    require_origin(root)
    version = refuse_tag_mismatch(root)
    require_clean(root, action="publish")
    fetch_main(root, action="publish")
    require_release_version(version, action="publish")
    if not is_merged_bump(root, version):
        raise ReleaseError(
            f"Refusing to publish: main is not the merged bump for {version}.\n"
            "Fix: check out the merged commit that changed causalpy/version.py "
            f"to {version}, make sure local main matches origin/main, and rerun. "
            "Do not tag a later commit. This target does not commit."
        )
    tag = version
    if tag_exists(root, tag):
        raise ReleaseError(
            f"Refusing to publish: tag {tag} already exists.\n"
            "Fix: do not delete or move the existing tag. Merge a new "
            "version-bump pull request, then publish that version."
        )
    try:
        git(root, ["tag", "-a", tag, "-m", f"Release {tag}"])
    except CommandError as exc:
        raise ReleaseError(
            f"Refusing to publish: creating tag {tag} failed.\n"
            "Fix: resolve the git tag error and rerun. This target does not "
            f"commit.\n{exc}"
        ) from exc
    try:
        git(
            root,
            ["push", "--no-follow-tags", "origin", f"refs/tags/{tag}"],
            strip_tokens=True,
        )
    except CommandError as exc:
        git(root, ["tag", "-d", tag], check=False)
        raise ReleaseError(
            f"Refusing to publish: pushing tag {tag} failed.\n"
            "Fix: authenticate with your own git credentials, not GITHUB_TOKEN, "
            "and rerun make release-publish. The local tag was removed because "
            "the remote did not accept it. This target does not commit.\n"
            f"{exc}"
        ) from exc
    print(f"Pushed tag {tag} to origin.")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        root = repo_root()
        if len(args) == 2 and args[0] == "bump":
            cmd_bump(root, args[1])
        elif args == ["publish"]:
            cmd_publish(root)
        else:
            raise ReleaseError(
                "Usage: python scripts/release.py bump {patch|minor|major}\n"
                "       python scripts/release.py publish"
            )
    except ReleaseError as exc:
        print(exc, file=sys.stderr)
        return 2
    except CommandError as exc:
        print(
            "Refusing to continue: a git command failed.\n"
            f"Fix: resolve the git error and rerun.\n{exc}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

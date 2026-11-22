#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Release targets must not open a real pull request, tag, or GitHub Release.

These tests use a temporary git repository and fake ``gh`` / ``uv`` binaries.
They do not call GitHub.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "release.py"
REAL_GIT = shutil.which("git")
assert REAL_GIT is not None


def _quote(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
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
        detail = (result.stderr or result.stdout or "").strip()
        raise AssertionError(f"{cmd} failed: {detail}")
    return result


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run([REAL_GIT, *args], cwd=repo)


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _fake_env(
    tmp: Path,
    *,
    uv_touch: bool = False,
    gh: bool = True,
) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    push_log = tmp / "push.log"
    gh_log = tmp / "gh.log"
    _write_executable(
        bin_dir / "git",
        "#!/bin/sh\n"
        'if [ "$1" = "push" ]; then\n'
        f"  printf 'GITHUB_TOKEN=%s\\n' \"${{GITHUB_TOKEN-<unset>}}\" >> {_quote(push_log)}\n"
        f"  printf 'GH_TOKEN=%s\\n' \"${{GH_TOKEN-<unset>}}\" >> {_quote(push_log)}\n"
        f"  printf '%s\\n' \"$*\" >> {_quote(push_log)}\n"
        "fi\n"
        f'exec {_quote(Path(REAL_GIT))} "$@"\n',
    )
    uv_script = "#!/bin/sh\nexit 0\n"
    if uv_touch:
        uv_script = "#!/bin/sh\nprintf '\\n# uv-lock-touched\\n' >> uv.lock\nexit 0\n"
    _write_executable(bin_dir / "uv", uv_script)
    if gh:
        _write_executable(
            bin_dir / "gh",
            "#!/bin/sh\n"
            f"printf '%s\\n' \"$*\" >> {_quote(gh_log)}\n"
            'prev=""\n'
            'for arg in "$@"; do\n'
            '  if [ "$prev" = "--body-file" ]; then\n'
            f"    printf '\\n--- body ---\\n' >> {_quote(gh_log)}\n"
            f'    /bin/cat "$arg" >> {_quote(gh_log)}\n'
            "  fi\n"
            "  prev=$arg\n"
            "done\n"
            "printf 'https://example.test/pull/1\\n'\n",
        )
    env = os.environ.copy()
    # Only the fake bin. /usr/bin on GitHub runners contains gh, which would
    # skip the missing-tool refusal and call the real CLI.
    env["PATH"] = str(bin_dir)
    env["GITHUB_TOKEN"] = "should-not-push-tags"
    env["GH_TOKEN"] = "should-not-push-tags"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env, push_log, gh_log


def _init_repo(tmp: Path, version: str = "1.2.3") -> Path:
    bare = tmp / "origin.git"
    repo = tmp / "repo"
    _run([REAL_GIT, "init", "--bare", "-b", "main", str(bare)])
    _run([REAL_GIT, "init", "-b", "main", str(repo)])
    _git(repo, "config", "user.email", "release-test@example.com")
    _git(repo, "config", "user.name", "Release Test")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "config", "tag.gpgsign", "false")
    package = repo / "causalpy"
    package.mkdir()
    (package / "version.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "causalpy"\ndynamic = ["version"]\n',
        encoding="utf-8",
    )
    (repo / "uv.lock").write_text(
        '[[package]]\nname = "causalpy"\nsource = { editable = "." }\n',
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")
    return repo


def _release(
    repo: Path,
    env: dict[str, str],
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return _run(
        [sys.executable, str(SCRIPT), *args],
        cwd=repo,
        env=env,
        check=False,
    )


def _sha(repo: Path, ref: str = "HEAD") -> str:
    return _git(repo, "rev-parse", ref).stdout.strip()


def _set_version_and_commit(repo: Path, new: str, *, push: bool = True) -> None:
    path = repo / "causalpy" / "version.py"
    text = path.read_text(encoding="utf-8")
    old = text.split('"')[1]
    path.write_text(text.replace(f'"{old}"', f'"{new}"', 1), encoding="utf-8")
    _git(repo, "add", "causalpy/version.py")
    _git(repo, "commit", "-m", f"Bump version {old} -> {new}")
    if push:
        _git(repo, "push", "origin", "main")


@pytest.mark.parametrize(
    ("part", "old", "new"),
    [
        ("patch", "1.2.3", "1.2.4"),
        ("patch", "1.2.9", "1.2.10"),
        ("minor", "1.2.3", "1.3.0"),
        ("minor", "1.9.4", "1.10.0"),
        ("major", "1.2.3", "2.0.0"),
        ("major", "0.9.9", "1.0.0"),
    ],
)
def test_bump_opens_pr_without_pushing_main_or_tagging(
    tmp_path: Path,
    part: str,
    old: str,
    new: str,
) -> None:
    repo = _init_repo(tmp_path, old)
    env, push_log, gh_log = _fake_env(tmp_path)
    start = _sha(repo, "main")

    result = _release(repo, env, "bump", part)

    assert result.returncode == 0, result.stderr
    assert "https://example.test/pull/1" in result.stdout
    assert _sha(repo, "main") == start
    assert _sha(repo, "origin/main") == start
    assert _git(repo, "branch", "--show-current").stdout.strip() == f"release-{new}"
    assert (
        f'__version__ = "{new}"'
        in _git(repo, "show", "HEAD:causalpy/version.py").stdout
    )
    assert (
        f'__version__ = "{old}"'
        in _git(repo, "show", "main:causalpy/version.py").stdout
    )
    assert _git(repo, "tag", "--list").stdout == ""
    assert _git(repo, "ls-remote", "--tags", "origin").stdout == ""
    pushes = push_log.read_text(encoding="utf-8")
    assert f"push --no-follow-tags -u origin release-{new}" in pushes
    assert "refs/tags" not in pushes
    assert "origin main" not in pushes
    body = gh_log.read_text(encoding="utf-8")
    assert "--base main" in body
    assert f"--head release-{new}" in body
    assert old in body and new in body
    assert "make release-publish" in body
    assert "Do not tag before this pull request merges" in body
    assert "`uv.lock` did not change" in body
    names = _git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        "HEAD",
    ).stdout.split()
    assert names == ["causalpy/version.py"]
    assert _git(repo, "diff", "main", "--", "pyproject.toml").stdout == ""
    lock = (repo / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "causalpy"' in lock
    assert "version =" not in lock


@pytest.mark.parametrize("version", ["1.0.0rc1", "1.0.0a1", "1.2", "v1.2.3"])
def test_bump_refuses_non_release_version(tmp_path: Path, version: str) -> None:
    repo = _init_repo(tmp_path, version)
    env, push_log, gh_log = _fake_env(tmp_path)

    result = _release(repo, env, "bump", "patch")

    assert result.returncode == 2
    assert "set a release version in causalpy/version.py first" in result.stderr
    assert "Do not guess a bump off a pre-release" in result.stderr
    assert _git(repo, "branch", "--show-current").stdout.strip() == "main"
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert not push_log.exists()
    assert not gh_log.exists()
    assert _git(repo, "tag", "--list").stdout == ""


def test_bump_refuses_dirty_tree_and_non_main(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    env, push_log, gh_log = _fake_env(tmp_path)
    (repo / "README").write_text("dirty\n", encoding="utf-8")

    dirty = _release(repo, env, "bump", "patch")

    assert dirty.returncode == 2
    assert "not clean" in dirty.stderr
    assert "does not commit on main" in dirty.stderr
    assert not push_log.exists()
    (repo / "README").unlink()
    _git(repo, "checkout", "-b", "feature")

    other = _release(repo, env, "bump", "minor")

    assert other.returncode == 2
    assert "not main" in other.stderr
    assert not gh_log.exists()


def test_bump_refuses_when_main_does_not_match_origin(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    env, push_log, _gh_log = _fake_env(tmp_path)
    _git(repo, "commit", "--allow-empty", "-m", "local only")
    local = _sha(repo)

    result = _release(repo, env, "bump", "patch")

    assert result.returncode == 2
    assert "does not match" in result.stderr
    assert "update local main to origin/main" in result.stderr
    assert _sha(repo) == local
    assert not push_log.exists()
    assert _git(repo, "tag", "--list").stdout == ""


def test_bump_includes_uv_lock_only_when_uv_changes_it(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    env, _push_log, gh_log = _fake_env(tmp_path, uv_touch=True)

    result = _release(repo, env, "bump", "patch")

    assert result.returncode == 0, result.stderr
    names = set(
        _git(
            repo,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "HEAD",
        ).stdout.split()
    )
    assert names == {"causalpy/version.py", "uv.lock"}
    lock = _git(repo, "show", "HEAD:uv.lock").stdout
    assert "# uv-lock-touched" in lock
    assert "version =" not in lock
    assert "`uv.lock` changed" in gh_log.read_text(encoding="utf-8")
    assert "did not add a version field" in gh_log.read_text(encoding="utf-8")


def test_bump_refuses_when_gh_is_missing(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    env, push_log, _gh_log = _fake_env(tmp_path, gh=False)

    result = _release(repo, env, "bump", "patch")

    assert result.returncode == 2
    assert "gh is not on PATH" in result.stderr
    assert "Do not push main" in result.stderr
    assert _git(repo, "branch", "--show-current").stdout.strip() == "main"
    assert not push_log.exists()
    assert _git(repo, "branch", "--list").stdout.strip() == "* main"


def test_publish_pushes_bare_tag_without_github_token(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "1.2.3")
    _set_version_and_commit(repo, "1.2.4")
    env, push_log, gh_log = _fake_env(tmp_path)
    start = _sha(repo)
    subject = _git(repo, "log", "-1", "--format=%s").stdout.strip()

    result = _release(repo, env, "publish")

    assert result.returncode == 0, result.stderr
    assert "Pushed tag 1.2.4" in result.stdout
    assert _sha(repo) == start
    assert _sha(repo, "origin/main") == start
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert _git(repo, "log", "-1", "--format=%s").stdout.strip() == subject
    remote = _git(repo, "ls-remote", "--tags", "origin").stdout
    assert "refs/tags/1.2.4" in remote
    assert "refs/tags/v1.2.4" not in remote
    pushes = push_log.read_text(encoding="utf-8")
    assert "GITHUB_TOKEN=<unset>" in pushes
    assert "GH_TOKEN=<unset>" in pushes
    assert "refs/tags/1.2.4" in pushes
    assert "origin main" not in pushes
    assert not gh_log.exists()


def test_publish_accepts_merge_commit(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "1.2.3")
    _git(repo, "checkout", "-b", "bump")
    _set_version_and_commit(repo, "1.3.0", push=False)
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "bump", "-m", "Merge bump")
    _git(repo, "push", "origin", "main")
    env, push_log, gh_log = _fake_env(tmp_path)

    result = _release(repo, env, "publish")

    assert result.returncode == 0, result.stderr
    assert "refs/tags/1.3.0" in _git(repo, "ls-remote", "--tags", "origin").stdout
    assert "refs/tags/1.3.0" in push_log.read_text(encoding="utf-8")
    assert not gh_log.exists()


def test_publish_refuses_when_main_is_not_the_merged_bump(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "1.2.3")
    _set_version_and_commit(repo, "1.2.4")
    (repo / "README").write_text("later\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-m", "later")
    _git(repo, "push", "origin", "main")
    env, push_log, _gh_log = _fake_env(tmp_path)
    start = _sha(repo)

    result = _release(repo, env, "publish")

    assert result.returncode == 2
    assert "main is not the merged bump" in result.stderr
    assert "Do not tag a later commit" in result.stderr
    assert "does not commit" in result.stderr
    assert _sha(repo) == start
    assert not push_log.exists()
    assert _git(repo, "ls-remote", "--tags", "origin").stdout == ""


def test_publish_refuses_when_tag_already_exists(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "1.2.3")
    _set_version_and_commit(repo, "1.2.4")
    _git(repo, "tag", "-a", "1.2.4", "-m", "already")
    _git(repo, "push", "origin", "refs/tags/1.2.4")
    env, push_log, _gh_log = _fake_env(tmp_path)
    start = _sha(repo)

    result = _release(repo, env, "publish")

    assert result.returncode == 2
    assert "tag 1.2.4 already exists" in result.stderr
    assert "do not delete or move the existing tag" in result.stderr
    assert _sha(repo) == start
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert not push_log.exists()


def test_publish_refuses_when_tag_would_not_equal_package_version(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path, "1.2.3")
    _set_version_and_commit(repo, "1.2.4")
    path = repo / "causalpy" / "version.py"
    path.write_text('__version__ = "1.2.5"\n', encoding="utf-8")
    env, push_log, _gh_log = _fake_env(tmp_path)
    start = _sha(repo)

    result = _release(repo, env, "publish")

    assert result.returncode == 2
    assert "would not equal the package version" in result.stderr
    assert "HEAD has 1.2.4, working tree has 1.2.5" in result.stderr
    assert "does not commit" in result.stderr
    assert _sha(repo) == start
    assert "causalpy/version.py" in _git(repo, "status", "--porcelain").stdout
    assert not push_log.exists()
    assert _git(repo, "tag", "--list").stdout == ""


def test_publish_refuses_prerelease_and_unpushed_main(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "1.2.3")
    _set_version_and_commit(repo, "1.0.0rc1")
    env, push_log, _gh_log = _fake_env(tmp_path)

    prerelease = _release(repo, env, "publish")

    assert prerelease.returncode == 2
    assert "set a release version in causalpy/version.py first" in prerelease.stderr
    assert not push_log.exists()
    assert _git(repo, "tag", "--list").stdout == ""

    _git(repo, "checkout", "HEAD~1", "--", "causalpy/version.py")
    _git(repo, "commit", "-m", "restore")
    _set_version_and_commit(repo, "1.2.4", push=False)
    ahead = _sha(repo)

    unpushed = _release(repo, env, "publish")

    assert unpushed.returncode == 2
    assert "does not match" in unpushed.stderr
    assert "does not commit" in unpushed.stderr
    assert _sha(repo) == ahead
    assert _git(repo, "tag", "--list").stdout == ""


def test_stale_remote_tracking_ref_does_not_hide_a_moved_main(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "1.2.3")
    _set_version_and_commit(repo, "1.2.4")
    other = tmp_path / "other"
    _run(
        [REAL_GIT, "clone", str(tmp_path / "origin.git"), str(other)],
        cwd=tmp_path,
    )
    _git(other, "config", "user.email", "release-test@example.com")
    _git(other, "config", "user.name", "Release Test")
    _git(other, "config", "commit.gpgsign", "false")
    (other / "note.txt").write_text("later\n", encoding="utf-8")
    _git(other, "add", "note.txt")
    _git(other, "commit", "-m", "later")
    _git(other, "push", "origin", "main")
    stale = _sha(repo, "origin/main")
    assert stale == _sha(repo)

    env, push_log, _gh_log = _fake_env(tmp_path, gh=False)
    published = _release(repo, env, "publish")

    assert published.returncode == 2, published.stderr
    assert "does not match" in published.stderr
    assert "does not commit" in published.stderr
    assert _git(repo, "tag", "--list").stdout == ""
    assert not push_log.exists()
    assert _sha(repo, "origin/main") != stale

    bumped = _release(repo, env, "bump", "patch")

    assert bumped.returncode == 2, bumped.stderr
    assert "does not match" in bumped.stderr
    assert "release-1.2.5" not in _git(repo, "branch", "--list").stdout
    assert _git(repo, "status", "--porcelain").stdout == ""


def test_bump_does_not_push_other_annotated_tags(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "1.2.3")
    _git(repo, "config", "push.followTags", "true")
    _git(repo, "tag", "-a", "0.9.0", "-m", "old")
    env, push_log, _gh_log = _fake_env(tmp_path)

    result = _release(repo, env, "bump", "patch")

    assert result.returncode == 0, result.stderr
    remote = _git(repo, "ls-remote", "--tags", "origin").stdout
    assert "refs/tags/0.9.0" not in remote
    assert "refs/tags" not in push_log.read_text(encoding="utf-8")


def test_publish_ignores_color_and_does_not_follow_tags(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path, "1.2.3")
    _set_version_and_commit(repo, "1.2.4")
    _git(repo, "config", "push.followTags", "true")
    _git(repo, "config", "color.ui", "always")
    _git(repo, "config", "color.diff", "always")
    _git(repo, "tag", "-a", "0.9.0", "-m", "old")
    env, push_log, gh_log = _fake_env(tmp_path, gh=False)

    result = _release(repo, env, "publish")

    assert result.returncode == 0, result.stderr
    remote = _git(repo, "ls-remote", "--tags", "origin").stdout
    assert "refs/tags/1.2.4" in remote
    assert "refs/tags/0.9.0" not in remote
    pushes = push_log.read_text(encoding="utf-8")
    assert "refs/tags/1.2.4" in pushes
    assert "refs/tags/0.9.0" not in pushes
    assert "GITHUB_TOKEN=<unset>" in pushes
    assert not gh_log.exists()

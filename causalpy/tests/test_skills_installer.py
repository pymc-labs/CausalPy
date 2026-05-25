#   Copyright 2026 - 2026 The PyMC Labs Developers
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
"""Tests for the skills installer, CLI, and platform adapters."""

from __future__ import annotations

import pytest

from causalpy.skills._installer import (
    VERSION_STAMP,
    check_version,
    detect_platforms,
    install,
    list_skills,
    uninstall,
)


class TestListSkills:
    """Verify bundled skill catalogue."""

    def test_returns_all_four_skills(self):
        skills = list_skills()
        assert len(skills) == 4
        assert "designing-experiments" in skills
        assert "loading-datasets" in skills
        assert "performing-causal-analysis" in skills
        assert "running-placebo-analysis" in skills


class TestDetectPlatforms:
    """Auto-detection of AI platforms from project directory structure."""

    def test_detects_cursor(self, tmp_path):
        (tmp_path / ".cursor").mkdir()
        assert "cursor" in detect_platforms(tmp_path)

    def test_detects_claude(self, tmp_path):
        (tmp_path / ".claude").mkdir()
        assert "claude" in detect_platforms(tmp_path)

    def test_detects_copilot_via_github_dir(self, tmp_path):
        (tmp_path / ".github").mkdir()
        assert "copilot" in detect_platforms(tmp_path)

    def test_detects_windsurf_via_dir(self, tmp_path):
        (tmp_path / ".windsurf").mkdir()
        assert "windsurf" in detect_platforms(tmp_path)

    def test_detects_windsurf_via_rules_file(self, tmp_path):
        (tmp_path / ".windsurfrules").touch()
        assert "windsurf" in detect_platforms(tmp_path)

    def test_empty_dir_returns_empty(self, tmp_path):
        assert detect_platforms(tmp_path) == []

    def test_detects_multiple(self, tmp_path):
        (tmp_path / ".cursor").mkdir()
        (tmp_path / ".claude").mkdir()
        detected = detect_platforms(tmp_path)
        assert "cursor" in detected
        assert "claude" in detected


class TestInstallCursor:
    """Install skills into a Cursor project."""

    def test_creates_skill_directories(self, tmp_path):
        (tmp_path / ".cursor").mkdir()
        result = install(project_dir=tmp_path, platform="cursor")
        assert "cursor" in result
        assert len(result["cursor"]) == 8

        skill_dir = tmp_path / ".cursor" / "skills" / "causalpy-designing-experiments"
        assert skill_dir.is_dir()
        assert (skill_dir / "SKILL.md").exists()

    def test_version_stamp_after_frontmatter(self, tmp_path):
        (tmp_path / ".cursor").mkdir()
        install(project_dir=tmp_path, platform="cursor")

        skill_md = (
            tmp_path
            / ".cursor"
            / "skills"
            / "causalpy-designing-experiments"
            / "SKILL.md"
        )
        content = skill_md.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        fm_end = content.index("---\n", 4) + 4
        after_fm = content[fm_end:]
        assert after_fm.startswith("<!-- causalpy-skills v")

    def test_reference_files_included(self, tmp_path):
        (tmp_path / ".cursor").mkdir()
        install(project_dir=tmp_path, platform="cursor")

        ref_dir = (
            tmp_path
            / ".cursor"
            / "skills"
            / "causalpy-performing-causal-analysis"
            / "reference"
        )
        assert ref_dir.is_dir()
        assert (ref_dir / "diff_in_diff.md").exists()
        assert (ref_dir / "interrupted_time_series.md").exists()
        assert (ref_dir / "synthetic_control.md").exists()


class TestInstallGeneric:
    """Install skills as a generic llms-causalpy.txt file."""

    def test_creates_llms_file(self, tmp_path):
        result = install(project_dir=tmp_path, platform="generic")
        assert "generic" in result
        assert len(result["generic"]) == 1

        llms_file = tmp_path / "llms-causalpy.txt"
        assert llms_file.exists()
        content = llms_file.read_text(encoding="utf-8")
        assert "CausalPy" in content
        assert "causalpy-skills" in content

    def test_fallback_to_generic_when_no_platform(self, tmp_path):
        result = install(project_dir=tmp_path)
        assert "generic" in result


class TestInstallCopilot:
    """Install skills into a Copilot project."""

    def test_creates_instructions_file(self, tmp_path):
        (tmp_path / ".github").mkdir()
        result = install(project_dir=tmp_path, platform="copilot")
        assert "copilot" in result

        instructions = tmp_path / ".github" / "copilot-instructions.md"
        assert instructions.exists()
        content = instructions.read_text(encoding="utf-8")
        assert "<!-- BEGIN causalpy-skills -->" in content
        assert "<!-- END causalpy-skills -->" in content

    def test_appends_to_existing_file(self, tmp_path):
        (tmp_path / ".github").mkdir()
        instructions = tmp_path / ".github" / "copilot-instructions.md"
        instructions.write_text("# Existing instructions\n\nKeep this.\n")

        install(project_dir=tmp_path, platform="copilot")
        content = instructions.read_text(encoding="utf-8")
        assert content.startswith("# Existing instructions")
        assert "<!-- BEGIN causalpy-skills -->" in content

    def test_replaces_existing_section(self, tmp_path):
        (tmp_path / ".github").mkdir()
        install(project_dir=tmp_path, platform="copilot")
        install(project_dir=tmp_path, platform="copilot")

        instructions = tmp_path / ".github" / "copilot-instructions.md"
        content = instructions.read_text(encoding="utf-8")
        assert content.count("<!-- BEGIN causalpy-skills -->") == 1


class TestUninstall:
    """Uninstall previously installed skills."""

    def test_uninstall_cursor(self, tmp_path):
        (tmp_path / ".cursor").mkdir()
        install(project_dir=tmp_path, platform="cursor")
        result = uninstall(project_dir=tmp_path, platform="cursor")

        assert "cursor" in result
        assert len(result["cursor"]) == 4
        skills_dir = tmp_path / ".cursor" / "skills"
        remaining = [d for d in skills_dir.iterdir() if d.name.startswith("causalpy-")]
        assert remaining == []

    def test_uninstall_generic(self, tmp_path):
        install(project_dir=tmp_path, platform="generic")
        assert (tmp_path / "llms-causalpy.txt").exists()

        uninstall(project_dir=tmp_path, platform="generic")
        assert not (tmp_path / "llms-causalpy.txt").exists()

    def test_uninstall_copilot_removes_section(self, tmp_path):
        (tmp_path / ".github").mkdir()
        instructions = tmp_path / ".github" / "copilot-instructions.md"
        instructions.write_text("# Keep\n")

        install(project_dir=tmp_path, platform="copilot")
        uninstall(project_dir=tmp_path, platform="copilot")

        content = instructions.read_text(encoding="utf-8")
        assert "causalpy-skills" not in content
        assert "Keep" in content

    def test_uninstall_noop_when_not_installed(self, tmp_path):
        (tmp_path / ".cursor").mkdir()
        result = uninstall(project_dir=tmp_path, platform="cursor")
        assert result["cursor"] == []


class TestCheckVersion:
    """Version checking for installed skills."""

    def test_returns_none_when_not_installed(self, tmp_path):
        assert check_version(project_dir=tmp_path) is None

    def test_returns_version_when_installed(self, tmp_path):
        (tmp_path / ".cursor").mkdir()
        install(project_dir=tmp_path, platform="cursor")
        version = check_version(project_dir=tmp_path)
        assert version is not None


class TestSymlinkSafety:
    """Verify _rmtree_if_exists does not follow symlinks outside the project."""

    def test_symlink_unlinked_without_following(self, tmp_path):
        (tmp_path / ".cursor" / "skills").mkdir(parents=True)
        target = tmp_path / "external_target"
        target.mkdir()
        (target / "secret.txt").write_text("do not delete")

        link = tmp_path / ".cursor" / "skills" / "causalpy-fake"
        link.symlink_to(target)

        from causalpy.skills._platforms._base import BasePlatformAdapter

        adapter_stub = type(
            "Stub",
            (BasePlatformAdapter,),
            {
                "install": lambda self, sd: [],
                "uninstall": lambda self: [],
                "get_installed_version": lambda self: None,
            },
        )(tmp_path, VERSION_STAMP)
        adapter_stub._rmtree_if_exists(link)

        assert not link.exists()
        assert target.is_dir()
        assert (target / "secret.txt").exists()

    def test_refuses_path_outside_project(self, tmp_path):

        external = tmp_path.parent / "outside_project"
        external.mkdir(exist_ok=True)

        from causalpy.skills._platforms._base import BasePlatformAdapter

        adapter_stub = type(
            "Stub",
            (BasePlatformAdapter,),
            {
                "install": lambda self, sd: [],
                "uninstall": lambda self: [],
                "get_installed_version": lambda self: None,
            },
        )(tmp_path, VERSION_STAMP)

        with pytest.raises(ValueError, match="outside project dir"):
            adapter_stub._rmtree_if_exists(external)


class TestInstallErrors:
    """Edge cases and error handling."""

    def test_nonexistent_project_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            install(project_dir="/nonexistent/path")

    def test_uninstall_nonexistent_project_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            uninstall(project_dir="/nonexistent/path")

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
"""Tests for resource-safe documentation notebook selection."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.run_notebooks import runner

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_NOTEBOOK_COLLECTIONS = set(runner.NOTEBOOK_COLLECTIONS)


@pytest.fixture
def notebook_collections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    gallery = tmp_path / "docs" / "source" / "notebooks"
    knowledgebase = tmp_path / "docs" / "source" / "knowledgebase"
    gallery.mkdir(parents=True)
    knowledgebase.mkdir(parents=True)
    for path in (
        gallery / "alpha-pymc.ipynb",
        gallery / "beta-sklearn.ipynb",
        gallery / "other.ipynb",
        gallery / "pymc-introduction.ipynb",
        gallery / "skipped.ipynb",
        gallery / "sklearn-overview.ipynb",
        knowledgebase / "concepts.ipynb",
    ):
        path.touch()

    collections = {"gallery": gallery, "knowledgebase": knowledgebase}
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "NOTEBOOK_COLLECTIONS", collections)
    monkeypatch.setattr(runner, "SKIP_NOTEBOOKS", {"gallery/skipped.ipynb"})
    return collections


def test_default_selection_uses_gallery_and_mock_skip_policy(
    notebook_collections: dict[str, Path],
) -> None:
    notebooks = runner.get_notebooks()

    assert [notebook.name for notebook in notebooks] == [
        "alpha-pymc.ipynb",
        "beta-sklearn.ipynb",
        "other.ipynb",
        "pymc-introduction.ipynb",
        "sklearn-overview.ipynb",
    ]


def test_named_knowledgebase_collection_is_discoverable(
    notebook_collections: dict[str, Path],
) -> None:
    notebooks = runner.get_notebooks(collections=["knowledgebase"])

    assert [notebook.name for notebook in notebooks] == ["concepts.ipynb"]


def test_multiple_exact_notebooks_can_cross_collections(
    notebook_collections: dict[str, Path],
) -> None:
    notebooks = runner.get_notebooks(
        notebook_paths=[
            "docs/source/knowledgebase/concepts.ipynb",
            "docs/source/notebooks/alpha-pymc.ipynb",
        ]
    )

    assert notebooks == [
        notebook_collections["knowledgebase"] / "concepts.ipynb",
        notebook_collections["gallery"] / "alpha-pymc.ipynb",
    ]


def test_exact_mock_incompatible_notebook_requires_full_mode(
    notebook_collections: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="Use --full"):
        runner.get_notebooks(notebook_paths=["docs/source/notebooks/skipped.ipynb"])

    notebooks = runner.get_notebooks(
        notebook_paths=["docs/source/notebooks/skipped.ipynb"], full=True
    )
    assert [notebook.name for notebook in notebooks] == ["skipped.ipynb"]


def test_collection_full_mode_does_not_bypass_skip_policy(
    notebook_collections: dict[str, Path],
) -> None:
    notebooks = runner.get_notebooks(full=True)

    assert "skipped.ipynb" not in [notebook.name for notebook in notebooks]


def test_exact_notebook_must_be_inside_documentation_collections(
    notebook_collections: dict[str, Path],
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.ipynb"
    outside.touch()

    with pytest.raises(ValueError, match="outside configured"):
        runner.get_notebooks(notebook_paths=[str(outside)])


def test_pattern_and_exclusion_filters_compose(
    notebook_collections: dict[str, Path],
) -> None:
    notebooks = runner.get_notebooks(pattern="*-pymc.ipynb")
    assert [notebook.name for notebook in notebooks] == ["alpha-pymc.ipynb"]

    notebooks = runner.get_notebooks(exclude_patterns=["pymc"])
    assert [notebook.name for notebook in notebooks] == [
        "beta-sklearn.ipynb",
        "other.ipynb",
        "sklearn-overview.ipynb",
    ]

    notebooks = runner.get_notebooks(exclude_patterns=["pymc", "sklearn"])
    assert [notebook.name for notebook in notebooks] == ["other.ipynb"]


def test_workflow_shards_are_exhaustive_and_non_overlapping(
    notebook_collections: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "test_notebook.yml").read_text()
    )
    notebook_job = workflow["jobs"]["notebooks"]
    strategy = notebook_job["strategy"]
    matrix = strategy["matrix"]["include"]
    run_step = next(
        step for step in notebook_job["steps"] if step.get("name") == "Run notebooks"
    )

    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert strategy["fail-fast"] is False
    assert strategy["max-parallel"] == 1
    assert {shard["collection"] for shard in matrix} == PRODUCTION_NOTEBOOK_COLLECTIONS
    assert run_step["env"] == {
        "NOTEBOOK_COLLECTION": "${{ matrix.collection }}",
        "NOTEBOOK_PATTERN": "${{ matrix.pattern }}",
        "NOTEBOOK_EXCLUDE_ONE": "${{ matrix.exclude_one }}",
        "NOTEBOOK_EXCLUDE_TWO": "${{ matrix.exclude_two }}",
    }
    for expected_fragment in (
        'args=(--collection "$NOTEBOOK_COLLECTION")',
        'args+=(--pattern "$NOTEBOOK_PATTERN")',
        'args+=(--exclude-pattern "$NOTEBOOK_EXCLUDE_ONE")',
        'args+=(--exclude-pattern "$NOTEBOOK_EXCLUDE_TWO")',
        'python scripts/run_notebooks/runner.py "${args[@]}"',
    ):
        assert expected_fragment in run_step["run"]
    assert matrix == [
        {
            "name": "gallery-pymc",
            "collection": "gallery",
            "pattern": "*-pymc.ipynb",
            "exclude_one": "",
            "exclude_two": "",
        },
        {
            "name": "gallery-sklearn",
            "collection": "gallery",
            "pattern": "*-sklearn.ipynb",
            "exclude_one": "",
            "exclude_two": "",
        },
        {
            "name": "gallery-other",
            "collection": "gallery",
            "pattern": "",
            "exclude_one": "*-pymc.ipynb",
            "exclude_two": "*-sklearn.ipynb",
        },
        {
            "name": "knowledgebase",
            "collection": "knowledgebase",
            "pattern": "",
            "exclude_one": "",
            "exclude_two": "",
        },
    ]

    shards: list[list[Path]] = []
    original_get_notebooks = runner.get_notebooks

    def record_selection(*args, **kwargs) -> list[Path]:
        selected = original_get_notebooks(*args, **kwargs)
        shards.append(selected)
        return selected

    monkeypatch.setattr(runner, "get_notebooks", record_selection)
    for shard in matrix:
        argv = ["--collection", shard["collection"], "--list"]
        if shard["pattern"]:
            argv.extend(["--pattern", shard["pattern"]])
        for exclude_pattern in (shard["exclude_one"], shard["exclude_two"]):
            if exclude_pattern:
                argv.extend(["--exclude-pattern", exclude_pattern])
        assert runner.main(argv) == 0
    selected = [notebook for shard in shards for notebook in shard]
    expected = [
        notebook_collections["gallery"] / "alpha-pymc.ipynb",
        notebook_collections["gallery"] / "beta-sklearn.ipynb",
        notebook_collections["gallery"] / "other.ipynb",
        notebook_collections["gallery"] / "pymc-introduction.ipynb",
        notebook_collections["gallery"] / "sklearn-overview.ipynb",
        notebook_collections["knowledgebase"] / "concepts.ipynb",
    ]

    assert len(selected) == len(set(selected))
    assert sorted(selected) == sorted(expected)


def test_explicit_notebooks_reject_other_selectors(
    notebook_collections: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        runner.get_notebooks(
            collections=["gallery"],
            notebook_paths=["docs/source/notebooks/alpha-pymc.ipynb"],
        )


def test_collection_symlink_cannot_escape_allowed_root(
    notebook_collections: dict[str, Path],
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.ipynb"
    outside.touch()
    (notebook_collections["gallery"] / "escape.ipynb").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes its configured collection"):
        runner.get_notebooks()


def test_notebook_symlink_cannot_cross_collection_boundary(
    notebook_collections: dict[str, Path],
) -> None:
    cross_collection_link = notebook_collections["gallery"] / "concepts.ipynb"
    cross_collection_link.symlink_to(
        notebook_collections["knowledgebase"] / "concepts.ipynb"
    )

    with pytest.raises(ValueError, match="escapes its configured collection"):
        runner.get_notebooks(collections=["gallery"])

    with pytest.raises(ValueError, match="escapes its configured collection"):
        runner.get_notebooks(notebook_paths=["docs/source/notebooks/concepts.ipynb"])


def test_collection_root_cannot_escape_repository(
    notebook_collections: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "external.ipynb").touch()
    monkeypatch.setattr(
        runner,
        "NOTEBOOK_COLLECTIONS",
        {**notebook_collections, "gallery": outside},
    )

    with pytest.raises(ValueError, match="root.*outside the repository"):
        runner.get_notebooks()


def test_collection_root_cannot_alias_another_collection(
    notebook_collections: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    linked_collection = tmp_path / "linked-notebooks"
    linked_collection.symlink_to(
        notebook_collections["knowledgebase"], target_is_directory=True
    )
    monkeypatch.setattr(
        runner,
        "NOTEBOOK_COLLECTIONS",
        {**notebook_collections, "gallery": linked_collection},
    )

    with pytest.raises(ValueError, match="root.*cannot be symlinks"):
        runner.get_notebooks()


def test_main_list_mode_never_executes_notebooks(
    notebook_collections: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        runner,
        "run_notebook",
        lambda path, *, full=False: executed.append((path, full)),
    )

    assert runner.main(["--list"]) == 0
    assert executed == []


def test_full_execution_allows_long_silent_cells(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    notebook = tmp_path / "notebook.ipynb"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner.papermill,
        "execute_notebook",
        lambda **kwargs: calls.append(kwargs),
    )

    runner.run_notebook(notebook, full=True)

    assert calls == [
        {
            "input_path": str(notebook),
            "output_path": str(notebook),
            "kernel_name": runner.KERNEL_NAME,
            "progress_bar": True,
            "cwd": notebook.parent,
            "iopub_timeout": runner.IOPUB_TIMEOUT_SECONDS,
            "raise_on_iopub_timeout": False,
        }
    ]


def test_mock_execution_discards_output_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text(
        '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
    )
    calls: list[dict[str, object]] = []
    temp_paths: list[Path] = []
    injected: list[list[object]] = []

    def execute_notebook(**kwargs: object) -> None:
        temp_path = Path(str(kwargs["input_path"]))
        assert temp_path.exists()
        temp_paths.append(temp_path)
        calls.append(kwargs)

    monkeypatch.setattr(runner.papermill, "execute_notebook", execute_notebook)
    monkeypatch.setattr(
        runner,
        "inject_mock_code",
        lambda cells: injected.append(cells),
    )

    runner.run_notebook(notebook)

    assert len(calls) == 1
    assert calls[0] == {
        "input_path": str(temp_paths[0]),
        "output_path": None,
        "kernel_name": runner.KERNEL_NAME,
        "progress_bar": True,
        "cwd": notebook.parent,
        "iopub_timeout": runner.IOPUB_TIMEOUT_SECONDS,
        "raise_on_iopub_timeout": False,
    }
    assert injected == [[]]
    assert not temp_paths[0].exists()


def test_kernel_path_prefers_active_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment_prefix = tmp_path / "env"
    monkeypatch.setattr(runner.sys, "prefix", str(environment_prefix))
    monkeypatch.setenv("JUPYTER_PATH", "/custom/jupyter")

    runner.configure_kernel_path()

    assert runner.os.environ["JUPYTER_PATH"].split(runner.os.pathsep) == [
        str(environment_prefix / "share" / "jupyter"),
        "/custom/jupyter",
    ]

    runner.configure_kernel_path()
    assert runner.os.environ["JUPYTER_PATH"].split(runner.os.pathsep) == [
        str(environment_prefix / "share" / "jupyter"),
        "/custom/jupyter",
    ]


def test_main_executes_exact_notebooks_serially_and_forwards_full(
    notebook_collections: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        runner,
        "run_notebook",
        lambda path, *, full=False: executed.append((path, full)),
    )

    assert (
        runner.main(
            [
                "--full",
                "--notebook",
                "docs/source/notebooks/alpha-pymc.ipynb",
                "--notebook",
                "docs/source/knowledgebase/concepts.ipynb",
            ]
        )
        == 0
    )
    assert executed == [
        (notebook_collections["gallery"] / "alpha-pymc.ipynb", True),
        (notebook_collections["knowledgebase"] / "concepts.ipynb", True),
    ]


def test_main_configures_kernel_before_execution(
    notebook_collections: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment_prefix = tmp_path / "env"
    expected_kernel_path = str(environment_prefix / "share" / "jupyter")
    observed_kernel_paths: list[str] = []
    monkeypatch.setattr(runner.sys, "prefix", str(environment_prefix))
    monkeypatch.delenv("JUPYTER_PATH", raising=False)
    monkeypatch.setattr(
        runner,
        "run_notebook",
        lambda path, *, full=False: observed_kernel_paths.append(
            runner.os.environ["JUPYTER_PATH"]
        ),
    )

    runner.main(["--notebook", "docs/source/notebooks/alpha-pymc.ipynb"])

    assert observed_kernel_paths == [expected_kernel_path]


def test_main_attempts_all_notebooks_before_reporting_failures(
    notebook_collections: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []

    def run_notebook(path: Path, *, full: bool = False) -> None:
        attempted.append(path.name)
        if path.name in {"alpha-pymc.ipynb", "other.ipynb"}:
            raise ValueError("notebook failed")

    monkeypatch.setattr(runner, "run_notebook", run_notebook)

    with pytest.raises(
        RuntimeError,
        match=(
            r"2 notebook\(s\) failed: alpha-pymc.ipynb: ValueError: notebook "
            r"failed; other.ipynb: ValueError: notebook failed"
        ),
    ):
        runner.main([])

    assert attempted == [
        "alpha-pymc.ipynb",
        "beta-sklearn.ipynb",
        "other.ipynb",
        "pymc-introduction.ipynb",
        "sklearn-overview.ipynb",
    ]


def test_main_rejects_empty_selection(
    notebook_collections: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="No notebooks matched"):
        runner.main(["--pattern", "does-not-exist-*.ipynb"])


def test_parallel_option_is_not_supported(
    notebook_collections: dict[str, Path],
) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--parallel"])


def test_injected_mock_builds_datatree_groups() -> None:
    injected_path = REPO_ROOT / "scripts" / "run_notebooks" / "injected.py"
    script = f"""
import runpy
import pymc as pm
import xarray as xr
namespace = runpy.run_path({str(injected_path)!r})
with pm.Model() as model:
    x = pm.Normal("x")
    pm.Normal("y", x, 1, observed=[0.0])
    pm.Normal("z", x, 1, observed=[1.0])
    result = namespace["mock_sample"](
        draws=3,
        random_seed=42,
        model=model,
        idata_kwargs={{"log_likelihood": ["y"]}},
    )
assert isinstance(result, xr.DataTree)
assert "posterior" in result
assert "sample_stats" in result
assert "log_likelihood" in result
assert set(result["log_likelihood"].data_vars) == {{"y"}}
assert {{"chain", "draw"}}.issubset(result["log_likelihood"]["y"].dims)
assert "prior" not in result
assert "prior_predictive" not in result
assert result["posterior"]["x"].sizes["draw"] == 100
# arviz-stats refuses multi-chain diagnostics on a single-chain posterior
# (`_mtc_c requires at least 2 chains`), which is what `az.plot_rank_dist` calls.
assert namespace["MOCK_CHAINS"] >= 2
assert result["posterior"]["x"].sizes["chain"] == namespace["MOCK_CHAINS"]
# sample_stats describes the posterior, so its chain count must agree with it.
assert result["sample_stats"]["diverging"].sizes["chain"] == namespace["MOCK_CHAINS"]
assert result["sample_stats"]["diverging"].sizes["draw"] == 100
# Chains must hold distinct draws, otherwise rank diagnostics compare a chain to itself.
assert not result["posterior"]["x"].isel(chain=0).equals(
    result["posterior"]["x"].isel(chain=1)
)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_injected_mock_retries_numba_division_by_zero() -> None:
    injected_path = REPO_ROOT / "scripts" / "run_notebooks" / "injected.py"
    script = f"""
import runpy
import pymc as pm
namespace = runpy.run_path({str(injected_path)!r})
sample_prior_predictive = pm.sample_prior_predictive
calls = []
def flaky_sample_prior_predictive(*args, **kwargs):
    calls.append(kwargs)
    if len(calls) == 1:
        raise ZeroDivisionError("numba prior RNG stream")
    return sample_prior_predictive(*args, **kwargs)
pm.sample_prior_predictive = flaky_sample_prior_predictive
with pm.Model() as model:
    pm.Normal("x")
    namespace["mock_sample"](random_seed=1040, model=model)
assert [call["random_seed"] for call in calls] == [1040, 1040]
assert calls[0].get("compile_kwargs") is None
assert calls[1]["compile_kwargs"] == {{"mode": namespace["FALLBACK_COMPILE_MODE"]}}
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_injected_mock_does_not_retry_other_errors() -> None:
    injected_path = REPO_ROOT / "scripts" / "run_notebooks" / "injected.py"
    script = f"""
import runpy
import pymc as pm
namespace = runpy.run_path({str(injected_path)!r})
calls = 0
def failing_sample_prior_predictive(*args, **kwargs):
    global calls
    calls += 1
    raise ValueError("model error")
pm.sample_prior_predictive = failing_sample_prior_predictive
with pm.Model() as model:
    pm.Normal("x")
    try:
        namespace["mock_sample"](random_seed=1040, model=model)
    except ValueError as error:
        assert str(error) == "model error"
    else:
        raise AssertionError("ValueError was not propagated")
assert calls == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_production_skip_configuration_is_consistent() -> None:
    skip_notebooks = set(
        yaml.safe_load(
            (REPO_ROOT / "scripts" / "run_notebooks" / "skip_notebooks.yml").read_text()
        )
    )
    assert skip_notebooks == {
        "gallery/instrumental-variables-pymc.ipynb",
        "gallery/instrumental-variables-weak-instruments.ipynb",
        "gallery/instrumental-variables-variable-selection-priors.ipynb",
        "gallery/interrupted-time-series-causalpy-vs-causalimpact.ipynb",
    }

    for key in skip_notebooks:
        collection, relative_path = key.split("/", maxsplit=1)
        notebook = runner.NOTEBOOK_COLLECTIONS[collection] / relative_path
        assert notebook.is_file()
        assert notebook.resolve() not in runner.get_notebooks(collections=[collection])
        with pytest.raises(ValueError, match="Use --full"):
            runner.get_notebooks(notebook_paths=[str(notebook)])
        assert runner.get_notebooks(notebook_paths=[str(notebook)], full=True) == [
            notebook.resolve()
        ]


def test_unknown_collection_is_rejected(
    notebook_collections: dict[str, Path],
) -> None:
    with pytest.raises(ValueError, match="Unknown notebook collection"):
        runner.get_notebooks(collections=["unknown"])

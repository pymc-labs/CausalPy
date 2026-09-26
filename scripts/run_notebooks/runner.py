"""Run CausalPy documentation notebooks serially.

Examples
--------
Run all notebooks (with mock injection, outputs discarded):

    python scripts/run_notebooks/runner.py

Run only PyMC notebooks:

    python scripts/run_notebooks/runner.py --pattern "*-pymc.ipynb"

Run only sklearn notebooks:

    python scripts/run_notebooks/runner.py --pattern "*-sklearn.ipynb"

Exclude PyMC and sklearn notebooks (run others):

    python scripts/run_notebooks/runner.py --exclude-pattern pymc --exclude-pattern sklearn

Full execution (no mock, saves outputs in place):

    python scripts/run_notebooks/runner.py --full

Full execution for a single notebook:

    python scripts/run_notebooks/runner.py --full \
        --notebook docs/source/notebooks/synthetic-control-sklearn.ipynb

Run the knowledgebase notebooks:

    python scripts/run_notebooks/runner.py --collection knowledgebase

"""

import argparse
import logging
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

# Monkey-patch nbclient to handle display_id=None for widget updates.
# This fixes an issue where ipywidgets/tqdm progress bars cause
# "assert display_id is not None" errors in nbclient.
import nbclient.client
import papermill
import yaml
from nbformat.notebooknode import NotebookNode
from papermill.iorw import load_notebook_node, write_ipynb

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
NOTEBOOK_COLLECTIONS = {
    "gallery": REPO_ROOT / "docs" / "source" / "notebooks",
    "knowledgebase": REPO_ROOT / "docs" / "source" / "knowledgebase",
}
KERNEL_NAME = "python3"
LOGGER = logging.getLogger(__name__)
IOPUB_TIMEOUT_SECONDS = 300

INJECTED_CODE_FILE = HERE / "injected.py"
INJECTED_CODE = INJECTED_CODE_FILE.read_text()

SKIP_NOTEBOOKS_FILE = HERE / "skip_notebooks.yml"
SKIP_NOTEBOOKS = set(yaml.safe_load(SKIP_NOTEBOOKS_FILE.read_text()))

_original_output = nbclient.client.NotebookClient.output


def _patched_output(self, outs, msg, display_id, cell_index):
    """Patched output method that catches assertion errors from widget updates."""
    try:
        return _original_output(self, outs, msg, display_id, cell_index)
    except AssertionError:
        # Silently skip messages that cause display_id assertion errors
        # (typically from ipywidgets/tqdm progress bar updates)
        return None


nbclient.client.NotebookClient.output = _patched_output


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def configure_kernel_path() -> None:
    """Prefer the active Python environment's kernels over user kernels."""
    environment_jupyter_path = str(Path(sys.prefix) / "share" / "jupyter")
    existing_paths = os.environ.get("JUPYTER_PATH", "").split(os.pathsep)
    remaining_paths = [
        path for path in existing_paths if path and path != environment_jupyter_path
    ]
    os.environ["JUPYTER_PATH"] = os.pathsep.join(
        [environment_jupyter_path, *remaining_paths]
    )


def generate_random_id() -> str:
    return str(uuid4())


def clear_cell_outputs(cells: list) -> None:
    """Clear all outputs from cells to avoid widget state issues with nbclient."""
    for cell in cells:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def inject_mock_code(cells: list) -> None:
    """Inject mock pm.sample code at the start of the notebook."""
    clear_cell_outputs(cells)
    cells.insert(
        0,
        NotebookNode(
            id=f"code-injection-{generate_random_id()}",
            execution_count=sum(map(ord, "Mock pm.sample")),
            cell_type="code",
            metadata={"tags": []},
            outputs=[],
            source=INJECTED_CODE,
        ),
    )


def run_notebook(notebook_path: Path, *, full: bool = False) -> None:
    """Run a notebook, optionally without mock injection and saving outputs in place.

    Parameters
    ----------
    notebook_path : Path
        Path to the notebook to execute.
    full : bool
        If True, execute without mock injection and overwrite the notebook
        with fresh outputs.  If False (default), inject mock code and
        discard outputs.
    """
    mode = "full" if full else "mock"
    LOGGER.info(f"Running notebook ({mode}): {notebook_path.name}")

    if full:
        papermill.execute_notebook(
            input_path=str(notebook_path),
            output_path=str(notebook_path),
            kernel_name=KERNEL_NAME,
            progress_bar=True,
            cwd=notebook_path.parent,
            iopub_timeout=IOPUB_TIMEOUT_SECONDS,
            raise_on_iopub_timeout=False,
        )
        return

    nb = load_notebook_node(str(notebook_path))
    inject_mock_code(nb.cells)

    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
            temp_path = Path(f.name)
            write_ipynb(nb, f.name)

        papermill.execute_notebook(
            input_path=str(temp_path),
            output_path=None,  # Discard output
            kernel_name=KERNEL_NAME,
            progress_bar=True,
            cwd=notebook_path.parent,
            iopub_timeout=IOPUB_TIMEOUT_SECONDS,
            raise_on_iopub_timeout=False,
        )
    except Exception:
        LOGGER.error(f"Error running notebook: {notebook_path.name}")
        raise
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                LOGGER.warning(
                    "Failed to delete temporary notebook file %s: %s",
                    temp_path,
                    cleanup_error,
                )


def get_notebooks(
    pattern: str | None = None,
    exclude_patterns: list[str] | None = None,
    collections: list[str] | None = None,
    notebook_paths: list[str] | None = None,
    *,
    full: bool = False,
) -> list[Path]:
    """Return selected notebooks after validating collection and skip policies.

    Explicit notebook paths are limited to the configured documentation
    collections. Mock-incompatible notebooks can be selected explicitly only
    in full mode; collection/pattern selection omits them in mock mode.
    """
    if notebook_paths and (collections or pattern or exclude_patterns):
        raise ValueError(
            "--notebook cannot be combined with --collection, --pattern, or "
            "--exclude-pattern."
        )

    selected_collections = collections or ["gallery"]
    unknown_collections = sorted(set(selected_collections) - set(NOTEBOOK_COLLECTIONS))
    if unknown_collections:
        raise ValueError(
            f"Unknown notebook collection(s): {', '.join(unknown_collections)}"
        )

    repo_root = REPO_ROOT.resolve()
    allowed_roots = {
        name: path.resolve() for name, path in NOTEBOOK_COLLECTIONS.items()
    }
    symlinked_roots = [
        name for name, path in NOTEBOOK_COLLECTIONS.items() if path.is_symlink()
    ]
    if symlinked_roots:
        raise ValueError(
            "Notebook collection root(s) cannot be symlinks: "
            f"{', '.join(sorted(symlinked_roots))}"
        )
    escaped_roots = [
        name
        for name, root in allowed_roots.items()
        if not root.is_relative_to(repo_root)
    ]
    if escaped_roots:
        raise ValueError(
            "Notebook collection root(s) are outside the repository: "
            f"{', '.join(sorted(escaped_roots))}"
        )
    roots = {name: allowed_roots[name] for name in selected_collections}

    def validate_path(
        path: Path,
        raw_path: str,
        expected_root: Path | None = None,
    ) -> Path:
        path = path.resolve()
        if path.suffix != ".ipynb" or not path.is_file():
            raise ValueError(f"Notebook does not exist: {raw_path}")
        if expected_root is not None and not path.is_relative_to(expected_root):
            raise ValueError(f"Notebook escapes its configured collection: {raw_path}")
        if not any(path.is_relative_to(root) for root in allowed_roots.values()):
            raise ValueError(
                f"Notebook is outside configured documentation collections: {raw_path}"
            )
        return path

    def notebook_key(path: Path) -> str:
        for name, root in allowed_roots.items():
            if path.is_relative_to(root):
                return f"{name}/{path.relative_to(root)}"
        raise ValueError(f"Notebook is outside configured collections: {path}")

    if notebook_paths:
        notebooks = []
        for raw_path in notebook_paths:
            path = Path(raw_path)
            if not path.is_absolute():
                path = REPO_ROOT / path
            lexical_path = Path(os.path.abspath(path))
            matching_roots = [
                allowed_roots[name]
                for name, collection_root in NOTEBOOK_COLLECTIONS.items()
                if lexical_path.is_relative_to(Path(os.path.abspath(collection_root)))
            ]
            if not matching_roots:
                raise ValueError(
                    "Notebook is outside configured documentation collections: "
                    f"{raw_path}"
                )
            notebooks.append(
                validate_path(path, raw_path, expected_root=matching_roots[0])
            )
    else:
        notebooks = [
            validate_path(notebook, str(notebook), expected_root=root)
            for root in roots.values()
            for notebook in root.glob("*.ipynb")
        ]

    skipped = [
        notebook for notebook in notebooks if notebook_key(notebook) in SKIP_NOTEBOOKS
    ]
    if skipped and notebook_paths and not full:
        names = ", ".join(sorted(notebook.name for notebook in skipped))
        raise ValueError(
            f"Mock execution is unsupported for: {names}. Use --full for explicit "
            "serialized execution."
        )
    if not notebook_paths or not full:
        notebooks = [notebook for notebook in notebooks if notebook not in skipped]

    if pattern:
        notebooks = [nb for nb in notebooks if Path(nb).match(pattern)]

    if exclude_patterns:
        for exc in exclude_patterns:
            if any(character in exc for character in "*?["):
                notebooks = [nb for nb in notebooks if not nb.match(exc)]
            else:
                notebooks = [nb for nb in notebooks if exc not in nb.name]

    if notebook_paths:
        return list(dict.fromkeys(notebooks))
    return sorted(set(notebooks))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CausalPy notebooks.")
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob pattern to filter notebooks (e.g., '*-pymc.ipynb')",
    )
    parser.add_argument(
        "--exclude-pattern",
        type=str,
        action="append",
        dest="exclude_patterns",
        help="Pattern to exclude from notebook names (can be used multiple times)",
    )
    parser.add_argument(
        "--collection",
        action="append",
        choices=sorted(NOTEBOOK_COLLECTIONS),
        dest="collections",
        help="Notebook collection to run (repeatable; defaults to gallery).",
    )
    parser.add_argument(
        "--notebook",
        action="append",
        dest="notebook_paths",
        help=(
            "Exact notebook path relative to the repository (repeatable). Paths must "
            "be inside a configured documentation collection."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Full execution: skip mock injection and overwrite notebooks with fresh outputs. This can take a long time.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="List the selected notebooks without executing or modifying them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Select and execute notebooks serially from command-line arguments."""
    setup_logging()
    configure_kernel_path()
    args = parse_args(argv)

    notebooks = get_notebooks(
        pattern=args.pattern,
        exclude_patterns=args.exclude_patterns,
        collections=args.collections,
        notebook_paths=args.notebook_paths,
        full=args.full,
    )

    LOGGER.info(f"Found {len(notebooks)} notebooks to run")
    for nb in notebooks:
        LOGGER.info(f"  - {nb.name}")

    if not notebooks:
        raise ValueError("No notebooks matched the requested selection.")

    if args.list_only:
        return 0

    if args.full:
        LOGGER.warning(
            "Full execution mode: notebooks will be run without mock injection "
            "and outputs will be saved in place. This can take a long time."
        )

    failures: list[tuple[Path, Exception]] = []
    for notebook in notebooks:
        try:
            run_notebook(notebook, full=args.full)
        except Exception as error:
            LOGGER.exception("Notebook failed: %s", notebook)
            failures.append((notebook, error))

    if failures:
        failure_details = "; ".join(
            f"{notebook.name}: {type(error).__name__}: {error}"
            for notebook, error in failures
        )
        raise RuntimeError(
            f"{len(failures)} notebook(s) failed: {failure_details}"
        ) from failures[0][1]

    LOGGER.info("All notebooks completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Script to run notebooks in docs/source/notebooks directory.

Examples
--------
Run all notebooks:

    python scripts/run_notebooks/runner.py

Run only PyMC notebooks:

    python scripts/run_notebooks/runner.py --pattern "*_pymc*.ipynb"

Run only sklearn notebooks:

    python scripts/run_notebooks/runner.py --pattern "*_skl*.ipynb"

Exclude PyMC and sklearn notebooks (run others):

    python scripts/run_notebooks/runner.py --exclude-pattern _pymc --exclude-pattern _skl

"""

import argparse
import logging
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
NOTEBOOKS_PATH = Path("docs/source/notebooks")
KERNEL_NAME = "python3"

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


def run_notebook(notebook_path: Path) -> None:
    """Run a notebook with mocked pm.sample."""
    logging.info(f"Running notebook: {notebook_path.name}")

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
        )
    except Exception as e:
        logging.error(f"Error running notebook: {notebook_path.name}")
        raise e
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                logging.warning(
                    "Failed to delete temporary notebook file %s: %s",
                    temp_path,
                    cleanup_error,
                )


def get_notebooks(
    pattern: str | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """Get list of notebooks to run, optionally filtered."""
    notebooks = list(NOTEBOOKS_PATH.glob("*.ipynb"))

    # Filter out notebooks that are incompatible with the mock
    notebooks = [nb for nb in notebooks if nb.name not in SKIP_NOTEBOOKS]

    if pattern:
        notebooks = [nb for nb in notebooks if Path(nb).match(pattern)]

    if exclude_patterns:
        for exc in exclude_patterns:
            notebooks = [nb for nb in notebooks if exc not in nb.name]

    return sorted(notebooks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CausalPy notebooks.")
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob pattern to filter notebooks (e.g., '*_pymc*.ipynb')",
    )
    parser.add_argument(
        "--exclude-pattern",
        type=str,
        action="append",
        dest="exclude_patterns",
        help="Pattern to exclude from notebook names (can be used multiple times)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run notebooks in parallel when possible.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()

    notebooks = get_notebooks(
        pattern=args.pattern,
        exclude_patterns=args.exclude_patterns,
    )

    logging.info(f"Found {len(notebooks)} notebooks to run")
    for nb in notebooks:
        logging.info(f"  - {nb.name}")

    if args.parallel:
        try:
            from joblib import Parallel, delayed
        except ImportError as exc:
            raise ImportError(
                "Parallel execution requires joblib. Install it or run without --parallel."
            ) from exc

        Parallel(n_jobs=-1)(delayed(run_notebook)(notebook) for notebook in notebooks)
    else:
        for notebook in notebooks:
            run_notebook(notebook)

    logging.info("All notebooks completed successfully!")

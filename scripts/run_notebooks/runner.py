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

import papermill
from nbformat.notebooknode import NotebookNode
from papermill.iorw import load_notebook_node, write_ipynb

HERE = Path(__file__).parent
NOTEBOOKS_PATH = Path("docs/source/notebooks")
KERNEL_NAME = "python3"

INJECTED_CODE_FILE = HERE / "injected.py"
INJECTED_CODE = INJECTED_CODE_FILE.read_text()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def inject_mock_code(cells: list) -> None:
    """Inject mock pm.sample code at the start of the notebook."""
    cells.insert(
        0,
        NotebookNode(
            id="mock-injection",
            execution_count=0,
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

    with NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
        write_ipynb(nb, f.name)
        try:
            papermill.execute_notebook(
                input_path=f.name,
                output_path=None,  # Discard output
                kernel_name=KERNEL_NAME,
                progress_bar=True,
                cwd=notebook_path.parent,
            )
        except Exception as e:
            logging.error(f"Error running notebook: {notebook_path.name}")
            raise e


def get_notebooks(
    pattern: str | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """Get list of notebooks to run, optionally filtered."""
    notebooks = list(NOTEBOOKS_PATH.glob("*.ipynb"))

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

    for notebook in notebooks:
        run_notebook(notebook)

    logging.info("All notebooks completed successfully!")

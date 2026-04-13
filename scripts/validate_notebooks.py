"""Validate Jupyter notebooks against the nbformat schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import nbformat
from nbformat import NotebookNode
from nbformat.validator import NotebookValidationError


def _extract_path_index(path_segments: list[Any], segment_name: str) -> int | None:
    """Return the integer index that follows a path segment name."""
    for i, segment in enumerate(path_segments[:-1]):
        if segment == segment_name and isinstance(path_segments[i + 1], int):
            return path_segments[i + 1]
    return None


def _get_output_type(
    notebook: NotebookNode,
    cell_index: int | None,
    output_index: int | None,
) -> str | None:
    """Return output_type for the failing output object if available."""
    if cell_index is None or output_index is None:
        return None

    try:
        output = notebook["cells"][cell_index]["outputs"][output_index]
    except (IndexError, KeyError, TypeError):
        return None

    if isinstance(output, dict):
        return output.get("output_type")
    return None


def _format_validation_error(
    notebook_path: Path,
    notebook: NotebookNode,
    error: NotebookValidationError,
) -> str:
    """Format a NotebookValidationError into actionable stderr output."""
    path_segments = list(getattr(error, "absolute_path", []))
    cell_index = _extract_path_index(path_segments, "cells")
    output_index = _extract_path_index(path_segments, "outputs")
    output_type = _get_output_type(notebook, cell_index, output_index)

    details = [f"{notebook_path}: notebook schema validation failed"]

    location_parts = []
    if cell_index is not None:
        location_parts.append(f"cell[{cell_index}]")
    if output_index is not None:
        location_parts.append(f"output[{output_index}]")
    if output_type is not None:
        location_parts.append(f"type={output_type}")
    if location_parts:
        details.append(f"  location: {' '.join(location_parts)}")

    if path_segments:
        details.append(f"  path: {'/'.join(str(segment) for segment in path_segments)}")

    details.append(f"  error: {error.message}")
    return "\n".join(details)


def validate_notebook(notebook_path: Path) -> tuple[bool, str | None]:
    """Validate a single notebook and return (is_valid, error_message)."""
    try:
        with notebook_path.open(encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
    except Exception as error:  # noqa: BLE001
        return False, f"{notebook_path}: failed to read notebook: {error}"

    try:
        nbformat.validate(notebook)
    except NotebookValidationError as error:
        return False, _format_validation_error(notebook_path, notebook, error)

    return True, None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate Jupyter notebooks against nbformat schema.",
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Notebook files to validate.",
    )
    return parser.parse_args()


def main() -> int:
    """Run validation for all provided notebook paths."""
    args = parse_args()
    if not args.notebooks:
        return 0

    invalid_count = 0
    for notebook_path in args.notebooks:
        is_valid, error_message = validate_notebook(notebook_path)
        if is_valid:
            continue
        invalid_count += 1
        if error_message is not None:
            print(error_message, file=sys.stderr)

    if invalid_count == 0:
        return 0

    notebook_word = "notebook" if invalid_count == 1 else "notebooks"
    print(f"Validation failed for {invalid_count} {notebook_word}.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

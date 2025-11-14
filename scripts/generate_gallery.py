#!/usr/bin/env python3
"""
Generate thumbnails for CausalPy documentation gallery.

This script scans notebooks in docs/source/notebooks/ and generates
thumbnails from the first plot in each notebook. The index.md file
should be maintained manually.
"""

import base64
import io
import sys
from pathlib import Path

try:
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
except ImportError:
    print(
        "Error: nbformat and nbconvert are required. Install with: pip install nbformat nbconvert"
    )
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Warning: Pillow not found. Thumbnails will not be generated.")
    Image = None  # type: ignore[assignment,misc]


def _find_image_in_notebook(nb) -> str | None:
    """Find first PNG image in notebook outputs."""
    for cell in nb.cells:
        if cell.cell_type == "code" and hasattr(cell, "outputs") and cell.outputs:
            for output in cell.outputs:
                if output.output_type in ("display_data", "execute_result"):
                    if image_data := output.get("data", {}).get("image/png"):
                        return image_data
    return None


def extract_first_image(notebook_path: Path, output_dir: Path) -> str | None:
    """Extract first image from notebook outputs (without executing if outputs exist)."""
    if Image is None:
        return None

    try:
        nb = nbformat.reads(notebook_path.read_text(encoding="utf-8"), as_version=4)

        # Try to find images in existing outputs first
        if image_data := _find_image_in_notebook(nb):
            return _save_thumbnail(notebook_path, output_dir, image_data)

        # Execute if notebook has no outputs
        if not any(
            cell.cell_type == "code" and hasattr(cell, "outputs") and cell.outputs
            for cell in nb.cells
        ):
            print(f"  Executing {notebook_path.name} to generate thumbnail...")
            try:
                ExecutePreprocessor(timeout=120, kernel_name="python3").preprocess(
                    nb, {"metadata": {"path": str(notebook_path.parent)}}
                )
                if image_data := _find_image_in_notebook(nb):
                    return _save_thumbnail(notebook_path, output_dir, image_data)
            except Exception as e:
                print(f"  Warning: Failed to execute {notebook_path.name}: {e}")

        return None
    except Exception as e:
        print(f"Warning: Could not generate thumbnail for {notebook_path.name}: {e}")
        return None


def _save_thumbnail(
    notebook_path: Path, output_dir: Path, image_data: str
) -> str | None:
    """Save thumbnail image from base64 data."""
    try:
        thumbnail_name = f"{notebook_path.stem}.png"
        thumbnail_path = output_dir / thumbnail_name

        # Decode and process image in memory
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        target_size = (400, 250)
        img.thumbnail(target_size, Image.Resampling.LANCZOS)

        # Create padded image and save
        new_img = Image.new("RGB", target_size, (255, 255, 255))
        new_img.paste(
            img,
            ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2),
        )
        new_img.save(thumbnail_path)

        # Path relative to document location (notebooks/)
        # Need to go up one level to source/, then into _static/thumbnails/
        return f"../_static/thumbnails/{thumbnail_name}"
    except Exception as e:
        print(f"Warning: Could not save thumbnail for {notebook_path.name}: {e}")
        return None


def main():
    """Main function to generate thumbnails only."""
    # Paths
    repo_root = Path(__file__).parent.parent
    notebooks_dir = repo_root / "docs" / "source" / "notebooks"
    thumbnails_dir = repo_root / "docs" / "source" / "_static" / "thumbnails"

    # Create thumbnails directory
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    # Find all notebooks
    notebook_files = sorted(notebooks_dir.glob("*.ipynb"))

    if not notebook_files:
        print("No notebooks found!")
        sys.exit(1)

    print(f"Found {len(notebook_files)} notebooks")

    # Process each notebook to generate thumbnails
    for nb_path in notebook_files:
        print(f"Processing {nb_path.name}...")
        extract_first_image(nb_path, thumbnails_dir)

    print(f"Thumbnails saved to {thumbnails_dir}")


if __name__ == "__main__":
    main()

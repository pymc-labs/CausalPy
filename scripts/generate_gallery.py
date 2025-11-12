#!/usr/bin/env python3
"""
Generate example gallery for CausalPy documentation.

This script scans notebooks in docs/source/notebooks/, extracts metadata,
generates thumbnails from the first plot in each notebook, and creates
a gallery page using sphinx-design cards.
"""

import base64
import io
import re
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


def load_categories_from_index(index_path: Path) -> dict[str, list[str]]:
    """
    Load category structure from existing index.md.

    Reads the markdown file and extracts:
    - Category names from ## headers
    - Notebook names from :link: fields under each category

    Returns
    -------
    dict[str, list[str]]
        Mapping from category name to list of notebook names (without .ipynb)
    """
    if not index_path.exists():
        return {}

    try:
        categories: dict[str, list[str]] = {}
        current_category = None
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                current_category = line[3:].strip()
                if current_category and current_category != "Example Gallery":
                    categories[current_category] = []
            elif current_category and (match := re.search(r":link:\s+(\S+)", line)):
                categories[current_category].append(match.group(1))
        return categories
    except Exception as e:
        print(f"Warning: Could not load categories from {index_path}: {e}")
        return {}


def get_notebook_category(filename: str, category_mapping: dict[str, list[str]]) -> str:
    """Determine the category for a notebook from the loaded mapping."""
    notebook_name = filename.replace(".ipynb", "")
    return next(
        (
            cat
            for cat, notebooks in category_mapping.items()
            if notebook_name in notebooks
        ),
        "Other",
    )


def extract_metadata(notebook_path: Path) -> str:
    """Extract title from notebook."""
    nb = nbformat.reads(notebook_path.read_text(encoding="utf-8"), as_version=4)

    # Look for title in first markdown cell
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            if match := re.search(r"^#+\s+(.+)$", cell.source.strip(), re.MULTILINE):
                return match.group(1).strip()

    # Fallback to filename-based title
    return notebook_path.stem.replace("_", " ").title()


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


def generate_gallery_markdown(
    notebooks_data: list[dict],
    output_path: Path,
    category_mapping: dict[str, list[str]],
):
    """Generate gallery markdown file with sphinx-design cards."""
    # Group notebooks by category
    categories: dict[str, list[dict]] = {}
    for nb_data in notebooks_data:
        categories.setdefault(nb_data["category"], []).append(nb_data)

    # Sort categories maintaining order from index.md
    sorted_categories = list(category_mapping.keys() & categories.keys()) + list(
        categories.keys() - category_mapping.keys()
    )

    # Generate markdown
    lines = ["# Example Gallery\n"]

    for category in sorted_categories:
        notebooks = sorted(categories[category], key=lambda x: x["filename"])

        lines.extend([f"## {category}\n", "::::{grid} 1 2 3 3\n", ":gutter: 3\n\n"])

        for nb in notebooks:
            doc_name = nb["filename"].replace(".ipynb", "")
            card_lines = [
                f":::{'{grid-item-card}'} {nb['title']}\n",
                ":class-card: sd-card-h-100\n",
            ]
            if nb.get("thumbnail"):
                card_lines.append(f":img-top: {nb['thumbnail']}\n")
            card_lines.extend([f":link: {doc_name}\n", ":link-type: doc\n", ":::\n"])
            lines.extend(card_lines)

        lines.append("::::\n\n")

    output_path.write_text("".join(lines), encoding="utf-8")


def main():
    """Main function to generate gallery."""
    # Paths
    repo_root = Path(__file__).parent.parent
    notebooks_dir = repo_root / "docs" / "source" / "notebooks"
    thumbnails_dir = repo_root / "docs" / "source" / "_static" / "thumbnails"
    output_file = notebooks_dir / "index.md"

    # Create thumbnails directory
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    # Load category structure from existing index.md
    category_mapping = load_categories_from_index(output_file)
    if category_mapping:
        print(f"Loaded {len(category_mapping)} categories from index.md")

    # Find all notebooks
    notebook_files = sorted(notebooks_dir.glob("*.ipynb"))

    if not notebook_files:
        print("No notebooks found!")
        sys.exit(1)

    print(f"Found {len(notebook_files)} notebooks")

    # Process each notebook
    notebooks_data = []
    for nb_path in notebook_files:
        print(f"Processing {nb_path.name}...")

        notebooks_data.append(
            {
                "filename": nb_path.name,
                "title": extract_metadata(nb_path),
                "category": get_notebook_category(nb_path.name, category_mapping),
                "thumbnail": extract_first_image(nb_path, thumbnails_dir),
            }
        )

    # Generate gallery markdown
    print("Generating gallery markdown...")
    generate_gallery_markdown(notebooks_data, output_file, category_mapping)

    print(f"Gallery generated successfully at {output_file}")
    print(f"Thumbnails saved to {thumbnails_dir}")


if __name__ == "__main__":
    main()

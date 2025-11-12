#!/usr/bin/env python3
"""
Generate example gallery for CausalPy documentation.

This script scans notebooks in docs/source/notebooks/, extracts metadata,
generates thumbnails from the first plot in each notebook, and creates
a gallery page using sphinx-design cards.
"""

import base64
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def load_categories_from_index(index_path: Path) -> Dict[str, List[str]]:
    """
    Load category structure from existing index.md.

    Reads the markdown file and extracts:
    - Category names from ## headers
    - Notebook names from :link: fields under each category

    Returns
    -------
    Dict[str, List[str]]
        Mapping from category name to list of notebook names (without .ipynb)
    """
    categories: Dict[str, List[str]] = {}
    current_category = None

    if not index_path.exists():
        return categories

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                # Check for category header (## Category Name)
                if line.startswith("## "):
                    current_category = line[3:].strip()
                    if current_category and current_category != "Example Gallery":
                        categories[current_category] = []
                # Check for notebook links under current category
                elif current_category and ":link:" in line:
                    # Extract notebook name from :link: notebook_name
                    link_match = re.search(r":link:\s+(\S+)", line)
                    if link_match:
                        notebook_name = link_match.group(1)
                        categories[current_category].append(notebook_name)
    except Exception as e:
        print(f"Warning: Could not load categories from {index_path}: {e}")

    return categories


def get_notebook_category(filename: str, category_mapping: Dict[str, List[str]]) -> str:
    """Determine the category for a notebook from the loaded mapping."""
    notebook_name = filename.replace(".ipynb", "")
    for category, notebooks in category_mapping.items():
        if notebook_name in notebooks:
            return category
    return "Other"


def extract_metadata(notebook_path: Path) -> Tuple[str, str]:
    """Extract title and description from notebook."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    title = None
    description = ""

    # Look for title in first markdown cell
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            source = cell.source.strip()
            # Look for H1 or H2 title
            title_match = re.match(r"^#+\s+(.+)$", source, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
                # Get description from rest of first markdown cell
                lines = source.split("\n")
                description_lines = []
                found_title = False
                for line in lines:
                    if re.match(r"^#+\s+", line):
                        found_title = True
                        continue
                    if found_title and line.strip():
                        # Skip MyST directives and formulas
                        stripped = line.strip()
                        if stripped.startswith(":::"):
                            break  # Stop at first MyST directive
                        if stripped.startswith("$$") or stripped.startswith("$"):
                            continue  # Skip math formulas
                        if stripped.startswith("*") and ":" in stripped:
                            continue  # Skip list items that are definitions
                        description_lines.append(stripped)
                        if len(description_lines) >= 2:  # Take first 2 meaningful lines
                            break
                description = " ".join(description_lines)
                break

    # Fallback to filename-based title
    if not title:
        title = notebook_path.stem.replace("_", " ").title()

    return title, description


def extract_first_image(notebook_path: Path, output_dir: Path) -> Optional[str]:
    """Extract first image from notebook outputs (without executing if outputs exist)."""
    if Image is None:
        return None

    try:
        # Read notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        # First, try to find images in existing outputs (no execution needed)
        for cell in nb.cells:
            if cell.cell_type == "code" and hasattr(cell, "outputs") and cell.outputs:
                for output in cell.outputs:
                    if (
                        output.output_type == "display_data"
                        or output.output_type == "execute_result"
                    ):
                        if "image/png" in output.get("data", {}):
                            image_data = output["data"]["image/png"]
                            return _save_thumbnail(
                                notebook_path, output_dir, image_data
                            )

        # If no images found in existing outputs, try executing (with short timeout)
        # Only execute if notebook appears to have no outputs
        has_outputs = any(
            cell.cell_type == "code" and hasattr(cell, "outputs") and cell.outputs
            for cell in nb.cells
        )

        if not has_outputs:
            print(f"  Executing {notebook_path.name} to generate thumbnail...")
            ep = ExecutePreprocessor(
                timeout=120, kernel_name="python3"
            )  # 2 min timeout
            try:
                ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
            except Exception as e:
                print(f"  Warning: Failed to execute {notebook_path.name}: {e}")
                return None

            # Find first image in outputs after execution
            for cell in nb.cells:
                if cell.cell_type == "code" and hasattr(cell, "outputs"):
                    for output in cell.outputs:
                        if (
                            output.output_type == "display_data"
                            or output.output_type == "execute_result"
                        ):
                            if "image/png" in output.get("data", {}):
                                image_data = output["data"]["image/png"]
                                return _save_thumbnail(
                                    notebook_path, output_dir, image_data
                                )

        return None
    except Exception as e:
        print(f"Warning: Could not generate thumbnail for {notebook_path.name}: {e}")
        return None


def _save_thumbnail(
    notebook_path: Path, output_dir: Path, image_data: str
) -> Optional[str]:
    """Save thumbnail image from base64 data."""
    try:
        thumbnail_name = f"{notebook_path.stem}.png"
        thumbnail_path = output_dir / thumbnail_name

        img_data = base64.b64decode(image_data)
        with open(thumbnail_path, "wb") as img_file:
            img_file.write(img_data)

        # Resize thumbnail to uniform square-like size (crop/pad to maintain aspect ratio)
        try:
            img = Image.open(thumbnail_path)
            # Target size for uniform thumbnails - more square-like
            target_size = (400, 250)

            # Calculate scaling to fit within target while maintaining aspect ratio
            img.thumbnail(target_size, Image.Resampling.LANCZOS)

            # Create a new image with target size and paste centered
            new_img = Image.new("RGB", target_size, (255, 255, 255))
            # Calculate position to center the image
            x_offset = (target_size[0] - img.size[0]) // 2
            y_offset = (target_size[1] - img.size[1]) // 2
            new_img.paste(img, (x_offset, y_offset))
            new_img.save(thumbnail_path)
        except Exception as e:
            print(f"Warning: Could not resize thumbnail for {notebook_path.name}: {e}")

        # Use relative path: from notebooks/ subdirectory, go up to source root, then to _static
        return f"../_static/thumbnails/{thumbnail_name}"
    except Exception as e:
        print(f"Warning: Could not save thumbnail for {notebook_path.name}: {e}")
        return None


def generate_gallery_markdown(
    notebooks_data: List[Dict],
    output_path: Path,
    category_mapping: Dict[str, List[str]],
):
    """Generate gallery markdown file with sphinx-design cards."""
    # Group notebooks by category
    categories: Dict[str, List[Dict]] = {}
    for nb_data in notebooks_data:
        category = nb_data["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(nb_data)

    # Sort categories - maintain order from index.md (order of appearance)
    # Use the order from category_mapping to preserve the structure
    sorted_categories = [cat for cat in category_mapping.keys() if cat in categories]
    # Add any categories found in notebooks but not in mapping (shouldn't happen, but handle gracefully)
    for cat in categories.keys():
        if cat not in sorted_categories:
            sorted_categories.append(cat)

    # Generate markdown
    lines = ["# Example Gallery\n"]

    for category in sorted_categories:
        if category not in categories:
            continue

        notebooks = categories[category]
        # Sort notebooks within category
        notebooks.sort(key=lambda x: x["filename"])

        lines.append(f"## {category}\n")
        lines.append("::::{grid} 1 2 3 3\n")
        lines.append(":gutter: 3\n\n")

        for nb in notebooks:
            # Title goes on the same line as grid-item-card (escape braces in f-string)
            card_lines = [f":::{'{grid-item-card}'} {nb['title']}\n"]
            # Add class to ensure uniform card height
            card_lines.append(":class-card: sd-card-h-100\n")

            if nb.get("thumbnail"):
                card_lines.append(f":img-top: {nb['thumbnail']}\n")

            # Use document name without extension (relative to current directory)
            # Since index.md is in notebooks/, links are relative to that directory
            doc_name = nb["filename"].replace(".ipynb", "")
            card_lines.append(f":link: {doc_name}\n")
            card_lines.append(":link-type: doc\n")
            card_lines.append(":::\n")
            lines.extend(card_lines)

        lines.append("::::\n\n")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


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

        # Extract metadata
        title, description = extract_metadata(nb_path)

        # Determine category from index.md structure
        category = get_notebook_category(nb_path.name, category_mapping)

        # Generate thumbnail
        thumbnail = extract_first_image(nb_path, thumbnails_dir)

        notebooks_data.append(
            {
                "filename": nb_path.name,
                "title": title,
                "description": description,
                "category": category,
                "thumbnail": thumbnail,
            }
        )

    # Generate gallery markdown
    print("Generating gallery markdown...")
    generate_gallery_markdown(notebooks_data, output_file, category_mapping)

    print(f"Gallery generated successfully at {output_file}")
    print(f"Thumbnails saved to {thumbnails_dir}")


if __name__ == "__main__":
    main()

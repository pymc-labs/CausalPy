#   Copyright 2024 The PyMC Labs Developers
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
"""
This is a simple script that converts the jupyter notebooks into markdown
for easier (and cleaner) parsing for the codespell check. Whitelisted words
are maintained within this directory in the `codespeel-whitelist.txt`. For
more information on this pre-commit hook please visit the github homepage
for the project: https://github.com/codespell-project/codespell.
"""

import argparse
import os
from glob import glob

import nbformat
from nbconvert import MarkdownExporter


def notebook_to_markdown(pattern: str, output_dir: str) -> None:
    """
    Utility to convert jupyter notebook to markdown files.

    :param pattern:
        str that is a glob appropriate pattern to search
    :param output_dir:
        str directory to save the markdown files to
    """
    for f_name in glob(pattern, recursive=True):
        with open(f_name, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        markdown_exporter = MarkdownExporter()
        (body, _) = markdown_exporter.from_notebook_node(nb)

        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(
            output_dir, os.path.splitext(os.path.basename(f_name))[0] + ".md"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pattern",
        help="the glob appropriate pattern to search for jupyter notebooks",
        default="docs/**/*.ipynb",
    )
    parser.add_argument(
        "-t",
        "--tempdir",
        help="temporary directory to save the converted notebooks",
        default="tmp_markdown",
    )
    args = parser.parse_args()
    notebook_to_markdown(args.pattern, args.tempdir)

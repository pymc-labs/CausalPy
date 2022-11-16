# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# autodoc_mock_imports
# This avoids autodoc breaking when it can't find packages imported in the code.
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_mock_imports
# autodoc_mock_imports = [
#     "arviz",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "patsy",
#     "pymc",
#     "scipy",
#     "seaborn",
#     "sklearn",
#     "xarray",
# ]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CausalPy"
copyright = "2022, Benjamin T. Vincent"
author = "Benjamin T. Vincent"

from causalpy.version import __version__

release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "myst_parser",
]

source_suffix = [".rst", ".md"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST options for working with markdown files. Info about extensions here https://myst-parser.readthedocs.io/en/latest/syntax/optional.html?highlight=math#admonition-directives
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "linkify"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
# TODO: version seems not to be displayed despite setting this to True
html_theme_options = {
    "display_version": True,
}

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# Add "Edit on Github" link. Replaces "view page source" ----------------------
html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "pymc-labs",  # Username
    "github_repo": "CausalPy",  # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

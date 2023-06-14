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

from causalpy.version import __version__

sys.path.insert(0, os.path.abspath("../"))

# autodoc_mock_imports
# This avoids autodoc breaking when it can't find packages imported in the code.
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_mock_imports # noqa: E501
autodoc_mock_imports = [
    "arviz",
    "matplotlib",
    "numpy",
    "pandas",
    "patsy",
    "pymc",
    "scipy",
    "seaborn",
    "sklearn",
    "xarray",
]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CausalPy"
copyright = "2022, Benjamin T. Vincent"
author = "Benjamin T. Vincent"


release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
]

source_suffix = [".rst", ".md"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"

# -- intersphinx config -------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
}

# -- nbsphinx config ----------------------------------------------------------
# Opt out of executing the notebooks remotely. This will save time in the remote build
# process on readthedocs. The notebooks in /docs/notebooks will be parsed/converted,
# but not re-executed.
nbsphinx_execute = "never"

# MyST options for working with markdown files.
# Info about extensions here https://myst-parser.readthedocs.io/en/latest/syntax/optional.html?highlight=math#admonition-directives # noqa: E501
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "linkify",
    "html_admonition",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "logo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
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
    "conf_py_path": "/docs/source/",  # Path in the checkout to the docs root
}

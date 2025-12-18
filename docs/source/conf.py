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
    "pymc-extras",
    "scipy",
    "seaborn",
    "sklearn",
    "xarray",
]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "CausalPy"
author = "PyMC Labs"
copyright = f"2024, {author}"


release = __version__
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings
extensions = [
    # extensions from sphinx base
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    # extensions provided by other packages
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",  # needed to plot in docstrings
    "myst_nb",
    "notfound.extension",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_sitemap",
    "sphinx_togglebutton",
]

nb_execution_mode = "off"

# configure copy button to avoid copying sphinx or console characters
copybutton_exclude = ".linenos, .gp"
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"

# bibtex config
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"


# numpydoc and autodoc typehints config
numpydoc_show_class_members = False
numpydoc_xref_param_type = True
# fmt: off
numpydoc_xref_ignore = {
    "of", "or", "optional", "default", "numeric", "type", "scalar", "1D", "2D", "3D", "nD", "array",
    "instance", "M", "N"
}
# fmt: on
numpydoc_xref_aliases = {
    "TensorVariable": ":class:`~pytensor.tensor.TensorVariable`",
    "RandomVariable": ":class:`~pytensor.tensor.random.RandomVariable`",
    "ndarray": ":class:`~numpy.ndarray`",
    "InferenceData": ":class:`~arviz.InferenceData`",
    "Model": ":class:`~pymc.Model`",
    "tensor_like": ":term:`tensor_like`",
    "unnamed_distribution": ":term:`unnamed_distribution`",
}
# don't add a return type section, use standard return with type info
typehints_document_rtype = False

# -- intersphinx config -------------------------------------------------------
intersphinx_mapping = {
    "arviz": ("https://python.arviz.org/en/stable/", None),
    "examples": ("https://www.pymc.io/projects/examples/en/latest/", None),
    "mpl": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pymc-extras": ("https://www.pymc.io/projects/extras/en/latest/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# MyST options for working with markdown files.
# Info about extensions here https://myst-parser.readthedocs.io/en/latest/syntax/optional.html?highlight=math#admonition-directives # noqa: E501
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "linkify",
    "html_admonition",
]

# sitemap extension configuration
site_url = "https://causalpy.readthedocs.io/"
sitemap_url_scheme = f"{{lang}}{version}/{{link}}"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "labs_sphinx_theme"
html_static_path = ["_static"]
html_extra_path = ["robots.txt"]
html_favicon = "_static/favicon_logo.png"
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "logo": {
        "image_light": "_static/flat_logo.png",
        "image_dark": "_static/flat_logo_darkmode.png",
    },
    "analytics": {"google_analytics_id": "G-3MCDG3M7X6"},
}
html_context = {
    "github_user": "pymc-labs",
    "github_repo": "CausalPy",
    "github_version": "main",
    "doc_path": "docs/source/",
    "default_mode": "light",
    "baseurl": "https://causalpy.readthedocs.io/",
}

# -- Options for autodoc ----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

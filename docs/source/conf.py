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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_extensions"))

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

# The version info for the project you're documenting
if os.environ.get("READTHEDOCS", False):
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if rtd_version.lower() == "latest":
        version = "dev"
else:
    version = "local"
    rtd_version = version

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
    "sphinxext.rediraffe",
    "strip_citation_labels",
]

# -- Redirects for renamed notebooks (issue #840) ---------------------------
# Maps old docnames to new docnames so legacy URLs keep resolving.
rediraffe_redirects = {
    "notebooks/ancova_pymc": "notebooks/ancova-pymc",
    "notebooks/did_pymc": "notebooks/difference-in-differences-pymc",
    "notebooks/did_pymc_banks": "notebooks/difference-in-differences-banks-pymc",
    "notebooks/did_skl": "notebooks/difference-in-differences-sklearn",
    "notebooks/geolift1": "notebooks/geolift-single-cell",
    "notebooks/inv_prop_latent": "notebooks/inverse-propensity-latent",
    "notebooks/inv_prop_pymc": "notebooks/inverse-propensity-pymc",
    "notebooks/its_covid": "notebooks/interrupted-time-series-covid",
    "notebooks/its_lift_test": "notebooks/interrupted-time-series-lift-test",
    "notebooks/its_post_intervention_analysis": "notebooks/interrupted-time-series-post-intervention-analysis",
    "notebooks/its_pymc": "notebooks/interrupted-time-series-pymc",
    "notebooks/its_pymc_comparative": "notebooks/interrupted-time-series-comparative-pymc",
    "notebooks/its_skl": "notebooks/interrupted-time-series-sklearn",
    "notebooks/iv_pymc": "notebooks/instrumental-variables-pymc",
    "notebooks/iv_vs_priors": "notebooks/instrumental-variables-variable-selection-priors",
    "notebooks/iv_weak_instruments": "notebooks/instrumental-variables-weak-instruments",
    "notebooks/multi_cell_geolift": "notebooks/multi-cell-geolift",
    "notebooks/panel_fixed_effects": "notebooks/panel-fixed-effects",
    "notebooks/piecewise_its_pymc": "notebooks/piecewise-interrupted-time-series-pymc",
    "notebooks/pipeline_workflow": "notebooks/pipeline-workflow",
    "notebooks/rd_donut_pymc": "notebooks/regression-discontinuity-donut-pymc",
    "notebooks/rd_pymc": "notebooks/regression-discontinuity-pymc",
    "notebooks/rd_pymc_drinking": "notebooks/regression-discontinuity-drinking-pymc",
    "notebooks/rd_skl": "notebooks/regression-discontinuity-sklearn",
    "notebooks/rd_skl_drinking": "notebooks/regression-discontinuity-drinking-sklearn",
    "notebooks/report_demo": "notebooks/reporting-demo",
    "notebooks/rkink_pymc": "notebooks/regression-kink-pymc",
    "notebooks/sc_pymc": "notebooks/synthetic-control-pymc",
    "notebooks/sc_pymc_brexit": "notebooks/synthetic-control-brexit-pymc",
    "notebooks/sc_skl": "notebooks/synthetic-control-sklearn",
    "notebooks/staggered_did_pymc": "notebooks/staggered-difference-in-differences-pymc",
}
rediraffe_branch = "main"
rediraffe_auto_redirect_perc = 0

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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".codespell"]
master_doc = "index"

# bibtex config
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"
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
myst_heading_anchors = 3  # auto-generate anchors for H1–H3, enabling #slug cross-refs
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "linkify",
    "html_admonition",
]

# sitemap extension configuration
site_url = "https://causalpy.readthedocs.io/"
sitemap_url_scheme = f"{{lang}}{rtd_version}/{{link}}"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "labs_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
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

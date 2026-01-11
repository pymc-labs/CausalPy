:::{image} _static/logo.png
:width: 60 %
:align: center
:alt: CausalPy logo
:::

<p align="center">

[![PyPI version](https://badge.fury.io/py/CausalPy.svg)](https://badge.fury.io/py/CausalPy)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/causalpy.svg?color=green)](https://anaconda.org/conda-forge/causalpy)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

![Build Status](https://github.com/pymc-labs/CausalPy/actions/workflows/ci.yml/badge.svg?branch=main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
![Interrogate](_static/interrogate_badge.svg)
[![codecov](https://codecov.io/gh/pymc-labs/CausalPy/branch/main/graph/badge.svg?token=FDKNAY5CZ9)](https://codecov.io/gh/pymc-labs/CausalPy)

![GitHub Repo stars](https://img.shields.io/github/stars/pymc-labs/causalpy?style=flat)
![Read the Docs](https://img.shields.io/readthedocs/causalpy)
[![Downloads](https://static.pepy.tech/badge/causalpy)](https://pepy.tech/project/causalpy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/causalpy)

</p>

```{include} ../../README.md
:start-after: "# CausalPy"
:end-before: "## Installation"
```

## Installation

To get the latest release you can use pip:

```bash
pip install CausalPy
```

or conda/mamba/micromamba:

```bash
conda install causalpy -c conda-forge    # or mamba/micromamba
```

Alternatively, if you want the very latest version of the package you can install from GitHub:

```bash
pip install git+https://github.com/pymc-labs/CausalPy.git
```

## Quickstart

```{include} ../../README.md
:start-after: "## Quickstart"
:end-before: "## Videos"
```

## Videos

<style>
.video-container {
    position: relative;
    padding-bottom: 56.25%; /* 16:9 aspect ratio */
    height: 0;
    overflow: hidden;
    max-width: 100%;
    background: #000;
}

.video-container iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border: 0;
}
</style>

<div class="video-container">
    <iframe src="https://www.youtube.com/embed/gV6wzTk3o1U" title="YouTube video player" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>

```{include} ../../README.md
:start-after: "## When CausalPy is a good fit"
:end-before: "## Roadmap"
```

## Getting Help

Have questions about using CausalPy? We're here to help!

- **Questions & Help**: Visit our [GitHub Discussions Q&A](https://github.com/pymc-labs/CausalPy/discussions/categories/q-a) to ask questions and get help from the community
- **Bug Reports & Feature Requests**: Open an [Issue](https://github.com/pymc-labs/CausalPy/issues) for bugs or feature requests
- **Documentation**: Browse the [knowledgebase](knowledgebase/index), [API documentation](api/index), and [examples](notebooks/index) for detailed guides

Please use GitHub Discussions for general questions rather than opening issues, so we can keep the issue tracker focused on bugs and enhancements.

## Support

This repository is supported by [PyMC Labs](https://www.pymc-labs.com).

For companies that want to use CausalPy in production, [PyMC Labs](https://www.pymc-labs.com) is available for consulting and training. We can help you build and deploy your models in production. We have experience with cutting edge Bayesian and causal modelling techniques which we have applied to a range of business domains.

:::{toctree}
:hidden:

knowledgebase/index
api/index
notebooks/index
:::

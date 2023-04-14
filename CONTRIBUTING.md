# CONTRIBUTING

This repository is under active development by a small number of contributors at the moment. Once the code and API has settled a bit we will open up and welcome contributions. But not yet.

## Setup for local development

1. Create a new environment using Python >=3.8, for example 3.10

```
conda create --name CausalPy python=3.10
```

2. Activate environment:

```
conda activate CausalPy
```

3. Install the package in editable mode

```
pip install -e .
```

4. Install development dependencies

```
pip install causalpy[dev]
pip install causalpy[docs]
pip install causalpy[test]
pip install causalpy[lint]
```

If that fails, try:

```
pip install 'causalpy[dev]'
pip install 'causalpy[docs]'
pip install 'causalpy[test]'
pip install 'causalpy[lint]'
```

It may also be necessary to [install](https://pandoc.org/installing.html) `pandoc`. On a mac, I run `brew install pandoc`.

5. You may also need to run this to get pre-commit checks working

```
pre-commit install
```

6. Note: You may have to run the following command to make Jupyter Lab aware of the `CausalPy` environment.

```
python -m ipykernel install --user --name CausalPy
```

## Building the documentation locally

Ensure the right packages (in `requirements-docs.txt`) are available in the environment. See the steps above.

A local build of the docs is achieved by:

```bash
cd docs
make html
```

Sometimes not all changes are recognised. In that case run this (again from within the `docs` folder):

```bash
make clean && make html
```

Docs are built in `docs/_build`, but these docs are _not_ committed to the GitHub repository due to `.gitignore`.

## Remote documentation

Documentation is hosted on https://causalpy.readthedocs.io/. New remote builds are triggered automatically whenever there is an update to the `main` branch.

The `.readthedocs.yaml` file contains the configurations for the remote build.

If there are autodoc issues/errors in remote builds of the docs, we need to add all package dependencies (in `requirements.txt`) into the list `autodoc_mock_imports` in `docs/config.py`.



## Overview of code structure

UML diagrams can be created with the command below. If you have not already done so, you may need to `pip install 'causalpy[lint]'` in order to install `pyreverse`.

```bash
pyreverse -o png causalpy --output-directory img
```

Classes
![](img/classes.png)

Packages
![](img/packages.png)

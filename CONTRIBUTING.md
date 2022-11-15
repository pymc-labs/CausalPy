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
pip install -r requirements-dev.txt
pip install -r requirements-docs.txt
```

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

Sometimes not all changes are recognised. In that case run:

```bash
make clean && make html
```

Docs are built in `docs/_build`, but these docs are _not_ committed to the GitHub repository due to `.gitignore`.

## New releases [work in progress]

1. Bump the release version in `causalpy/version.py`. This is automatically read by `setup.py` and `docs/config.py`.
2. Update on pypi.org. In the root directory:
  - `python setup.py sdist`
  - update to pypi.org with `twine upload dist/*`
3. Readthedocs:
  - Docs should be built remotely every time there is a pull request
  - See here https://docs.readthedocs.io/en/stable/tutorial/#versioning-documentation for versioning the docs
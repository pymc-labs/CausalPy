.PHONY: init lint check_lint test uml html cleandocs doctest

init:
	python -m pip install -e . --no-deps

lint:
	ruff check --fix .
	ruff format .

check_lint:
	ruff check .
	ruff format --diff --check .
	interrogate .

doctest:
	pytest --doctest-modules --ignore=causalpy/tests/ causalpy/ --config-file=causalpy/tests/conftest.py

test:
	python -m pytest

uml:
	pyreverse -o png causalpy --output-directory docs/source/_static --ignore tests

# Docs build commands

html:
	sphinx-build -b html docs/source docs/_build

cleandocs:
	rm -rf docs/_build
	rm -rf docs/source/api/generated

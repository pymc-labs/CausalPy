.PHONY: init lint check_lint test

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
	pytest --doctest-modules --ignore=causalpy/tests/ causalpy/

test:
	pytest

uml:
	pyreverse -o png causalpy --output-directory docs/source/_static --ignore tests

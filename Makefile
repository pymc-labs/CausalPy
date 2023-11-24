.PHONY: init lint check_lint test

init:
	python -m pip install -e .

lint:
	pip install causalpy[lint]
	ruff check --fix .
	ruff format .

check_lint:
	pip install causalpy[lint]
	ruff check .
	ruff format --diff --check .
	nbqa black --check .
	nbqa ruff .
	interrogate .

doctest:
	pip install causalpy[test]
	pytest --doctest-modules causalpy/

test:
	pip install causalpy[test]
	pytest

uml:
	pip install pylint
	pyreverse -o png causalpy --output-directory docs/source/_static --ignore tests

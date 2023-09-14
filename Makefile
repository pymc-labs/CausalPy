.PHONY: init lint check_lint test

init:
	python -m pip install -e .

lint:
	pip install causalpy[lint]
	isort .
	black .

check_lint:
	pip install causalpy[lint]
	flake8 .
	isort --check-only .
	black --diff --check --fast .
	nbqa black --check .
	nbqa isort --check-only .
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

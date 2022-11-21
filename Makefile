.PHONY: init lint check_lint test

init:
	python -m pip install -e .

lint:
	pip install -r requirements-lint.txt
	isort .
	black .

check_lint:
	pip install -r requirements-lint.txt
	flake8 .
	isort --check-only .
	black --diff --check --fast .

test:
	pip install -r requirements-test.txt
	pytest

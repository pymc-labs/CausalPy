#################################################################################
# GLOBALS                                                                       #
#################################################################################

PACKAGE_DIR = causalpy

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: init setup lint check_lint test uml html cleandocs doctest help

init: ## Install the package in editable mode
	python -m pip install -e . --no-deps

setup: ## Set up complete dev environment (run after conda activate CausalPy)
	python -m pip install --no-deps -e .
	pip install -e '.[dev,docs,test,lint]'
	pre-commit install
	@echo "Development environment ready!"

lint: ## Run ruff linter and formatter
	ruff check --fix .
	ruff format .

check_lint: ## Check code formatting and linting without making changes
	ruff check .
	ruff format --diff --check .
	interrogate .

doctest: ## Run doctests for the causalpy module
	python -m pytest --doctest-modules --ignore=causalpy/tests/ causalpy/ --config-file=causalpy/tests/conftest.py

test: ## Run all tests with pytest
	python -m pytest

uml: ## Generate UML diagrams from code
	pyreverse -o png causalpy --output-directory docs/source/_static --ignore tests

html: ## Build HTML documentation with Sphinx
	sphinx-build -b html docs/source docs/_build

cleandocs: ## Clean the documentation build directories
	rm -rf docs/_build
	rm -rf docs/source/api/generated


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

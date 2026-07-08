#################################################################################
# GLOBALS                                                                       #
#################################################################################

PACKAGE_DIR = causalpy

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: init setup lint check_lint check-exports check-architecture test test-patch-cov uml gallery html cleandocs doctest run_notebooks_full help

DIFF_COVER_COMPARE_BRANCH ?= $(shell if git show-ref --verify --quiet refs/remotes/upstream/main; then printf "upstream/main"; else printf "origin/main"; fi)
DIFF_COVER_FAIL_UNDER ?= 95

init: ## Install the package in editable mode
	python -m pip install -e . --no-deps

setup: ## Set up complete dev environment (run inside CausalPy env, e.g. conda run -n CausalPy make setup)
	python -m pip install --no-deps -e .
	python -m pip install -e '.[dev,docs,test,lint]'
	prek install -f
	@echo "Development environment ready!"

lint: ## Run ruff linter and formatter
	ruff check --fix .
	ruff format .

check_lint: ## Check code formatting and linting without making changes
	ruff check .
	ruff format --diff --check .

check-exports: ## Verify experiment/check public API export wiring
	python scripts/check_public_exports.py --check

check-architecture: ## Verify ARCHITECTURE.md experiment inventory matches code
	python scripts/check_architecture_inventory.py --check

doctest: ## Run doctests for the causalpy module
	python -m pytest --doctest-modules --ignore=causalpy/tests/ causalpy/ --config-file=causalpy/tests/conftest.py

test: ## Run all tests with pytest
	python -m pytest

test-patch-cov: ## Run tests and fail if patch coverage versus the base branch is too low
	python -m pytest --cov-report=xml --no-cov-on-fail
	diff-cover coverage.xml --compare-branch=$(DIFF_COVER_COMPARE_BRANCH) --fail-under=$(DIFF_COVER_FAIL_UNDER)

uml: ## Generate UML diagrams from code
	pyreverse -o png causalpy --output-directory docs/source/_static --ignore tests

gallery: ## Generate example gallery from notebooks
	python scripts/generate_gallery.py

html: gallery ## Build HTML documentation with Sphinx
	sphinx-build -b html docs/source docs/_build

run_notebooks_full: ## Re-execute all notebooks and save outputs in place (slow)
	python scripts/run_notebooks/runner.py --full

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

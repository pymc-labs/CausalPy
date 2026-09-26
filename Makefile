#################################################################################
# GLOBALS                                                                       #
#################################################################################

PACKAGE_DIR = causalpy

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: init setup setup-conda lint check_lint typecheck check-exports check-architecture test test-nightly test-correctness test-patch-cov uml gallery html cleandocs doctest run_notebooks_full help

# Patch coverage must be measured against the branch the PR actually targets.
# While the 1.0 transition branch exists, work branched from it targets it, not
# `main` -- and it is hundreds of commits ahead of `main`, so comparing against
# `main` reports the whole migration as "the patch" and the gate stops meaning
# anything. Prefer the transition branch when HEAD descends from it (a branch
# cut from `main` never does), otherwise fall back to `main`. Override
# DIFF_COVER_COMPARE_BRANCH when a PR deliberately targets something else.
DIFF_COVER_COMPARE_BRANCH ?= $(shell \
	for ref in upstream/pymc6_and_pymcmarketing1_migration origin/pymc6_and_pymcmarketing1_migration; do \
		if git show-ref --verify --quiet "refs/remotes/$$ref" \
			&& git merge-base --is-ancestor "$$ref" HEAD; then \
			printf "%s" "$$ref"; exit 0; \
		fi; \
	done; \
	if git show-ref --verify --quiet refs/remotes/upstream/main; then \
		printf "upstream/main"; \
	else \
		printf "origin/main"; \
	fi)
DIFF_COVER_FAIL_UNDER ?= 96
# diff-cover (10.3.0, 10.4.1) matches exclude patterns against the basename
# and then the absolute path, never the repo-relative path. The pattern goes
# through fnmatch, so a checkout path containing [ ? or * disables it.
DIFF_COVER_EXCLUDE ?= $(CURDIR)/$(PACKAGE_DIR)/tests/*

init: ## Install the package in editable mode
	python -m pip install -e . --no-deps

setup: ## Set up complete dev environment with uv (default; see CONTRIBUTING.md for the conda alternative)
	uv sync --locked --extra dev --extra docs --extra test --extra lint
	uv run prek install -f
	@echo "Development environment ready! Run commands with 'uv run <command>', e.g. 'uv run make test'."

setup-conda: ## Set up dev environment inside an already-active conda/micromamba env (alternative to 'setup')
	python -m pip install --no-deps -e .
	python -m pip install -e '.[dev,docs,test,lint]'
	prek install -f
	@echo "Conda development environment ready!"

lint: ## Run ruff linter and formatter
	ruff check --fix .
	ruff format .

check_lint: ## Check code formatting and linting without making changes
	ruff check .
	ruff format --diff --check .

typecheck: ## Run mypy over causalpy (scope and per-module allowlist in pyproject.toml)
	mypy

check-exports: ## Verify public API export and documentation wiring
	python scripts/check_public_exports.py --check

check-architecture: ## Verify ARCHITECTURE.md experiment inventory matches code
	python scripts/check_architecture_inventory.py --check

doctest: ## Run doctests for the causalpy module (slow doctests included)
	python -m pytest --doctest-modules -p causalpy.tests.doctest_sampling --ignore=causalpy/tests/ causalpy/ -m ""

test: ## Run default tests with pytest
	python -m pytest

test-nightly: ## Run the non-correctness tests that are skipped by default and run nightly in CI
	python -m pytest -m "nightly and not correctness" --no-cov

test-correctness: ## Run statistical correctness tests
	python -m pytest -o addopts='' -m correctness --no-cov

test-patch-cov: ## Run tests and fail if patch coverage versus the base branch is too low
	python -m pytest --cov-report=xml --no-cov-on-fail
	diff-cover coverage.xml --compare-branch=$(DIFF_COVER_COMPARE_BRANCH) --fail-under=$(DIFF_COVER_FAIL_UNDER) --exclude '$(DIFF_COVER_EXCLUDE)'

uml: ## Generate UML diagrams from code
	pyreverse -o png causalpy --output-directory docs/source/_static --ignore tests

gallery: ## Regenerate index.md and thumbnails from gallery.yaml
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

.PHONY: help install test test-cov lint format format-check typecheck docstrings \
        docs docs-serve build clean ci ci-docker check release install-pre-commit
.DEFAULT_GOAL := help

# Local mirror of the canon reusable CI (gojiplus/py-canon). CI itself syncs
# --all-groups --all-extras, so these targets do the same: skipping an extra
# here means the optional-feature tests silently skip rather than fail.

help: ## Show this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install all dependency groups and extras
	uv sync --all-groups --all-extras

test: ## Run tests
	uv run --all-extras pytest tests/ -v

test-cov: ## Run tests with coverage
	uv run --all-extras pytest tests/ -v --cov=rowvoi --cov-report=term-missing --cov-report=html

lint: ## Run linting checks
	uv run ruff check .
	uv run deptry .

format: ## Format code
	uv run ruff format .

format-check: ## Check code formatting
	uv run ruff format --check .

typecheck: ## Run pyright
	uv run pyright

docstrings: ## Check docstring/signature agreement
	uv run pydoclint --config pyproject.toml rowvoi/

docs: ## Build documentation (warnings are errors, as in CI)
	uv run --group docs sphinx-build -W -b html docs docs/_build/html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

build: ## Build package
	uv build

check: ## Run the preen fleet-conformance check
	uvx preen check . --explain

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .mypy_cache/
	rm -rf htmlcov/ .coverage coverage.xml docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

ci: ## Run all CI checks locally
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) typecheck
	$(MAKE) docstrings
	$(MAKE) test

ci-docker: ## Run CI in Docker (standard Python image)
	docker run --rm -v $(PWD):/app -w /app python:3.11 sh -c \
		"pip install uv && \
		uv sync --all-groups --all-extras && \
		uv run ruff check . && \
		uv run ruff format --check . && \
		uv run deptry . && \
		uv run pytest tests/ -v"

install-pre-commit: ## Install pre-commit hooks
	uv tool install pre-commit
	pre-commit install

release: ## Cut a release (run with VERSION=x.y.z)
	@if [ -z "$(VERSION)" ]; then echo "Please specify VERSION=x.y.z"; exit 1; fi
	uvx preen release $(VERSION)

.PHONY: help install test lint format type-check docs build clean ci-docker
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install development dependencies
	uv sync --extra dev --extra docs

test: ## Run tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage
	uv run pytest tests/ -v --cov=rowvoi --cov-report=term-missing --cov-report=html

lint: ## Run linting checks
	uv run ruff check .

format: ## Format code
	uv run ruff format .

format-check: ## Check code formatting
	uv run ruff format --check .

type-check: ## Run type checking
	uv run mypy rowvoi/

docs: ## Build documentation
	cd docs && uv run sphinx-build -b html . _build/html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

build: ## Build package
	uv build

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

ci: ## Run all CI checks locally
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) type-check
	$(MAKE) test

ci-docker: ## Run CI in Docker (standard Python image)
	docker run --rm -v $(PWD):/app -w /app python:3.11 sh -c \
		"pip install uv && \
		uv sync --extra dev && \
		uv run ruff check . && \
		uv run ruff format --check . && \
		uv run mypy rowvoi/ && \
		uv run pytest tests/ -v"

install-pre-commit: ## Install pre-commit hooks
	pip install pre-commit
	pre-commit install

release: ## Create a new release (run with VERSION=x.y.z)
	@if [ -z "$(VERSION)" ]; then echo "Please specify VERSION=x.y.z"; exit 1; fi
	sed -i 's/version = ".*"/version = "$(VERSION)"/' pyproject.toml
	git add pyproject.toml
	git commit -m "Bump version to $(VERSION)"
	git tag v$(VERSION)
	@echo "Release $(VERSION) ready. Push with: git push && git push --tags"
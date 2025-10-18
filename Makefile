TOKEN=${PYPI_TOKEN}

.PHONY: install
install: ## Install the uv environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy cupid_matching/*.py
	@uv run mypy cupid_matching/examples/*.py


.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run pytest --doctest-modules tests/test*.py


.PHONY: build
build: clean-build ## Build wheel file using uv
	@echo "🚀 Creating wheel file"
	@uv build

.PHONY: clean-build
clean-build: ## clean build artifacts
	@rm -rf dist

.PHONY: publish
publish: ## publish a release to pypi.
	@echo "🚀 Publishing: Dry run."
	@uv publish --dry-run -u __token__ -p pypi-${TOKEN}
	@echo "🚀 Publishing."
	@uv publish -u __token__ -p pypi-${TOKEN}

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@cp README.md docs/index.md
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@cp README.md docs/index.md
	@uv run mkdocs serve

.PHONY: docs-deploy
docs-deploy: ## Build and deploy the documentation on Github pages
	@cp README.md docs/index.md
	@uv run mkdocs gh-deploy

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

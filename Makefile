# Makefile for simple-vascx development

.PHONY: help install install-dev test test-cov lint format clean build docker-test docker-test-gpu

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e ".[dev,test]"

test:  ## Run tests
	pytest -v

test-cov:  ## Run tests with coverage
	pytest -v --cov=simple_vascx --cov-report=term-missing --cov-report=html

lint:  ## Run linting checks
	flake8 src/ tests/
	mypy src/

format:  ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

docker-test:  ## Run tests in Docker (CPU)
	docker-compose up test-cpu

docker-test-gpu:  ## Run tests in Docker (GPU)
	docker-compose up test-gpu

docker-dev:  ## Start development environment in Docker
	docker-compose run dev

docker-build:  ## Build and verify package in Docker
	docker-compose up build

docker-clean:  ## Clean Docker images and volumes
	docker-compose down -v
	docker rmi simple-vascx:test simple-vascx:cuda || true

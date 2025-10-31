# Quick Start Guide for vascx-simplify

## Installation Options

### Option 1: Install from source (development mode)
```bash
# Clone the repository
git clone https://github.com/kapong/vascx_simplify.git
cd vascx_simplify

# Install in editable mode
pip install -e .

# Or with all dev dependencies
pip install -e ".[dev,test]"
```

### Option 2: Install with uv (faster)
```bash
# Install uv if you haven't already
pip install uv

# Install the package
uv pip install -e .

# Or with dev dependencies
uv pip install -e ".[dev,test]"
```

### Option 3: Build and install wheel
```bash
# Build the package
python -m build

# Install the wheel
pip install dist/vascx_simplify-0.1.0-py3-none-any.whl
```

## Testing

### Local Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=simple_vascx --cov-report=html

# Run specific test file
pytest tests/test_inference.py -v
```

### Docker Testing
```bash
# CPU tests
docker-compose up test-cpu

# GPU tests (requires nvidia-docker)
docker-compose up test-gpu

# Development environment
docker-compose run dev
```

### Using Make commands (Linux/Mac)
```bash
make help           # Show all available commands
make install-dev    # Install with dev dependencies
make test           # Run tests
make test-cov       # Run tests with coverage
make docker-test    # Run tests in Docker
```

## Development Workflow

1. **Setup environment**
   ```bash
   pip install -e ".[dev,test]"
   ```

2. **Make changes to code**
   - Edit files in `src/simple_vascx/`

3. **Format code**
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

4. **Run tests**
   ```bash
   pytest -v
   ```

5. **Check lint**
   ```bash
   flake8 src/ tests/
   ```

## Building for Distribution

```bash
# Build source distribution and wheel
python -m build

# Check the package
twine check dist/*

# Upload to PyPI (when ready)
twine upload dist/*
```

## Adding Author Information

Before publishing, update these files with your information:

1. **pyproject.toml**
   ```toml
   authors = [
       {name = "Your Name", email = "your.email@example.com"},
   ]
   ```

2. **LICENSE**
   - Replace `[TODO: Add Author Name]` with your name

3. **README.md**
   - Update the Authors section
   - Update GitHub repository URLs
   - Update citation information

4. **pyproject.toml URLs**
   - Update `YOUR-USERNAME` in all repository URLs

## Publishing Checklist

- [ ] Update author information in `pyproject.toml`
- [ ] Update LICENSE with your name
- [ ] Update README.md with correct repository URLs
- [ ] Update citation information
- [ ] Run all tests: `pytest`
- [ ] Check code formatting: `black --check src/ tests/`
- [ ] Build package: `python -m build`
- [ ] Test installation: `pip install dist/*.whl`
- [ ] Tag release: `git tag v0.1.0`
- [ ] Push to GitHub: `git push --tags`
- [ ] Publish to PyPI: `twine upload dist/*`

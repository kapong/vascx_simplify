# PyPI Publishing Setup Guide

This document explains how to set up automatic publishing to PyPI using GitHub Actions.

## Overview

The GitHub Actions workflow (`.github/workflows/build-and-publish.yml`) automatically:
- Builds the package on every push to main and pull requests
- Tests installation across Python 3.8-3.12
- Publishes to TestPyPI on pushes to main branch
- Publishes to PyPI when you create a version tag (e.g., `v0.1.0`)

## Prerequisites

1. **PyPI Account**: Create accounts at:
   - [PyPI](https://pypi.org/account/register/)
   - [TestPyPI](https://test.pypi.org/account/register/) (optional, for testing)

2. **GitHub Repository**: Your code must be in a GitHub repository

## Setup Instructions

### 1. Configure PyPI Trusted Publishing

GitHub Actions uses OpenID Connect (OIDC) for secure authentication with PyPI, eliminating the need for API tokens.

#### For PyPI (Production):

1. Go to your PyPI account settings: https://pypi.org/manage/account/
2. Scroll to "Publishing" section
3. Click "Add a new pending publisher"
4. Fill in:
   - **PyPI Project Name**: `vascx_simplify` (or your package name)
   - **Owner**: Your GitHub username/organization
   - **Repository name**: Your repository name
   - **Workflow name**: `build-and-publish.yml`
   - **Environment name**: `pypi`
5. Click "Add"

#### For TestPyPI (Optional, for testing):

1. Go to TestPyPI: https://test.pypi.org/manage/account/
2. Follow the same steps as above, but use environment name: `testpypi`

### 2. Configure GitHub Environments (Optional but Recommended)

For additional security, set up GitHub environments:

1. Go to your GitHub repository
2. Navigate to **Settings** → **Environments**
3. Create environment named `pypi`:
   - Click "New environment"
   - Name: `pypi`
   - (Optional) Add protection rules:
     - Required reviewers
     - Wait timer
     - Deployment branches: Only protected branches or selected branches
4. Repeat for `testpypi` if using TestPyPI

### 3. Update Package Version

Before publishing a new release:

1. Update version in `pyproject.toml`:
   ```toml
   [project]
   version = "0.1.1"  # Increment version
   ```

2. Update version in `src/vascx_simplify/__init__.py`:
   ```python
   __version__ = "0.1.1"
   ```

3. Commit changes:
   ```bash
   git add pyproject.toml src/vascx_simplify/__init__.py
   git commit -m "Bump version to 0.1.1"
   git push origin main
   ```

### 4. Create and Push a Release Tag

To trigger publication to PyPI:

```bash
# Create an annotated tag
git tag -a v0.1.1 -m "Release version 0.1.1"

# Push the tag to GitHub
git push origin v0.1.1
```

The workflow will automatically:
1. Build the package
2. Run tests across all supported Python versions
3. Publish to PyPI (if all tests pass)

## Workflow Triggers

The workflow runs on:

- **Pull Requests to main**: Build and test only
- **Push to main branch**: Build, test, and publish to TestPyPI
- **Version tags (v*)**: Build, test, and publish to PyPI
- **Manual trigger**: Via GitHub Actions UI (workflow_dispatch)

## Monitoring

1. Go to your repository on GitHub
2. Click the **Actions** tab
3. View workflow runs and their status

## Troubleshooting

### Build Fails

- Check that `pyproject.toml` is properly configured
- Ensure all dependencies are correctly specified
- Review the build logs in GitHub Actions

### Publishing Fails

- Verify Trusted Publishing is configured correctly on PyPI
- Check that the package name matches PyPI configuration
- Ensure the version number hasn't been used before
- Verify GitHub environment names match (`pypi`, `testpypi`)

### Import Test Fails

- Ensure `__version__` is defined in `src/vascx_simplify/__init__.py`
- Check that package structure is correct
- Verify all dependencies are specified in `pyproject.toml`

## Manual Publishing (Alternative)

If you prefer manual publishing:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Check the distribution
twine check dist/*

# Upload to TestPyPI (test first)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (1.0.0): Incompatible API changes
- **MINOR** (0.1.0): Add functionality (backwards compatible)
- **PATCH** (0.0.1): Bug fixes (backwards compatible)

## Security Best Practices

1. ✅ Use Trusted Publishing (OIDC) instead of API tokens
2. ✅ Enable required reviewers for production environment
3. ✅ Test with TestPyPI before publishing to PyPI
4. ✅ Use protected branches for main
5. ✅ Review all changes before creating release tags

## Resources

- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions for PyPI](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Python Packaging Guide](https://packaging.python.org/)

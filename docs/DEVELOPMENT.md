# Development Workflow Guide

## Table of Contents
1. [Git Commit Strategy](#git-commit-strategy)
2. [Backward Compatibility Checking](#backward-compatibility-checking)
3. [File Management](#file-management)
4. [Code Review Checklist](#code-review-checklist)
5. [Testing Workflow](#testing-workflow)

## Git Commit Strategy

### Principle: One Logical Task = One Commit

**DO:**
- Commit after completing each logical unit of work
- Keep commits focused and atomic
- Write clear, descriptive commit messages
- Test before committing

**DON'T:**
- Commit work-in-progress code
- Combine unrelated changes in one commit
- Commit without testing
- Create "WIP" or "temp" commits

### Commit Message Format

Use conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `refactor`: Code restructuring (no behavior change)
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```bash
# Good commit messages
git commit -m "feat: add gaussian importance map for sliding window inference"
git commit -m "perf: optimize preprocessing with mixed precision (2.5x speedup)"
git commit -m "fix: preserve tensor device in preprocessing pipeline"
git commit -m "refactor: simplify RANSAC ellipse fitting logic"
git commit -m "docs: add performance optimization guide"

# Bad commit messages
git commit -m "update"
git commit -m "fixes"
git commit -m "WIP"
git commit -m "changes to inference.py"
```

### Commit Workflow

```bash
# 1. Make changes
# ... edit files ...

# 2. Check what changed
git status
git diff

# 3. Review changes carefully
git diff src/vascx_simplify/inference.py

# 4. Test the changes
python examples/01_artery_vein.py
python -m pytest tests/  # if tests exist

# 5. Stage specific files
git add src/vascx_simplify/inference.py
git add src/vascx_simplify/preprocess.py

# 6. Commit with descriptive message
git commit -m "perf: optimize sliding window with batched inference"

# 7. Verify commit
git log --oneline -5
git show HEAD
```

### When to Commit

**Commit when you:**
- ✅ Complete a feature or fix
- ✅ Finish refactoring a function/class
- ✅ Add documentation
- ✅ Fix a bug
- ✅ Optimize performance (and verify it works)
- ✅ Add a new utility function

**Don't commit when:**
- ❌ Code doesn't work yet
- ❌ Tests are failing
- ❌ Output has changed unexpectedly
- ❌ You have temporary debug code
- ❌ You have commented-out code blocks

## Backward Compatibility Checking

### Before Every Commit: Check Differences

```bash
# View all changes since last commit
git diff

# View changes in specific file
git diff src/vascx_simplify/inference.py

# View changes excluding whitespace
git diff -w

# View staged changes
git diff --staged

# View changes in specific function
git diff -U10 src/vascx_simplify/inference.py | grep -A20 "def sliding_window"
```

### After Commit: Verify Changes

```bash
# Compare with previous commit
git diff HEAD~1

# Compare with previous commit (specific file)
git diff HEAD~1 src/vascx_simplify/inference.py

# Compare with specific commit
git diff abc1234 HEAD

# Show commit details
git show HEAD

# View commit history
git log --oneline -10
git log --oneline --graph --all -10
```

### Verify Output Consistency

**Critical: Always verify that outputs remain the same after changes!**

```python
# Create a test script: test_consistency.py
import torch
import numpy as np
from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface

def test_output_consistency():
    """Test that refactoring doesn't change outputs."""
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load model
    model_path = from_huggingface('Eyened/vascx:artery_vein/av_july24.pt')
    model = EnsembleSegmentation(model_path, VASCXTransform())
    
    # Create test input
    test_image = torch.randn(1, 3, 512, 512)
    
    # Run multiple times (should be deterministic)
    with torch.no_grad():
        output1 = model.predict(test_image)
        output2 = model.predict(test_image)
    
    # Verify consistency
    assert torch.allclose(output1, output2), "Non-deterministic output!"
    
    # Save expected output (first time only)
    # torch.save(output1, 'expected_output.pt')
    
    # Compare with expected output
    expected = torch.load('expected_output.pt')
    
    if not torch.allclose(output1, expected, rtol=1e-5, atol=1e-6):
        diff = torch.abs(output1 - expected).max().item()
        print(f"⚠️  WARNING: Output changed! Max difference: {diff}")
        print("If this is intentional, update expected_output.pt")
        return False
    
    print("✓ Output consistency verified")
    return True

if __name__ == "__main__":
    test_output_consistency()
```

**Workflow:**

```bash
# 1. Before making changes: capture baseline
python test_consistency.py  # Saves expected_output.pt

# 2. Make your changes
# ... edit code ...

# 3. Test that output is unchanged
python test_consistency.py

# 4. If output changed unexpectedly, investigate
git diff src/  # What did I change?

# 5. If output change is intentional, document it
git commit -m "fix: correct bias in preprocessing (changes output)"
```

### Using Git Bisect to Find Issues

If you discover a regression:

```bash
# Find which commit introduced the issue
git bisect start
git bisect bad HEAD  # Current commit is bad
git bisect good v0.1.0  # Last known good version

# Git will checkout commits for testing
# Test each commit:
python test_consistency.py
git bisect good  # or git bisect bad

# Repeat until found
# Git will identify the problematic commit
git bisect reset  # Done
```

## File Management

### Files to Commit

**DO commit:**
- ✅ Source code (`.py`, `.pyx`)
- ✅ Documentation (`.md`, `.rst`)
- ✅ Configuration (`pyproject.toml`, `setup.py`, `.gitignore`)
- ✅ Examples (`examples/*.py`)
- ✅ Tests (`tests/*.py`)
- ✅ License and readme files
- ✅ Requirements files

**DON'T commit:**
- ❌ Cached files (`__pycache__/`, `*.pyc`)
- ❌ Build artifacts (`dist/`, `build/`, `*.egg-info/`)
- ❌ Model weights (`*.pt`, `*.pth`, `*.onnx`) - use HuggingFace Hub
- ❌ Data files (`*.jpg`, `*.png`, `*.npy`) - except tiny test images
- ❌ IDE files (`.vscode/`, `.idea/`) - except shared settings
- ❌ Virtual environments (`venv/`, `.env/`)
- ❌ Temporary files (`*.tmp`, `*.bak`, `~`)
- ❌ Log files (`*.log`)
- ❌ OS files (`.DS_Store`, `Thumbs.db`)

### Proper .gitignore

```bash
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pt
*.pth
*.onnx
*.pb

# Data files
*.jpg
*.jpeg
*.png
*.npy
*.npz
*.h5
*.hdf5

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
*.cover

# Misc
*.log
*.tmp
*.bak
```

### Cleaning Up Accidentally Committed Files

```bash
# Remove file from git but keep locally
git rm --cached model.pt
git commit -m "chore: remove accidentally committed model file"

# Remove file from history (if sensitive)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/file" \
  --prune-empty --tag-name-filter cat -- --all

# Simpler with git-filter-repo (install separately)
git filter-repo --path path/to/file --invert-paths
```

### File Organization Best Practices

```
vascx_simplify/
├── .github/
│   └── copilot-instructions.md     # Copilot configuration
├── docs/
│   ├── PERFORMANCE.md              # Performance guide
│   └── DEVELOPMENT.md              # This file
├── examples/
│   ├── 01_artery_vein.py          # Minimal working examples
│   ├── 02_disc_segment.py
│   └── HRF_07_dr.jpg              # Small test image OK
├── src/
│   └── vascx_simplify/
│       ├── __init__.py            # Public API
│       ├── inference.py           # Inference utilities
│       ├── preprocess.py          # Preprocessing
│       └── utils.py               # Helper functions
├── tests/                         # Optional
│   └── test_consistency.py
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Code Review Checklist

### Before Committing

- [ ] **Code Quality**
  - [ ] No commented-out code
  - [ ] No debug print statements
  - [ ] No TODO/FIXME comments (fix them or create issues)
  - [ ] Type hints added to all functions
  - [ ] Docstrings added to public APIs
  - [ ] Code follows project style guide

- [ ] **Functionality**
  - [ ] Code runs without errors
  - [ ] Output matches expected results
  - [ ] Edge cases handled
  - [ ] No regression in existing features

- [ ] **Performance**
  - [ ] No unnecessary CPU↔GPU transfers
  - [ ] Mixed precision used appropriately
  - [ ] No memory leaks
  - [ ] Batch operations used where possible

- [ ] **Git**
  - [ ] Commit message is clear and descriptive
  - [ ] Changes are focused on one task
  - [ ] No unnecessary files included
  - [ ] Diff reviewed with `git diff`

### Self-Review Process

```bash
# 1. Check what you're about to commit
git diff --staged

# 2. Review changes section by section
# Ask yourself:
# - Is this change necessary?
# - Does it break anything?
# - Is it the simplest solution?
# - Are there side effects?

# 3. Run tests
python examples/01_artery_vein.py
python -m pytest tests/

# 4. Check code quality
# - No debug prints?
# - No commented code?
# - Type hints present?
# - Docstrings added?

# 5. Verify performance
# - Run benchmark if performance-critical
# - Check memory usage
# - Profile if unsure

# 6. If everything looks good, commit
git commit -m "feat: add feature X"

# 7. Immediately review your commit
git show HEAD
```

## Testing Workflow

### Manual Testing

```bash
# Test each example after changes
python examples/01_artery_vein.py
python examples/02_disc_segment.py
python examples/03_fovea_regression.py
python examples/04_quality_classify.py

# If all pass, you're good to commit!
```

### Automated Testing (Optional)

```python
# tests/test_inference.py
import pytest
import torch
from vascx_simplify import sliding_window_inference

def test_sliding_window_basic():
    """Test basic sliding window inference."""
    inputs = torch.randn(1, 3, 512, 512)
    
    def dummy_predictor(x):
        return torch.randn(x.shape[0], 2, x.shape[2], x.shape[3])
    
    output = sliding_window_inference(
        inputs,
        roi_size=(256, 256),
        sw_batch_size=4,
        predictor=dummy_predictor
    )
    
    assert output.shape == (1, 2, 512, 512)
    assert output.device == inputs.device

def test_sliding_window_device_preservation():
    """Test that device is preserved."""
    if torch.cuda.is_available():
        inputs = torch.randn(1, 3, 512, 512).cuda()
        
        def dummy_predictor(x):
            return torch.randn(x.shape[0], 2, x.shape[2], x.shape[3]).to(x.device)
        
        output = sliding_window_inference(
            inputs,
            roi_size=(256, 256),
            sw_batch_size=4,
            predictor=dummy_predictor
        )
        
        assert output.device.type == 'cuda'
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=vascx_simplify --cov-report=html

# Run specific test
python -m pytest tests/test_inference.py::test_sliding_window_basic

# Run with verbose output
python -m pytest tests/ -v
```

### Continuous Testing

```bash
# Install pytest-watch for auto-testing
pip install pytest-watch

# Auto-run tests on file changes
ptw -- tests/
```

## Quick Reference Commands

### Git Commands

```bash
# Status and diff
git status                          # What's changed?
git diff                           # Show unstaged changes
git diff --staged                  # Show staged changes
git diff HEAD~1                    # Compare with last commit

# Staging
git add <file>                     # Stage specific file
git add -p                         # Stage interactively
git reset <file>                   # Unstage file

# Committing
git commit -m "type: message"      # Commit staged changes
git commit --amend                 # Fix last commit
git commit --amend --no-edit       # Add to last commit

# History
git log --oneline -10              # Last 10 commits
git log --graph --all              # Visual history
git show HEAD                      # Show last commit
git show HEAD~1                    # Show commit before last

# Comparison
git diff HEAD~1                    # Compare with last commit
git diff abc1234 HEAD              # Compare specific commits
git diff main..feature-branch      # Compare branches

# Undoing (use carefully!)
git reset --soft HEAD~1            # Undo commit, keep changes
git reset --hard HEAD~1            # Undo commit, discard changes
git revert HEAD                    # Create new commit that undoes last commit
```

### Testing Commands

```bash
# Run examples
python examples/01_artery_vein.py

# Run tests
python -m pytest tests/
python -m pytest tests/ -v
python -m pytest tests/ --cov=vascx_simplify

# Run consistency check
python test_consistency.py

# Profile performance
python -m cProfile -o profile.out examples/01_artery_vein.py
```

### File Management

```bash
# Check what will be committed
git status
git diff --staged

# Remove file from staging
git reset <file>

# Remove file from git (keep locally)
git rm --cached <file>

# Clean untracked files (preview)
git clean -n

# Clean untracked files (do it)
git clean -fd
```

## Common Workflows

### Adding a New Feature

```bash
# 1. Create feature branch (optional)
git checkout -b feature/gaussian-importance

# 2. Implement feature
# ... edit files ...

# 3. Test thoroughly
python examples/01_artery_vein.py
python test_consistency.py

# 4. Review changes
git diff

# 5. Commit
git add src/vascx_simplify/inference.py
git commit -m "feat: add gaussian importance map for sliding window"

# 6. Verify
git show HEAD
git diff HEAD~1

# 7. Merge to main (if using branches)
git checkout main
git merge feature/gaussian-importance
```

### Fixing a Bug

```bash
# 1. Reproduce the bug
# ... create test case ...

# 2. Fix the bug
# ... edit files ...

# 3. Verify fix works
python test_consistency.py

# 4. Review changes
git diff

# 5. Commit with clear description
git commit -m "fix: correct device handling in preprocessing"

# 6. Verify output unchanged (unless bug was in output)
python test_consistency.py
```

### Optimizing Performance

```bash
# 1. Benchmark current performance
python benchmark.py  # Create this script

# 2. Make optimization
# ... edit files ...

# 3. Benchmark again
python benchmark.py

# 4. Verify output unchanged
python test_consistency.py

# 5. Commit with performance metrics
git commit -m "perf: optimize sliding window with batching (2.5x speedup)"

# 6. Document performance gain
# Add note in commit message or PERFORMANCE.md
```

## Summary

1. **Commit frequently** - After each logical task
2. **Write clear messages** - Follow conventional commits format
3. **Check diffs** - Always review with `git diff` before and after commit
4. **Test everything** - Run examples and tests before committing
5. **Verify consistency** - Ensure outputs remain the same (unless fixing bugs)
6. **Keep it clean** - Don't commit unnecessary files
7. **Document changes** - Update docs when adding features

**Remember:** Good commits tell a story of how the code evolved!

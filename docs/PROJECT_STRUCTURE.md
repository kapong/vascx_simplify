# Project Structure Overview

This document explains the organization of the vascx_simplify project and the purpose of each directory and key file.

## Directory Structure

```
vascx_simplify/
├── .github/                    # GitHub-specific configuration
│   └── copilot-instructions.md # GitHub Copilot project guidelines
│
├── docs/                       # Comprehensive documentation
│   ├── README.md              # Documentation overview
│   ├── QUICK_REFERENCE.md     # Cheat sheet for common tasks
│   ├── BEST_PRACTICES.md      # Code organization and patterns
│   ├── PERFORMANCE.md         # GPU optimization techniques
│   └── DEVELOPMENT.md         # Git workflow and testing
│
├── examples/                   # Working code examples
│   ├── 01_artery_vein.py      # Artery/vein segmentation
│   ├── 02_disc_segment.py     # Optic disc segmentation
│   ├── 03_fovea_regression.py # Fovea detection
│   ├── 04_quality_classify.py # Image quality assessment
│   └── HRF_07_dr.jpg          # Test image
│
├── src/                        # Source code
│   └── vascx_simplify/
│       ├── __init__.py        # Public API exports
│       ├── inference.py       # Inference utilities
│       ├── preprocess.py      # Preprocessing pipelines
│       └── utils.py           # Helper functions
│
├── tests/                      # Unit tests (optional)
│   └── test_*.py              # Test files
│
├── .gitignore                 # Git ignore patterns
├── LICENSE                    # MIT License
├── README.md                  # Project overview and usage
├── ACKNOWLEDGMENTS.md         # Credits and acknowledgments
├── MANIFEST.in                # Package manifest
├── pyproject.toml             # Project configuration (PEP 621)
├── setup.py                   # Setup script (legacy)
└── requirements.txt           # Python dependencies
```

## Key Files

### Configuration Files

#### `pyproject.toml`
Modern Python project configuration (PEP 621). Contains:
- Project metadata (name, version, description)
- Dependencies
- Build system requirements
- Tool configurations (pytest, black, isort)

#### `setup.py`
Legacy setup script for backward compatibility. Minimal implementation that defers to `pyproject.toml`.

#### `requirements.txt`
Pin-free dependency list for installation:
```
torch>=1.10.0
kornia>=0.6.0
numpy>=1.21.0
...
```

#### `.gitignore`
Excludes from git:
- Python cache files (`__pycache__/`, `*.pyc`)
- Build artifacts (`dist/`, `build/`)
- Model weights (`*.pt`, `*.pth`)
- IDE files (`.vscode/`, `.idea/`)

### Documentation Files

#### `.github/copilot-instructions.md`
**For:** GitHub Copilot AI assistant
**Purpose:** Configure Copilot to follow project conventions
**Contains:**
- Code style guidelines
- Performance optimization rules
- Output consistency requirements
- Git workflow expectations

#### `docs/README.md`
**For:** All developers
**Purpose:** Documentation directory overview
**Contains:**
- Guide summaries
- Quick start instructions
- Common questions
- Example code snippets

#### `docs/QUICK_REFERENCE.md`
**For:** Daily development
**Purpose:** Fast lookup for common tasks
**Contains:**
- Essential commands
- Code patterns
- Checklists
- Performance targets

#### `docs/BEST_PRACTICES.md`
**For:** Writing new code
**Purpose:** Code organization and design patterns
**Contains:**
- Naming conventions
- Type hints
- Error handling
- When NOT to abstract

#### `docs/PERFORMANCE.md`
**For:** Optimizing performance
**Purpose:** GPU optimization techniques
**Contains:**
- Memory management
- Mixed precision (FP16/FP32)
- Batch processing
- Profiling tools

#### `docs/DEVELOPMENT.md`
**For:** Git workflow
**Purpose:** Development process and testing
**Contains:**
- Commit strategy
- Backward compatibility checking
- File management
- Testing workflow

### Source Code

#### `src/vascx_simplify/__init__.py`
**Public API exports.** Only items exported here are considered public.

```python
from .inference import (
    sliding_window_inference,
    EnsembleSegmentation,
    ClassificationEnsemble,
    HeatmapRegressionEnsemble,
)
from .preprocess import (
    FundusContrastEnhance,
    VASCXTransform,
)
from .utils import (
    from_huggingface,
)

__version__ = "0.1.2"
```

#### `src/vascx_simplify/inference.py`
**Inference utilities.** Contains:
- `sliding_window_inference()` - For large images
- `EnsembleSegmentation` - Segmentation models
- `ClassificationEnsemble` - Classification models
- `HeatmapRegressionEnsemble` - Regression models

#### `src/vascx_simplify/preprocess.py`
**Preprocessing pipelines.** Contains:
- `FundusContrastEnhance` - GPU-accelerated contrast enhancement
- `VASCXTransform` - Standard preprocessing transform

#### `src/vascx_simplify/utils.py`
**Helper functions.** Contains:
- `from_huggingface()` - Download models from HuggingFace Hub

### Examples

#### `examples/01_artery_vein.py`
Demonstrates artery/vein segmentation:
- Load model from HuggingFace
- Preprocess fundus image
- Run segmentation
- Visualize results (arteries=red, veins=blue)

#### `examples/02_disc_segment.py`
Demonstrates optic disc segmentation:
- Load disc segmentation model
- Process image
- Visualize disc mask

#### `examples/03_fovea_regression.py`
Demonstrates fovea detection:
- Load heatmap regression model
- Detect fovea coordinates
- Visualize location

#### `examples/04_quality_classify.py`
Demonstrates quality assessment:
- Load classification model
- Classify image quality
- Output: Reject/Usable/Good

## File Organization Principles

### What Goes Where

**Source Code (`src/vascx_simplify/`)**
- Core functionality
- Public APIs
- Reusable components
- GPU-accelerated operations

**Examples (`examples/`)**
- Working demonstrations
- Usage patterns
- Minimal, focused code
- Small test images only

**Documentation (`docs/`)**
- Developer guides
- Best practices
- Performance tips
- Workflow instructions

**Tests (`tests/`)**
- Unit tests
- Integration tests
- Performance benchmarks
- Consistency checks

**Configuration (root)**
- Project metadata
- Dependencies
- Build configuration
- Tool settings

### What NOT to Commit

**Never commit:**
- ❌ Model weights (`*.pt`, `*.pth`)
- ❌ Large data files (`*.npy`, `*.h5`)
- ❌ Cache files (`__pycache__/`, `*.pyc`)
- ❌ Build artifacts (`dist/`, `build/`)
- ❌ IDE settings (`.vscode/`, `.idea/`)
- ❌ Virtual environments (`venv/`, `.env/`)
- ❌ Temporary files (`*.tmp`, `*.bak`)

**Exception:** Small test images (<1MB) in `examples/` are OK.

## Dependencies

### Core Dependencies

```python
# Deep Learning
torch>=1.10.0              # PyTorch
kornia>=0.6.0              # Computer vision ops

# Scientific Computing
numpy>=1.21.0              # Numerical operations
scipy>=1.7.0               # Scientific functions
scikit-learn>=1.0.0        # Machine learning utilities

# Utilities
huggingface-hub>=0.10.0    # Model downloads
```

### Why These Versions?

- **Python >= 3.12**: Modern features, better performance
- **PyTorch >= 1.10**: FP16 support, modern features
- **Kornia >= 0.6**: GPU-accelerated transforms
- **Minimal deps**: Easier installation, fewer conflicts

### Optional Dependencies

```python
# Development
dev = [
    "black>=22.0.0",        # Code formatting
    "flake8>=5.0.0",        # Linting
    "isort>=5.10.0",        # Import sorting
    "mypy>=0.990",          # Type checking
    "pytest>=7.0.0",        # Testing
]
```

## Project Workflow

### Typical Development Flow

1. **Setup**
   ```bash
   git clone https://github.com/kapong/vascx_simplify.git
   cd vascx_simplify
   pip install -e .
   ```

2. **Read Documentation**
   - Start with `docs/README.md`
   - Check `docs/QUICK_REFERENCE.md` for patterns

3. **Make Changes**
   - Edit files in `src/vascx_simplify/`
   - Follow `docs/BEST_PRACTICES.md`

4. **Test**
   ```bash
   python examples/01_artery_vein.py
   python -m pytest tests/
   ```

5. **Review**
   ```bash
   git diff
   git diff --staged
   ```

6. **Commit**
   ```bash
   git add <files>
   git commit -m "type: description"
   ```

7. **Verify**
   ```bash
   git diff HEAD~1
   git show HEAD
   ```

## Design Decisions

### Why This Structure?

**Single `src/` directory**
- Clear separation of source and other files
- Standard Python packaging structure
- Easy to install with `pip install -e .`

**Flat module structure**
- Only 4 files: `__init__.py`, `inference.py`, `preprocess.py`, `utils.py`
- Easy to navigate
- No unnecessary nesting

**Examples separate from source**
- Examples are not part of the package
- Can be run independently
- Show real-world usage

**Comprehensive docs**
- Separate guides for different purposes
- Quick reference for daily use
- In-depth guides for deep dives

### Why Minimal Dependencies?

- **Easier installation**: Fewer things to break
- **Faster installs**: Less to download
- **Fewer conflicts**: Less chance of version issues
- **Clearer purpose**: Only what's needed

### Why Copilot Instructions?

GitHub Copilot can:
- Suggest code following project conventions
- Avoid common anti-patterns
- Follow performance best practices
- Generate consistent code style

The `.github/copilot-instructions.md` file trains Copilot on project-specific patterns.

## Quick Navigation

### I want to...

**Learn the codebase:**
1. Read `README.md` (project overview)
2. Read `docs/README.md` (documentation overview)
3. Read `docs/BEST_PRACTICES.md` (conventions)
4. Look at `examples/` (usage)

**Write new code:**
1. Check `docs/QUICK_REFERENCE.md` (patterns)
2. Follow `docs/BEST_PRACTICES.md` (style)
3. Review `src/vascx_simplify/` (existing code)

**Optimize performance:**
1. Read `docs/PERFORMANCE.md` (techniques)
2. Profile current code
3. Apply optimizations
4. Benchmark improvements

**Make commits:**
1. Test changes
2. Review with `git diff`
3. Follow `docs/DEVELOPMENT.md` (workflow)
4. Commit with clear message
5. Verify with `git diff HEAD~1`

**Debug issues:**
1. Check `docs/PERFORMANCE.md` (profiling)
2. Use profiling decorators
3. Review with `git diff` (what changed?)
4. Test with examples

## Summary

This project is organized for:
- **Simplicity**: Flat structure, minimal files
- **Clarity**: Clear separation of concerns
- **Performance**: GPU-optimized, best practices
- **Maintainability**: Comprehensive docs, clear workflow
- **Quality**: Testing, profiling, code review

**Key Principle:** Keep it simple, make it fast, document well.

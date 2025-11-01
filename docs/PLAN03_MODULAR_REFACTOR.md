# PLAN03: Modular Refactoring - Split inference.py and preprocess.py

## Goals
- Split large monolithic files into smaller, focused modules
- Improve code discoverability and maintainability
- Maintain 100% backward compatibility with existing API
- Preserve all performance optimizations
- Keep the same import paths for users

## Motivation
Currently, `inference.py` (665 lines) and `preprocess.py` (1102 lines) contain multiple classes and utilities that serve different purposes. This makes the codebase harder to navigate and maintain. By splitting into focused modules:

1. **Easier navigation**: Developers can quickly find specific functionality
2. **Better organization**: Related code grouped together logically
3. **Clearer boundaries**: Each module has a single responsibility
4. **Improved testing**: Smaller modules are easier to test in isolation
5. **Scalability**: Adding new features doesn't bloat existing files

## Current Structure

### inference.py (665 lines)
- `sliding_window_inference()` - Sliding window utility
- `_create_gaussian_importance_map()` - Helper function
- `_initialize_output_tensor()` - Helper function
- `EnsembleBase` - Base class for all ensemble models
- `EnsembleSegmentation` - Segmentation ensemble
- `ClassificationEnsemble` - Classification ensemble
- `RegressionEnsemble` - Regression ensemble
- `HeatmapRegressionEnsemble` - Heatmap regression ensemble

### preprocess.py (1102 lines)
- `FundusContrastEnhance` - GPU-accelerated fundus contrast enhancement (~988 lines)
- `VASCXTransform` - Transform pipeline (~98 lines)

## Proposed Structure

```
src/vascx_simplify/
├── __init__.py              # Public API exports (unchanged for users)
├── utils.py                 # Utilities (HuggingFace, etc.)
├── inference/
│   ├── __init__.py          # Re-export all inference classes/functions
│   ├── sliding_window.py    # Sliding window inference + helpers
│   ├── base.py              # EnsembleBase abstract class
│   ├── segmentation.py      # EnsembleSegmentation
│   ├── classification.py    # ClassificationEnsemble
│   ├── regression.py        # RegressionEnsemble
│   └── heatmap.py           # HeatmapRegressionEnsemble
└── preprocess/
    ├── __init__.py          # Re-export all preprocessing classes
    ├── contrast.py          # FundusContrastEnhance
    └── transform.py         # VASCXTransform
```

## Technical Approach

### 1. Module Organization Principles
- **One class per file** for ensemble models (except base)
- **Related utilities together** (sliding window + helpers)
- **Re-export everything** from `__init__.py` to maintain backward compatibility
- **Keep imports clean** - no circular dependencies

### 2. Backward Compatibility Strategy
The existing public API must remain unchanged:

```python
# These imports MUST still work after refactoring
from vascx_simplify import (
    EnsembleSegmentation,
    ClassificationEnsemble,
    RegressionEnsemble,
    HeatmapRegressionEnsemble,
    sliding_window_inference,
    FundusContrastEnhance,
    VASCXTransform,
)
```

This is achieved by:
1. Creating subpackage `__init__.py` files that re-export everything
2. Updating main `src/vascx_simplify/__init__.py` to import from subpackages
3. No changes needed in example scripts or user code

### 3. Import Path Design

**inference/__init__.py:**
```python
from .sliding_window import sliding_window_inference
from .base import EnsembleBase
from .segmentation import EnsembleSegmentation
from .classification import ClassificationEnsemble
from .regression import RegressionEnsemble
from .heatmap import HeatmapRegressionEnsemble

__all__ = [
    "sliding_window_inference",
    "EnsembleBase",
    "EnsembleSegmentation",
    "ClassificationEnsemble",
    "RegressionEnsemble",
    "HeatmapRegressionEnsemble",
]
```

**preprocess/__init__.py:**
```python
from .contrast import FundusContrastEnhance
from .transform import VASCXTransform

__all__ = [
    "FundusContrastEnhance",
    "VASCXTransform",
]
```

**Main __init__.py remains the same:**
```python
from .inference import (
    EnsembleSegmentation,
    ClassificationEnsemble,
    RegressionEnsemble,
    HeatmapRegressionEnsemble,
    sliding_window_inference,
)
from .preprocess import FundusContrastEnhance, VASCXTransform
# ... rest unchanged
```

### 4. File Split Details

#### inference/ module breakdown:
1. **sliding_window.py** (~125 lines)
   - `sliding_window_inference()` function
   - `_create_gaussian_importance_map()` helper
   - `_initialize_output_tensor()` helper
   - Constants: `GAUSSIAN_SIGMA_FRACTION`, `MIN_WEIGHT_THRESHOLD`

2. **base.py** (~225 lines)
   - `EnsembleBase` abstract class
   - Common methods: `_load_models()`, `_build_ensemble()`, `get_ensemble_info()`
   - Batch processing utilities

3. **segmentation.py** (~45 lines)
   - `EnsembleSegmentation` class
   - Inherits from `EnsembleBase`

4. **classification.py** (~80 lines)
   - `ClassificationEnsemble` class
   - Classification-specific logic

5. **regression.py** (~75 lines)
   - `RegressionEnsemble` class
   - Regression-specific logic

6. **heatmap.py** (~60 lines)
   - `HeatmapRegressionEnsemble` class
   - Heatmap regression logic

#### preprocess/ module breakdown:
1. **contrast.py** (~988 lines)
   - `FundusContrastEnhance` class
   - All contrast enhancement methods and constants

2. **transform.py** (~98 lines)
   - `VASCXTransform` class
   - Transform pipeline logic

## Implementation Steps

### Phase 1: Create New Module Structure (No Breaking Changes)
1. ✅ Create branch: `plan/modular-refactor`
2. ✅ Create plan document: `docs/PLAN03_MODULAR_REFACTOR.md`
3. Create directory structure:
   - `src/vascx_simplify/inference/`
   - `src/vascx_simplify/preprocess/`

### Phase 2: Split inference.py
4. Create `inference/__init__.py` (empty initially)
5. Extract `sliding_window_inference()` to `inference/sliding_window.py`:
   - Move function + helpers + constants
   - Test imports and functionality
6. Extract `EnsembleBase` to `inference/base.py`:
   - Move base class + common methods
   - Update imports
7. Extract each ensemble class to separate files:
   - `inference/segmentation.py` → `EnsembleSegmentation`
   - `inference/classification.py` → `ClassificationEnsemble`
   - `inference/regression.py` → `RegressionEnsemble`
   - `inference/heatmap.py` → `HeatmapRegressionEnsemble`
8. Complete `inference/__init__.py` with all re-exports
9. Update main `__init__.py` to import from `inference/`
10. **Verify**: Run all examples, ensure outputs match

### Phase 3: Split preprocess.py
11. Create `preprocess/__init__.py` (empty initially)
12. Extract `FundusContrastEnhance` to `preprocess/contrast.py`
13. Extract `VASCXTransform` to `preprocess/transform.py`
14. Complete `preprocess/__init__.py` with all re-exports
15. Update main `__init__.py` to import from `preprocess/`
16. **Verify**: Run all examples, ensure outputs match

### Phase 4: Cleanup and Testing
17. Delete old `inference.py` and `preprocess.py` files
18. Run lint and format: `black src/ examples/` + `isort src/ examples/`
19. Run all examples sequentially to verify:
    - `01_artery_vein.py`
    - `02_disc_segment.py`
    - `03_fovea_regression.py`
    - `04_quality_classify.py`
    - `05_batch_fovea.py`
20. Check imports in all example files still work
21. Verify no circular import issues
22. Git commit with message: `refactor: split inference and preprocess into modular structure`

### Phase 5: Documentation
23. Update `docs/PROJECT_STRUCTURE.md` with new structure
24. Update `docs/QUICK_REFERENCE.md` if needed
25. Add migration notes (though API is unchanged)
26. Git commit with message: `docs: update structure documentation for modular refactor`

### Phase 6: Merge
27. Review all changes with `git diff main`
28. Ensure all tests pass and examples run
29. Merge to main: `git checkout main && git merge plan/modular-refactor`
30. Consider creating a patch version release

## Success Criteria

### Functional Requirements
- [ ] All existing imports work without changes
- [ ] All example scripts run successfully with same outputs
- [ ] No circular import issues
- [ ] Module structure is clear and intuitive

### Code Quality Requirements
- [ ] Each file has a single, clear responsibility
- [ ] No code duplication introduced
- [ ] All files pass `black` and `isort` formatting
- [ ] Type hints preserved in all new files
- [ ] Docstrings preserved for public APIs

### Performance Requirements
- [ ] No performance degradation (same GPU operations)
- [ ] No additional import overhead
- [ ] Memory usage unchanged

### Documentation Requirements
- [ ] `PROJECT_STRUCTURE.md` updated
- [ ] Module-level docstrings in all `__init__.py` files
- [ ] README.md examples still work

## Testing Strategy

### 1. Import Testing
```python
# Test 1: Verify old imports still work
from vascx_simplify import (
    EnsembleSegmentation,
    ClassificationEnsemble,
    RegressionEnsemble,
    HeatmapRegressionEnsemble,
    sliding_window_inference,
    FundusContrastEnhance,
    VASCXTransform,
)
print("✅ All imports successful")

# Test 2: Verify new imports also work
from vascx_simplify.inference import EnsembleSegmentation
from vascx_simplify.preprocess import VASCXTransform
print("✅ New import paths work")
```

### 2. Output Consistency Testing
Run each example script and verify:
- No errors during execution
- Output files generated successfully
- Visual inspection of outputs (same as before)

### 3. Performance Testing
```python
import time
import torch

# Before and after timing comparison
model = EnsembleSegmentation(...)
image = torch.randn(1, 3, 512, 512).cuda()

start = time.time()
output = model.predict(image)
elapsed = time.time() - start

# Should be ~same as before (<5% variance acceptable)
print(f"Inference time: {elapsed:.3f}s")
```

### 4. Manual Testing Checklist
- [ ] `python examples/01_artery_vein.py` - runs without errors
- [ ] `python examples/02_disc_segment.py` - runs without errors
- [ ] `python examples/03_fovea_regression.py` - runs without errors
- [ ] `python examples/04_quality_classify.py` - runs without errors
- [ ] `python examples/05_batch_fovea.py` - runs without errors
- [ ] No import errors or warnings
- [ ] Output files match expected results

## Risks and Mitigations

### Risk 1: Circular Imports
**Mitigation**: 
- Keep base classes in separate files
- Use type hints with `TYPE_CHECKING` if needed
- Avoid cross-imports between submodules

### Risk 2: Breaking User Code
**Mitigation**:
- Maintain all existing import paths via `__init__.py` re-exports
- Test import compatibility before merging
- Document both old and new import styles

### Risk 3: Import Performance Overhead
**Mitigation**:
- Python caches imports automatically
- Re-exports add negligible overhead
- Profile if concerned, but shouldn't be measurable

### Risk 4: Incomplete Module Extraction
**Mitigation**:
- Use grep to find all references before moving code
- Test each module independently after extraction
- Verify no orphaned functions/classes

## Notes
- This refactoring is purely structural - no logic changes
- All performance optimizations preserved
- Users won't need to change any code
- Internal development becomes much cleaner
- Foundation for future feature additions

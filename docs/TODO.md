# Refactoring TODO

## ✅ Phase 1: Safe Constants & Type Hints (COMPLETED)

### Phase 1.1: Extract Constants in inference.py ✅
- [x] Add `GAUSSIAN_SIGMA_FRACTION = 0.125` constant
- [x] Add `MIN_WEIGHT_THRESHOLD = 1e-8` constant
- [x] Replace magic numbers in code with named constants
- [x] Commit: `refactor: extract constants and add type hints to inference.py`

### Phase 1.2: Add Type Hints to inference.py ✅
- [x] Add typing imports (`Tuple`, `Callable`, `Optional`, `Union`, `Dict`, `Any`)
- [x] Add type hints to `sliding_window_inference()`
- [x] Add type hints to all `EnsembleBase` methods
- [x] Add type hints to all ensemble subclasses
- [x] Remove unused variable `num_wins`

### Phase 1.3: Extract Constants in preprocess.py ✅
- [x] Add circle detection constants (`MAX_RADIUS_DIVISOR`, `MIN_RADIUS_DIVISOR`)
- [x] Add scaling constants (`CIRCLE_SCALE_FACTOR`, `BORDER_MARGIN_FRACTION`, `CIRCLE_REFIT_SCALE`)
- [x] Add enhancement constants (`DEFAULT_SIGMA_FRACTION`, `DEFAULT_CONTRAST_FACTOR`, `REDUCED_BLUR_RESOLUTION`)
- [x] Replace all hardcoded magic numbers (0.99, 0.01, 0.95, 256, etc.)
- [x] Update `__init__` to use class constants as defaults
- [x] Commit: `refactor: extract constants and add type hints to preprocess.py`

### Phase 1.4: Add Type Hints to preprocess.py ✅
- [x] Add typing imports
- [x] Add type hints to all `FundusContrastEnhance` methods
- [x] Add type hints to all `VASCXTransform` methods
- [x] Type hint all internal helper methods

### Phase 1 Verification ✅
- [x] Create test script to verify refactoring
- [x] Verify all modules import correctly
- [x] Verify all constants accessible and correct
- [x] Verify class signatures remain compatible
- [x] Verify no functional changes

**Status:** ✅ COMPLETE - 2 commits, 100% backward compatible, no breaking changes

---

## 🔄 Phase 2: Extract Helper Functions (TODO)

### Priority: Medium | Risk: Low-Medium

### Phase 2.1: Extract Coordinate Transform Helpers
**File:** `preprocess.py`

**Objective:** Reduce code duplication in matrix operations

**Refactorings:**
- [ ] Extract `_invert_affine_matrix()` helper
  ```python
  def _invert_affine_matrix(M: torch.Tensor, device: torch.device) -> torch.Tensor:
      """Convert 2x3 affine matrix to 3x3 and invert."""
      M_3x3 = torch.cat([M.float(), torch.tensor([[0., 0., 1.]], device=device)], dim=0)
      return torch.inverse(M_3x3)
  ```
- [ ] Extract `_transform_point_to_original()` helper for single point transformation
- [ ] Replace 3 occurrences of matrix inversion logic with helper
- [ ] Test: Verify outputs unchanged in `_get_bounds()`

**Expected Impact:**
- Reduce ~15 lines of duplicated code
- Improve readability of coordinate transform logic

### Phase 2.2: Extract Grid Management Helpers
**File:** `preprocess.py`

**Objective:** Simplify grid caching mechanism

**Refactorings:**
- [ ] Extract `_get_or_create_grid()` method
  ```python
  def _get_or_create_grid(
      self, h: int, w: int, device: torch.device, dtype: torch.dtype
  ) -> Tuple[torch.Tensor, torch.Tensor]:
      """Get or create cached coordinate grids."""
      cache_key = (h, w, device, dtype)
      if cache_key not in self._grid_cache:
          self._grid_cache[cache_key] = self._create_coordinate_grid(h, w, device, dtype)
      return self._grid_cache[cache_key]
  ```
- [ ] Extract `_create_coordinate_grid()` method
- [ ] Replace 3 occurrences of grid creation logic
- [ ] Test: Verify caching still works correctly

**Expected Impact:**
- Reduce ~20 lines of duplicated code
- Clearer separation of grid creation vs. caching

### Phase 2.3: Extract Shape Handling in inference.py
**File:** `inference.py`

**Objective:** Simplify tensor shape initialization

**Refactorings:**
- [ ] Extract `_initialize_output_tensor()` helper
  ```python
  def _initialize_output_tensor(
      first_pred: torch.Tensor,
      batch_size: int, height: int, width: int,
      device: torch.device, dtype: torch.dtype
  ) -> Tuple[torch.Tensor, torch.Tensor]:
      """Initialize output tensor with correct shape based on prediction dimensions."""
      if first_pred.dim() == 5:  # (B, M, C, H, W)
          _, n_models, n_classes, _, _ = first_pred.shape
          output = torch.zeros((batch_size, n_models, n_classes, height, width), 
                              device=device, dtype=dtype)
          importance_map_exp = importance_map[None, None, None, :, :]
      else:  # (B, C, H, W)
          _, n_classes, _, _ = first_pred.shape
          output = torch.zeros((batch_size, n_classes, height, width), 
                              device=device, dtype=dtype)
          importance_map_exp = importance_map[None, None, :, :]
      return output, importance_map_exp
  ```
- [ ] Refactor `sliding_window_inference()` to use helper
- [ ] Test: Verify sliding window outputs unchanged

**Expected Impact:**
- Reduce complexity in main sliding window function
- Easier to understand output tensor initialization

---

## 🔨 Phase 3: Reduce Code Duplication (TODO)

### Priority: Medium | Risk: Medium

### Phase 3.1: Consolidate Ensemble Class Duplication
**File:** `inference.py`

**Objective:** Extract common patterns from ensemble subclasses

**Refactorings:**
- [ ] Extract `_prepare_input()` in `EnsembleBase`
  ```python
  def _prepare_input(self, img: torch.Tensor) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
      """Common input preparation logic."""
      img, bounds = self.transforms(img)
      return img.to(self.device).unsqueeze(dim=0), bounds
  ```
- [ ] Extract `_run_inference()` wrapper in `EnsembleBase`
- [ ] Simplify `predict()` methods in subclasses to use base helpers
- [ ] Test: Run all 4 examples, verify outputs identical

**Expected Impact:**
- Reduce ~15-20 lines of duplicated code
- Clearer separation of concerns in ensemble classes

### Phase 3.2: Consolidate Edge Mirroring Logic
**File:** `preprocess.py`

**Objective:** Reduce duplication in `_mirror_image()`

**Refactorings:**
- [ ] Extract `_mirror_edge()` helper method
  ```python
  def _mirror_edge(
      self, mirrored: torch.Tensor, bound_val: int, 
      is_2d: bool, dim_2d: int, dim_3d: int, direction: str
  ) -> None:
      """Mirror a single edge (top/bottom/left/right)."""
      # Unified logic for all edges
  ```
- [ ] Parameterize edge mirroring with configuration
- [ ] Replace 4 if-blocks with loop over edge configs
- [ ] Test: Verify mirroring outputs unchanged

**Expected Impact:**
- Reduce ~40 lines of duplicated code
- Easier to maintain edge mirroring logic

### Phase 3.3: Extract Line Fitting Logic
**File:** `preprocess.py`

**Objective:** Simplify `_fit_lines()` method

**Refactorings:**
- [ ] Extract `_fit_line_ransac()` helper for single line fitting
- [ ] Extract horizontal vs vertical line logic into helper
- [ ] Reduce duplication in left/right and top/bottom fitting
- [ ] Test: Verify line detection unchanged

**Expected Impact:**
- Reduce ~15 lines of duplicated code
- Clearer line fitting logic

---

## 🔪 Phase 4: Split Large Methods (TODO)

### Priority: Low-Medium | Risk: Medium

### Phase 4.1: Split `_get_bounds()` Method
**File:** `preprocess.py` (Currently ~100 lines)

**Objective:** Break down complex method into logical units

**Refactorings:**
- [ ] Extract `_scale_image_for_detection()`
  - Handles image scaling to RESOLUTION
- [ ] Extract `_fit_lines_if_needed()`
  - Conditional line fitting based on circle_fraction
- [ ] Extract `_transform_bounds_to_original()`
  - Transform center, radius, lines back to original coordinates
- [ ] Refactor `_get_bounds()` to orchestrate these steps
- [ ] Test: Verify bounds detection outputs unchanged

**Expected Impact:**
- Split 100-line method into 4 focused methods (~20-30 lines each)
- Each method has single responsibility
- Easier to test and maintain individual steps

### Phase 4.2: Split `_mirror_image()` Method
**File:** `preprocess.py` (Currently ~80 lines)

**Objective:** Separate edge mirroring from circle mirroring

**Refactorings:**
- [ ] Extract `_mirror_edges()` method
  - Handles all rectangular edge mirroring
- [ ] Extract `_mirror_circle()` method
  - Handles circular boundary mirroring
- [ ] Keep `_mirror_image()` as orchestrator
- [ ] Test: Verify mirroring outputs unchanged

**Expected Impact:**
- Split 80-line method into 3 focused methods
- Clearer separation of edge vs circle mirroring

### Phase 4.3: Split `_enhance_contrast()` Method (Optional)
**File:** `preprocess.py` (Currently ~70 lines)

**Objective:** Separate blur from unsharp mask application

**Refactorings:**
- [ ] Extract `_compute_blur_at_reduced_resolution()`
- [ ] Extract `_apply_unsharp_mask()`
- [ ] Keep `_enhance_contrast()` as orchestrator
- [ ] Test: Verify enhancement outputs unchanged

**Expected Impact:**
- Split 70-line method into 3 focused methods
- Each step can be tested independently

---

## 📋 Testing Strategy

### Per-Phase Testing
After each refactoring:
1. Run `python test_refactoring.py` - verify structure
2. Run example files to verify outputs unchanged:
   - `python examples/01_artery_vein.py` (if models available)
   - `python examples/02_disc_segment.py`
   - `python examples/03_fovea_regression.py`
   - `python examples/04_quality_classify.py`
3. Check execution time - no slowdowns
4. Run `git diff` to review changes
5. Commit with descriptive message

### Final Integration Test
After all phases:
- Compare outputs with baseline (before Phase 2)
- Verify memory usage unchanged
- Run all examples end-to-end
- Performance benchmark (if available)

---

## 🎯 Success Criteria

### Code Quality Metrics
- [ ] Reduce total lines of code by ~10-15% through deduplication
- [ ] No method longer than 80 lines
- [ ] All public methods have type hints
- [ ] All magic numbers replaced with named constants
- [ ] Each method has single responsibility

### Compatibility Metrics
- [ ] ✅ Same calling interface (100%)
- [ ] ✅ Same numerical outputs (bit-for-bit)
- [ ] ✅ Same performance characteristics (±5%)
- [ ] ✅ Zero breaking changes

### Maintainability Metrics
- [ ] Easier to add new ensemble types
- [ ] Easier to modify preprocessing steps
- [ ] Better IDE support with type hints
- [ ] Self-documenting code with named constants

---

## ⚠️ Anti-Patterns to Avoid

Throughout all phases:

❌ **Don't change computation order**
- Floating point operations are order-sensitive
- Always preserve exact sequence of operations

❌ **Don't change dtype conversions**
- Keep float16/float32 boundaries identical
- Preserve GPU/CPU device handling

❌ **Don't change tensor operations**
- Keep same PyTorch functions
- Maintain same parameters (mode, align_corners, etc.)

❌ **Don't refactor RANSAC/scipy operations**
- External library calls are black boxes
- Keep as-is unless fixing bugs

❌ **Don't change grid_sample parameters**
- Highly sensitive to parameter changes
- Results can differ significantly

---

## 📝 Notes

- Each phase is independent and can be skipped if desired
- Commit after each logical refactoring (not per-file)
- Always verify with test_refactoring.py before committing
- Keep commits atomic and descriptive
- Document any intentional behavior changes (should be none)

---

**Last Updated:** 2025-11-01  
**Phase 1 Completion:** ✅ 2025-11-01  
**Next Phase:** Phase 2 (Optional - at maintainer's discretion)

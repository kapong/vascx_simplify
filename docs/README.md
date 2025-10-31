# Documentation

This directory contains comprehensive guides for developing and optimizing the vascx_simplify library.

## Available Guides

### 📘 [BEST_PRACTICES.md](BEST_PRACTICES.md)
**Best practices for writing clean, maintainable code**

Topics covered:
- Code organization and file structure
- Naming conventions (snake_case, PascalCase, UPPER_CASE)
- Type hints and documentation
- Error handling with clear messages
- Common design patterns
- When NOT to abstract (avoiding over-engineering)
- Testing patterns

**Read this if you're:**
- New to the codebase
- Writing new features
- Wondering about naming or structure conventions

### ⚡ [PERFORMANCE.md](PERFORMANCE.md)
**Performance optimization techniques for GPU-accelerated code**

Topics covered:
- GPU memory management
- Mixed precision (FP16 vs FP32)
- Batch processing strategies
- Avoiding CPU↔GPU transfers
- Memory profiling and benchmarking
- Performance testing patterns

**Read this if you're:**
- Optimizing slow code
- Working with GPU operations
- Seeing memory issues
- Want to understand FP16/FP32 tradeoffs

### 🔧 [DEVELOPMENT.md](DEVELOPMENT.md)
**Development workflow and git best practices**

Topics covered:
- Git commit strategy (one task = one commit)
- Backward compatibility checking with `git diff`
- File management (what to commit, what to ignore)
- Code review checklist
- Testing workflow
- Common git commands reference

**Read this if you're:**
- Making commits to the repo
- Checking backward compatibility
- Setting up development environment
- Need git workflow guidance

## Quick Start

### For New Contributors

1. **Start here:** Read [BEST_PRACTICES.md](BEST_PRACTICES.md) to understand code conventions
2. **Before committing:** Check [DEVELOPMENT.md](DEVELOPMENT.md) for git workflow
3. **Optimizing code:** Refer to [PERFORMANCE.md](PERFORMANCE.md) for GPU optimization

### For GitHub Copilot

The [.github/copilot-instructions.md](../.github/copilot-instructions.md) file configures GitHub Copilot to follow project conventions automatically. It includes:
- Project-specific best practices
- Performance optimization guidelines
- Output consistency requirements
- File organization rules
- Git workflow expectations

## Key Principles

All guides emphasize these core principles:

1. **Keep it simple** - Don't over-engineer
2. **Optimize wisely** - Profile first, then optimize
3. **Preserve outputs** - No breaking changes without explicit intent
4. **Stay on GPU** - Avoid CPU↔GPU transfers during processing
5. **Clean commits** - One task per commit, verify with `git diff`
6. **No junk files** - Keep repo clean and minimal

## Examples

### Example: GPU-Accelerated Class

```python
class EfficientProcessor:
    """GPU-accelerated processor with cached constants."""
    
    # Pre-compute at class level
    RESOLUTION = 256
    _mask_np = np.arange(RESOLUTION) > RESOLUTION // 2
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        # Convert to GPU once
        self.mask = torch.from_numpy(self._mask_np).to(device)
    
    def process(self, image: torch.Tensor) -> torch.Tensor:
        """Process using cached GPU tensors."""
        return image * self.mask
```

### Example: Commit Workflow

```bash
# 1. Make changes
# ... edit files ...

# 2. Test
python examples/01_artery_vein.py

# 3. Review changes
git diff

# 4. Commit
git add src/vascx_simplify/inference.py
git commit -m "perf: optimize sliding window with batching (2.5x speedup)"

# 5. Verify
git diff HEAD~1
```

### Example: Performance Testing

```python
@profile_memory
def test_inference():
    image = torch.randn(1, 3, 512, 512).cuda()
    output = model(image)
    return output

# Output:
# test_inference Memory Profile:
#   Start: 150.0 MB
#   End: 180.0 MB
#   Peak: 250.0 MB
#   Delta: 30.0 MB
```

## Common Questions

**Q: When should I create abstractions?**  
A: Follow the "Rule of Three" - only abstract when you have 3+ similar implementations. See [BEST_PRACTICES.md](BEST_PRACTICES.md#when-not-to-abstract).

**Q: How do I optimize GPU code?**  
A: Keep tensors on GPU, use FP16 for compute-intensive ops, process in batches. See [PERFORMANCE.md](PERFORMANCE.md#gpu-memory-management).

**Q: What commit message format should I use?**  
A: Use conventional commits: `<type>: <description>`. See [DEVELOPMENT.md](DEVELOPMENT.md#commit-message-format).

**Q: How do I verify backward compatibility?**  
A: Use `git diff HEAD~1` to review changes and run consistency tests. See [DEVELOPMENT.md](DEVELOPMENT.md#backward-compatibility-checking).

**Q: When should I use FP16 vs FP32?**  
A: Use FP16 for convolutions/blur (2-4x faster), FP32 for RANSAC/coordinates (precision). See [PERFORMANCE.md](PERFORMANCE.md#mixed-precision-training).

## Performance Targets

Reference performance on RTX 3090:

| Operation | Resolution | FP32 | FP16 | Speedup |
|-----------|-----------|------|------|---------|
| Preprocessing | 512×512 | 15ms | 5ms | 3.0x |
| Segmentation | 512×512 | 25ms | 10ms | 2.5x |
| Sliding Window | 2048×2048 | 250ms | 100ms | 2.5x |

## Contributing

When adding new documentation:

1. Follow existing structure and format
2. Include practical examples
3. Add to this README's table of contents
4. Keep language clear and concise
5. Test all code examples

## Additional Resources

- **Project README**: [../README.md](../README.md) - Library overview and usage
- **Examples**: [../examples/](../examples/) - Working code examples
- **Source Code**: [../src/vascx_simplify/](../src/vascx_simplify/) - Implementation

## Feedback

Found an issue or have suggestions? Please open an issue or submit a pull request!

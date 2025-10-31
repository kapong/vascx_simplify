# Acknowledgments

## ⚠️ IMPORTANT: This is NOT Original Work

**This project is a simplified rewrite of the original [rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models) repository by [Eyened](https://github.com/Eyened).**

All core algorithms, preprocessing logic, model architectures, and inference implementations are derived from the original work. This rewrite only reorganizes the code into a more accessible package structure.

### Original Repository Features:
- Fundus image contrast enhancement algorithms
- GPU-accelerated preprocessing with mixed precision
- Ensemble model inference for segmentation, classification, and regression tasks
- Sliding window inference for large medical images

### This Rewrite:
- Reorganized code as a pip-installable package
- Simplified API and improved documentation
- Added packaging infrastructure (pyproject.toml, setup.py)
- Removed testing infrastructure for simplicity

**All credit for the algorithms and models goes to the original authors of [rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models).**

---

## AI-Assisted Development

**This rewrite was developed with EXTENSIVE assistance from AI tools:**

- **GitHub Copilot**: Code completion, refactoring suggestions
- **ChatGPT / Claude**: Architecture planning, documentation, packaging setup

AI assistance was used for:
- Package structure and build configuration (pyproject.toml, setup.py, MANIFEST.in)
- Code reorganization from original repository
- Documentation writing (README.md, docstrings, comments)
- Docker containerization and docker-compose setup
- Build scripts (Makefile, build.ps1)
- CI/CD configuration
- Code formatting and linting setup

**The core algorithms, preprocessing logic, and model inference code are from the original work by Eyened.**

## Dependencies

This project builds upon excellent open-source libraries:
- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **[Kornia](https://kornia.github.io/)**: Differentiable computer vision library
- **[scikit-learn](https://scikit-learn.org/)**: Machine learning tools (RANSAC, preprocessing)
- **[SciPy](https://scipy.org/)**: Scientific computing (image processing)
- **[NumPy](https://numpy.org/)**: Numerical computing
- **[HuggingFace Hub](https://huggingface.co/)**: Model hosting and distribution

## Inspiration

The sliding window inference implementation is inspired by [MONAI](https://monai.io/), a PyTorch-based framework for deep learning in healthcare imaging.

## Contributors

- **Original Author (Eyened)**: Core algorithms and model architectures
- **kapong**: Packaging, refactoring, and documentation
- **AI Tools**: Development assistance (see above)

## License

This project maintains compatibility with the original work's licensing. Please refer to the LICENSE file for details.

---

Thank you to all the developers and researchers whose work made this project possible!

# Acknowledgments

## Original Work

This project is a **simplified rewrite** of the original [rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models) repository by [Eyened](https://github.com/Eyened).

The original repository provided the core algorithms and model architectures for:
- Fundus image contrast enhancement
- GPU-accelerated preprocessing with mixed precision
- Ensemble model inference for segmentation, classification, and regression tasks
- Sliding window inference for large medical images

This rewrite focuses on:
- Making the code more accessible as a pip-installable package
- Improving code organization and documentation
- Adding comprehensive testing infrastructure
- Providing Docker-based testing environments
- Simplifying the API for easier integration

## AI-Assisted Development

This project was developed with assistance from AI tools, including:
- **GitHub Copilot/Claude**: Code completion and suggestions
- **ChatGPT/Claude**: Documentation, testing strategy, and package structure

AI assistance was primarily used for:
- Package structure and build configuration (pyproject.toml, setup.py)
- Test suite development and pytest configuration
- Docker containerization and docker-compose setup
- Documentation writing (README, guides, examples)
- CI/CD pipeline configuration (GitHub Actions)
- Code formatting and linting setup

The core algorithms, preprocessing logic, and model inference code are based on the original work referenced above.

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

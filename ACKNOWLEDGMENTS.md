# Acknowledgments

## Original Work

This is a simplified rewrite of [rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models) by [Eyened](https://github.com/Eyened). All core algorithms, preprocessing logic, model architectures, and inference implementations are derived from the original work. We are grateful to the Eyened team for making their research and code publicly available.

## Key Improvements in This Rewrite

This rewrite focuses on:
- **Performance Optimization**: GPU-accelerated preprocessing with mixed precision support (float16/float32)
- **Modular Architecture**: Clean separation of concerns with organized submodules (`inference/`, `preprocess/`, `utils/`)
- **Simplified API**: Intuitive interface with minimal boilerplate code
- **Modern Practices**: Type hints, comprehensive documentation, and adherence to Python best practices
- **Minimal Dependencies**: Reduced dependency footprint for easier installation and maintenance
- **Production Ready**: Batch processing, memory-efficient sliding window inference, and robust error handling

## AI-Assisted Development

This rewrite was developed with extensive assistance from AI tools (GitHub Copilot, ChatGPT, Claude) for:
- Package structure and build configuration
- Code reorganization and refactoring
- Performance optimization strategies
- Documentation and comprehensive examples
- Best practices implementation
- Testing and validation approaches

The AI tools helped accelerate development while maintaining code quality and consistency.

## Core Dependencies

This project builds upon excellent open-source libraries:

### Deep Learning & Computer Vision
- **[PyTorch](https://pytorch.org/)**: Core deep learning framework providing GPU acceleration
- **[Kornia](https://kornia.github.io/)**: Differentiable computer vision library for GPU-accelerated image transformations
- **[Pillow](https://python-pillow.org/)**: Python Imaging Library for image I/O

### Scientific Computing
- **[NumPy](https://numpy.org/)**: Fundamental package for numerical computing
- **[SciPy](https://scipy.org/)**: Advanced scientific computing and optimization algorithms
- **[scikit-learn](https://scikit-learn.org/)**: Machine learning utilities and preprocessing tools

### Infrastructure
- **[HuggingFace Hub](https://huggingface.co/)**: Model hosting, distribution, and version management

## Inspiration & Design Influences

- **[MONAI](https://monai.io/)**: The sliding window inference implementation with Gaussian importance maps is inspired by MONAI's medical imaging framework
- **[timm](https://github.com/huggingface/pytorch-image-models)**: Model architecture patterns and ensemble inference techniques
- **[torchvision](https://pytorch.org/vision/)**: Standard practices for PyTorch-based computer vision libraries

## Community & Support

Special thanks to:
- The PyTorch community for excellent documentation and support
- The medical imaging community for advancing open-source research
- Early users and contributors who provided valuable feedback

## Author

**Phongphan Phienphanich** (Kapong)  
Email: garpong@gmail.com  
GitHub: [@kapong](https://github.com/kapong)

## Citation

If you use this library in your research, please cite the original work:

```bibtex
@software{vascx_simplify,
  author = {Phienphanich, Phongphan},
  title = {VASCX Simplify: GPU-Accelerated Vessel Analysis and Fundus Image Processing Toolkit},
  year = {2024},
  url = {https://github.com/kapong/vascx_simplify}
}
```

And acknowledge the original implementation:

```bibtex
@software{rtnls_vascx_models,
  author = {Eyened},
  title = {rtnls_vascx_models},
  url = {https://github.com/Eyened/rtnls_vascx_models}
}
```

## License

GNU Affero General Public License v3.0 (AGPL-3.0) - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests on [GitHub](https://github.com/kapong/vascx_simplify).

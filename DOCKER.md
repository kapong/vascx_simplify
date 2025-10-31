# Docker Configuration

## Building and Running Tests

### CPU Testing
```bash
# Build and run tests
docker-compose up test-cpu

# Rebuild from scratch
docker-compose build --no-cache test-cpu
docker-compose up test-cpu
```

### GPU Testing (Requires nvidia-docker)
```bash
# Build and run tests with GPU
docker-compose up test-gpu

# Check GPU availability
docker-compose run test-gpu nvidia-smi
```

### Development Environment
```bash
# Start interactive shell
docker-compose run dev

# Inside container:
pytest -v
python -c "import simple_vascx; print(simple_vascx.__version__)"
```

### Build Package
```bash
# Build wheel and test installation
docker-compose up build
```

## Manual Docker Commands

### CPU Testing
```bash
# Build image
docker build -t simple-vascx:test -f Dockerfile .

# Run tests
docker run --rm simple-vascx:test

# Interactive mode
docker run --rm -it simple-vascx:test /bin/bash
```

### GPU Testing
```bash
# Build CUDA image
docker build -t simple-vascx:cuda -f Dockerfile.cuda .

# Run with GPU
docker run --rm --gpus all simple-vascx:cuda

# Interactive mode
docker run --rm --gpus all -it simple-vascx:cuda /bin/bash
```

## Notes

- The CPU Dockerfile uses Python 3.11-slim for minimal size
- The CUDA Dockerfile uses NVIDIA's CUDA 11.8 base image with cuDNN 8
- Both images install the package in editable mode with test dependencies
- Volumes are used to persist pip cache across runs
- Test coverage reports are generated in the `htmlcov/` directory

# PowerShell script for building and testing simple-vascx on Windows
# Usage: .\build.ps1 [command]

param(
    [Parameter(Position=0)]
    [ValidateSet('install', 'install-dev', 'test', 'test-cov', 'lint', 'format', 'clean', 'build', 'docker-test', 'help')]
    [string]$Command = 'help'
)

function Show-Help {
    Write-Host "VASCX Simplify Build Script" -ForegroundColor Cyan
    Write-Host "===========================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\build.ps1 [command]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Green
    Write-Host "  install       - Install the package"
    Write-Host "  install-dev   - Install with development dependencies"
    Write-Host "  test          - Run tests"
    Write-Host "  test-cov      - Run tests with coverage"
    Write-Host "  lint          - Run linting checks"
    Write-Host "  format        - Format code with black and isort"
    Write-Host "  clean         - Clean build artifacts"
    Write-Host "  build         - Build package"
    Write-Host "  docker-test   - Run tests in Docker"
    Write-Host "  help          - Show this help message"
    Write-Host ""
}

function Install-Package {
    Write-Host "Installing vascx-simplify..." -ForegroundColor Green
    pip install -e .
}

function Install-Dev {
    Write-Host "Installing vascx-simplify with dev dependencies..." -ForegroundColor Green
    pip install -e ".[dev,test]"
}

function Run-Tests {
    Write-Host "Running tests..." -ForegroundColor Green
    pytest -v
}

function Run-TestsCoverage {
    Write-Host "Running tests with coverage..." -ForegroundColor Green
    pytest -v --cov=simple_vascx --cov-report=term-missing --cov-report=html
    Write-Host ""
    Write-Host "Coverage report generated in htmlcov/index.html" -ForegroundColor Cyan
}

function Run-Lint {
    Write-Host "Running lint checks..." -ForegroundColor Green
    flake8 src/ tests/
    mypy src/
}

function Run-Format {
    Write-Host "Formatting code..." -ForegroundColor Green
    black src/ tests/
    isort src/ tests/
}

function Clean-Artifacts {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Green
    
    # Remove build directories
    $dirs = @("build", "dist", "htmlcov", ".pytest_cache", "*.egg-info")
    foreach ($dir in $dirs) {
        Get-ChildItem -Path . -Directory -Recurse -Filter $dir -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    }
    
    # Remove Python cache
    Get-ChildItem -Path . -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Get-ChildItem -Path . -File -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue | Remove-Item -Force
    
    # Remove coverage files
    if (Test-Path ".coverage") { Remove-Item ".coverage" -Force }
    
    Write-Host "Clean complete!" -ForegroundColor Green
}

function Build-Package {
    Write-Host "Building package..." -ForegroundColor Green
    python -m build
    Write-Host ""
    Write-Host "Build complete! Files in dist/" -ForegroundColor Cyan
}

function Run-DockerTest {
    Write-Host "Running tests in Docker..." -ForegroundColor Green
    docker-compose up test-cpu
}

# Main script logic
switch ($Command) {
    'install' { Install-Package }
    'install-dev' { Install-Dev }
    'test' { Run-Tests }
    'test-cov' { Run-TestsCoverage }
    'lint' { Run-Lint }
    'format' { Run-Format }
    'clean' { Clean-Artifacts }
    'build' { Build-Package }
    'docker-test' { Run-DockerTest }
    'help' { Show-Help }
    default { Show-Help }
}

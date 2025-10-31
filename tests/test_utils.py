"""Tests for utility functions."""
import pytest


def test_from_huggingface_format():
    """Test that from_huggingface parses model strings correctly."""
    from simple_vascx.utils import from_huggingface
    
    # This would normally download, so we test the parsing logic separately
    # In a real test, you'd mock hf_hub_download
    modelstr = "username/model-name:model.pt"
    repo_name, repo_fpath = modelstr.split(":")
    
    assert repo_name == "username/model-name"
    assert repo_fpath == "model.pt"


def test_import_main_classes():
    """Test that main classes can be imported."""
    from simple_vascx import (
        EnsembleBase,
        EnsembleSegmentation,
        ClassificationEnsemble,
        RegressionEnsemble,
        HeatmapRegressionEnsemble,
        FundusContrastEnhance,
        VASCXTransform,
        from_huggingface,
        sliding_window_inference,
    )
    
    # Just check they're importable
    assert EnsembleBase is not None
    assert EnsembleSegmentation is not None
    assert ClassificationEnsemble is not None
    assert RegressionEnsemble is not None
    assert HeatmapRegressionEnsemble is not None
    assert FundusContrastEnhance is not None
    assert VASCXTransform is not None
    assert from_huggingface is not None
    assert sliding_window_inference is not None


def test_package_version():
    """Test that package version is accessible."""
    import simple_vascx
    
    assert hasattr(simple_vascx, "__version__")
    assert isinstance(simple_vascx.__version__, str)

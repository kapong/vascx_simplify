"""VASCX Simplify: A PyTorch library for vessel and fundus image analysis.

This is a simplified rewrite of the original rtnls_vascx_models project.
Original work: https://github.com/Eyened/rtnls_vascx_models
"""

__version__ = "0.1.2"

from .inference import (
    sliding_window_inference,
    EnsembleBase,
    EnsembleSegmentation,
    ClassificationEnsemble,
    RegressionEnsemble,
    HeatmapRegressionEnsemble,
)
from .preprocess import FundusContrastEnhance, VASCXTransform
from .utils import from_huggingface

__all__ = [
    "sliding_window_inference",
    "EnsembleBase",
    "EnsembleSegmentation",
    "ClassificationEnsemble",
    "RegressionEnsemble",
    "HeatmapRegressionEnsemble",
    "FundusContrastEnhance",
    "VASCXTransform",
    "from_huggingface",
]

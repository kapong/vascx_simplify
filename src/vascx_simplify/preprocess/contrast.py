from typing import Any, Dict, Optional, Tuple, Union

import kornia.filters as KF
import kornia.geometry as K_geom
import kornia.geometry.conversions as Kconv
import kornia.geometry.transform as K
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import sobel
from sklearn.linear_model import RANSACRegressor

# Import utility functions from vascx_simplify.utils
from ..utils.geometry import create_coordinate_grid
from ..utils.transforms import (
    create_affine_matrix,
    invert_affine_matrix,
    transform_point,
)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleFundusEnhance:
    """Simple fundus contrast enhancement with padding-based pseudo bounds.
    
    Drop-in replacement for FundusContrastEnhance with simpler logic:
    - No circle/line detection (faster)
    - Uses padding + resize for bounds (pseudo bounds)
    - Automatic black region masking
    - Compatible bounds/unbounds API
    
    Performance: ~2-3x faster than FundusContrastEnhance
    """
    
    # Enhancement constants (match FundusContrastEnhance)
    DEFAULT_SIGMA_FRACTION = 0.05
    DEFAULT_CONTRAST_FACTOR = 4
    DEFAULT_BLACK_THRESHOLD = 10
    
    def __init__(
        self,
        square_size: Optional[int] = 1024,
        sigma_fraction: Optional[float] = None,
        contrast_factor: Optional[float] = None,
        black_threshold: Optional[int] = None,
        use_fp16: bool = True,
    ):
        """
        Args:
            square_size: Output size (None to keep original)
            sigma_fraction: Blur strength as fraction of radius (default 0.05)
            contrast_factor: Enhancement strength (default 4.0)
            black_threshold: Pixels below this are considered black (default 10)
            use_fp16: Use float16 for compute-intensive ops (default True, only on CUDA)
        """
        self.square_size = square_size
        self.sigma_fraction = (
            sigma_fraction if sigma_fraction is not None else self.DEFAULT_SIGMA_FRACTION
        )
        self.contrast_factor = (
            contrast_factor if contrast_factor is not None else self.DEFAULT_CONTRAST_FACTOR
        )
        self.black_threshold = (
            black_threshold if black_threshold is not None else self.DEFAULT_BLACK_THRESHOLD
        )
        self.use_fp16 = use_fp16
        
        # Cache for compute dtype per device
        self._compute_dtype_cache = {}
    
    def _get_compute_dtype(self, device: torch.device) -> torch.dtype:
        """Get compute dtype based on device and user preference."""
        if device not in self._compute_dtype_cache:
            if self.use_fp16 and device.type == "cuda":
                self._compute_dtype_cache[device] = torch.float16
            else:
                self._compute_dtype_cache[device] = torch.float32
        return self._compute_dtype_cache[device]
    
    def _get_pseudo_bounds(self, image: torch.Tensor, padding: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """Create pseudo bounds - only essential info for undo operations."""
        h, w = image.shape[-2:]
        return {
            "hw": (h, w),  # Original image dimensions
            "square_size": self.square_size,  # Target size (for undo operations)
            "padding": padding if padding is not None else (0, 0, 0, 0),  # Padding applied (left, right, top, bottom)
        }
    
    def _enhance_image(
        self, image: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Apply contrast enhancement with automatic masking."""
        compute_dtype = self._get_compute_dtype(device)
        
        # Create mask: detect black/near-black regions
        if image.shape[0] > 1:
            # For RGB: a pixel is non-black if any channel > threshold
            mask = (image.max(dim=0)[0] > self.black_threshold).unsqueeze(0)  # [1, H, W]
        else:
            # For grayscale
            mask = (image[0] > self.black_threshold).unsqueeze(0)  # [1, H, W]
        
        # Auto-scale sigma based on image size
        h, w = image.shape[-2:]
        effective_radius = min(h, w) / 2
        sigma = self.sigma_fraction * effective_radius
        
        # Normalize to [0, 1] and convert to compute dtype
        image_norm = image.unsqueeze(0).to(compute_dtype) / 255.0  # [1, C, H, W]
        
        # Apply mask BEFORE blurring to prevent boundary artifacts
        mask_fp = mask.unsqueeze(0).to(compute_dtype)  # [1, 1, H, W]
        image_norm = image_norm * mask_fp
        
        # Compute Gaussian blur
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)
        
        blurred = KF.gaussian_blur2d(image_norm, (kernel_size, kernel_size), (sigma, sigma))
        
        # Apply unsharp mask (use float32 for final computation)
        blurred_fp32 = blurred.float()
        image_norm_fp32 = image_norm.float()
        
        enhanced = torch.clamp(
            self.contrast_factor * (image_norm_fp32 - blurred_fp32) + 0.5, 0, 1
        )
        
        # Convert to uint8
        enhanced = (enhanced * 255).to(torch.uint8).squeeze(0)  # [C, H, W]
        
        # Apply mask again to ensure clean borders
        enhanced = enhanced * mask
        
        return enhanced
    
    def _pad_and_resize_image(
        self, image: torch.Tensor, target_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """Pad image to square, then resize to target size.
        
        Returns:
            resized: Resized image [C, target_size, target_size]
            padding: (pad_left, pad_right, pad_top, pad_bottom) applied
        """
        compute_dtype = self._get_compute_dtype(device)
        
        # Ensure 3D [C, H, W]
        if image.dim() == 2:
            image = image.unsqueeze(0)
        
        h, w = image.shape[-2:]
        
        # Calculate padding to make square (pad to max dimension)
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        
        # Distribute padding evenly (add extra to right/bottom if odd)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad with zeros (black)
        # F.pad expects (left, right, top, bottom) for 2D padding
        padded = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        
        # Resize to target size
        padded_batch = padded.unsqueeze(0).to(compute_dtype)  # [1, C, H, W]
        resized = F.interpolate(
            padded_batch,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        
        # Convert back to uint8
        resized = resized.squeeze(0).float().clamp(0, 255)  # [C, H, W]
        return resized.to(torch.uint8), (pad_left, pad_right, pad_top, pad_bottom)
    
    def __call__(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            img: torch.Tensor [C, H, W] uint8
        Returns:
            tuple: (rgb, ce_img, bounds) - compatible with FundusContrastEnhance API
        """
        device = img.device
        
        # Ensure uint8
        if img.dtype != torch.uint8:
            img = img.to(torch.uint8)
        
        # Resize if square_size specified
        if self.square_size is not None:
            resized_img, padding = self._pad_and_resize_image(img, self.square_size, device)
            org_bounds = self._get_pseudo_bounds(img, padding)
            ce_img = self._enhance_image(resized_img, device)
            return resized_img, ce_img, org_bounds
        else:
            org_bounds = self._get_pseudo_bounds(img)
            ce_img = self._enhance_image(img, device)
            return img, ce_img, org_bounds
    
    def undo_bounds(
        self,
        bounded_image: torch.Tensor,
        hw: Tuple[int, int],
        square_size: Optional[int] = None,
        padding: Optional[Tuple[int, int, int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Resize back to original size and remove padding (pseudo undo_bounds).
        
        Args:
            bounded_image: Image tensor to resize back
            hw: Original (height, width) tuple
            square_size: The square size used (optional, uses self.square_size if None)
            padding: The padding used (left, right, top, bottom), optional
            **kwargs: Ignored (for API compatibility)
        """
        # Use instance square_size if not provided
        if square_size is None:
            square_size = self.square_size
            
        if square_size is None:
            return bounded_image
        
        device = bounded_image.device
        compute_dtype = self._get_compute_dtype(device)
        h, w = hw
        
        # Use default padding if not provided
        if padding is None:
            padding = (0, 0, 0, 0)
        pad_left, pad_right, pad_top, pad_bottom = padding
        
        # Handle different input shapes
        original_shape = bounded_image.shape
        if bounded_image.dim() == 2:
            # [H, W] -> [1, 1, H, W]
            bounded_batch = bounded_image.unsqueeze(0).unsqueeze(0).to(compute_dtype)
        elif bounded_image.dim() == 3:
            # [C, H, W] -> [1, C, H, W]
            bounded_batch = bounded_image.unsqueeze(0).to(compute_dtype)
        elif bounded_image.dim() == 4:
            # [B, C, H, W] already has batch
            bounded_batch = bounded_image.to(compute_dtype)
        else:
            raise ValueError(f"Unexpected tensor shape: {bounded_image.shape}")
        
        # First, resize back to padded dimensions
        max_dim = max(h, w)
        resized = F.interpolate(
            bounded_batch,
            size=(max_dim, max_dim),
            mode="bilinear",
            align_corners=False,
        )
        
        # Remove padding by cropping
        # Crop from [pad_top : max_dim - pad_bottom, pad_left : max_dim - pad_right]
        if len(resized.shape) == 4:
            undone = resized[:, :, pad_top:max_dim - pad_bottom, pad_left:max_dim - pad_right]
        else:
            undone = resized[:, pad_top:max_dim - pad_bottom, pad_left:max_dim - pad_right]
        
        # Restore original shape
        if len(original_shape) == 2:
            undone = undone.squeeze(0).squeeze(0)  # Back to [H, W]
        elif len(original_shape) == 3:
            undone = undone.squeeze(0)  # Back to [C, H, W]
        # else keep as [B, C, H, W]
        
        # Convert back to original dtype
        undone = undone.float()
        if bounded_image.dtype == torch.uint8:
            undone = undone.clamp(0, 255)
        return undone.to(bounded_image.dtype)
    
    def undo_bounds_points(
        self,
        points_tensor: torch.Tensor,
        hw: Tuple[int, int],
        square_size: Optional[int] = None,
        padding: Optional[Tuple[int, int, int, int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Scale points back to original coordinates and account for padding.
        
        Points are in [x, y] format (column 0 = x, column 1 = y).
        
        Args:
            points_tensor: Points in bounded space [N, 2] or [B, K, 2]
            hw: Original (height, width) tuple
            square_size: The square size used (optional, uses self.square_size if None)
            padding: The padding used (left, right, top, bottom), optional
            **kwargs: Ignored (for API compatibility)
        """
        # Use instance square_size if not provided
        if square_size is None:
            square_size = self.square_size
            
        if square_size is None:
            return points_tensor
        
        h, w = hw
        
        # Use default padding if not provided
        if padding is None:
            padding = (0, 0, 0, 0)
        pad_left, pad_right, pad_top, pad_bottom = padding
        
        # Calculate padded dimensions
        max_dim = max(h, w)
        
        # Handle different shapes: [B, K, 2] or [N, 2]
        original_shape = points_tensor.shape
        if points_tensor.dim() == 3:
            # [B, K, 2] -> flatten to [B*K, 2]
            batch_size, num_keypoints = points_tensor.shape[:2]
            points_flat = points_tensor.reshape(-1, 2)
        else:
            # Already [N, 2]
            points_flat = points_tensor
        
        # Step 1: Scale from square_size to padded (max_dim x max_dim)
        scale = max_dim / square_size
        points_scaled = points_flat.clone()
        points_scaled[:, 0] = points_flat[:, 0] * scale  # x coordinate
        points_scaled[:, 1] = points_flat[:, 1] * scale  # y coordinate
        
        # Step 2: Remove padding offset
        points_scaled[:, 0] = points_scaled[:, 0] - pad_left  # x coordinate
        points_scaled[:, 1] = points_scaled[:, 1] - pad_top   # y coordinate
        
        # Restore original shape
        if len(original_shape) == 3:
            points_scaled = points_scaled.reshape(original_shape)
        
        return points_scaled


def simple_fundus_enhance(
    image: Union[torch.Tensor, np.ndarray],
    sigma: Optional[float] = None,
    contrast_factor: float = 4.0,
    device: Optional[str] = None,
    use_fp16: bool = True,
    black_threshold: int = 10,
) -> torch.Tensor:
    """Simple fundus contrast enhancement function (functional API).

    Applies unsharp masking: contrast_factor * (image - blur(image)) + 0.5
    Automatically detects and masks black/near-black regions (fundus borders).

    Sigma automatically scales with image size to match FundusContrastEnhance behavior:
    - For 1024×1024: sigma ≈ 25.6 (matches FundusContrastEnhance default)
    - For other sizes: sigma = 0.05 * min(height, width) / 2

    Args:
        image: Input image as torch.Tensor [C, H, W] or [H, W] or numpy array
        sigma: Gaussian blur sigma (None = auto-scale based on image size)
        contrast_factor: Contrast enhancement multiplier (default 4.0)
        device: Target device ('cuda' or 'cpu', auto-detected if None)
        use_fp16: Use float16 for blur computation (faster on CUDA)
        black_threshold: Pixels below this value are considered black (default 10)

    Returns:
        Enhanced image as uint8 tensor [C, H, W], with black regions masked
    """
    # Convert numpy to torch if needed
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    # Handle device
    if device is None:
        device = DEFAULT_DEVICE
    image = image.to(device)

    # Ensure 3D tensor [C, H, W]
    original_shape = image.shape
    if image.ndim == 2:
        image = image.unsqueeze(0)  # [1, H, W]
    elif image.ndim == 4:
        image = image.squeeze(0)  # Remove batch dimension if present

    # Create mask: detect black/near-black regions (fundus borders)
    # Use max across channels to preserve any non-black content
    if image.shape[0] > 1:
        # For RGB: a pixel is non-black if any channel > threshold
        mask = (image.max(dim=0)[0] > black_threshold).unsqueeze(0)  # [1, H, W]
    else:
        # For grayscale: pixel is non-black if value > threshold
        mask = (image[0] > black_threshold).unsqueeze(0)  # [1, H, W]

    # Auto-scale sigma based on image size if not provided
    # Matches FundusContrastEnhance: sigma_fraction=0.05, radius=size/2
    # For 1024×1024: sigma = 0.05 * 512 = 25.6
    if sigma is None:
        h, w = image.shape[-2:]
        effective_radius = min(h, w) / 2
        sigma = 0.05 * effective_radius

    # Determine compute dtype
    compute_dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32

    # Normalize to [0, 1] and convert to compute dtype
    image_norm = image.unsqueeze(0).to(compute_dtype) / 255.0  # [1, C, H, W]

    # Apply mask BEFORE blurring to prevent boundary artifacts
    # This stops black border pixels from bleeding into fundus during blur
    mask_fp = mask.unsqueeze(0).to(compute_dtype)  # [1, 1, H, W]
    image_norm = image_norm * mask_fp

    # Compute Gaussian blur (only on masked region)
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)

    blurred = KF.gaussian_blur2d(image_norm, (kernel_size, kernel_size), (sigma, sigma))

    # Apply unsharp mask (use float32 for final computation)
    blurred_fp32 = blurred.float()
    image_norm_fp32 = image_norm.float()

    enhanced = torch.clamp(
        contrast_factor * (image_norm_fp32 - blurred_fp32) + 0.5, 0, 1
    )

    # Convert to uint8
    enhanced = (enhanced * 255).to(torch.uint8).squeeze(0)  # [C, H, W]

    # Apply mask again to ensure clean borders
    enhanced = enhanced * mask

    # Restore original shape if needed
    if len(original_shape) == 2:
        enhanced = enhanced.squeeze(0)  # Back to [H, W]
    elif len(original_shape) == 4:
        enhanced = enhanced.unsqueeze(0)  # Back to [1, C, H, W]

    return enhanced


class FundusContrastEnhance:
    """GPU-accelerated fundus contrast enhancement with mixed precision support.

    Uses float16 for compute-intensive operations (blur, warp) and float32 for
    precision-critical operations (RANSAC, coordinate transforms).

    Performance gains with float16:
    - 2-4x faster on modern GPUs (Tensor Cores)
    - 50% memory usage reduction
    - ~10-15ms → ~5-8ms total pipeline
    """

    # Image processing resolution constants
    RESOLUTION = 256
    CENTER = RESOLUTION // 2

    # Circle detection constants
    MAX_RADIUS_DIVISOR = 1.8  # Maximum radius = RESOLUTION / 1.8
    MIN_RADIUS_DIVISOR = 4  # Minimum radius = RESOLUTION / 4
    MAX_R = RESOLUTION // MAX_RADIUS_DIVISOR  # ~142 pixels
    MIN_R = RESOLUTION // MIN_RADIUS_DIVISOR  # 64 pixels
    INLIER_DIST_THRESHOLD = RESOLUTION // 256  # 1 pixel for 256×256

    # Circle and mask scaling constants
    CIRCLE_SCALE_FACTOR = 0.99  # Scale factor for circle boundary
    BORDER_MARGIN_FRACTION = 0.01  # Margin for rectangular bounds (1%)
    CIRCLE_REFIT_SCALE = 0.95  # Scale for circle refitting

    # Enhancement constants
    DEFAULT_SIGMA_FRACTION = 0.05  # Gaussian blur sigma as fraction of radius
    DEFAULT_CONTRAST_FACTOR = 4  # Contrast enhancement multiplier
    REDUCED_BLUR_RESOLUTION = 256  # Resolution for efficient blur operation

    # Pre-compute masks and costs (numpy for initialization only)
    r = np.arange(RESOLUTION)
    q = RESOLUTION // 8
    f = 0.3
    RECT_MASKS = {
        "bottom": (q < r) & (r < 3 * q),
        "top": (5 * q < r) & (r < 7 * q),
        "right": (r < (1 - f) * q) | (r > (7 + f) * q),
        "left": ((3 + f) * q < r) & (r < (5 - f) * q),
    }
    CORNER_MASK = ~np.logical_or.reduce(list(RECT_MASKS.values()))

    n = RESOLUTION - MIN_R
    COST_DIST = 0.01 * np.subtract.outer(np.arange(n), np.arange(n)) ** 2
    th = np.arange(RESOLUTION) * 2 * np.pi / RESOLUTION
    COST_TH = np.cos(th)
    SIN_TH = np.sin(th)

    def __init__(
        self,
        square_size: Optional[int] = 1024,
        sigma_fraction: Optional[float] = None,
        contrast_factor: Optional[int] = None,
        return_bounds: bool = True,
        use_fp16: bool = True,
    ):
        """
        Args:
            square_size: Output size (None to keep original)
            sigma_fraction: Blur strength (default 0.05)
            contrast_factor: Enhancement strength (default 4)
            use_fp16: Use float16 for compute-intensive ops (default True, only on CUDA)
        """
        self.square_size = square_size
        self.sigma_fraction = (
            sigma_fraction if sigma_fraction is not None else self.DEFAULT_SIGMA_FRACTION
        )
        self.contrast_factor = (
            contrast_factor if contrast_factor is not None else self.DEFAULT_CONTRAST_FACTOR
        )
        self.return_bounds = return_bounds
        self.use_fp16 = use_fp16

        # FP16 only supported on CUDA, fallback to FP32 on CPU
        # Will be determined per-device when processing
        self._compute_dtype_cache = {}

        # Pre-initialize RANSAC for line fitting (reusable)
        self.ransac_line = RANSACRegressor(
            residual_threshold=self.INLIER_DIST_THRESHOLD, random_state=42
        )

        # Cache for grid coordinates (avoids recreating meshgrids)
        self._grid_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_compute_dtype(self, device: torch.device) -> torch.dtype:
        """Get compute dtype based on device and user preference.

        FP16 only works on CUDA. CPU operations fallback to FP32.
        """
        if device not in self._compute_dtype_cache:
            if self.use_fp16 and device.type == "cuda":
                self._compute_dtype_cache[device] = torch.float16
            else:
                self._compute_dtype_cache[device] = torch.float32
        return self._compute_dtype_cache[device]

    def _invert_affine_matrix(self, M: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Convert 2x3 affine matrix to 3x3 and invert (float32 for precision).

        Args:
            M: 2x3 affine transformation matrix
            device: Target device

        Returns:
            3x3 inverted matrix (float32)
        """
        return invert_affine_matrix(M, device)

    def _transform_point_to_original(
        self, point: Tuple[float, float], M_inv: torch.Tensor, device: torch.device
    ) -> np.ndarray:
        """Transform a point from normalized space back to original coordinates.

        Args:
            point: (x, y) coordinates in normalized space
            M_inv: 3x3 inverse transformation matrix (float32)
            device: Target device

        Returns:
            (x, y) coordinates in original space as numpy array
        """
        return transform_point(point, M_inv, device)

    def _get_or_create_grid(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or create cached coordinate grids for given dimensions.

        Args:
            h: Grid height
            w: Grid width
            device: Target device
            dtype: Data type for grid

        Returns:
            Tuple of (x_grid, y_grid) coordinate meshgrids
        """
        cache_key = (h, w, device, dtype)
        if cache_key not in self._grid_cache:
            self._grid_cache[cache_key] = create_coordinate_grid(h, w, device, dtype)
        return self._grid_cache[cache_key]

    def _create_coordinate_grid(
        self, h: int, w: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create coordinate grids for given dimensions.

        Args:
            h: Grid height
            w: Grid width
            device: Target device
            dtype: Data type for grid

        Returns:
            Tuple of (x_grid, y_grid) coordinate meshgrids
        """
        return create_coordinate_grid(h, w, device, dtype)

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            img: torch.Tensor [C, H, W] uint8
        Returns:
            tuple: (rgb, ce_img, bounds) all as torch.Tensor
        """
        device = img.device

        # Ensure uint8
        if img.dtype != torch.uint8:
            img = img.to(torch.uint8)

        # Process entirely on GPU with mixed precision
        rgb, ce_img, bounds = self._process(img, device)

        return rgb, ce_img, bounds

    def _process(
        self, image: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Main processing pipeline - all GPU operations with mixed precision."""
        org_bounds = self._get_bounds(image, device)

        if self.square_size is not None:
            image = self._crop_to_square(image, org_bounds, device)
            bounds = self._update_bounds_after_crop(org_bounds)
        else:
            bounds = org_bounds

        ce = self._enhance_contrast(image, bounds, device)

        return image, ce, org_bounds

    def _scale_image_for_detection(
        self, gray: torch.Tensor, h: int, w: int, device: torch.device, compute_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Scale grayscale image to standard resolution for detection.

        Returns:
            gray_scaled: Scaled image tensor
            M: Affine transformation matrix
            scale: Scaling factor applied
        """
        scale = min(self.RESOLUTION / h, self.RESOLUTION / w)
        # Create matrix in compute dtype for warp_affine
        M = self._create_affine_matrix_torch(
            (h, w), self.RESOLUTION, scale, (h // 2, w // 2), device, dtype=compute_dtype
        )

        # GPU warp - use compute dtype for speed
        gray_batch = gray.unsqueeze(0).unsqueeze(0).to(compute_dtype)  # [1, 1, H, W]
        gray_scaled = K.warp_affine(
            gray_batch,
            M.unsqueeze(0),
            (self.RESOLUTION, self.RESOLUTION),
            mode="bilinear",
            padding_mode="zeros",
        )
        gray_scaled = gray_scaled.squeeze(0).squeeze(0).float()  # [H, W] - back to float32

        return gray_scaled, M, scale

    def _fit_lines_if_needed(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        radius: float,
        center: Tuple[float, float],
        circle_fraction: float,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Fit lines if circle fraction is below threshold.

        Returns:
            Dictionary of lines (empty if circle_fraction > 0.85)
        """
        if circle_fraction > 0.85:
            return {}
        return self._fit_lines(xs, ys, radius, center, circle_fraction)

    def _transform_bounds_to_original(
        self,
        center: Tuple[float, float],
        radius: float,
        lines: Dict[str, Tuple[np.ndarray, np.ndarray]],
        M: torch.Tensor,
        scale: float,
        device: torch.device,
    ) -> Tuple[Tuple[float, float], float, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """Transform detected bounds back to original image coordinates.

        Returns:
            center_orig: Center in original coordinates
            radius_orig: Radius in original coordinates
            lines_orig: Lines in original coordinates
        """
        # Transform back to original coordinates using GPU (use float32 for precision)
        M_inv = self._invert_affine_matrix(M, device)
        center_orig = self._transform_point_to_original(center, M_inv, device)

        lines_orig = {}
        for k, (p0, p1) in lines.items():
            p0_orig = self._transform_point_to_original(p0, M_inv, device)
            p1_orig = self._transform_point_to_original(p1, M_inv, device)
            lines_orig[k] = (p0_orig, p1_orig)

        radius_orig = radius / scale

        return center_orig, radius_orig, lines_orig

    def _get_bounds(self, image: torch.Tensor, device: torch.device) -> Dict[str, Any]:
        """Detect fundus boundaries using GPU where possible."""
        compute_dtype = self._get_compute_dtype(device)

        # Extract grayscale (first channel if RGB)
        if image.dim() == 3 and image.shape[0] >= 1:
            gray = image[0]  # [H, W]
        else:
            gray = image

        h, w = gray.shape[-2:]

        # Scale to standard resolution
        gray_scaled, M, scale = self._scale_image_for_detection(gray, h, w, device, compute_dtype)

        # Edge detection (uses scipy sobel - small CPU operation)
        xs, ys = self._detect_edges(gray_scaled, device)

        # Circle fitting (uses sklearn RANSAC on 256 points)
        radius, center, circle_fraction = self._fit_circle(xs, ys)

        # Line fitting if needed
        lines = self._fit_lines_if_needed(xs, ys, radius, center, circle_fraction)

        # Transform back to original coordinates
        center_orig, radius_orig, lines_orig = self._transform_bounds_to_original(
            center, radius, lines, M, scale, device
        )

        return {"center": center_orig, "radius": radius_orig, "lines": lines_orig, "hw": (h, w)}

    def _detect_edges(
        self, image: torch.Tensor, device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect edge points using polar transform.

        Note: Uses scipy sobel on small 256×256 array (~1ms CPU overhead).
        """
        # GPU polar transform using grid_sample (use compute dtype)
        polar = self._linearPolar(image, (self.CENTER, self.CENTER), self.MAX_R, device)
        edge_region = polar[:, self.MIN_R :].float().cpu().numpy()

        # Sobel gradient (scipy on CPU - small array)
        edge_max = edge_region.max()
        if edge_max > 0:
            gradients = sobel(edge_region / edge_max, 1)
        else:
            gradients = np.zeros_like(edge_region)

        # Dynamic programming shortest path
        costs = gradients.copy()
        path = np.zeros_like(costs, dtype=int)
        for y in range(1, self.RESOLUTION):
            total_costs = costs[y - 1] + self.COST_DIST
            costs[y] += total_costs.min(axis=1)
            path[y] = total_costs.argmin(axis=1)

        # Backtrack
        indices = [costs[-1].argmin()]
        for i in range(self.RESOLUTION - 2, -1, -1):
            indices.append(path[i + 1, indices[-1]])
        radii = self.MIN_R + np.array(indices[::-1])

        # Convert to Cartesian
        r = self.MAX_R * radii / self.RESOLUTION
        return self.CENTER + r * self.COST_TH, self.CENTER + r * self.SIN_TH

    def _linearPolar(
        self,
        image: torch.Tensor,
        center: Tuple[float, float],
        max_radius: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert image to polar coordinates using GPU grid_sample."""
        compute_dtype = self._get_compute_dtype(device)
        cy, cx = center

        # Create polar grid on GPU (use compute dtype)
        theta = torch.linspace(0, 2 * np.pi, self.RESOLUTION, device=device, dtype=compute_dtype)
        radius = torch.linspace(0, max_radius, self.RESOLUTION, device=device, dtype=compute_dtype)

        theta_grid, radius_grid = torch.meshgrid(theta, radius, indexing="ij")

        # Convert to Cartesian
        x = cx + radius_grid * torch.cos(theta_grid)
        y = cy + radius_grid * torch.sin(theta_grid)

        # Normalize to [-1, 1] for grid_sample
        x_norm = 2 * x / (self.RESOLUTION - 1) - 1
        y_norm = 2 * y / (self.RESOLUTION - 1) - 1

        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        # Sample using GPU (input already in compute dtype)
        image_batch = image.unsqueeze(0).unsqueeze(0).to(compute_dtype)  # [1, 1, H, W]
        polar = F.grid_sample(
            image_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        return polar.squeeze(0).squeeze(0)  # [H, W]

    def _fit_circle(self, xs: np.ndarray, ys: np.ndarray) -> Tuple[float, np.ndarray, float]:
        """RANSAC circle fitting.

        Note: Uses sklearn RANSAC on 256 points (~1ms). Too small to benefit from GPU.
        """
        pts = np.c_[xs, ys]
        n = len(pts)
        best_inliers = None
        best_count = 0
        rng = np.random.default_rng(seed=42)

        for _ in range(1000):
            # Sample and fit
            attempts = 0
            while attempts < 100:
                indices = rng.choice(n, 3, replace=False)
                B = np.c_[pts[indices], np.ones(3)]
                d = (pts[indices] ** 2).sum(axis=1)
                y = np.linalg.lstsq(B, d, rcond=None)[0]
                center = 0.5 * y[:2]
                radius = np.sqrt(y[2] + (center**2).sum())
                if self.MIN_R < radius < self.MAX_R:
                    break
                attempts += 1

            # Count inliers
            distances = np.abs(np.sqrt(((pts - center) ** 2).sum(axis=1)) - radius)
            inliers = distances < self.INLIER_DIST_THRESHOLD
            count = inliers.sum()

            if count > best_count:
                best_inliers = inliers
                best_count = count
                if count > n / 2:
                    break

        circle_fraction = best_count / n if best_count > 0.2 * n else 0

        if circle_fraction > 0:
            # Refit with inliers
            B = np.c_[pts[best_inliers], np.ones(best_count)]
            d = (pts[best_inliers] ** 2).sum(axis=1)
            y = np.linalg.lstsq(B, d, rcond=None)[0]
            center = 0.5 * y[:2]
            radius = np.sqrt(y[2] + (center**2).sum())

        return radius, center, circle_fraction

    def _fit_line_ransac(
        self, x_data: np.ndarray, y_data: np.ndarray, mask: np.ndarray, is_horizontal: bool
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Fit a single line using RANSAC.

        Args:
            x_data: X coordinates of all points
            y_data: Y coordinates of all points
            mask: Boolean mask selecting points for this line
            is_horizontal: If True, fit horizontal line (predict y from x)
                          If False, fit vertical line (predict x from y)

        Returns:
            Tuple of (x_vals, y_vals) for the fitted line, or None if fit fails
        """
        if is_horizontal:
            # Horizontal line: y = a*x + b
            self.ransac_line.fit(x_data[mask].reshape(-1, 1), y_data[mask])
        else:
            # Vertical line: x = a*y + b
            self.ransac_line.fit(y_data[mask].reshape(-1, 1), x_data[mask])

        # Check if fit is good (majority inliers)
        if self.ransac_line.inlier_mask_.mean() <= 0.5:
            return None

        a = self.ransac_line.estimator_.coef_[0]
        b = self.ransac_line.estimator_.intercept_

        if is_horizontal:
            # For horizontal lines: return (x_vals, y_vals)
            x_vals = np.array([0, self.RESOLUTION])
            y_vals = a * x_vals + b
            return (x_vals, y_vals)
        else:
            # For vertical lines: return (y_vals, x_vals) reversed
            x_vals = np.array([0, self.RESOLUTION])
            y_vals = a * x_vals + b
            return (y_vals[::-1], x_vals[::-1])

    def _fit_lines(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        radius: float,
        center: np.ndarray,
        circle_fraction: float,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Fit lines to edge regions using sklearn RANSAC."""
        lines = {}

        # Fit vertical lines (left, right) - predict x from y
        for loc in ["left", "right"]:
            mask = self.RECT_MASKS[loc]
            result = self._fit_line_ransac(xs, ys, mask, is_horizontal=False)
            if result is not None:
                lines[loc] = result

        # Fit horizontal lines (top, bottom) - predict y from x
        for loc in ["top", "bottom"]:
            mask = self.RECT_MASKS[loc]
            result = self._fit_line_ransac(xs, ys, mask, is_horizontal=True)
            if result is not None:
                lines[loc] = result

        return lines

    def _crop_to_square(
        self, image: torch.Tensor, bounds: Dict[str, Any], device: torch.device
    ) -> torch.Tensor:
        """Crop and resize to square using kornia (GPU with mixed precision)."""
        compute_dtype = self._get_compute_dtype(device)
        cy, cx = bounds["center"]
        scale = self.square_size / (2 * bounds["radius"])
        # Use compute dtype for matrix (must match image for warp_affine)
        M = self._create_affine_matrix_torch(
            bounds["hw"], self.square_size, scale, (cy, cx), device, dtype=compute_dtype
        )

        # Warp image on GPU (use compute dtype for speed)
        if image.dim() == 2:
            image = image.unsqueeze(0)  # [1, H, W]

        image_batch = image.unsqueeze(0).to(compute_dtype)  # [1, C, H, W]
        warped = K.warp_affine(
            image_batch,
            M.unsqueeze(0),
            (self.square_size, self.square_size),
            mode="bilinear",
            padding_mode="zeros",
        )

        # Convert back to uint8
        warped = warped.squeeze(0).float().clamp(0, 255)  # [C, H, W]
        return warped.to(torch.uint8)

    def _update_bounds_after_crop(self, bounds: Dict[str, Any]) -> Dict[str, Any]:
        """Update bounds for cropped image."""
        return {
            "center": (self.square_size / 2, self.square_size / 2),
            "radius": bounds["radius"] * (self.square_size / (2 * bounds["radius"])),
            "lines": {},
            "hw": (self.square_size, self.square_size),
        }

    def _compute_blur_at_reduced_resolution(
        self,
        mirrored: torch.Tensor,
        bounds: Dict[str, Any],
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute Gaussian blur at reduced resolution for efficiency.

        Warps image to 256×256, applies blur, then warps back to original size.
        Uses compute dtype (typically float16) for speed.

        Returns:
            Blurred image at original size (in compute dtype)
        """
        # Efficient blur at reduced resolution
        ce_res = self.REDUCED_BLUR_RESOLUTION
        cy, cx = bounds["center"]
        scale_ce = ce_res / (2 * bounds["radius"])
        # Use compute dtype for matrix (must match image for warp_affine)
        M_ce = self._create_affine_matrix_torch(
            bounds["hw"], ce_res, scale_ce, (cy, cx), device, dtype=compute_dtype
        )

        # Warp to smaller size (use compute dtype for speed)
        if mirrored.dim() == 2:
            mirrored = mirrored.unsqueeze(0)

        mirrored_batch = mirrored.unsqueeze(0).to(compute_dtype) / 255.0  # [1, C, H, W]
        mirrored_small = K.warp_affine(
            mirrored_batch,
            M_ce.unsqueeze(0),
            (ce_res, ce_res),
            mode="bilinear",
            padding_mode="zeros",
        )

        # Gaussian blur using kornia (use compute dtype - major speedup with fp16)
        sigma = self.sigma_fraction * (ce_res / 2)
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

        blurred_small = KF.gaussian_blur2d(
            mirrored_small, (kernel_size, kernel_size), (sigma, sigma)
        )

        # Upsample blur back to original size (keep in compute dtype)
        h, w = bounds["hw"]
        # Use helper to invert matrix (returns 3x3 float32)
        M_ce_inv_3x3 = self._invert_affine_matrix(M_ce, device)
        M_ce_inv = M_ce_inv_3x3[:2].to(
            compute_dtype
        )  # Extract 2x3 and convert back for warp_affine

        blurred = K.warp_affine(
            blurred_small, M_ce_inv.unsqueeze(0), (w, h), mode="bilinear", padding_mode="zeros"
        )

        return blurred

    def _apply_unsharp_mask(
        self,
        image: torch.Tensor,
        blurred: torch.Tensor,
        bounds: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        """Apply unsharp mask for contrast enhancement.

        Uses float32 for final computation to preserve quality.

        Returns:
            Enhanced image as uint8 tensor
        """
        # Unsharp mask (use float32 for final computation to preserve quality)
        image_norm = image.unsqueeze(0).float() / 255.0  # [1, C, H, W]
        blurred_fp32 = blurred.float()

        enhanced = torch.clamp(self.contrast_factor * (image_norm - blurred_fp32) + 0.5, 0, 1)
        enhanced = (enhanced * 255).to(torch.uint8).squeeze(0)  # [C, H, W]

        # Apply mask (GPU)
        mask = self._make_mask(bounds, device)
        enhanced = enhanced * mask.unsqueeze(0)

        return enhanced

    def _enhance_contrast(
        self, image: torch.Tensor, bounds: Dict[str, Any], device: torch.device
    ) -> torch.Tensor:
        """Apply contrast enhancement using kornia (GPU with mixed precision).

        Major optimization: Blur at 256×256 then upsample instead of full resolution.
        Uses float16 for blur (2-4x faster) and float32 for final unsharp mask.
        """
        compute_dtype = self._get_compute_dtype(device)
        mirrored = self._mirror_image(image, bounds, device)

        # Compute blur at reduced resolution
        blurred = self._compute_blur_at_reduced_resolution(mirrored, bounds, device, compute_dtype)

        # Apply unsharp mask
        enhanced = self._apply_unsharp_mask(image, blurred, bounds, device)

        return enhanced

    def _mirror_edge(
        self,
        mirrored: torch.Tensor,
        bound_val: int,
        size: int,
        content_min: int,
        content_max: int,
        is_top_or_left: bool,
        is_2d: bool,
        dim_2d: int,
        dim_3d: int,
    ) -> None:
        """Mirror a single edge (top/bottom/left/right) in-place.

        Args:
            mirrored: Tensor to modify in-place
            bound_val: The boundary value (min_y, max_y, min_x, or max_x)
            size: Total size along this dimension (h or w)
            content_min: Content region minimum
            content_max: Content region maximum
            is_top_or_left: True for top/left edges, False for bottom/right
            is_2d: Whether tensor is 2D
            dim_2d: Dimension to flip for 2D tensors
            dim_3d: Dimension to flip for 3D tensors
        """
        if is_top_or_left:
            # Top or left edge
            if bound_val <= 0:
                return
            flip_size = min(bound_val, content_max - content_min)
            if is_2d:
                if dim_2d == 0:  # top edge
                    mirrored[:bound_val] = torch.flip(
                        mirrored[bound_val : bound_val + flip_size], dims=[dim_2d]
                    )[:bound_val]
                else:  # left edge
                    mirrored[:, :bound_val] = torch.flip(
                        mirrored[:, bound_val : bound_val + flip_size], dims=[dim_2d]
                    )[:, :bound_val]
            else:
                if dim_3d == 1:  # top edge
                    mirrored[:, :bound_val] = torch.flip(
                        mirrored[:, bound_val : bound_val + flip_size], dims=[dim_3d]
                    )[:, :bound_val]
                else:  # left edge
                    mirrored[:, :, :bound_val] = torch.flip(
                        mirrored[:, :, bound_val : bound_val + flip_size], dims=[dim_3d]
                    )[:, :, :bound_val]
        else:
            # Bottom or right edge
            if bound_val >= size:
                return
            flip_size = min(size - bound_val, content_max - content_min)
            if is_2d:
                if dim_2d == 0:  # bottom edge
                    mirrored[bound_val:] = torch.flip(
                        mirrored[bound_val - flip_size : bound_val], dims=[dim_2d]
                    )[: size - bound_val]
                else:  # right edge
                    mirrored[:, bound_val:] = torch.flip(
                        mirrored[:, bound_val - flip_size : bound_val], dims=[dim_2d]
                    )[:, : size - bound_val]
            else:
                if dim_3d == 1:  # bottom edge
                    mirrored[:, bound_val:] = torch.flip(
                        mirrored[:, bound_val - flip_size : bound_val], dims=[dim_3d]
                    )[:, : size - bound_val]
                else:  # right edge
                    mirrored[:, :, bound_val:] = torch.flip(
                        mirrored[:, :, bound_val - flip_size : bound_val], dims=[dim_3d]
                    )[:, :, : size - bound_val]

    def _mirror_edges(
        self, mirrored: torch.Tensor, rect: Dict[str, int], h: int, w: int, d: int, is_2d: bool
    ) -> None:
        """Mirror all rectangular edges in-place.

        Args:
            mirrored: Tensor to modify in-place
            rect: Rectangle bounds dictionary
            h: Image height
            w: Image width
            d: Border margin in pixels
            is_2d: Whether tensor is 2D
        """
        min_y = max(rect["min_y"] + d, 0)
        max_y = min(rect["max_y"] - d, h)
        min_x = max(rect["min_x"] + d, 0)
        max_x = min(rect["max_x"] - d, w)

        # Mirror edges using torch.flip (GPU) - works directly on uint8
        # Top edge
        self._mirror_edge(mirrored, min_y, h, min_y, max_y, True, is_2d, 0, 1)
        # Bottom edge
        self._mirror_edge(mirrored, max_y, h, min_y, max_y, False, is_2d, 0, 1)
        # Left edge
        self._mirror_edge(mirrored, min_x, w, min_x, max_x, True, is_2d, 1, 2)
        # Right edge
        self._mirror_edge(mirrored, max_x, w, min_x, max_x, False, is_2d, 1, 2)

    def _mirror_circle(
        self,
        mirrored: torch.Tensor,
        cx: float,
        cy: float,
        radius: float,
        h: int,
        w: int,
        device: torch.device,
        compute_dtype: torch.dtype,
        is_2d: bool,
    ) -> None:
        """Mirror pixels outside circular boundary in-place.

        Args:
            mirrored: Tensor to modify in-place
            cx: Circle center x-coordinate
            cy: Circle center y-coordinate
            radius: Circle radius
            h: Image height
            w: Image width
            device: Device for computation
            compute_dtype: Data type for computation
            is_2d: Whether tensor is 2D
        """
        # Mirror circle using cached grid coordinates (GPU, use compute dtype)
        x_grid, y_grid = self._get_or_create_grid(h, w, device, compute_dtype)

        r_sq_norm = ((x_grid - cx) / (self.CIRCLE_SCALE_FACTOR * radius)) ** 2 + (
            (y_grid - cy) / (self.CIRCLE_SCALE_FACTOR * radius)
        ) ** 2
        outside = r_sq_norm > 1

        scale = 1.0 / r_sq_norm[outside]
        x_in = torch.clamp(torch.round(cx + (x_grid[outside] - cx) * scale).long(), 0, w - 1)
        y_in = torch.clamp(torch.round(cy + (y_grid[outside] - cy) * scale).long(), 0, h - 1)

        if is_2d:
            mirrored[outside] = mirrored[y_in, x_in]
        else:
            mirrored[:, outside] = mirrored[:, y_in, x_in]

    def _mirror_image(
        self, image: torch.Tensor, bounds: Dict[str, Any], device: torch.device
    ) -> torch.Tensor:
        """Mirror pixels at boundaries using torch operations (GPU).

        Uses uint8 directly - no need for float conversion here.
        """
        compute_dtype = self._get_compute_dtype(device)
        cx, cy = bounds["center"]
        radius = bounds["radius"]
        h, w = bounds["hw"]

        if image.dim() == 2:
            mirrored = image.clone()
            is_2d = True
        else:
            mirrored = image.clone()
            is_2d = False

        # Get rectangular bounds
        d = int(self.BORDER_MARGIN_FRACTION * radius)
        rect = self._get_rect_bounds(bounds["lines"], bounds["center"], radius, (h, w))

        # Mirror edges
        self._mirror_edges(mirrored, rect, h, w, d, is_2d)

        # Mirror circle
        self._mirror_circle(mirrored, cx, cy, radius, h, w, device, compute_dtype, is_2d)

        return mirrored

    def _make_mask(self, bounds: Dict[str, Any], device: torch.device) -> torch.Tensor:
        """Create binary mask using torch (GPU, use compute dtype for speed)."""
        compute_dtype = self._get_compute_dtype(device)
        cx, cy = bounds["center"]
        radius = bounds["radius"]
        h, w = bounds["hw"]
        d = int(self.BORDER_MARGIN_FRACTION * radius)

        # Get or create cached grid (use compute dtype)
        x_grid, y_grid = self._get_or_create_grid(h, w, device, compute_dtype)

        r_norm = torch.sqrt(
            ((x_grid - cx) / (self.CIRCLE_SCALE_FACTOR * radius)) ** 2
            + ((y_grid - cy) / (self.CIRCLE_SCALE_FACTOR * radius)) ** 2
        )
        mask = r_norm < 1

        rect = self._get_rect_bounds(bounds["lines"], bounds["center"], radius, (h, w))
        mask[: rect["min_y"] + d] = False
        mask[rect["max_y"] - d :] = False
        mask[:, : rect["min_x"] + d] = False
        mask[:, rect["max_x"] - d :] = False

        return mask

    def _get_rect_bounds(
        self,
        lines: Dict[str, Tuple[np.ndarray, np.ndarray]],
        center: Tuple[float, float],
        radius: float,
        hw: Tuple[int, int],
    ) -> Dict[str, int]:
        """Extract rectangular bounds from lines."""
        h, w = hw
        bounds = {"min_y": 0, "max_y": h, "min_x": 0, "max_x": w}

        for loc in ["bottom", "top", "left", "right"]:
            if loc not in lines:
                continue

            p0, p1 = np.array(lines[loc][0]), np.array(lines[loc][1])
            d = p1 - p0
            a = d.dot(d)
            b = 2 * d.dot(p0 - np.array(center))
            c = p0.dot(p0) + np.array(center).dot(center) - 2 * p0.dot(center) - radius**2

            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                continue

            sqrt_disc = np.sqrt(discriminant)
            t1, t2 = (-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)
            intersects = [p0 + t1 * d, p0 + t2 * d]

            if loc == "bottom":
                y_vals = [intersects[0][1], intersects[1][1]]
                bounds["max_y"] = min(bounds["max_y"], int(np.floor(min(y_vals))))
            elif loc == "top":
                y_vals = [intersects[0][1], intersects[1][1]]
                bounds["min_y"] = max(bounds["min_y"], int(np.ceil(max(y_vals))))
            elif loc == "left":
                x_vals = [intersects[0][0], intersects[1][0]]
                bounds["min_x"] = max(bounds["min_x"], int(np.ceil(max(x_vals))))
            elif loc == "right":
                x_vals = [intersects[0][0], intersects[1][0]]
                bounds["max_x"] = min(bounds["max_x"], int(np.floor(min(x_vals))))

        return bounds

    def undo_bounds(
        self,
        bounded_image: torch.Tensor,
        center: Tuple[float, float],
        radius: float,
        hw: Tuple[int, int],
        **kwargs,
    ) -> torch.Tensor:
        """Reverses a specific center-radius crop-and-scale operation."""
        device = bounded_image.device
        compute_dtype = self._get_compute_dtype(device)
        y_center, x_center = center
        orig_radius = radius

        crop_center = self.square_size / 2.0
        scale = crop_center / orig_radius

        # Create matrix in compute dtype (must match image dtype for grid_sample)
        M_torch = torch.tensor(
            [
                [
                    [scale, 0, (crop_center - x_center * scale)],
                    [0, scale, (crop_center - y_center * scale)],
                ]
            ],
            dtype=compute_dtype,
            device=device,
        )

        M_torch = K.invert_affine_transform(M_torch)

        # Convert image to compute dtype if needed
        if bounded_image.dtype != compute_dtype:
            bounded_fp = bounded_image.to(compute_dtype)
        else:
            bounded_fp = bounded_image

        undone_image = K.warp_affine(
            bounded_fp, M_torch, hw, mode="nearest", padding_mode="zeros", align_corners=False
        )

        # Convert back to original dtype
        return undone_image.to(bounded_image.dtype)

    def undo_bounds_points(
        self, points_tensor: torch.Tensor, center: Tuple[float, float], radius: float, **kwargs
    ) -> torch.Tensor:
        """Reverses a specific center-radius crop-and-scale operation for points.

        Note: Point transformation uses matrix multiplication, not grid_sample,
        so we use float32 for numerical precision regardless of compute_dtype.
        """
        device = points_tensor.device
        y_center, x_center = center
        orig_radius = radius

        crop_center = self.square_size / 2.0
        scale = crop_center / orig_radius

        # Use float32 for point transformation (precision critical)
        M_torch = torch.tensor(
            [
                [
                    [scale, 0, (crop_center - x_center * scale)],
                    [0, scale, (crop_center - y_center * scale)],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )

        M_torch = K.invert_affine_transform(M_torch)
        T_3x3 = Kconv.convert_affinematrix_to_homography(M_torch)
        batch_size = points_tensor.shape[0]
        T_batch = T_3x3.repeat(batch_size, 1, 1)

        # Use float32 for point transformation (precision critical)
        points_fp32 = points_tensor.float()
        undone_points = K_geom.transform_points(T_batch, points_fp32)

        return undone_points.to(points_tensor.dtype)

    @staticmethod
    def _create_affine_matrix_torch(
        in_size: Tuple[int, int],
        out_size: int,
        scale: float,
        center: Tuple[float, float],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create affine transformation matrix as torch tensor.

        Always use float32 for affine matrices to ensure precision.
        """
        return create_affine_matrix(in_size, out_size, scale, center, device, dtype)

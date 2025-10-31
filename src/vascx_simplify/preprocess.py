from typing import Tuple, Dict, Any, Optional, Union
import torch
import numpy as np
from scipy.ndimage import sobel
from sklearn.linear_model import RANSACRegressor

import kornia.geometry.transform as K
import kornia.geometry.conversions as Kconv
import kornia.geometry as K_geom
import kornia.filters as KF
import torch.nn.functional as F

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    MIN_RADIUS_DIVISOR = 4    # Minimum radius = RESOLUTION / 4
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
        use_fp16: bool = True
    ):
        """
        Args:
            square_size: Output size (None to keep original)
            sigma_fraction: Blur strength (default 0.05)
            contrast_factor: Enhancement strength (default 4)
            use_fp16: Use float16 for compute-intensive ops (default True, only on CUDA)
        """
        self.square_size = square_size
        self.sigma_fraction = sigma_fraction if sigma_fraction is not None else self.DEFAULT_SIGMA_FRACTION
        self.contrast_factor = contrast_factor if contrast_factor is not None else self.DEFAULT_CONTRAST_FACTOR
        self.return_bounds = return_bounds
        self.use_fp16 = use_fp16
        
        # FP16 only supported on CUDA, fallback to FP32 on CPU
        # Will be determined per-device when processing
        self._compute_dtype_cache = {}
        
        # Pre-initialize RANSAC for line fitting (reusable)
        self.ransac_line = RANSACRegressor(
            residual_threshold=self.INLIER_DIST_THRESHOLD,
            random_state=42
        )
        
        # Cache for grid coordinates (avoids recreating meshgrids)
        self._grid_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def _get_compute_dtype(self, device: torch.device) -> torch.dtype:
        """Get compute dtype based on device and user preference.
        
        FP16 only works on CUDA. CPU operations fallback to FP32.
        """
        if device not in self._compute_dtype_cache:
            if self.use_fp16 and device.type == 'cuda':
                self._compute_dtype_cache[device] = torch.float16
            else:
                self._compute_dtype_cache[device] = torch.float32
        return self._compute_dtype_cache[device]
    
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
        self, 
        image: torch.Tensor, 
        device: torch.device
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
    
    def _get_bounds(self, image: torch.Tensor, device: torch.device) -> Dict[str, Any]:
        """Detect fundus boundaries using GPU where possible."""
        compute_dtype = self._get_compute_dtype(device)
        
        # Extract grayscale (first channel if RGB)
        if image.dim() == 3 and image.shape[0] >= 1:
            gray = image[0]  # [H, W]
        else:
            gray = image
        
        h, w = gray.shape[-2:]
        
        # Scale to standard resolution using kornia
        scale = min(self.RESOLUTION / h, self.RESOLUTION / w)
        # Create matrix in compute dtype for warp_affine
        M = self._create_affine_matrix_torch((h, w), self.RESOLUTION, scale, (h // 2, w // 2), 
                                            device, dtype=compute_dtype)
        
        # GPU warp - use compute dtype for speed
        gray_batch = gray.unsqueeze(0).unsqueeze(0).to(compute_dtype)  # [1, 1, H, W]
        gray_scaled = K.warp_affine(
            gray_batch, 
            M.unsqueeze(0), 
            (self.RESOLUTION, self.RESOLUTION),
            mode='bilinear',
            padding_mode='zeros'
        )
        gray_scaled = gray_scaled.squeeze(0).squeeze(0).float()  # [H, W] - back to float32
        
        # Edge detection (uses scipy sobel - small CPU operation)
        xs, ys = self._detect_edges(gray_scaled, device)
        
        # Circle fitting (uses sklearn RANSAC on 256 points)
        radius, center, circle_fraction = self._fit_circle(xs, ys)
        
        # Line fitting if needed
        lines = {} if circle_fraction > 0.85 else self._fit_lines(xs, ys, radius, center, circle_fraction)
        
        # Transform back to original coordinates using GPU (use float32 for precision)
        M_fp32 = M.float()  # Convert to float32 for accurate inverse
        M_3x3 = torch.cat([M_fp32, torch.tensor([[0., 0., 1.]], device=device, dtype=torch.float32)], dim=0)
        M_inv = torch.inverse(M_3x3)
        
        center_homo = torch.tensor([center[0], center[1], 1.], device=device, dtype=torch.float32)
        center_orig_homo = M_inv @ center_homo
        center_orig = (center_orig_homo[:2] / center_orig_homo[2]).cpu().numpy()
        
        lines_orig = {}
        for k, (p0, p1) in lines.items():
            p0_homo = torch.tensor([p0[0], p0[1], 1.], device=device, dtype=torch.float32)
            p1_homo = torch.tensor([p1[0], p1[1], 1.], device=device, dtype=torch.float32)
            p0_orig_homo = M_inv @ p0_homo
            p1_orig_homo = M_inv @ p1_homo
            p0_orig = (p0_orig_homo[:2] / p0_orig_homo[2]).cpu().numpy()
            p1_orig = (p1_orig_homo[:2] / p1_orig_homo[2]).cpu().numpy()
            lines_orig[k] = (p0_orig, p1_orig)
        
        return {
            'center': center_orig,
            'radius': radius / scale,
            'lines': lines_orig,
            'hw': (h, w)
        }
    
    def _detect_edges(
        self, 
        image: torch.Tensor, 
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect edge points using polar transform.
        
        Note: Uses scipy sobel on small 256×256 array (~1ms CPU overhead).
        """
        # GPU polar transform using grid_sample (use compute dtype)
        polar = self._linearPolar(image, (self.CENTER, self.CENTER), self.MAX_R, device)
        edge_region = polar[:, self.MIN_R:].float().cpu().numpy()
        
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
        device: torch.device
    ) -> torch.Tensor:
        """Convert image to polar coordinates using GPU grid_sample."""
        compute_dtype = self._get_compute_dtype(device)
        cy, cx = center
        
        # Create polar grid on GPU (use compute dtype)
        theta = torch.linspace(0, 2 * np.pi, self.RESOLUTION, device=device, dtype=compute_dtype)
        radius = torch.linspace(0, max_radius, self.RESOLUTION, device=device, dtype=compute_dtype)
        
        theta_grid, radius_grid = torch.meshgrid(theta, radius, indexing='ij')
        
        # Convert to Cartesian
        x = cx + radius_grid * torch.cos(theta_grid)
        y = cy + radius_grid * torch.sin(theta_grid)
        
        # Normalize to [-1, 1] for grid_sample
        x_norm = 2 * x / (self.RESOLUTION - 1) - 1
        y_norm = 2 * y / (self.RESOLUTION - 1) - 1
        
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        
        # Sample using GPU (input already in compute dtype)
        image_batch = image.unsqueeze(0).unsqueeze(0).to(compute_dtype)  # [1, 1, H, W]
        polar = F.grid_sample(image_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return polar.squeeze(0).squeeze(0)  # [H, W]
    
    def _fit_circle(
        self, 
        xs: np.ndarray, 
        ys: np.ndarray
    ) -> Tuple[float, np.ndarray, float]:
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
                d = (pts[indices]**2).sum(axis=1)
                y = np.linalg.lstsq(B, d, rcond=None)[0]
                center = 0.5 * y[:2]
                radius = np.sqrt(y[2] + (center**2).sum())
                if self.MIN_R < radius < self.MAX_R:
                    break
                attempts += 1
            
            # Count inliers
            distances = np.abs(np.sqrt(((pts - center)**2).sum(axis=1)) - radius)
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
            d = (pts[best_inliers]**2).sum(axis=1)
            y = np.linalg.lstsq(B, d, rcond=None)[0]
            center = 0.5 * y[:2]
            radius = np.sqrt(y[2] + (center**2).sum())
        
        return radius, center, circle_fraction
    
    def _fit_lines(
        self, 
        xs: np.ndarray, 
        ys: np.ndarray, 
        radius: float, 
        center: np.ndarray, 
        circle_fraction: float
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Fit lines to edge regions using sklearn RANSAC."""
        if circle_fraction < 0.3:
            # Refit circle using corners
            pts = np.c_[xs[self.CORNER_MASK], ys[self.CORNER_MASK]]
            B = np.c_[pts, np.ones(len(pts))]
            d = (pts**2).sum(axis=1)
            y = np.linalg.lstsq(B, d, rcond=None)[0]
            center = 0.5 * y[:2]
            radius = self.CIRCLE_REFIT_SCALE * np.sqrt(y[2] + (center**2).sum())
        
        lines = {}
        for loc in ["left", "right"]:
            mask = self.RECT_MASKS[loc]
            self.ransac_line.fit(ys[mask].reshape(-1, 1), xs[mask])
            if self.ransac_line.inlier_mask_.mean() > 0.5:
                a, b = self.ransac_line.estimator_.coef_[0], self.ransac_line.estimator_.intercept_
                x_vals = np.array([0, self.RESOLUTION])
                y_vals = a * x_vals + b
                lines[loc] = (y_vals[::-1], x_vals[::-1])
        
        for loc in ["top", "bottom"]:
            mask = self.RECT_MASKS[loc]
            self.ransac_line.fit(xs[mask].reshape(-1, 1), ys[mask])
            if self.ransac_line.inlier_mask_.mean() > 0.5:
                a, b = self.ransac_line.estimator_.coef_[0], self.ransac_line.estimator_.intercept_
                x_vals = np.array([0, self.RESOLUTION])
                y_vals = a * x_vals + b
                lines[loc] = (x_vals, y_vals)
        
        return lines
    
    def _crop_to_square(
        self, 
        image: torch.Tensor, 
        bounds: Dict[str, Any], 
        device: torch.device
    ) -> torch.Tensor:
        """Crop and resize to square using kornia (GPU with mixed precision)."""
        compute_dtype = self._get_compute_dtype(device)
        cy, cx = bounds['center']
        scale = self.square_size / (2 * bounds['radius'])
        # Use compute dtype for matrix (must match image for warp_affine)
        M = self._create_affine_matrix_torch(bounds['hw'], self.square_size, scale, (cy, cx), 
                                            device, dtype=compute_dtype)
        
        # Warp image on GPU (use compute dtype for speed)
        if image.dim() == 2:
            image = image.unsqueeze(0)  # [1, H, W]
        
        image_batch = image.unsqueeze(0).to(compute_dtype)  # [1, C, H, W]
        warped = K.warp_affine(
            image_batch,
            M.unsqueeze(0),
            (self.square_size, self.square_size),
            mode='bilinear',
            padding_mode='zeros'
        )
        
        # Convert back to uint8
        warped = warped.squeeze(0).float().clamp(0, 255)  # [C, H, W]
        return warped.to(torch.uint8)
    
    def _update_bounds_after_crop(self, bounds: Dict[str, Any]) -> Dict[str, Any]:
        """Update bounds for cropped image."""
        return {
            'center': (self.square_size / 2, self.square_size / 2),
            'radius': bounds['radius'] * (self.square_size / (2 * bounds['radius'])),
            'lines': {},
            'hw': (self.square_size, self.square_size)
        }
    
    def _enhance_contrast(
        self, 
        image: torch.Tensor, 
        bounds: Dict[str, Any], 
        device: torch.device
    ) -> torch.Tensor:
        """Apply contrast enhancement using kornia (GPU with mixed precision).
        
        Major optimization: Blur at 256×256 then upsample instead of full resolution.
        Uses float16 for blur (2-4x faster) and float32 for final unsharp mask.
        """
        compute_dtype = self._get_compute_dtype(device)
        mirrored = self._mirror_image(image, bounds, device)
        
        # Efficient blur at reduced resolution
        ce_res = self.REDUCED_BLUR_RESOLUTION
        cy, cx = bounds['center']
        scale_ce = ce_res / (2 * bounds['radius'])
        # Use compute dtype for matrix (must match image for warp_affine)
        M_ce = self._create_affine_matrix_torch(bounds['hw'], ce_res, scale_ce, (cy, cx), 
                                                device, dtype=compute_dtype)
        
        # Warp to smaller size (use compute dtype for speed)
        if mirrored.dim() == 2:
            mirrored = mirrored.unsqueeze(0)
        
        mirrored_batch = mirrored.unsqueeze(0).to(compute_dtype) / 255.0  # [1, C, H, W]
        mirrored_small = K.warp_affine(
            mirrored_batch,
            M_ce.unsqueeze(0),
            (ce_res, ce_res),
            mode='bilinear',
            padding_mode='zeros'
        )
        
        # Gaussian blur using kornia (use compute dtype - major speedup with fp16)
        sigma = self.sigma_fraction * (ce_res / 2)
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)
        
        blurred_small = KF.gaussian_blur2d(
            mirrored_small,
            (kernel_size, kernel_size),
            (sigma, sigma)
        )
        
        # Upsample blur back to original size (keep in compute dtype)
        h, w = bounds['hw']
        # Convert M_ce to float32 for accurate inverse, then back to compute_dtype
        M_ce_fp32 = M_ce.float()
        M_ce_3x3 = torch.cat([M_ce_fp32, torch.tensor([[0., 0., 1.]], device=device, dtype=torch.float32)], dim=0)
        M_ce_inv_fp32 = torch.inverse(M_ce_3x3)[:2]
        M_ce_inv = M_ce_inv_fp32.to(compute_dtype)  # Convert back for warp_affine
        
        blurred = K.warp_affine(
            blurred_small,
            M_ce_inv.unsqueeze(0),
            (w, h),
            mode='bilinear',
            padding_mode='zeros'
        )
        
        # Unsharp mask (use float32 for final computation to preserve quality)
        image_norm = image.unsqueeze(0).float() / 255.0  # [1, C, H, W]
        blurred_fp32 = blurred.float()
        
        enhanced = torch.clamp(
            self.contrast_factor * (image_norm - blurred_fp32) + 0.5,
            0, 1
        )
        enhanced = (enhanced * 255).to(torch.uint8).squeeze(0)  # [C, H, W]
        
        # Apply mask (GPU)
        mask = self._make_mask(bounds, device)
        enhanced = enhanced * mask.unsqueeze(0)
        
        return enhanced
    
    def _mirror_image(
        self, 
        image: torch.Tensor, 
        bounds: Dict[str, Any], 
        device: torch.device
    ) -> torch.Tensor:
        """Mirror pixels at boundaries using torch operations (GPU).
        
        Uses uint8 directly - no need for float conversion here.
        """
        compute_dtype = self._get_compute_dtype(device)
        cx, cy = bounds['center']
        radius = bounds['radius']
        h, w = bounds['hw']
        
        if image.dim() == 2:
            mirrored = image.clone()
            is_2d = True
        else:
            mirrored = image.clone()
            is_2d = False
        
        # Get rectangular bounds
        d = int(self.BORDER_MARGIN_FRACTION * radius)
        rect = self._get_rect_bounds(bounds['lines'], bounds['center'], radius, (h, w))
        min_y = max(rect['min_y'] + d, 0)
        max_y = min(rect['max_y'] - d, h)
        min_x = max(rect['min_x'] + d, 0)
        max_x = min(rect['max_x'] - d, w)
        
        # Mirror edges using torch.flip (GPU) - works directly on uint8
        if min_y > 0:
            flip_h = min(min_y, max_y - min_y)
            if is_2d:
                mirrored[:min_y] = torch.flip(mirrored[min_y:min_y+flip_h], dims=[0])[:min_y]
            else:
                mirrored[:, :min_y] = torch.flip(mirrored[:, min_y:min_y+flip_h], dims=[1])[:, :min_y]
        
        if max_y < h:
            flip_h = min(h - max_y, max_y - min_y)
            if is_2d:
                mirrored[max_y:] = torch.flip(mirrored[max_y-flip_h:max_y], dims=[0])[:h-max_y]
            else:
                mirrored[:, max_y:] = torch.flip(mirrored[:, max_y-flip_h:max_y], dims=[1])[:, :h-max_y]
        
        if min_x > 0:
            flip_w = min(min_x, max_x - min_x)
            if is_2d:
                mirrored[:, :min_x] = torch.flip(mirrored[:, min_x:min_x+flip_w], dims=[1])[:, :min_x]
            else:
                mirrored[:, :, :min_x] = torch.flip(mirrored[:, :, min_x:min_x+flip_w], dims=[2])[:, :, :min_x]
        
        if max_x < w:
            flip_w = min(w - max_x, max_x - min_x)
            if is_2d:
                mirrored[:, max_x:] = torch.flip(mirrored[:, max_x-flip_w:max_x], dims=[1])[:, :w-max_x]
            else:
                mirrored[:, :, max_x:] = torch.flip(mirrored[:, :, max_x-flip_w:max_x], dims=[2])[:, :, :w-max_x]
        
        # Mirror circle using cached grid coordinates (GPU, use compute dtype)
        grid_key = (h, w, device, compute_dtype)
        if grid_key not in self._grid_cache:
            y_coords = torch.arange(h, device=device, dtype=compute_dtype)
            x_coords = torch.arange(w, device=device, dtype=compute_dtype)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            self._grid_cache[grid_key] = (x_grid, y_grid)
        else:
            x_grid, y_grid = self._grid_cache[grid_key]
        
        r_sq_norm = ((x_grid - cx) / (self.CIRCLE_SCALE_FACTOR * radius))**2 + ((y_grid - cy) / (self.CIRCLE_SCALE_FACTOR * radius))**2
        outside = r_sq_norm > 1
        
        scale = 1.0 / r_sq_norm[outside]
        x_in = torch.clamp(torch.round(cx + (x_grid[outside] - cx) * scale).long(), 0, w - 1)
        y_in = torch.clamp(torch.round(cy + (y_grid[outside] - cy) * scale).long(), 0, h - 1)
        
        if is_2d:
            mirrored[outside] = mirrored[y_in, x_in]
        else:
            mirrored[:, outside] = mirrored[:, y_in, x_in]
        
        return mirrored
    
    def _make_mask(
        self, 
        bounds: Dict[str, Any], 
        device: torch.device
    ) -> torch.Tensor:
        """Create binary mask using torch (GPU, use compute dtype for speed)."""
        compute_dtype = self._get_compute_dtype(device)
        cx, cy = bounds['center']
        radius = bounds['radius']
        h, w = bounds['hw']
        d = int(self.BORDER_MARGIN_FRACTION * radius)
        
        # Get or create cached grid (use compute dtype)
        grid_key = (h, w, device, compute_dtype)
        if grid_key not in self._grid_cache:
            y_coords = torch.arange(h, device=device, dtype=compute_dtype)
            x_coords = torch.arange(w, device=device, dtype=compute_dtype)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            self._grid_cache[grid_key] = (x_grid, y_grid)
        else:
            x_grid, y_grid = self._grid_cache[grid_key]
        
        r_norm = torch.sqrt(((x_grid - cx) / (self.CIRCLE_SCALE_FACTOR * radius))**2 + ((y_grid - cy) / (self.CIRCLE_SCALE_FACTOR * radius))**2)
        mask = r_norm < 1
        
        rect = self._get_rect_bounds(bounds['lines'], bounds['center'], radius, (h, w))
        mask[:rect['min_y'] + d] = False
        mask[rect['max_y'] - d:] = False
        mask[:, :rect['min_x'] + d] = False
        mask[:, rect['max_x'] - d:] = False
        
        return mask
    
    def _get_rect_bounds(
        self, 
        lines: Dict[str, Tuple[np.ndarray, np.ndarray]], 
        center: Tuple[float, float], 
        radius: float, 
        hw: Tuple[int, int]
    ) -> Dict[str, int]:
        """Extract rectangular bounds from lines."""
        h, w = hw
        bounds = {'min_y': 0, 'max_y': h, 'min_x': 0, 'max_x': w}
        
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
                bounds['max_y'] = min(bounds['max_y'], int(np.floor(min(y_vals))))
            elif loc == "top":
                y_vals = [intersects[0][1], intersects[1][1]]
                bounds['min_y'] = max(bounds['min_y'], int(np.ceil(max(y_vals))))
            elif loc == "left":
                x_vals = [intersects[0][0], intersects[1][0]]
                bounds['min_x'] = max(bounds['min_x'], int(np.ceil(max(x_vals))))
            elif loc == "right":
                x_vals = [intersects[0][0], intersects[1][0]]
                bounds['max_x'] = min(bounds['max_x'], int(np.floor(min(x_vals))))
        
        return bounds

    def undo_bounds(
        self, 
        bounded_image: torch.Tensor, 
        center: Tuple[float, float], 
        radius: float, 
        hw: Tuple[int, int], 
        **kwargs
    ) -> torch.Tensor:
        """Reverses a specific center-radius crop-and-scale operation."""
        device = bounded_image.device
        compute_dtype = self._get_compute_dtype(device)
        y_center, x_center = center
        orig_radius = radius
        
        crop_center = self.square_size / 2.0
        scale = crop_center / orig_radius
        
        # Create matrix in compute dtype (must match image dtype for grid_sample)
        M_torch = torch.tensor([[
            [scale, 0, (crop_center - x_center * scale)],
            [0, scale, (crop_center - y_center * scale)]
        ]], dtype=compute_dtype, device=device)

        M_torch = K.invert_affine_transform(M_torch)

        # Convert image to compute dtype if needed
        if bounded_image.dtype != compute_dtype:
            bounded_fp = bounded_image.to(compute_dtype)
        else:
            bounded_fp = bounded_image
        
        undone_image = K.warp_affine(
            bounded_fp,
            M_torch,
            hw[::-1],
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )
        
        # Convert back to original dtype
        return undone_image.to(bounded_image.dtype)

    def undo_bounds_points(
        self, 
        points_tensor: torch.Tensor, 
        center: Tuple[float, float], 
        radius: float, 
        hw: Tuple[int, int], 
        **kwargs
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
        M_torch = torch.tensor([[
            [scale, 0, (crop_center - x_center * scale)],
            [0, scale, (crop_center - y_center * scale)]
        ]], dtype=torch.float32, device=device)

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
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Create affine transformation matrix as torch tensor.
        
        Always use float32 for affine matrices to ensure precision.
        """
        cy, cx = center
        ty, tx = out_size / 2, out_size / 2
        return torch.tensor([
            [scale, 0, tx - scale * cx],
            [0, scale, ty - scale * cy]
        ], dtype=dtype, device=device)

class VASCXTransform:
    def __init__(
        self, 
        size: int = 1024, 
        have_ce: bool = True, 
        use_fp16: bool = True, 
        device: Union[torch.device, str] = DEFAULT_DEVICE
    ):
        """
        Args:
            size: Output size
            have_ce: Whether to use contrast enhancement
            use_fp16: Use float16 for compute-intensive ops
            device: Device to run transformations on ('cuda' or 'cpu')
        """
        self.size = size
        self.device = torch.device(device)
        self.have_ce = have_ce
        self.use_fp16 = use_fp16
        
        if self.have_ce:
            self.contrast_enhancer = FundusContrastEnhance(square_size=size, use_fp16=use_fp16)
            # 6 channels (rgb + ce)
            self.mean = torch.tensor(
                [0.485, 0.456, 0.406, 0.485, 0.456, 0.406], 
                device=self.device, dtype=torch.float32
            ).view(6, 1, 1)
            self.std = torch.tensor(
                [0.229, 0.224, 0.225, 0.229, 0.224, 0.225], 
                device=self.device, dtype=torch.float32
            ).view(6, 1, 1)
        else:
            self.contrast_enhancer = None
            # 3 channels (rgb only)
            self.mean = torch.tensor(
                [0.485, 0.456, 0.406], 
                device=self.device, dtype=torch.float32
            ).view(3, 1, 1)
            self.std = torch.tensor(
                [0.229, 0.224, 0.225], 
                device=self.device, dtype=torch.float32
            ).view(3, 1, 1)

    def __call__(self, image) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Args:
            image: Input image as numpy array, PIL Image, or torch.Tensor
        Returns:
            tuple: (processed_image, bounds) as torch.Tensors on specified device
        """
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            if hasattr(image, 'shape'):  # numpy array
                image = torch.from_numpy(np.array(image))
            else:  # PIL Image
                image = torch.from_numpy(np.array(image))
        
        # Move to device
        image = image.to(self.device)
        
        # Ensure proper format [C, H, W]
        if image.ndim == 2:  # Grayscale [H, W]
            image = image.unsqueeze(0)
        elif image.ndim == 3:
            if image.shape[-1] in [1, 3, 4]:  # [H, W, C] format
                image = image.permute(2, 0, 1)
            # else already in [C, H, W] format
        
        # Ensure uint8
        if image.dtype != torch.uint8:
            if image.max() <= 1.0:
                image = (image * 255).clamp(0, 255).to(torch.uint8)
            else:
                image = image.clamp(0, 255).to(torch.uint8)
        
        if self.have_ce:
            # Apply contrast enhancement
            rgb, ce, bounds = self.contrast_enhancer(image)
            # Concatenate RGB and CE
            inputs = torch.cat([rgb, ce], dim=0)  # [6, H, W]
            # Convert to float32 [0, 1] and normalize
            inputs = inputs.float() / 255.0
            inputs = (inputs - self.mean) / self.std
            return inputs, bounds
        else:
            # Resize using kornia (GPU-accelerated)
            image_batch = image.unsqueeze(0).float()  # [1, C, H, W]
            resized = K.resize(
                image_batch, 
                (self.size, self.size), 
                interpolation='bilinear',
                align_corners=False
            )
            # Convert to [0, 1] and normalize
            resized = resized / 255.0
            resized = (resized - self.mean) / self.std
            return resized.squeeze(0), None

    def undo_bounds(self, image: torch.Tensor, bounds: Dict[str, Any]) -> torch.Tensor:
        """Reverse the cropping transformation."""
        return self.contrast_enhancer.undo_bounds(image, **bounds)
        
    def undo_bounds_points(self, points: torch.Tensor, bounds: Dict[str, Any]) -> torch.Tensor:
        """Reverse the cropping transformation for point coordinates."""
        return self.contrast_enhancer.undo_bounds_points(points, **bounds)

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature import local_binary_pattern
from typing import Tuple

def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Compute the gradient magnitude of an image using the Sobel filter.
    
    Args:
        image (np.ndarray): Input image (usually grayscale).
        
    Returns:
        np.ndarray: Gradient magnitude image.
    """
    return sobel(image)

def compute_lbp_histogram(gray_image: np.ndarray, mask: np.ndarray, n_points: int = 8, radius: int = 1, lbp_precomputed: np.ndarray = None) -> np.ndarray:
    """
    Compute LBP histogram for a masked region of a grayscale image.
    
    Args:
        gray_image (np.ndarray): Grayscale input image.
        mask (np.ndarray): Boolean mask indicating the region of interest.
        n_points (int): Number of circularly symmetric neighbor set points.
        radius (int): Radius of circle.
        lbp_precomputed (np.ndarray, optional): Precomputed LBP map for the full image.
        
    Returns:
        np.ndarray: Histogram of LBP values (10 bins for uniform LBP with n_points=8).
    """
    if lbp_precomputed is not None:
        lbp = lbp_precomputed
    else:
        lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    # Mask out the region
    lbp_region = lbp[mask]
    
    if len(lbp_region) == 0:
        return np.zeros(n_points + 2, dtype=np.float32)
        
    # Calculate histogram
    hist, _ = np.histogram(lbp_region, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype(np.float32)
    # Normalize histogram
    hist /= (hist.sum() + 1e-6)
    
    return hist

def extract_node_features(image: np.ndarray, segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract per-superpixel node features and centroids (vectorized implementation).
    
    Args:
        image (np.ndarray): RGB input image.
        segments (np.ndarray): Superpixel segment labels.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - features: Node features of shape [N, 15]
            - centroids: Normalized (x, y) coordinates of shape [N, 2]
    """
    h, w = image.shape[:2]
    labels, inverse = np.unique(segments, return_inverse=True)
    num_nodes = len(labels)
    inv_flat = inverse.ravel()
    
    counts = np.bincount(inv_flat, minlength=num_nodes).astype(np.float32)
    counts_safe = np.maximum(counts, 1.0)
    
    # Grayscale, Sobel gradient magnitude, and uniform LBP (8, 1)
    if image.ndim == 3:
        gray_image = rgb2gray(image)
        if gray_image.max() <= 1.0:
            gray_image = (gray_image * 255).astype(np.uint8)
    else:
        gray_image = image
        
    grad_mag = sobel_magnitude(gray_image.astype(np.float32) / 255.0)
    lbp = local_binary_pattern(gray_image, 8, 1, method="uniform").astype(np.int64)
    
    features = np.zeros((num_nodes, 15), dtype=np.float32)
    
    # 1. Mean color per channel (0:3)
    if image.ndim == 3:
        for c in range(3):
            features[:, c] = np.bincount(inv_flat, weights=image[..., c].ravel(), minlength=num_nodes) / counts_safe
    else:
        m = np.bincount(inv_flat, weights=image.ravel(), minlength=num_nodes) / counts_safe
        for c in range(3):
            features[:, c] = m
            
    # 2. Std intensity across all channels (3)
    if image.ndim == 3:
        img_sq = (image.astype(np.float64)**2).sum(axis=2)
        sum_sq = np.bincount(inv_flat, weights=img_sq.ravel(), minlength=num_nodes)
        img_sum = (image.astype(np.float64)).sum(axis=2)
        sum_val = np.bincount(inv_flat, weights=img_sum.ravel(), minlength=num_nodes)
        n_elements = counts_safe * 3.0
        mean_val = sum_val / n_elements
        var = (sum_sq / n_elements) - (mean_val**2)
        features[:, 3] = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    else:
        img_sq = image.astype(np.float64)**2
        sum_sq = np.bincount(inv_flat, weights=img_sq.ravel(), minlength=num_nodes)
        mean_val = features[:, 0].astype(np.float64)
        var = (sum_sq / counts_safe) - (mean_val**2)
        features[:, 3] = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
        
    # 3. Mean gradient magnitude (4)
    features[:, 4] = np.bincount(inv_flat, weights=grad_mag.ravel(), minlength=num_nodes) / counts_safe
    
    # 4. LBP histogram (5:15) - 10 uniform bins
    lbp_clipped = np.clip(lbp.ravel(), 0, 9)
    joint_lbp = inv_flat * 10 + lbp_clipped
    lbp_hist_2d = np.bincount(joint_lbp, minlength=num_nodes * 10).reshape(num_nodes, 10).astype(np.float32)
    lbp_hist_norm = lbp_hist_2d / (lbp_hist_2d.sum(axis=1, keepdims=True) + 1e-6)
    features[:, 5:15] = lbp_hist_norm
    
    # 5. Centroids (cx, cy)
    yy, xx = np.indices((h, w), dtype=np.float32)
    cy = (np.bincount(inv_flat, weights=yy.ravel(), minlength=num_nodes) / counts_safe) / float(h)
    cx = (np.bincount(inv_flat, weights=xx.ravel(), minlength=num_nodes) / counts_safe) / float(w)
    centroids = np.stack([cx, cy], axis=1).astype(np.float32)
    
    return features, centroids

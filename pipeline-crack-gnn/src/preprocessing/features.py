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

def compute_lbp_histogram(gray_image: np.ndarray, mask: np.ndarray, n_points: int = 8, radius: int = 1) -> np.ndarray:
    """
    Compute LBP histogram for a masked region of a grayscale image.
    
    Args:
        gray_image (np.ndarray): Grayscale input image.
        mask (np.ndarray): Boolean mask indicating the region of interest.
        n_points (int): Number of circularly symmetric neighbor set points.
        radius (int): Radius of circle.
        
    Returns:
        np.ndarray: Histogram of LBP values (10 bins for uniform LBP with n_points=8).
    """
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
    Extract per-superpixel node features and centroids.
    
    Args:
        image (np.ndarray): RGB input image.
        segments (np.ndarray): Superpixel segment labels.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - features: Node features of shape [N, 15]
            - centroids: Normalized (x, y) coordinates of shape [N, 2]
    """
    labels = np.unique(segments)
    num_nodes = len(labels)
    
    features = np.zeros((num_nodes, 15), dtype=np.float32)
    centroids = np.zeros((num_nodes, 2), dtype=np.float32)
    
    # Convert image to grayscale for LBP and Sobel
    if image.ndim == 3:
        gray_image = rgb2gray(image)
        if gray_image.max() <= 1.0:
            gray_image = (gray_image * 255).astype(np.uint8)
    else:
        gray_image = image
        
    grad_mag = sobel_magnitude(gray_image.astype(np.float32) / 255.0)
    
    h, w = image.shape[:2]
    
    for i, label in enumerate(labels):
        mask = (segments == label)
        
        # 1. Mean intensity per channel (3)
        if image.ndim == 3:
            mean_color = image[mask].mean(axis=0)
        else:
            mean_color = np.array([image[mask].mean()] * 3)
            
        # 2. Std intensity (1 scalar across all channels)
        std_intensity = image[mask].std()
        
        # 3. Mean gradient magnitude (1)
        mean_grad = grad_mag[mask].mean()
        
        # 4. LBP histogram (10 bins)
        lbp_hist = compute_lbp_histogram(gray_image, mask)
        
        # Concatenate features
        features[i, 0:3] = mean_color
        features[i, 3] = std_intensity
        features[i, 4] = mean_grad
        features[i, 5:15] = lbp_hist
        
        # Calculate centroids
        y_coords, x_coords = np.nonzero(mask)
        cy = y_coords.mean() / h
        cx = x_coords.mean() / w
        centroids[i] = [cx, cy]
        
    return features, centroids

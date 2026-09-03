import os
import numpy as np
import skimage.io
from skimage.segmentation import slic

def compute_superpixels(image: np.ndarray, n_segments: int = 300, compactness: float = 10.0, sigma: float = 1.0, start_label: int = 1) -> np.ndarray:
    """
    Compute SLIC superpixels for an image.
    
    Args:
        image (np.ndarray): Input image.
        n_segments (int): The (approximate) number of labels in the segmented output image.
        compactness (float): Balances color proximity and space proximity.
        sigma (float): Width of Gaussian smoothing kernel.
        start_label (int): The label for the first segment.
        
    Returns:
        np.ndarray: Integer mask indicating segment labels.
    """
    segments = slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=start_label)
    return segments

def compute_and_cache_superpixels(image_path: str, output_dir: str, **kwargs) -> np.ndarray:
    """
    Loads an image, computes superpixels, and caches the result as a numpy array.
    
    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the superpixel numpy array.
        **kwargs: Additional arguments for compute_superpixels.
        
    Returns:
        np.ndarray: The computed segment labels.
    """
    image = skimage.io.imread(image_path)
    segments = compute_superpixels(image, **kwargs)
    
    os.makedirs(output_dir, exist_ok=True)
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{image_id}.npy")
    np.save(output_path, segments)
    
    return segments

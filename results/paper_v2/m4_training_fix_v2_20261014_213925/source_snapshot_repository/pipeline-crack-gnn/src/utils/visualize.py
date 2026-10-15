import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from typing import Dict, List, Optional

def overlay_superpixels(image: np.ndarray, segments: np.ndarray, ax: Optional[plt.Axes] = None):
    """
    Display superpixel boundaries overlay on the image.
    
    Args:
        image (np.ndarray): Original image.
        segments (np.ndarray): Superpixel labels.
        ax (Optional[plt.Axes]): Matplotlib axes to draw on.
    """
    out = mark_boundaries(image, segments)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(out)
        ax.axis('off')
        plt.show()
    else:
        ax.imshow(out)
        ax.axis('off')

def overlay_predictions(image: np.ndarray, segments: np.ndarray, node_preds: np.ndarray, ax: Optional[plt.Axes] = None):
    """
    Colors superpixels by crack (red) / no-crack (original color) prediction.
    
    Args:
        image (np.ndarray): Original image.
        segments (np.ndarray): Superpixel labels.
        node_preds (np.ndarray): Binary predictions for each node (superpixel).
        ax (Optional[plt.Axes]): Matplotlib axes to draw on.
    """
    colored_image = image.copy()
    
    # If grayscale, convert to RGB
    if colored_image.ndim == 2:
        colored_image = np.stack((colored_image,)*3, axis=-1)
        
    labels = np.unique(segments)
    
    for i, label in enumerate(labels):
        if i < len(node_preds) and node_preds[i] == 1:
            mask = (segments == label)
            # Create a red overlay
            red_patch = np.zeros_like(colored_image)
            red_patch[mask] = [255, 0, 0] if colored_image.max() > 1 else [1.0, 0, 0]
            
            # Blend
            colored_image[mask] = (0.5 * colored_image[mask] + 0.5 * red_patch[mask]).astype(colored_image.dtype)
            
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(colored_image)
        ax.axis('off')
        plt.show()
    else:
        ax.imshow(colored_image)
        ax.axis('off')

def plot_dirichlet_energy(energy_per_layer: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plots layer index vs Dirichlet energy for multiple models.
    
    Args:
        energy_per_layer (Dict[str, List[float]]): Dictionary mapping model names to lists of energy values.
        save_path (Optional[str]): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    for model_name, energies in energy_per_layer.items():
        layers = range(1, len(energies) + 1)
        plt.plot(layers, energies, marker='o', label=model_name)
        
    plt.xlabel('Layer Index')
    plt.ylabel('Dirichlet Energy')
    plt.title('Over-smoothing Analysis: Dirichlet Energy vs Layer')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_curves(train_losses: List[float], val_losses: List[float], val_f1s: List[float], save_path: Optional[str] = None):
    """
    Plot training/validation loss and F1 scores over epochs.
    
    Args:
        train_losses (List[float]): Training losses.
        val_losses (List[float]): Validation losses.
        val_f1s (List[float]): Validation F1 scores.
        save_path (Optional[str]): Path to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # F1 plot
    epochs_f1 = range(1, len(val_f1s) + 1)
    ax2.plot(epochs_f1, val_f1s, color='green', label='Validation F1')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1 Score')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

def compute_f1(preds: np.ndarray, labels: np.ndarray, average: str = 'binary') -> float:
    """
    Compute the F1 score.
    
    Args:
        preds (np.ndarray): Binary predictions.
        labels (np.ndarray): Ground truth labels.
        average (str): Averaging method for multi-class/binary.
        
    Returns:
        float: F1 score.
    """
    return f1_score(labels, preds, average=average)

def compute_iou(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Intersection over Union for the crack class (label 1).
    
    Args:
        preds (np.ndarray): Binary predictions.
        labels (np.ndarray): Ground truth labels.
        
    Returns:
        float: IoU score.
    """
    intersection = np.logical_and(preds == 1, labels == 1).sum()
    union = np.logical_or(preds == 1, labels == 1).sum()
    
    if union == 0:
        return 0.0
    return intersection / union

def compute_roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    
    Args:
        probs (np.ndarray): Predicted probabilities for the positive class.
        labels (np.ndarray): Ground truth labels.
        
    Returns:
        float: ROC AUC score.
    """
    # ROC AUC requires both classes in labels to be present
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, probs)

def dirichlet_energy(x: torch.Tensor, edge_index: torch.Tensor) -> float:
    """
    Measure over-smoothing using Dirichlet energy.
    E = mean of ||x_i - x_j||^2 over all edges.
    
    Args:
        x (torch.Tensor): Node embeddings of shape [N, F].
        edge_index (torch.Tensor): Edge indices of shape [2, E].
        
    Returns:
        float: Dirichlet energy.
    """
    src, dst = edge_index
    diff = x[src] - x[dst]
    energy = (diff.norm(dim=1) ** 2).mean().item()
    return energy

def avg_pairwise_cosine_sim(x: torch.Tensor, sample_size: int = 1000) -> float:
    """
    Compute the average cosine similarity on a random sample of node pairs.
    
    Args:
        x (torch.Tensor): Node embeddings of shape [N, F].
        sample_size (int): Number of pairs to sample.
        
    Returns:
        float: Average pairwise cosine similarity.
    """
    num_nodes = x.size(0)
    if num_nodes < 2:
        return 1.0
        
    # Sample random pairs
    idx1 = torch.randint(0, num_nodes, (sample_size,))
    idx2 = torch.randint(0, num_nodes, (sample_size,))
    
    # Filter out self-pairs
    valid = idx1 != idx2
    if not valid.any():
        return 1.0
        
    idx1 = idx1[valid]
    idx2 = idx2[valid]
    
    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(x[idx1], x[idx2], dim=1)
    return cos_sim.mean().item()

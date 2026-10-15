import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from typing import Tuple, Optional

def build_pirm_graph(intensity_features: np.ndarray, k: int = 8, sigma: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a PIRM graph using k-NN in intensity feature space.
    
    Args:
        intensity_features (np.ndarray): Node features.
        k (int): Number of nearest neighbors.
        sigma (float): Gaussian similarity kernel width.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - edge_index of shape [2, E]
            - edge_weight of shape [E]
    """
    # Ensure k is not larger than the number of nodes - 1
    k = min(k, len(intensity_features) - 1)
    
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(intensity_features)
    distances, indices = nbrs.kneighbors(intensity_features)
    
    # Exclude self-loops (the first neighbor is usually the node itself with distance 0)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    num_nodes = intensity_features.shape[0]
    
    # Construct edge index
    sources = np.repeat(np.arange(num_nodes), k)
    targets = indices.flatten()
    edge_index = np.vstack([sources, targets])
    
    # Construct edge weight using Gaussian similarity
    edge_weight = np.exp(-(distances.flatten() ** 2) / (2 * sigma ** 2))
    
    return edge_index, edge_weight

def symmetrize_graph(edge_index: np.ndarray, edge_weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetrize graph by adding reverse edges and averaging duplicate weights.
    
    Args:
        edge_index (np.ndarray): Edge indices [2, E].
        edge_weight (np.ndarray): Edge weights [E].
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Symmetrized edge_index and edge_weight.
    """
    # Create directed edges mapping
    edges_dict = {}
    
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        w = edge_weight[i]
        
        # Ensure consistent ordering for undirected edge
        edge = (min(u, v), max(u, v))
        if edge not in edges_dict:
            edges_dict[edge] = []
        edges_dict[edge].append(w)
        
    new_edge_index = []
    new_edge_weight = []
    
    for (u, v), weights in edges_dict.items():
        avg_w = np.mean(weights)
        # Add both directions
        new_edge_index.append([u, v])
        new_edge_index.append([v, u])
        new_edge_weight.extend([avg_w, avg_w])
        
    new_edge_index = np.array(new_edge_index).T
    new_edge_weight = np.array(new_edge_weight)
    
    return new_edge_index, new_edge_weight

def build_pyg_data(node_features: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray, 
                   centroids: np.ndarray, node_labels: Optional[np.ndarray] = None) -> Data:
    """
    Package arrays into a PyTorch Geometric Data object.
    
    Args:
        node_features (np.ndarray): Node features.
        edge_index (np.ndarray): Edge indices.
        edge_weight (np.ndarray): Edge weights.
        centroids (np.ndarray): Node positions.
        node_labels (Optional[np.ndarray]): Node labels for classification.
        
    Returns:
        Data: PyG Data object.
    """
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_weight, dtype=torch.float)
    pos = torch.tensor(centroids, dtype=torch.float)
    
    y = None
    if node_labels is not None:
        y = torch.tensor(node_labels, dtype=torch.long)
        
    data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr, pos=pos, y=y)
    
    return data

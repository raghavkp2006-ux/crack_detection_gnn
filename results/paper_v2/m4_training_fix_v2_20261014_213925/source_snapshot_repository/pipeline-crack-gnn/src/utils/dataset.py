"""
PyTorch Geometric dataset loader for cached crack graph .pt files.
"""

import os
import glob
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch.utils.data import Subset
from typing import List, Tuple


class CrackGraphDataset(Dataset):
    """
    Loads cached .pt graph files from a flat directory.

    Each .pt file is a PyG Data object with:
        - x: [N, d_in] node features
        - edge_index: [2, E] edge indices
        - edge_attr: [E] or [E, 1] edge weights
        - pos: [N, 2] normalized centroid coordinates
        - y: [N] binary node labels (0=no-crack, 1=crack)
    """

    def __init__(self, root: str, transform=None, pre_transform=None):
        """
        Args:
            root: Directory containing .pt graph files.
        """
        self._graph_dir = root
        self._file_list = sorted(glob.glob(os.path.join(root, "*.pt")))
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return [os.path.basename(f) for f in self._file_list]

    def download(self):
        pass

    def process(self):
        pass

    def len(self) -> int:
        """Returns the number of graphs in the dataset."""
        return len(self._file_list)

    def get(self, idx: int) -> Data:
        """
        Gets the graph Data object at the specified index.

        Args:
            idx: Index to retrieve.

        Returns:
            PyG Data object.
        """
        data = torch.load(self._file_list[idx], weights_only=False)
        return data


def get_splits(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create a random train/val/test index split.

    Args:
        dataset: The PyG dataset.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    train_end = int(train_ratio * num_samples)
    val_end = train_end + int(val_ratio * num_samples)

    train_indices = indices[:train_end].tolist()
    val_indices = indices[train_end:val_end].tolist()
    test_indices = indices[val_end:].tolist()

    return train_indices, val_indices, test_indices

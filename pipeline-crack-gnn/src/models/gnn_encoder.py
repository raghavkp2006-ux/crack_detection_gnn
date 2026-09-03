import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm
import torch.nn.functional as F

class DeepGNN(nn.Module):
    """
    Baseline deep GNN to demonstrate over-smoothing.
    Uses SAGEConv layers without residual connections.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 8, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
            
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 2)
        self.layer_outputs = []

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for DeepGNN.
        Returns logits of shape [N, 2].
        """
        self.layer_outputs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            self.layer_outputs.append(x)
            
        logits = self.head(x)
        return logits


class ShallowGNN(nn.Module):
    """
    Shallow GNN with configurable number of layers and optional residual connections.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3, use_residual: bool = True):
        super().__init__()
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
            
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 2)
        self.layer_outputs = []

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ShallowGNN.
        Returns logits of shape [N, 2].
        """
        self.layer_outputs = []
        for i in range(self.num_layers):
            prev_x = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            
            if self.use_residual and self.num_layers >= 2 and i > 0:
                if prev_x.shape == x.shape:
                    x = x + prev_x
            
            self.layer_outputs.append(x)
            
        logits = self.head(x)
        return logits

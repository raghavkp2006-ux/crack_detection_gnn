import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm
import torch.nn.functional as F

from .gnn_encoder import DeepGNN, ShallowGNN
from .positional_encoding import PositionalEncodingFusion
from .transformer_block import TransformerStack

class HybridGNNTransformer(nn.Module):
    """
    Hybrid GNN + Transformer model for pipeline crack detection.
    Combines a shallow GNN encoder with Positional Encoding and a Transformer stack.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, gnn_layers: int = 2, 
                 transformer_layers: int = 2, heads: int = 4, pe_dim: int = 32, 
                 num_freqs: int = 8, gnn_dropout: float = 0.3, transformer_dropout: float = 0.1, 
                 use_pe: bool = True):
        super().__init__()
        self.use_pe = use_pe
        self.gnn_layers = gnn_layers
        
        # Shallow GNN encoder
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if gnn_layers > 0:
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
            for _ in range(gnn_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                self.bns.append(BatchNorm(hidden_dim))
        else:
            self.proj = nn.Linear(in_dim, hidden_dim)
            
        self.gnn_dropout = nn.Dropout(gnn_dropout)
        
        # Positional Encoding Fusion
        if use_pe:
            self.pe_fusion = PositionalEncodingFusion(hidden_dim=hidden_dim, pe_dim=pe_dim, num_freqs=num_freqs)
            
        # Transformer Stack
        self.transformer = TransformerStack(dim=hidden_dim, num_layers=transformer_layers, 
                                            heads=heads, dropout=transformer_dropout)
                                            
        # Classification head
        self.head = nn.Linear(hidden_dim, 2)
        
        self.gnn_layer_outputs = []

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Hybrid model.
        Returns logits of shape [N, 2].
        """
        self.gnn_layer_outputs = []
        
        # GNN encoding
        if self.gnn_layers > 0:
            for i in range(self.gnn_layers):
                prev_x = x
                x = self.convs[i](x, edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                x = self.gnn_dropout(x)
                
                # Residual connection
                if self.gnn_layers >= 2 and i > 0:
                    if prev_x.shape == x.shape:
                        x = x + prev_x
                        
                self.gnn_layer_outputs.append(x)
        else:
            x = self.proj(x)
            
        # PE fusion
        if self.use_pe:
            x = self.pe_fusion(x, pos)
            
        # Transformer
        x = self.transformer(x)
        
        # Head
        logits = self.head(x)
        return logits


def build_model(config: dict) -> nn.Module:
    """
    Factory function to build a model based on the configuration.
    
    Args:
        config: Dictionary containing model configuration parameters
        
    Returns:
        Instantiated PyTorch module
    """
    model_config = config.get('model', {})
    model_type = model_config.get('type')
    
    in_dim = model_config.get('in_dim', 1)
    hidden_dim = model_config.get('hidden_dim', 64)
    dropout = model_config.get('dropout', 0.3)
    
    if model_type == 'deep_gnn':
        num_layers = model_config.get('num_layers', 8)
        return DeepGNN(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        
    elif model_type == 'shallow_gnn':
        num_layers = model_config.get('num_layers', 2)
        use_residual = model_config.get('use_residual', True)
        return ShallowGNN(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, 
                          dropout=dropout, use_residual=use_residual)
                          
    elif model_type == 'hybrid':
        gnn_layers = model_config.get('gnn_layers', 2)
        transformer_layers = model_config.get('transformer_layers', 2)
        heads = model_config.get('heads', 4)
        pe_dim = model_config.get('pe_dim', 32)
        num_freqs = model_config.get('num_freqs', 8)
        transformer_dropout = model_config.get('transformer_dropout', 0.1)
        use_pe = model_config.get('use_pe', True)
        
        return HybridGNNTransformer(
            in_dim=in_dim, hidden_dim=hidden_dim, gnn_layers=gnn_layers,
            transformer_layers=transformer_layers, heads=heads, pe_dim=pe_dim,
            num_freqs=num_freqs, gnn_dropout=dropout, transformer_dropout=transformer_dropout,
            use_pe=use_pe
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

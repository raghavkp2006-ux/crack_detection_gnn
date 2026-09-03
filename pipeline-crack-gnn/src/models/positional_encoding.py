import torch
import torch.nn as nn
import math

def sinusoidal_pe(coords: torch.Tensor, num_freqs: int = 8) -> torch.Tensor:
    """
    Sinusoidal positional encoding for 2D centroid coordinates.
    
    Args:
        coords: Tensor of shape [N, 2] normalized to [0, 1]
        num_freqs: Number of frequency bands
        
    Returns:
        Tensor of shape [N, 4 * num_freqs]
    """
    device = coords.device
    freqs = (2 ** torch.arange(num_freqs, device=device)) * math.pi
    args = coords.unsqueeze(-1) * freqs  # [N, 2, num_freqs]
    
    sin_enc = torch.sin(args)
    cos_enc = torch.cos(args)
    
    # Concatenate sin and cos along the last dimension
    enc = torch.cat([sin_enc, cos_enc], dim=-1)  # [N, 2, 2 * num_freqs]
    
    # Flatten the last two dimensions to [N, 4 * num_freqs]
    enc = enc.view(coords.shape[0], -1)  
    
    return enc

class PositionalEncodingFusion(nn.Module):
    """
    Module to compute sinusoidal PE from 2D positions and fuse with node features.
    """
    def __init__(self, hidden_dim: int, pe_dim: int = 32, num_freqs: int = 8):
        super().__init__()
        self.num_freqs = num_freqs
        self.proj = nn.Linear(hidden_dim + pe_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Computes PE from positions, concatenates with features, and projects back.
        
        Args:
            x: Node features of shape [N, hidden_dim]
            pos: 2D Coordinates of shape [N, 2]
            
        Returns:
            Fused features of shape [N, hidden_dim]
        """
        pe = sinusoidal_pe(pos, num_freqs=self.num_freqs)
        x_concat = torch.cat([x, pe], dim=-1)
        x_fused = self.proj(x_concat)
        return x_fused

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer encoder block with optional relative positional bias.
    """
    def __init__(self, dim: int = 64, heads: int = 4, ffn_mult: int = 4, dropout: float = 0.1,
                 use_rel_bias: bool = False, init_temperature: float = 1.0):
        super().__init__()
        self.use_rel_bias = use_rel_bias
        if use_rel_bias:
            self.temperature = nn.Parameter(torch.tensor(init_temperature, dtype=torch.float32))
            
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, dist: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [N, dim]
            dist: Optional pairwise distance matrix of shape [N, N]
            
        Returns:
            Output tensor of shape [N, dim]
        """
        # MultiheadAttention with batch_first=True expects [batch_size, seq_len, embed_dim]
        # Unsqueeze to treat the graph as a single sequence of length N: [1, N, dim]
        x_seq = x.unsqueeze(0)
        
        # Relative positional bias: attn_mask = -dist / temperature
        attn_mask = None
        if self.use_rel_bias and dist is not None:
            temp = torch.clamp(self.temperature, min=1e-3)
            attn_mask = -dist / temp  # [N, N]
            
        # Pre-LN and attention
        normed_x = self.norm1(x_seq)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x, attn_mask=attn_mask)
        x_seq = x_seq + self.dropout(attn_out)
        
        # Pre-LN and FFN
        normed_x2 = self.norm2(x_seq)
        ffn_out = self.ffn(normed_x2)
        x_seq = x_seq + self.dropout(ffn_out)
        
        # Squeeze back to [N, dim]
        return x_seq.squeeze(0)


class TransformerStack(nn.Module):
    """
    Stack of multiple TransformerBlock layers with optional relative positional bias.
    """
    def __init__(self, dim: int = 64, num_layers: int = 2, heads: int = 4, ffn_mult: int = 4, 
                 dropout: float = 0.1, use_rel_bias: bool = False, init_temperature: float = 1.0):
        super().__init__()
        self.use_rel_bias = use_rel_bias
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim, heads=heads, ffn_mult=ffn_mult, dropout=dropout,
                use_rel_bias=use_rel_bias, init_temperature=init_temperature
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [N, dim]
            pos: Optional centroid coordinates [N, 2]
            
        Returns:
            Output tensor of shape [N, dim]
        """
        dist = None
        if self.use_rel_bias and pos is not None:
            dist = torch.cdist(pos, pos)  # [N, N]
            
        for layer in self.layers:
            x = layer(x, dist=dist)
        return x


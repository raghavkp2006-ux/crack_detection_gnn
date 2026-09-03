import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer encoder block.
    """
    def __init__(self, dim: int = 64, heads: int = 4, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [N, dim]
            
        Returns:
            Output tensor of shape [N, dim]
        """
        # MultiheadAttention with batch_first=True expects [batch_size, seq_len, embed_dim]
        # We unsqueeze to treat the whole graph as a single sequence of length N
        # Resulting shape: [1, N, dim]
        x_seq = x.unsqueeze(0)
        
        # Pre-LN and attention
        normed_x = self.norm1(x_seq)
        attn_out, _ = self.attn(normed_x, normed_x, normed_x)
        x_seq = x_seq + self.dropout(attn_out)
        
        # Pre-LN and FFN
        normed_x2 = self.norm2(x_seq)
        ffn_out = self.ffn(normed_x2)
        x_seq = x_seq + self.dropout(ffn_out)
        
        # Squeeze back to [N, dim]
        return x_seq.squeeze(0)


class TransformerStack(nn.Module):
    """
    Stack of multiple TransformerBlock layers.
    """
    def __init__(self, dim: int = 64, num_layers: int = 2, heads: int = 4, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim=dim, heads=heads, ffn_mult=ffn_mult, dropout=dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [N, dim]
            
        Returns:
            Output tensor of shape [N, dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x

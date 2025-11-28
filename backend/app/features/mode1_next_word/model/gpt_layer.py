"""
GPT Decoder Layer

Single GPT-style decoder layer (decoder-only, no cross-attention).
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from app.shared.attention.attention import MultiHeadAttention


class GPTDecoderLayer(nn.Module):
    """
    Single GPT-style decoder layer (decoder-only, no cross-attention).

    Components:
    1. Masked Self-Attention
    2. Feed-Forward Network
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Causal mask (batch_size, 1, seq_len, seq_len)

        Returns:
            output: Processed tensor
            viz_data: Visualization data
        """
        # Self-attention with residual connection
        attn_output, attn_viz_data = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        viz_data = {
            'attention': attn_viz_data,
            'output': x.detach().cpu().numpy().tolist()
        }

        return x, viz_data

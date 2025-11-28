"""
GPT Model

GPT-style decoder-only transformer for next token prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from app.shared.embeddings.embeddings import InputEmbedding
from app.shared.layers.layers import create_causal_mask
from .gpt_layer import GPTDecoderLayer


class GPTModel(nn.Module):
    """
    GPT-style decoder-only transformer for next token prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 100,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Embedding layer
        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            padding_idx=padding_idx
        )

        # Decoder layers
        self.layers = nn.ModuleList([
            GPTDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: Input token IDs (batch_size, seq_len)
            mask: Optional mask

        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
            viz_data: Complete visualization data
        """
        batch_size, seq_len = x.size()

        # Create causal mask
        if mask is None:
            mask = create_causal_mask(seq_len, x.device)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # Embedding
        embeddings, embedding_viz = self.embedding(x)

        # Store layer-wise outputs
        layer_viz_data = []
        current = embeddings

        for i, layer in enumerate(self.layers):
            current, layer_viz = layer(current, mask)
            layer_viz_data.append({
                'layer_idx': i,
                'attention': layer_viz['attention'],
                'output': layer_viz['output']
            })

        # Output projection
        logits = self.output_projection(current)

        # Compile visualization data
        viz_data = {
            'embedding': embedding_viz,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'layer_wise_details': layer_viz_data,
            'logits': logits.detach().cpu().numpy().tolist(),
            'final_hidden': current.detach().cpu().numpy().tolist()
        }

        return logits, viz_data

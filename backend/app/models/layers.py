"""
Transformer Layer Components

This module implements:
1. Feed-Forward Network: Position-wise transformation
2. Encoder Layer: Self-attention + FFN
3. Decoder Layer: Self-attention + Cross-attention + FFN

Educational Focus: Building blocks of the transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Applied to each position separately and identically.
    Consists of two linear transformations with a ReLU activation.

    Architecture: Linear -> ReLU -> Dropout -> Linear
    Typically: d_model -> d_ff (expansion) -> d_model (projection back)

    Common pattern: d_ff = 4 * d_model (expansion factor of 4)

    Why? Allows the model to learn complex non-linear transformations
    at each position independently.

    Args:
        d_model: Model dimension (e.g., 512)
        d_ff: Hidden dimension (e.g., 2048)
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # First linear layer: expand dimension
        self.linear1 = nn.Linear(d_model, d_ff)

        # Second linear layer: project back to d_model
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Apply position-wise feed-forward network.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            output: Transformed tensor (batch_size, seq_len, d_model)
            viz_data: Visualization data including activations
        """
        # Save for residual connection
        residual = x

        # Step 1: Expand to d_ff
        hidden = self.linear1(x)

        # Step 2: Apply ReLU activation
        activated = F.relu(hidden)

        # Step 3: Apply dropout
        activated = self.dropout(activated)

        # Step 4: Project back to d_model
        output = self.linear2(activated)

        # Step 5: Apply dropout
        output = self.dropout(output)

        # Step 6: Add residual connection and normalize
        output = self.layer_norm(output + residual)

        # Extract visualization data
        viz_data = {
            "input_shape": list(x.shape),
            "hidden_dim": self.d_ff,
            "expansion_factor": self.d_ff / self.d_model,
            "hidden_activations": hidden.detach().cpu().numpy().tolist(),
            "activated_values": activated.detach().cpu().numpy().tolist(),
            "output_shape": list(output.shape),
            "activation_stats": {
                "mean": float(activated.mean()),
                "std": float(activated.std()),
                "sparsity": float((activated == 0).float().mean()),  # % of zeros after ReLU
            }
        }

        return output, viz_data


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.

    Components (in order):
    1. Multi-Head Self-Attention
    2. Add & Normalize (residual connection + layer norm)
    3. Position-wise Feed-Forward Network
    4. Add & Normalize (residual connection + layer norm)

    The encoder processes the input sequence and learns contextual
    representations. Each token attends to all other tokens.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Process input through encoder layer.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional padding mask

        Returns:
            output: Encoded tensor (batch_size, seq_len, d_model)
            viz_data: Visualization data from both sublayers
        """
        # Sublayer 1: Multi-Head Self-Attention
        # In self-attention, Q, K, V all come from the same source (x)
        attn_output, attn_viz = self.self_attention(
            q=x, k=x, v=x, mask=mask
        )

        # Sublayer 2: Feed-Forward Network
        output, ff_viz = self.feed_forward(attn_output)

        # Combine visualization data
        viz_data = {
            "sublayer_1_self_attention": attn_viz,
            "sublayer_2_feed_forward": ff_viz,
            "final_output": output.detach().cpu().numpy().tolist()
        }

        return output, viz_data


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.

    Components (in order):
    1. Masked Multi-Head Self-Attention (causal)
    2. Add & Normalize
    3. Multi-Head Cross-Attention (attend to encoder output)
    4. Add & Normalize
    5. Position-wise Feed-Forward Network
    6. Add & Normalize

    The decoder generates output tokens autoregressively.
    Masked attention ensures each position can only attend to earlier positions.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Masked self-attention (for autoregressive generation)
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        # Cross-attention (attend to encoder output)
        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )

        # Feed-forward network
        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Process input through decoder layer.

        Args:
            x: Decoder input (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output (batch_size, src_seq_len, d_model)
            src_mask: Mask for source sequence (padding)
            tgt_mask: Mask for target sequence (causal + padding)

        Returns:
            output: Decoded tensor (batch_size, tgt_seq_len, d_model)
            viz_data: Visualization data from all sublayers
        """
        # Sublayer 1: Masked Self-Attention
        # Decoder attends to its own previous outputs (with causal masking)
        self_attn_output, self_attn_viz = self.self_attention(
            q=x, k=x, v=x, mask=tgt_mask
        )

        # Sublayer 2: Cross-Attention
        # Decoder attends to encoder output
        # Q comes from decoder, K and V come from encoder
        cross_attn_output, cross_attn_viz = self.cross_attention(
            q=self_attn_output,
            k=encoder_output,
            v=encoder_output,
            mask=src_mask
        )

        # Sublayer 3: Feed-Forward Network
        output, ff_viz = self.feed_forward(cross_attn_output)

        # Combine visualization data
        viz_data = {
            "sublayer_1_masked_self_attention": self_attn_viz,
            "sublayer_2_cross_attention": cross_attn_viz,
            "sublayer_3_feed_forward": ff_viz,
            "final_output": output.detach().cpu().numpy().tolist()
        }

        return output, viz_data


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (autoregressive) mask for decoder self-attention.

    The mask prevents positions from attending to future positions.
    Position i can only attend to positions <= i.

    Returns:
        Mask of shape (seq_len, seq_len) where mask[i][j] = True if i >= j, else False

    Example for seq_len=4:
        [[True, False, False, False],
         [True, True, False, False],
         [True, True, True, False],
         [True, True, True, True]]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Create mask for padded positions.

    Args:
        seq: Input sequence (batch_size, seq_len)
        pad_idx: Padding token index

    Returns:
        Mask of shape (batch_size, 1, 1, seq_len) where mask[i][j] = 1 if not padding
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask

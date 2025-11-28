"""
Attention Mechanisms for Transformer

This module implements:
1. Scaled Dot-Product Attention: Core attention mechanism
2. Multi-Head Attention: Parallel attention with different learned projections

Educational Focus: Understanding "Attention is All You Need"
This is the HEART of the transformer!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention: The fundamental attention mechanism.

    Given Query (Q), Key (K), and Value (V) matrices:
    1. Compute attention scores: Q @ K^T
    2. Scale by sqrt(d_k) to prevent gradient vanishing
    3. Apply softmax to get attention weights
    4. Multiply weights with Values to get output

    Formula: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Why scaling? Large dot products push softmax into regions with tiny gradients.
    Scaling by sqrt(d_k) keeps values in a reasonable range.
    """

    def __init__(self, temperature: Optional[float] = None, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature  # Will be set to sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute scaled dot-product attention.

        Args:
            q: Query tensor (batch_size, n_heads, seq_len_q, d_k)
            k: Key tensor (batch_size, n_heads, seq_len_k, d_k)
            v: Value tensor (batch_size, n_heads, seq_len_v, d_v)
            mask: Optional mask (batch_size, 1, seq_len_q, seq_len_k)
                  - Used for padding or causal masking
                  - Values of 0 will be masked (set to -inf before softmax)

        Returns:
            output: Attention output (batch_size, n_heads, seq_len_q, d_v)
            viz_data: Visualization data including attention weights
        """
        d_k = q.size(-1)

        # Set temperature if not already set
        if self.temperature is None:
            self.temperature = math.sqrt(d_k)

        # Step 1: Compute attention scores (Q @ K^T)
        # Shape: (batch_size, n_heads, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # Step 2: Scale by sqrt(d_k)
        attn_scores = attn_scores / self.temperature

        # Step 3: Apply mask (if provided)
        if mask is not None:
            # Replace masked positions with large negative value
            # This makes them ~0 after softmax
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Step 4: Apply softmax to get attention weights
        # Softmax normalizes across the key dimension (last dimension)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout to attention weights (regularization)
        attn_weights_dropped = self.dropout(attn_weights)

        # Step 5: Apply attention weights to values
        # Shape: (batch_size, n_heads, seq_len_q, d_v)
        output = torch.matmul(attn_weights_dropped, v)

        # Extract visualization data
        viz_data = {
            "attention_scores": attn_scores.detach().cpu().numpy().tolist(),
            "attention_weights": attn_weights.detach().cpu().numpy().tolist(),
            "query_shape": list(q.shape),
            "key_shape": list(k.shape),
            "value_shape": list(v.shape),
            "output_shape": list(output.shape),
            "temperature": self.temperature,
            "has_mask": mask is not None
        }

        return output, viz_data


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention: Run multiple attention mechanisms in parallel.

    Instead of one attention function, we use multiple attention "heads".
    Each head learns different aspects of the relationships between tokens.

    Example: One head might learn syntactic relationships, another semantic.

    Process:
    1. Project Q, K, V into n_heads different subspaces
    2. Apply scaled dot-product attention in parallel for each head
    3. Concatenate all head outputs
    4. Apply final linear projection

    Args:
        d_model: Model dimension (e.g., 512)
        n_heads: Number of attention heads (e.g., 8)
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.d_v = d_model // n_heads  # Typically same as d_k

        # Linear projections for Q, K, V
        # We use single matrices and split into heads later (more efficient)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final output projection
        self.w_o = nn.Linear(d_model, d_model)

        # Core attention mechanism
        self.attention = ScaledDotProductAttention(
            temperature=math.sqrt(self.d_k),
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Apply multi-head attention.

        Args:
            q: Query (batch_size, seq_len_q, d_model)
            k: Key (batch_size, seq_len_k, d_model)
            v: Value (batch_size, seq_len_v, d_model)
            mask: Optional attention mask

        Returns:
            output: Attention output (batch_size, seq_len_q, d_model)
            viz_data: Detailed visualization data for each head
        """
        batch_size = q.size(0)
        seq_len_q = q.size(1)
        seq_len_k = k.size(1)

        # Save residual connection
        residual = q

        # Step 1: Linear projections and split into heads
        # Shape: (batch_size, seq_len, n_heads, d_k)
        # Then transpose to: (batch_size, n_heads, seq_len, d_k)
        q = self.w_q(q).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, seq_len_k, self.n_heads, self.d_v).transpose(1, 2)

        # Expand mask for heads if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len_q, seq_len_k)

        # Step 2: Apply scaled dot-product attention
        attn_output, attn_viz = self.attention(q, k, v, mask=mask)

        # Step 3: Concatenate heads
        # Transpose back and reshape to (batch_size, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )

        # Step 4: Apply final linear projection
        output = self.w_o(attn_output)
        output = self.dropout(output)

        # Step 5: Add residual connection and layer normalization
        output = self.layer_norm(output + residual)

        # Extract per-head visualization data
        viz_data = {
            "n_heads": self.n_heads,
            "d_k": self.d_k,
            "attention_per_head": self._extract_head_data(attn_viz),
            "q_projection": q.detach().cpu().numpy().tolist(),
            "k_projection": k.detach().cpu().numpy().tolist(),
            "v_projection": v.detach().cpu().numpy().tolist(),
            "concatenated_output": attn_output.detach().cpu().numpy().tolist(),
            "final_output": output.detach().cpu().numpy().tolist(),
        }

        return output, viz_data

    def _extract_head_data(self, attn_viz: dict) -> list:
        """
        Extract visualization data for each attention head separately.

        This allows users to inspect what each head is "paying attention to".
        """
        import numpy as np

        attn_weights = np.array(attn_viz["attention_weights"])

        # Handle the shape - attention_weights should be 4D
        # Shape: (batch_size, n_heads, seq_len_q, seq_len_k)
        # Squeeze out any singleton dimensions (e.g., from nested lists)
        attn_weights = np.squeeze(attn_weights)

        # If still not 4D after squeezing, add back necessary dimensions
        while len(attn_weights.shape) < 4:
            attn_weights = np.expand_dims(attn_weights, axis=0)

        if len(attn_weights.shape) != 4:
            # If still not 4D after processing, there's an issue
            raise ValueError(f"Expected 4D attention weights after processing, got shape: {attn_weights.shape}")

        batch_size, n_heads, seq_len_q, seq_len_k = attn_weights.shape

        head_data = []
        for head_idx in range(n_heads):
            head_data.append({
                "head_index": head_idx,
                "attention_weights": attn_weights[:, head_idx, :, :].tolist(),
                "avg_attention_entropy": self._compute_entropy(
                    attn_weights[:, head_idx, :, :]
                ),
                # Which tokens does this head focus on most?
                "max_attention_positions": np.argmax(
                    attn_weights[:, head_idx, :, :], axis=-1
                ).tolist()
            })

        return head_data

    @staticmethod
    def _compute_entropy(attn_weights: torch.Tensor) -> float:
        """
        Compute entropy of attention distribution.

        High entropy = attention is spread out (looking at many tokens)
        Low entropy = attention is focused (looking at few tokens)
        """
        import numpy as np

        # Avoid log(0)
        attn_weights = np.array(attn_weights) + 1e-9
        entropy = -np.sum(attn_weights * np.log(attn_weights), axis=-1)
        return float(np.mean(entropy))

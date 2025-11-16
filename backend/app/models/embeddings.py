"""
Embedding Layers for Transformer

This module implements:
1. Token Embeddings: Convert token IDs to dense vectors
2. Positional Encodings: Add position information to embeddings

Educational Focus: Understanding how text becomes meaningful vectors
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class TokenEmbedding(nn.Module):
    """
    Convert token IDs to dense vector representations.

    In transformers, each token (word/subword) is mapped to a learned vector
    in a continuous space. Similar tokens should have similar embeddings.

    Args:
        vocab_size: Number of unique tokens in vocabulary
        d_model: Dimension of embedding vectors (typically 512 or 768)
        padding_idx: Token ID used for padding (default: 0)
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Convert token IDs to embeddings.

        Args:
            x: Token IDs of shape (batch_size, seq_len)

        Returns:
            embeddings: Shape (batch_size, seq_len, d_model)
            viz_data: Dictionary containing visualization data
        """
        # Scale embeddings by sqrt(d_model) - this is from "Attention is All You Need"
        # Helps maintain reasonable variance across the network
        embeddings = self.embedding(x) * math.sqrt(self.d_model)

        # Extract visualization data
        viz_data = {
            "token_ids": x.detach().cpu().numpy().tolist(),
            "embeddings": embeddings.detach().cpu().numpy().tolist(),
            "embedding_norm": torch.norm(embeddings, dim=-1).detach().cpu().numpy().tolist(),
            "vocab_size": self.embedding.num_embeddings,
            "d_model": self.d_model
        }

        return embeddings, viz_data


class PositionalEncoding(nn.Module):
    """
    Add positional information to token embeddings using sinusoidal functions.

    Since transformers process all positions in parallel (unlike RNNs),
    we need to explicitly encode position information. The sinusoidal
    approach allows the model to learn relative positions easily.

    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Dimension of embeddings
        max_len: Maximum sequence length to pre-compute
        dropout: Dropout rate for regularization
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Create positional encoding matrix
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # Compute the div_term for the sinusoidal functions
        # This creates different frequencies for each dimension
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Add positional encoding to embeddings.

        Args:
            x: Token embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            output: Position-aware embeddings (batch_size, seq_len, d_model)
            viz_data: Dictionary containing visualization data
        """
        batch_size, seq_len, _ = x.size()

        # Add positional encoding to embeddings
        # We only use the first seq_len positions
        positional_encoding = self.pe[:, :seq_len, :]
        x = x + positional_encoding

        # Extract visualization data
        viz_data = {
            "positional_encoding": positional_encoding.detach().cpu().numpy().tolist(),
            "embeddings_with_position": x.detach().cpu().numpy().tolist(),
            "sequence_length": seq_len,
            "encoding_pattern": self._get_encoding_pattern(seq_len)
        }

        output = self.dropout(x)

        return output, viz_data

    def _get_encoding_pattern(self, seq_len: int) -> dict:
        """
        Extract encoding patterns for visualization.

        Returns frequency information for different dimensions.
        """
        pe_slice = self.pe[0, :seq_len, :].detach().cpu().numpy()

        return {
            "full_pattern": pe_slice.tolist(),
            "first_dim": pe_slice[:, 0].tolist(),  # Lowest frequency
            "middle_dim": pe_slice[:, self.d_model // 2].tolist(),
            "last_dim": pe_slice[:, -1].tolist(),  # Highest frequency
        }


class InputEmbedding(nn.Module):
    """
    Complete input embedding: Token Embedding + Positional Encoding.

    This combines token embeddings with positional information to create
    the final input representations for the transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Complete embedding transformation.

        Args:
            x: Token IDs (batch_size, seq_len)

        Returns:
            output: Complete embeddings (batch_size, seq_len, d_model)
            viz_data: Combined visualization data from both components
        """
        # Step 1: Token embedding
        token_emb, token_viz = self.token_embedding(x)

        # Step 2: Add positional encoding
        output, pos_viz = self.positional_encoding(token_emb)

        # Combine visualization data
        viz_data = {
            "step_1_token_embedding": token_viz,
            "step_2_positional_encoding": pos_viz,
            "final_output": output.detach().cpu().numpy().tolist()
        }

        return output, viz_data

"""
Complete Transformer Architecture

This module implements the full transformer model as described in
"Attention is All You Need" (Vaswani et al., 2017).

Educational Focus: How all components work together for sequence-to-sequence tasks
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List

from .embeddings import InputEmbedding
from .layers import EncoderLayer, DecoderLayer, create_causal_mask, create_padding_mask


class TransformerEncoder(nn.Module):
    """
    Stack of N encoder layers.

    The encoder processes the input sequence and produces contextualized
    representations. Each layer refines these representations.

    Args:
        vocab_size: Size of source vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        d_ff: Feed-forward hidden dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
        padding_idx: Padding token index
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        # Input embedding layer
        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            padding_idx=padding_idx
        )

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Encode source sequence.

        Args:
            src: Source token IDs (batch_size, src_seq_len)
            src_mask: Source padding mask (batch_size, 1, 1, src_seq_len)

        Returns:
            output: Encoded representations (batch_size, src_seq_len, d_model)
            viz_data: Comprehensive visualization data from all layers
        """
        # Create padding mask if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src)

        # Step 1: Embed input tokens
        x, embedding_viz = self.embedding(src)

        # Step 2: Pass through encoder layers
        layer_outputs = []
        layer_viz_data = []

        for i, layer in enumerate(self.layers):
            x, layer_viz = layer(x, mask=src_mask)
            layer_outputs.append(x.detach().cpu().numpy().tolist())
            layer_viz_data.append(layer_viz)

        # Compile visualization data
        viz_data = {
            "embedding": embedding_viz,
            "n_layers": self.n_layers,
            "layer_wise_outputs": layer_outputs,
            "layer_wise_details": layer_viz_data,
            "final_output": x.detach().cpu().numpy().tolist()
        }

        return x, viz_data


class TransformerDecoder(nn.Module):
    """
    Stack of N decoder layers.

    The decoder generates output tokens autoregressively, attending to both
    its own previous outputs and the encoder's output.

    Args:
        vocab_size: Size of target vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of decoder layers
        d_ff: Feed-forward hidden dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
        padding_idx: Padding token index
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        # Input embedding layer
        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            padding_idx=padding_idx
        )

        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Decode target sequence.

        Args:
            tgt: Target token IDs (batch_size, tgt_seq_len)
            encoder_output: Encoder output (batch_size, src_seq_len, d_model)
            src_mask: Source padding mask
            tgt_mask: Target causal + padding mask

        Returns:
            output: Logits over vocabulary (batch_size, tgt_seq_len, vocab_size)
            viz_data: Comprehensive visualization data
        """
        batch_size, tgt_seq_len = tgt.size()

        # Create causal mask for autoregressive generation
        if tgt_mask is None:
            # Combine causal mask with padding mask
            causal_mask = create_causal_mask(tgt_seq_len, tgt.device)
            padding_mask = create_padding_mask(tgt)
            tgt_mask = causal_mask & padding_mask

        # Step 1: Embed target tokens
        x, embedding_viz = self.embedding(tgt)

        # Step 2: Pass through decoder layers
        layer_outputs = []
        layer_viz_data = []

        for i, layer in enumerate(self.layers):
            x, layer_viz = layer(
                x=x,
                encoder_output=encoder_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )
            layer_outputs.append(x.detach().cpu().numpy().tolist())
            layer_viz_data.append(layer_viz)

        # Step 3: Project to vocabulary
        logits = self.output_projection(x)

        # Compile visualization data
        viz_data = {
            "embedding": embedding_viz,
            "n_layers": self.n_layers,
            "layer_wise_outputs": layer_outputs,
            "layer_wise_details": layer_viz_data,
            "logits": logits.detach().cpu().numpy().tolist(),
            "predicted_tokens": torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
        }

        return logits, viz_data


class Transformer(nn.Module):
    """
    Complete Transformer Model for Sequence-to-Sequence Tasks.

    Classic architecture from "Attention is All You Need":
    - Encoder processes source sequence
    - Decoder generates target sequence autoregressively
    - Cross-attention allows decoder to attend to encoder output

    Common Use Cases:
    - Machine Translation (e.g., English → French)
    - Text Summarization
    - Question Answering

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension (default: 512)
        n_heads: Number of attention heads (default: 8)
        n_encoder_layers: Number of encoder layers (default: 6)
        n_decoder_layers: Number of decoder layers (default: 6)
        d_ff: Feed-forward hidden dimension (default: 2048)
        max_len: Maximum sequence length (default: 5000)
        dropout: Dropout rate (default: 0.1)
        padding_idx: Padding token index (default: 0)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            padding_idx=padding_idx
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            padding_idx=padding_idx
        )

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """
        Initialize model parameters using Xavier uniform initialization.

        This helps with training stability and convergence.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Complete forward pass through transformer.

        Args:
            src: Source token IDs (batch_size, src_seq_len)
            tgt: Target token IDs (batch_size, tgt_seq_len)
            src_mask: Source padding mask
            tgt_mask: Target causal + padding mask

        Returns:
            output: Logits over target vocabulary (batch_size, tgt_seq_len, tgt_vocab_size)
            viz_data: Complete visualization data from encoder and decoder
        """
        # Step 1: Encode source sequence
        encoder_output, encoder_viz = self.encoder(src, src_mask)

        # Step 2: Decode target sequence
        decoder_output, decoder_viz = self.decoder(
            tgt=tgt,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )

        # Compile complete visualization data
        viz_data = {
            "source_sequence": src.detach().cpu().numpy().tolist(),
            "target_sequence": tgt.detach().cpu().numpy().tolist(),
            "encoder": encoder_viz,
            "decoder": decoder_viz,
            "architecture_info": {
                "d_model": self.encoder.d_model,
                "n_encoder_layers": self.encoder.n_layers,
                "n_decoder_layers": self.decoder.n_layers,
            }
        }

        return decoder_output, viz_data

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence only."""
        encoder_output, _ = self.encoder(src, src_mask)
        return encoder_output

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence given encoder output."""
        decoder_output, _ = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return decoder_output

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 50,
        start_token: int = 1,
        end_token: int = 2,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Generate target sequence autoregressively (for inference).

        Args:
            src: Source token IDs (batch_size, src_seq_len)
            max_len: Maximum generation length
            start_token: Start-of-sequence token ID
            end_token: End-of-sequence token ID
            src_mask: Source padding mask

        Returns:
            generated: Generated token IDs (batch_size, generated_len)
            step_viz: Visualization data for each generation step
        """
        batch_size = src.size(0)
        device = src.device

        # Encode source once
        encoder_output = self.encode(src, src_mask)

        # Start with start token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        step_viz = []

        for step in range(max_len):
            # Decode current sequence
            logits = self.decode(
                tgt=generated,
                encoder_output=encoder_output,
                src_mask=src_mask
            )

            # Get next token prediction (greedy decoding)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch_size, 1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Store visualization data for this step
            step_viz.append({
                "step": step,
                "generated_so_far": generated.detach().cpu().numpy().tolist(),
                "next_token_logits": next_token_logits.detach().cpu().numpy().tolist(),
                "next_token": next_token.item()
            })

            # Stop if all sequences have generated end token
            if (next_token == end_token).all():
                break

        return generated, step_viz

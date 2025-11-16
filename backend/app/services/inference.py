"""
Transformer Inference Service

Handles model loading, inference, and result formatting for the API.
"""

import torch
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..models.transformer import Transformer


class TransformerService:
    """
    Service for running transformer inference and extracting visualization data.

    This class manages:
    - Model initialization and loading
    - Tokenization (simple character-level for demo)
    - Forward pass execution
    - Visualization data extraction
    """

    def __init__(
        self,
        src_vocab_size: int = 1000,
        tgt_vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        d_ff: int = 1024,
        max_len: int = 100,
        dropout: float = 0.1,
        device: Optional[str] = None
    ):
        """
        Initialize transformer service.

        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension (smaller for educational purposes)
            n_heads: Number of attention heads
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        ).to(self.device)

        # Set to evaluation mode
        self.model.eval()

        # Simple character-level tokenizer for demo
        # In production, use tiktoken, sentencepiece, or transformers tokenizer
        self.char_to_idx = {}
        self.idx_to_char = {}
        self._init_simple_tokenizer()

    def _init_simple_tokenizer(self):
        """
        Initialize a simple character-level tokenizer for demonstration.

        Special tokens:
        - 0: <PAD>
        - 1: <SOS> (start of sequence)
        - 2: <EOS> (end of sequence)
        - 3: <UNK> (unknown)
        """
        self.char_to_idx = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

        # Add common characters
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:\'"-()'
        for idx, char in enumerate(chars, start=4):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char

    def tokenize(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add <SOS> and <EOS>

        Returns:
            List of token IDs
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.char_to_idx['<SOS>'])

        for char in text:
            tokens.append(self.char_to_idx.get(char, self.char_to_idx['<UNK>']))

        if add_special_tokens:
            tokens.append(self.char_to_idx['<EOS>'])

        return tokens

    def detokenize(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        chars = []
        for token_id in token_ids:
            char = self.idx_to_char.get(token_id, '<UNK>')
            if char not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                chars.append(char)
        return ''.join(chars)

    @torch.no_grad()
    def run_inference(
        self,
        source_text: str,
        target_text: Optional[str] = None,
        generate: bool = False,
        max_gen_len: int = 50
    ) -> Dict:
        """
        Run transformer inference and extract visualization data.

        Args:
            source_text: Source text to encode
            target_text: Target text (for teacher forcing) or None
            generate: Whether to generate target autoregressively
            max_gen_len: Maximum generation length

        Returns:
            Dictionary containing:
            - source_tokens: Source token IDs
            - target_tokens: Target token IDs (or generated)
            - output_logits: Model output logits
            - visualization_data: Complete viz data from all layers
            - decoded_output: Decoded output text
        """
        # Tokenize source
        src_tokens = self.tokenize(source_text)
        src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=self.device)

        if generate:
            # Autoregressive generation
            generated_ids, step_viz = self.model.generate(
                src=src_tensor,
                max_len=max_gen_len,
                start_token=self.char_to_idx['<SOS>'],
                end_token=self.char_to_idx['<EOS>']
            )

            # Get final visualization by running forward pass
            tgt_tensor = generated_ids
            output_logits, viz_data = self.model(src_tensor, tgt_tensor)

            decoded_output = self.detokenize(generated_ids[0].tolist())

            return {
                "source_text": source_text,
                "source_tokens": src_tokens,
                "target_tokens": generated_ids[0].tolist(),
                "output_logits": output_logits[0].detach().cpu().numpy().tolist(),
                "visualization_data": viz_data,
                "generation_steps": step_viz,
                "decoded_output": decoded_output,
                "mode": "generation"
            }

        else:
            # Teacher forcing mode
            if target_text is None:
                target_text = source_text  # Echo task for demo

            tgt_tokens = self.tokenize(target_text)
            tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long, device=self.device)

            # Forward pass
            output_logits, viz_data = self.model(src_tensor, tgt_tensor)

            # Get predictions
            predicted_ids = torch.argmax(output_logits, dim=-1)[0].tolist()
            decoded_output = self.detokenize(predicted_ids)

            return {
                "source_text": source_text,
                "target_text": target_text,
                "source_tokens": src_tokens,
                "target_tokens": tgt_tokens,
                "predicted_tokens": predicted_ids,
                "output_logits": output_logits[0].detach().cpu().numpy().tolist(),
                "visualization_data": viz_data,
                "decoded_output": decoded_output,
                "mode": "teacher_forcing"
            }

    def get_model_info(self) -> Dict:
        """
        Get model architecture information.

        Returns:
            Dictionary with model specifications
        """
        return {
            "architecture": "Transformer (Encoder-Decoder)",
            "d_model": self.model.encoder.d_model,
            "n_encoder_layers": self.model.encoder.n_layers,
            "n_decoder_layers": self.model.decoder.n_layers,
            "vocab_size": len(self.char_to_idx),
            "device": str(self.device),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

    def load_pretrained(self, checkpoint_path: str):
        """
        Load pretrained model weights.

        Args:
            checkpoint_path: Path to model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded pretrained model from {checkpoint_path}")

    def save_checkpoint(self, save_path: str, metadata: Optional[Dict] = None):
        """
        Save model checkpoint.

        Args:
            save_path: Path to save checkpoint
            metadata: Optional metadata to save with checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'metadata': metadata or {}
        }
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

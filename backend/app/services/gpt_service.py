"""
GPT-Style Next Word Prediction Service (Mode 1)

Implements decoder-only transformer for next token prediction,
similar to GPT architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..models.embeddings import InputEmbedding
from ..models.layers import create_causal_mask
from ..models.attention import MultiHeadAttention


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


class GPTService:
    """
    Service for GPT-style next word prediction with visualization.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_len: int = 100,
        dropout: float = 0.1,
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize GPT service.

        Args:
            vocab_size: Vocabulary size (ignored if checkpoint_path is provided)
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of layers
            d_ff: Feed-forward dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
            device: Device to use
            checkpoint_path: Path to pretrained checkpoint (optional)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize tokenizer first (might be loaded from checkpoint)
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.tokenization_level = 'char'  # Default to character-level

        # Load from checkpoint if provided
        if checkpoint_path is not None:
            self._load_from_checkpoint(checkpoint_path)
        else:
            # Initialize new model with simple tokenizer
            self.model = GPTModel(
                vocab_size=vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                max_len=max_len,
                dropout=dropout
            ).to(self.device)

            self._init_simple_tokenizer()

        self.model.eval()

    def _load_from_checkpoint(self, checkpoint_path: str):
        """
        Load model and tokenizer from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        import os

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract model config
        model_config = checkpoint.get('model_config', {})
        vocab_size = len(checkpoint['tokenizer_vocab'])

        # Get max_len from checkpoint state dict (from positional encoding)
        pe_shape = checkpoint['model_state_dict']['embedding.positional_encoding.pe'].shape
        max_len_from_checkpoint = pe_shape[1]  # [1, max_len, d_model]

        # Initialize model with checkpoint config
        self.model = GPTModel(
            vocab_size=vocab_size,
            d_model=model_config.get('d_model', 256),
            n_heads=model_config.get('n_heads', 4),
            n_layers=model_config.get('n_layers', 4),
            d_ff=model_config.get('d_ff', 1024),
            max_len=max_len_from_checkpoint,
            dropout=0.1,
            padding_idx=0
        ).to(self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load tokenizer vocabulary
        self.char_to_idx = checkpoint['tokenizer_vocab']
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.tokenization_level = checkpoint.get('tokenizer_level', 'char')  # Get tokenization level from checkpoint

        print(f"[OK] Loaded checkpoint (epoch {checkpoint.get('epoch', 'unknown')}, loss: {checkpoint.get('loss', 0):.4f})")
        print(f"[OK] Vocabulary size: {vocab_size}")
        print(f"[OK] Tokenization level: {self.tokenization_level}")

    def _init_simple_tokenizer(self):
        """Initialize character-level tokenizer."""
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
        """Tokenize text to IDs."""
        tokens = []

        if add_special_tokens:
            tokens.append(self.char_to_idx['<SOS>'])

        if self.tokenization_level == 'word':
            # Word-level tokenization
            import re
            # Simple word tokenization (same as in training)
            text = re.sub(r'([.,!?;:])', r' \1 ', text)
            words = text.lower().split()
            for word in words:
                tokens.append(self.char_to_idx.get(word, self.char_to_idx['<UNK>']))
        else:
            # Character-level tokenization
            for char in text:
                tokens.append(self.char_to_idx.get(char, self.char_to_idx['<UNK>']))

        return tokens

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs to text."""
        tokens = []
        for token_id in token_ids:
            token = self.idx_to_char.get(token_id, '<UNK>')
            if token not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                tokens.append(token)

        if self.tokenization_level == 'word':
            # Word-level: join with spaces
            return ' '.join(tokens)
        else:
            # Character-level: concatenate
            return ''.join(tokens)

    @torch.no_grad()
    def predict_next_word(self, input_text: str) -> Dict:
        """
        Predict next token based on input text.

        Args:
            input_text: Input text context

        Returns:
            Dictionary with prediction results and visualization data
        """
        # Tokenize
        tokens = self.tokenize(input_text, add_special_tokens=True)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Forward pass
        logits, viz_data = self.model(input_tensor)

        # Get predictions for the last position
        last_logits = logits[0, -1, :]  # (vocab_size,)

        # Apply softmax to get probabilities
        probabilities = torch.softmax(last_logits, dim=0)

        # Get top predictions
        top_k = 10
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(probabilities)))

        top_predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            token = self.idx_to_char.get(int(idx), '<UNK>')
            top_predictions.append({
                'token': token,
                'token_id': int(idx),
                'probability': float(prob)
            })

        # Get predicted token
        predicted_idx = top_indices[0].item()
        predicted_token = self.idx_to_char.get(predicted_idx, '<UNK>')

        # Extract step-by-step visualization data
        steps_data = self._extract_steps_visualization(
            tokens=tokens,
            viz_data=viz_data,
            logits=logits[0].cpu().numpy(),
            probabilities=probabilities.cpu().numpy()
        )

        return {
            'input_text': input_text,
            'predicted_token': predicted_token,
            'predicted_word': predicted_token,  # For char-level, word = token
            'confidence': float(top_probs[0]),
            'top_predictions': top_predictions,
            'steps': steps_data,
            'raw_visualization': viz_data
        }

    def _extract_steps_visualization(
        self,
        tokens: List[int],
        viz_data: Dict,
        logits: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict:
        """
        Extract step-by-step visualization data for frontend.
        """
        # Step 1: Tokenization
        token_strings = [self.idx_to_char.get(t, '<UNK>') for t in tokens]

        # Step 2: Embeddings
        embedding_data = viz_data['embedding']
        # The embedding forward returns 'final_output' which is already a list
        if 'final_output' in embedding_data:
            final_output_list = embedding_data['final_output']
            # It's a batch, so get the first item
            if isinstance(final_output_list, list) and len(final_output_list) > 0:
                embeddings_array = np.array(final_output_list[0])  # First batch item
                embedding_shape = list(embeddings_array.shape)
                # Get sample values (first 16 dimensions for each token)
                sample_values = embeddings_array[:, :16].tolist() if embeddings_array.shape[1] >= 16 else embeddings_array.tolist()
            else:
                embedding_shape = [len(tokens), self.model.d_model]
                sample_values = [[0.0] * 16 for _ in tokens]
        else:
            embedding_shape = [len(tokens), self.model.d_model]
            sample_values = [[0.0] * 16 for _ in tokens]

        # Step 3: Attention
        attention_shape = [len(tokens), len(tokens)]

        # Step 4: Feedforward
        ff_shape = [len(tokens), self.model.d_model]

        # Step 5: Output layer
        logits_shape = list(logits.shape)
        softmax_shape = list(probabilities.shape)

        return {
            'tokenization': {
                'tokens': token_strings,
                'token_ids': tokens
            },
            'embeddings': {
                'shape': embedding_shape,
                'sample_values': sample_values
            },
            'attention': {
                'num_heads': self.model.n_heads,
                'num_layers': self.model.n_layers,
                'attention_shape': attention_shape
            },
            'feedforward': {
                'hidden_dim': self.model.layers[0].feed_forward[0].out_features if len(self.model.layers) > 0 else 1024,
                'output_shape': ff_shape
            },
            'output': {
                'logits_shape': logits_shape,
                'softmax_shape': softmax_shape
            }
        }

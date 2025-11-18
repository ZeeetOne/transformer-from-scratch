"""
Dataset utilities for GPT-style language model training.

Prepares text corpus for next-word prediction using shifted target technique.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import re


class TextDataset(Dataset):
    """
    Dataset for language modeling with next-word prediction.

    Uses shifted target technique:
    - Input: [token1, token2, token3, token4]
    - Target: [token2, token3, token4, token5]

    Each token must predict the next token in sequence.
    """

    def __init__(
        self,
        text_data: str,
        tokenizer: 'Tokenizer',
        max_seq_len: int = 50,
        min_seq_len: int = 5
    ):
        """
        Args:
            text_data: Raw text corpus
            tokenizer: Tokenizer instance (character or word level)
            max_seq_len: Maximum sequence length
            min_seq_len: Minimum sequence length to include
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len

        # Tokenize entire corpus
        self.tokens = tokenizer.tokenize(text_data, add_special_tokens=False)

        # Create sequences using sliding window
        self.sequences = self._create_sequences()

        print(f"Dataset created:")
        print(f"  Total tokens: {len(self.tokens)}")
        print(f"  Total sequences: {len(self.sequences)}")
        print(f"  Vocabulary size: {len(tokenizer.vocab)}")

    def _create_sequences(self) -> List[List[int]]:
        """
        Create training sequences from tokens using sliding window.

        For efficient training, we create overlapping sequences.
        """
        sequences = []

        # Sliding window with stride
        stride = max(1, self.max_seq_len // 2)  # 50% overlap

        for i in range(0, len(self.tokens) - self.min_seq_len, stride):
            # Get sequence of max_seq_len + 1 (extra token for target)
            end_idx = min(i + self.max_seq_len + 1, len(self.tokens))
            seq = self.tokens[i:end_idx]

            if len(seq) >= self.min_seq_len + 1:  # Need at least 1 token for target
                sequences.append(seq)

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get training pair: (input_sequence, target_sequence)

        Returns:
            input_ids: Input token IDs [seq_len]
            target_ids: Target token IDs [seq_len] (shifted by 1)
        """
        seq = self.sequences[idx]

        # Add <SOS> token at the beginning
        input_seq = [self.tokenizer.char_to_idx['<SOS>']] + seq[:-1]
        target_seq = seq  # Target is shifted by 1 (next token)

        # Convert to tensors
        input_ids = torch.tensor(input_seq, dtype=torch.long)
        target_ids = torch.tensor(target_seq, dtype=torch.long)

        return input_ids, target_ids


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader to handle variable-length sequences.

    Pads sequences to the same length within a batch.
    """
    input_ids_list, target_ids_list = zip(*batch)

    # Find max length in this batch
    max_len = max(len(ids) for ids in input_ids_list)

    # Pad sequences
    padded_inputs = []
    padded_targets = []

    for input_ids, target_ids in zip(input_ids_list, target_ids_list):
        # Pad to max_len
        pad_len = max_len - len(input_ids)

        if pad_len > 0:
            # Pad with <PAD> token (index 0)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            target_ids = torch.cat([target_ids, torch.zeros(pad_len, dtype=torch.long)])

        padded_inputs.append(input_ids)
        padded_targets.append(target_ids)

    # Stack into batches
    input_batch = torch.stack(padded_inputs)
    target_batch = torch.stack(padded_targets)

    return input_batch, target_batch


class SimpleTokenizer:
    """
    Simple character-level or word-level tokenizer.
    """

    def __init__(self, tokenization_level: str = 'char'):
        """
        Args:
            tokenization_level: 'char' or 'word'
        """
        self.level = tokenization_level
        self.char_to_idx: Dict[str, int] = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }
        self.idx_to_char: Dict[int, str] = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_built = False

    def build_vocab(self, text: str):
        """Build vocabulary from text corpus."""
        if self.level == 'char':
            # Character-level: each character is a token
            chars = sorted(set(text))
            for idx, char in enumerate(chars, start=len(self.char_to_idx)):
                if char not in self.char_to_idx:
                    self.char_to_idx[char] = idx
                    self.idx_to_char[idx] = char
        else:
            # Word-level: split by whitespace and punctuation
            words = self._word_tokenize(text)
            unique_words = sorted(set(words))
            for idx, word in enumerate(unique_words, start=len(self.char_to_idx)):
                if word not in self.char_to_idx:
                    self.char_to_idx[word] = idx
                    self.idx_to_char[idx] = word

        self.vocab_built = True
        print(f"Vocabulary built: {len(self.char_to_idx)} tokens ({self.level}-level)")

    def _word_tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        # Split on whitespace and keep punctuation separate
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        words = text.lower().split()
        return words

    def tokenize(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Convert text to token IDs."""
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        tokens = []

        if add_special_tokens:
            tokens.append(self.char_to_idx['<SOS>'])

        if self.level == 'char':
            for char in text:
                tokens.append(self.char_to_idx.get(char, self.char_to_idx['<UNK>']))
        else:
            words = self._word_tokenize(text)
            for word in words:
                tokens.append(self.char_to_idx.get(word, self.char_to_idx['<UNK>']))

        if add_special_tokens:
            tokens.append(self.char_to_idx['<EOS>'])

        return tokens

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = []
        for token_id in token_ids:
            token = self.idx_to_char.get(token_id, '<UNK>')
            if token not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                tokens.append(token)

        if self.level == 'char':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)

    @property
    def vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        return self.char_to_idx

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.char_to_idx)


def load_text_corpus(file_path: str) -> str:
    """Load text corpus from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def create_dataloader(
    text_data: str,
    tokenizer: SimpleTokenizer,
    batch_size: int = 32,
    max_seq_len: int = 50,
    shuffle: bool = True
) -> DataLoader:
    """
    Create DataLoader for training.

    Args:
        text_data: Raw text corpus
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    dataset = TextDataset(
        text_data=text_data,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch
    )

    return dataloader

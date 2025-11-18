"""
Training service for GPT-style language model.

Implements complete training loop with:
- Cross-entropy loss calculation
- Backpropagation and optimization
- Model checkpoint saving/loading
- Training metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import os
import json
from pathlib import Path
import time

from ..services.gpt_service import GPTModel
from .dataset import SimpleTokenizer


class GPTTrainer:
    """
    Trainer for GPT-style language model.

    Handles complete training pipeline from data loading to model saving.
    """

    def __init__(
        self,
        model: GPTModel,
        tokenizer: SimpleTokenizer,
        device: Optional[str] = None,
        learning_rate: float = 1e-4,
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Args:
            model: GPT model instance
            tokenizer: Tokenizer instance
            device: Device to train on
            learning_rate: Learning rate for optimizer
            checkpoint_dir: Directory to save checkpoints
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # Loss function: Cross-Entropy Loss
        # Ignores padding tokens (index 0) in loss calculation
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Optimizer: Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': 0
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            epoch: Current epoch number

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            # Move to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits, _ = self.model(input_ids)

            # Reshape for cross-entropy loss
            # logits: (batch_size, seq_len, vocab_size)
            # target_ids: (batch_size, seq_len)
            batch_size, seq_len, vocab_size = logits.shape

            # Flatten: (batch_size * seq_len, vocab_size) and (batch_size * seq_len)
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = target_ids.view(-1)

            # Calculate loss
            loss = self.criterion(logits_flat, targets_flat)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate on validation set.

        Args:
            dataloader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass
            logits, _ = self.model(input_ids)

            # Calculate loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = target_ids.view(-1)

            loss = self.criterion(logits_flat, targets_flat)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        save_every: int = 5
    ):
        """
        Complete training loop.

        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs to train
            val_loader: Optional validation data loader
            save_every: Save checkpoint every N epochs
        """
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        print(f"Training batches: {len(train_loader)}")
        print("=" * 60)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            start_time = time.time()

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)

                # Update learning rate based on validation loss
                self.scheduler.step(val_loss)

                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Time: {time.time() - start_time:.2f}s")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt', epoch, val_loss)
                    print(f"  [BEST] Best model saved (val_loss: {val_loss:.4f})")
            else:
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Time: {time.time() - start_time:.2f}s")

            # Save periodic checkpoints
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt', epoch, train_loss)

            self.history['epochs'] = epoch + 1

        # Save final model
        self.save_checkpoint('final_model.pt', num_epochs - 1, train_loss)
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

    def save_checkpoint(self, filename: str, epoch: int, loss: float):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            epoch: Current epoch
            loss: Current loss
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history,
            'tokenizer_vocab': self.tokenizer.vocab,
            'tokenizer_level': self.tokenizer.level,
            'model_config': {
                'd_model': self.model.d_model,
                'n_heads': self.model.n_heads,
                'n_layers': self.model.n_layers,
            }
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)

        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Loss: {checkpoint['loss']:.4f}")

        return checkpoint

    @torch.no_grad()
    def generate_sample(self, prompt: str, max_length: int = 50) -> str:
        """
        Generate text sample from prompt.

        Args:
            prompt: Input prompt text
            max_length: Maximum tokens to generate

        Returns:
            Generated text
        """
        self.model.eval()

        # Tokenize prompt
        tokens = self.tokenizer.tokenize(prompt, add_special_tokens=True)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        generated = tokens.copy()

        for _ in range(max_length):
            # Forward pass
            logits, _ = self.model(input_ids)

            # Get logits for last token
            next_token_logits = logits[0, -1, :]

            # Sample next token (greedy for now)
            next_token = torch.argmax(next_token_logits).item()

            # Stop if <EOS>
            if next_token == self.tokenizer.char_to_idx.get('<EOS>', 2):
                break

            # Add to sequence
            generated.append(next_token)

            # Update input
            input_ids = torch.tensor([generated], dtype=torch.long, device=self.device)

        # Decode
        return self.tokenizer.detokenize(generated)


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.

    Perplexity = exp(loss)

    Lower perplexity = better model
    """
    return torch.exp(torch.tensor(loss)).item()

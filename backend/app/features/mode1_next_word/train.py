"""
Training Script for GPT-Style Language Model (Mode 1)

This script trains a Mini-GPT model from scratch on a text corpus.

Usage (from backend/ directory):
    python -m app.features.mode1_next_word.train --corpus app/features/mode1_next_word/data/sample_corpus.txt --epochs 50

The trained model will be saved in the checkpoints/ directory.
"""

import argparse
import sys
from pathlib import Path

import torch
from .model.gpt_model import GPTModel
from .training.dataset import SimpleTokenizer, load_text_corpus, create_dataloader
from .training.trainer import GPTTrainer


def main():
    # Get script directory for relative paths
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description='Train GPT-style language model (Mode 1)')
    parser.add_argument('--corpus', type=str,
                        default=str(script_dir / 'data' / 'sample_corpus.txt'),
                        help='Path to training corpus (.txt file)')
    parser.add_argument('--level', type=str, default='word', choices=['char', 'word'],
                        help='Tokenization level: char or word')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--d-model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--d-ff', type=int, default=1024,
                        help='Feed-forward dimension')
    parser.add_argument('--max-seq-len', type=int, default=50,
                        help='Maximum sequence length')
    parser.add_argument('--checkpoint-dir', type=str,
                        default=str(script_dir / 'checkpoints'),
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, default: auto)')

    args = parser.parse_args()

    print("=" * 70)
    print("GPT-STYLE LANGUAGE MODEL TRAINING")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Corpus: {args.corpus}")
    print(f"  Tokenization: {args.level}-level")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Model Dimension: {args.d_model}")
    print(f"  Attention Heads: {args.n_heads}")
    print(f"  Transformer Layers: {args.n_layers}")
    print(f"  Feed-Forward Dimension: {args.d_ff}")
    print(f"  Max Sequence Length: {args.max_seq_len}")
    print("=" * 70)

    # Step 1: Load and prepare data
    print("\nStep 1: Loading corpus...")
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {args.corpus}")
        sys.exit(1)

    text_data = load_text_corpus(str(corpus_path))
    print(f"[OK] Loaded {len(text_data)} characters")

    # Step 2: Build tokenizer
    print("\nStep 2: Building tokenizer...")
    tokenizer = SimpleTokenizer(tokenization_level=args.level)
    tokenizer.build_vocab(text_data)

    # Step 3: Create data loaders with 80/20 train/validation split
    print("\nStep 3: Creating data loaders with validation split...")

    # Tokenize the entire corpus
    all_tokens = tokenizer.tokenize(text_data, add_special_tokens=False)

    # Split tokens into train (80%) and validation (20%)
    train_size = int(0.8 * len(all_tokens))
    train_tokens = all_tokens[:train_size]
    val_tokens = all_tokens[train_size:]

    # Reconstruct text from tokens for each split
    train_text = tokenizer.detokenize(train_tokens)
    val_text = tokenizer.detokenize(val_tokens)

    print(f"  Train tokens: {len(train_tokens)}")
    print(f"  Validation tokens: {len(val_tokens)}")

    # Create train dataloader
    train_loader = create_dataloader(
        text_data=train_text,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        shuffle=True
    )

    # Create validation dataloader
    val_loader = create_dataloader(
        text_data=val_text,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        shuffle=False  # Don't shuffle validation data
    )

    # Step 4: Initialize model
    print("\nStep 4: Initializing model...")
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_len=args.max_seq_len + 10,
        dropout=0.1,
        padding_idx=0
    )

    print(f"[OK] Model initialized")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Step 5: Initialize trainer
    print("\nStep 5: Initializing trainer...")
    trainer = GPTTrainer(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )

    # Step 6: Train model with validation
    print("\nStep 6: Training model with validation...")
    try:
        trainer.train(
            train_loader=train_loader,
            num_epochs=args.epochs,
            val_loader=val_loader,  # Use validation split
            save_every=10
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model...")
        trainer.save_checkpoint('interrupted_model.pt',
                                trainer.history['epochs'],
                                trainer.history['train_loss'][-1] if trainer.history['train_loss'] else 0)

    # Step 7: Test the trained model
    print("\n" + "=" * 70)
    print("TESTING TRAINED MODEL")
    print("=" * 70)

    test_prompts = [
        "I eat",
        "You eat",
        "She likes",
        "The weather is",
        "Today is"
    ]

    for prompt in test_prompts:
        generated = trainer.generate_sample(prompt, max_length=10)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print()

    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to: {args.checkpoint_dir}/")
    print(f"  - best_model.pt (best validation loss)")
    print(f"  - final_model.pt (last epoch)")
    print(f"\nTo use this model for inference:")
    print(f"  1. Copy best_model.pt to a known location")
    print(f"  2. Update GPTService to load this checkpoint")
    print(f"  3. Restart the API server")


if __name__ == '__main__':
    main()

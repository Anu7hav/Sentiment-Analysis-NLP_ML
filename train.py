"""
Sentiment Analysis using Deep Learning — Main Training Script
Trains LSTM and Transformer classifiers; evaluates with accuracy and F1-score.

Usage:
    python train.py --model all          # Train both LSTM and Transformer
    python train.py --model lstm         # Train BiLSTM + Attention only
    python train.py --model transformer  # Train Transformer only
    python train.py --data path/to/imdb.csv
"""

import argparse
import torch
from torch.utils.data import DataLoader

from utils.data_utils import (
    generate_sample_data, load_imdb, prepare_data, collate_fn
)
from utils.trainer import SentimentTrainer
from models.lstm_classifier import LSTMClassifier, SimpleLSTMClassifier
from models.transformer_classifier import TransformerClassifier


BATCH_SIZE = 64
MAX_LEN    = 256
N_EPOCHS   = 10
PATIENCE   = 3


def train_lstm(train_ds, val_ds, test_ds, vocab_size):
    print("\n" + "="*55)
    print("BiLSTM + Attention Classifier")
    print("="*55)

    model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_classes=2,
        dropout=0.3,
        use_attention=True
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, collate_fn=collate_fn)

    trainer = SentimentTrainer(model, lr=2e-4, num_classes=2)
    trainer.fit(train_loader, val_loader, n_epochs=N_EPOCHS,
                patience=PATIENCE, save_path='best_lstm.pt')

    print("\n--- Test Set Results ---")
    trainer.print_classification_report(test_loader)
    metrics, _, _ = trainer.evaluate(test_loader)
    print(f"Accuracy: {metrics['Accuracy']:.4f} | F1: {metrics['F1']:.4f} | "
          f"Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f}")
    return trainer, metrics


def train_transformer(train_ds, val_ds, test_ds, vocab_size):
    print("\n" + "="*55)
    print("Transformer Encoder Classifier")
    print("="*55)

    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=3,
        ffn_dim=256,
        max_len=MAX_LEN,
        num_classes=2,
        dropout=0.1
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE*2, collate_fn=collate_fn)

    trainer = SentimentTrainer(model, lr=1e-4, num_classes=2)
    trainer.fit(train_loader, val_loader, n_epochs=N_EPOCHS,
                patience=PATIENCE, save_path='best_transformer.pt')

    print("\n--- Test Set Results ---")
    trainer.print_classification_report(test_loader)
    metrics, _, _ = trainer.evaluate(test_loader)
    print(f"Accuracy: {metrics['Accuracy']:.4f} | F1: {metrics['F1']:.4f} | "
          f"Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f}")
    return trainer, metrics


def demo_inference(trainer, vocab, model_name='Model'):
    """Run a few example predictions."""
    print(f"\n--- {model_name} Inference Demo ---")
    samples = [
        "This movie was absolutely fantastic! The acting was superb.",
        "Terrible film. Complete waste of time, boring and poorly written.",
        "It was okay, nothing special but not bad either.",
        "One of the best movies I have ever seen. Highly recommended!",
    ]
    for text in samples:
        result = trainer.predict(text, vocab)
        print(f"  [{result['label']:8s} {result['confidence']*100:.1f}%] {text[:60]}...")


def main(args):
    print("Sentiment Analysis using Deep Learning (PyTorch)")

    # Load data
    if args.data:
        df = load_imdb(args.data)
    else:
        print("No data file specified — using synthetic data.")
        print("For real training: download IMDB dataset from Kaggle and pass --data path/to/imdb.csv\n")
        df = generate_sample_data(n=3000)

    # Prepare datasets
    train_ds, val_ds, test_ds, vocab = prepare_data(
        df, max_len=MAX_LEN, min_freq=2
    )
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size:,}")

    results = {}

    if args.model in ('all', 'lstm'):
        lstm_trainer, lstm_metrics = train_lstm(train_ds, val_ds, test_ds, vocab_size)
        results['BiLSTM+Attn'] = lstm_metrics
        demo_inference(lstm_trainer, vocab, 'BiLSTM+Attention')

    if args.model in ('all', 'transformer'):
        tf_trainer, tf_metrics = train_transformer(train_ds, val_ds, test_ds, vocab_size)
        results['Transformer'] = tf_metrics
        demo_inference(tf_trainer, vocab, 'Transformer')

    # Summary
    if len(results) > 1:
        print("\n" + "="*55)
        print("Model Comparison Summary")
        print("="*55)
        print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10} {'Precision':>12} {'Recall':>10}")
        print("-"*55)
        for name, m in results.items():
            print(f"{name:<20} {m['Accuracy']:>10.4f} {m['F1']:>10.4f} "
                  f"{m['Precision']:>12.4f} {m['Recall']:>10.4f}")

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all',
                        choices=['all', 'lstm', 'transformer'])
    parser.add_argument('--data', default=None,
                        help='Path to IMDB CSV (reviewerID, sentiment columns)')
    args = parser.parse_args()
    main(args)

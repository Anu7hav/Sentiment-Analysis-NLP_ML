"""
LSTM-based Sentiment Classifier (PyTorch)
Supports unidirectional and bidirectional LSTM with attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    """Additive attention over LSTM hidden states."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden*2)
        energy = torch.tanh(self.attn(lstm_out))   # (batch, seq_len, hidden)
        scores = self.v(energy).squeeze(-1)         # (batch, seq_len)
        weights = F.softmax(scores, dim=1)          # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden*2)
        return context, weights


class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM + Attention for Sentiment Classification.

    Architecture:
        Embedding → BiLSTM → Attention → Dropout → FC → Sigmoid/Softmax
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, num_classes=2, dropout=0.3,
                 pad_idx=0, use_attention=True, pretrained_embeddings=None):
        super().__init__()
        self.use_attention = use_attention
        self.num_classes = num_classes

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(
                torch.FloatTensor(pretrained_embeddings)
            )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )

        self.attention = Attention(hidden_dim) if use_attention else None
        self.dropout = nn.Dropout(dropout)

        lstm_out_dim = hidden_dim * 2  # bidirectional

        if not use_attention:
            lstm_out_dim = hidden_dim * 2  # last hidden state

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes if num_classes > 2 else 1)
        )

    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))   # (batch, seq, embed)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, _) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, _) = self.lstm(embedded)

        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)
            out = self.dropout(context)
        else:
            # Concatenate final forward & backward hidden states
            out = torch.cat([hidden[-2], hidden[-1]], dim=1)
            out = self.dropout(out)

        logits = self.fc(out)

        if self.num_classes == 2:
            return logits.squeeze(-1)   # Binary: BCEWithLogitsLoss
        return logits                   # Multi-class: CrossEntropyLoss


class SimpleLSTMClassifier(nn.Module):
    """
    Simple unidirectional LSTM for ablation / baseline comparison.
    """

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128,
                 num_layers=1, num_classes=2, dropout=0.5, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1 if num_classes == 2 else num_classes)

    def forward(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        out = self.dropout(hidden[-1])
        return self.fc(out).squeeze(-1)

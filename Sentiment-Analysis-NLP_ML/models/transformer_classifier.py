"""
Transformer-based Sentiment Classification (PyTorch)
Implements a lightweight Transformer encoder from scratch + optional BERT fine-tuning wrapper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head attention."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        weights = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.out_proj(out)


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block: Attention + FFN + LayerNorm."""

    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerClassifier(nn.Module):
    """
    Lightweight Transformer Encoder for Sentiment Classification.

    Architecture:
        Embedding + PosEnc → N x EncoderBlock → CLS pooling → FC → Logits

    Uses [CLS] token (index 1) pooling, same as BERT convention.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3,
                 ffn_dim=256, max_len=256, num_classes=2, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1 if num_classes == 2 else num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        for p in self.fc.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (batch, seq_len) token indices
        mask = (x != self.pad_idx)  # (batch, seq_len)

        x_emb = self.pos_enc(self.token_emb(x) * math.sqrt(self.d_model))

        out = x_emb
        for layer in self.encoder:
            out = layer(out, mask)

        out = self.norm(out)

        # Mean pooling over non-pad tokens
        mask_expanded = mask.unsqueeze(-1).float()
        pooled = (out * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled).squeeze(-1)


# ─────────────────────────────────────────────────────────────────
# HuggingFace BERT Wrapper (requires: pip install transformers)
# ─────────────────────────────────────────────────────────────────

try:
    from transformers import BertModel

    class BERTSentimentClassifier(nn.Module):
        """
        Fine-tune BERT for binary/multi-class sentiment classification.
        Uses bert-base-uncased with a classification head on [CLS].
        """

        def __init__(self, num_classes=2, dropout=0.3, freeze_bert=False):
            super().__init__()
            self.bert = BertModel.from_pretrained('bert-base-uncased')

            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False

            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(768, 1 if num_classes == 2 else num_classes)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            cls = self.dropout(out.pooler_output)   # [CLS] representation
            return self.fc(cls).squeeze(-1)

except ImportError:
    pass  # transformers not installed; BERT wrapper unavailable

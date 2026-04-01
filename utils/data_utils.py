"""
Text Preprocessing Pipeline for Sentiment Analysis
Covers: cleaning, tokenization, vocabulary building, and Dataset classes.
"""

import re
import string
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# ─────────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────────

def clean_text(text: str, remove_stopwords=False) -> str:
    """
    Standard NLP text cleaning pipeline.
    Steps: lowercase → remove HTML → remove URLs → expand contractions
           → remove punctuation/digits → collapse whitespace
    """
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)                     # HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)              # URLs
    text = re.sub(r"n't", " not", text)                      # contractions
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r'[^a-z\s]', ' ', text)                   # keep letters only
    text = re.sub(r'\s+', ' ', text).strip()

    if remove_stopwords:
        STOPWORDS = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'is', 'it', 'of', 'that', 'this', 'was', 'with', 'as',
            'by', 'from', 'are', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        }
        text = ' '.join(w for w in text.split() if w not in STOPWORDS)

    return text


# ─────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────

class Vocabulary:
    """
    Word-level vocabulary with frequency-based filtering.
    Special tokens: <PAD>=0, <UNK>=1, <CLS>=2, <SEP>=3
    """
    PAD, UNK, CLS, SEP = 0, 1, 2, 3
    SPECIAL = ['<PAD>', '<UNK>', '<CLS>', '<SEP>']

    def __init__(self, min_freq=2, max_size=30000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {}
        self.idx2word = {}

    def build(self, texts: list) -> 'Vocabulary':
        """Build vocab from list of cleaned text strings."""
        counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Keep min_freq words, sort by frequency
        vocab_words = [w for w, c in counter.most_common() if c >= self.min_freq]
        vocab_words = vocab_words[:self.max_size - len(self.SPECIAL)]

        all_tokens = self.SPECIAL + vocab_words
        self.word2idx = {w: i for i, w in enumerate(all_tokens)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        print(f"Vocabulary size: {len(self.word2idx):,} tokens "
              f"(covering {len(vocab_words):,} words, min_freq={self.min_freq})")
        return self

    def encode(self, text: str, max_len=256, add_special=True) -> list:
        """Convert text string to list of token indices."""
        tokens = text.split()[:max_len - (2 if add_special else 0)]
        indices = [self.word2idx.get(t, self.UNK) for t in tokens]
        if add_special:
            indices = [self.CLS] + indices + [self.SEP]
        return indices

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────────
# Dataset Classes
# ─────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification."""

    def __init__(self, texts: list, labels: list, vocab: Vocabulary,
                 max_len=256):
        self.encodings = [vocab.encode(t, max_len=max_len) for t in texts]
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = torch.LongTensor(self.encodings[idx])
        label = torch.FloatTensor([self.labels[idx]])
        return ids, label


def collate_fn(batch):
    """Pad sequences to same length within a batch."""
    ids, labels = zip(*batch)
    padded = pad_sequence(ids, batch_first=True, padding_value=0)
    lengths = torch.LongTensor([len(x) for x in ids])
    labels = torch.cat(labels)
    return padded, labels, lengths


# ─────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────

def load_imdb(filepath: str) -> pd.DataFrame:
    """
    Load IMDB movie reviews CSV.
    Download: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    Expected columns: review, sentiment (positive/negative)
    """
    df = pd.read_csv(filepath)
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    df['text'] = df['review'].apply(clean_text)
    print(f"IMDB: {len(df):,} reviews | Positive: {df['label'].sum():,}")
    return df


def load_sst2(filepath: str) -> pd.DataFrame:
    """
    Load SST-2 dataset (tab-separated: sentence, label).
    Download: https://gluebenchmark.com/tasks
    """
    df = pd.read_csv(filepath, sep='\t')
    df.columns = ['text', 'label']
    df['text'] = df['text'].apply(clean_text)
    return df


def generate_sample_data(n=2000, seed=42) -> pd.DataFrame:
    """Generate synthetic sentiment data for testing."""
    rng = np.random.RandomState(seed)
    positive_words = ['great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                      'love', 'brilliant', 'perfect', 'outstanding', 'superb']
    negative_words = ['terrible', 'awful', 'horrible', 'disappointing', 'boring',
                      'waste', 'poor', 'bad', 'worst', 'dreadful']
    filler = ['the', 'movie', 'film', 'was', 'really', 'quite', 'very',
              'this', 'it', 'a', 'an', 'and', 'of', 'in']

    texts, labels = [], []
    for _ in range(n):
        label = rng.randint(0, 2)
        words = rng.choice(positive_words if label else negative_words, size=rng.randint(3, 6)).tolist()
        words += rng.choice(filler, size=rng.randint(5, 10)).tolist()
        rng.shuffle(words)
        texts.append(' '.join(words))
        labels.append(label)

    return pd.DataFrame({'text': texts, 'label': labels})


def prepare_data(df: pd.DataFrame, text_col='text', label_col='label',
                 test_size=0.15, val_size=0.1, min_freq=2, max_len=256, seed=42):
    """
    Full data preparation pipeline:
    1. Train/val/test split
    2. Build vocabulary on train
    3. Return Dataset objects + vocab
    """
    train_val, test = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                  stratify=train_val[label_col], random_state=seed)

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    vocab = Vocabulary(min_freq=min_freq).build(train[text_col].tolist())

    train_ds = SentimentDataset(train[text_col].tolist(), train[label_col].tolist(), vocab, max_len)
    val_ds   = SentimentDataset(val[text_col].tolist(),   val[label_col].tolist(),   vocab, max_len)
    test_ds  = SentimentDataset(test[text_col].tolist(),  test[label_col].tolist(),  vocab, max_len)

    return train_ds, val_ds, test_ds, vocab

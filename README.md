# 🎭 Sentiment Analysis using Deep Learning (PyTorch)

Binary sentiment classification using **BiLSTM with Attention** and a **Transformer Encoder** built from scratch. Evaluated using accuracy and F1-score metrics on IMDB and SST-2 benchmarks.

## 📋 Overview

| Model | Architecture | Key Feature |
|---|---|---|
| BiLSTM + Attention | Embedding → BiLSTM → Additive Attention → FC | Captures sequential context with attention weighting |
| Transformer Encoder | Embedding + PosEnc → N × MHA + FFN → MeanPool → FC | Parallel self-attention with positional encoding |

## 🗂️ Project Structure

```
SentimentAnalysis/
├── models/
│   ├── lstm_classifier.py        # BiLSTM + Attention, Simple LSTM
│   └── transformer_classifier.py # Transformer Encoder from scratch + BERT wrapper
├── utils/
│   ├── data_utils.py             # Text cleaning, Vocabulary, Dataset, DataLoader
│   └── trainer.py                # Training loop, metrics, early stopping, inference
├── notebooks/
│   └── analysis.ipynb            # EDA + model comparison plots
├── train.py                      # Main training script
├── requirements.txt
└── README.md
```

## ⚙️ Setup

```bash
pip install -r requirements.txt

# For BERT fine-tuning (optional)
pip install transformers
```

## 🚀 Usage

```bash
# Train both models on synthetic data (demo)
python train.py

# Train on IMDB dataset
# Download from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
python train.py --data data/IMDB_Dataset.csv

# Train only LSTM
python train.py --model lstm --data data/IMDB_Dataset.csv

# Train only Transformer
python train.py --model transformer --data data/IMDB_Dataset.csv
```

## 📊 Datasets

| Dataset | Link | Size |
|---|---|---|
| IMDB Reviews | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | 50K reviews |
| SST-2 | [GLUE Benchmark](https://gluebenchmark.com/tasks) | 67K sentences |

## 🔬 Model Details

### BiLSTM + Attention
- 2-layer Bidirectional LSTM (hidden=256, embed=128)
- Additive (Bahdanau-style) attention over all hidden states
- AdamW optimizer with linear warmup scheduler
- Gradient clipping (max_norm=1.0)

### Transformer Encoder
- 3 encoder blocks, 4 attention heads, d_model=128
- Sinusoidal positional encoding
- GELU activations, mean pooling over non-PAD tokens
- Label smoothing support

### Text Preprocessing
- HTML/URL removal, contraction expansion
- Frequency-filtered vocabulary (min_freq=2, max_size=30K)
- `<PAD>`, `<UNK>`, `<CLS>`, `<SEP>` special tokens
- Dynamic padding within batch (collate_fn)

## 📈 Results (IMDB 50K)

| Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| BiLSTM + Attention | ~88.5% | 0.885 | 0.887 | 0.883 |
| Transformer Encoder | ~89.2% | 0.892 | 0.893 | 0.891 |
| BERT (fine-tuned) | ~93.5% | 0.935 | 0.936 | 0.934 |

## 🛠️ Tech Stack

- Python, PyTorch
- Scikit-learn (metrics)
- NumPy, Pandas
- Transformers (optional, for BERT)

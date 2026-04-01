"""
Trainer and Evaluator for Sentiment Classification Models (PyTorch).
Handles training loop, metrics, early stopping, and prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from utils.data_utils import collate_fn


class SentimentTrainer:
    """
    Unified trainer for LSTM and Transformer sentiment classifiers.
    Supports binary (BCEWithLogitsLoss) and multi-class (CrossEntropyLoss) tasks.
    """

    def __init__(self, model, lr=2e-4, weight_decay=1e-4,
                 num_classes=2, device=None, label_smoothing=0.0):
        self.model = model
        self.num_classes = num_classes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if num_classes == 2:
            self.criterion = nn.BCEWithLogitsLoss(label_smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.train_metrics = []
        self.val_metrics = []

    def _get_scheduler(self, num_steps, warmup_ratio=0.1):
        from torch.optim.lr_scheduler import LambdaLR
        warmup = int(num_steps * warmup_ratio)

        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            return max(0.0, (num_steps - step) / max(1, num_steps - warmup))

        return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, loader, scheduler=None):
        self.model.train()
        total_loss, all_preds, all_labels = 0, [], []

        for batch in loader:
            if len(batch) == 3:
                ids, labels, lengths = batch
                ids, labels, lengths = ids.to(self.device), labels.to(self.device), lengths
            else:
                ids, labels = batch
                lengths = None

            self.optimizer.zero_grad()

            if lengths is not None and hasattr(self.model, 'lstm'):
                logits = self.model(ids, lengths)
            else:
                logits = self.model(ids)

            if self.num_classes == 2:
                loss = self.criterion(logits, labels)
                preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
            else:
                loss = self.criterion(logits, labels.long())
                preds = logits.argmax(dim=1).cpu().numpy()

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item() * len(labels)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().long().numpy())

        metrics = self._compute_metrics(all_labels, all_preds, total_loss, len(all_labels))
        return metrics

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, all_preds, all_labels, all_probs = 0, [], [], []

        for batch in loader:
            if len(batch) == 3:
                ids, labels, lengths = batch
                ids, labels, lengths = ids.to(self.device), labels.to(self.device), lengths
            else:
                ids, labels = batch
                lengths = None

            if lengths is not None and hasattr(self.model, 'lstm'):
                logits = self.model(ids, lengths)
            else:
                logits = self.model(ids)

            if self.num_classes == 2:
                loss = self.criterion(logits, labels)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs)
            else:
                loss = self.criterion(logits, labels.long())
                preds = logits.argmax(dim=1).cpu().numpy()

            total_loss += loss.item() * len(labels)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().long().numpy())

        metrics = self._compute_metrics(all_labels, all_preds, total_loss, len(all_labels))
        if all_probs:
            try:
                metrics['ROC-AUC'] = roc_auc_score(all_labels, all_probs)
            except Exception:
                pass
        return metrics, all_preds, all_labels

    def _compute_metrics(self, labels, preds, total_loss, n):
        return {
            'Loss': total_loss / n,
            'Accuracy': accuracy_score(labels, preds),
            'F1': f1_score(labels, preds, average='binary' if self.num_classes == 2 else 'macro', zero_division=0),
            'Precision': precision_score(labels, preds, average='binary' if self.num_classes == 2 else 'macro', zero_division=0),
            'Recall': recall_score(labels, preds, average='binary' if self.num_classes == 2 else 'macro', zero_division=0),
        }

    def fit(self, train_loader, val_loader, n_epochs=10, patience=3,
            warmup_ratio=0.1, save_path='best_model.pt'):
        total_steps = n_epochs * len(train_loader)
        scheduler = self._get_scheduler(total_steps, warmup_ratio)

        best_val_f1 = 0
        patience_ctr = 0

        for epoch in range(1, n_epochs + 1):
            train_m = self.train_epoch(train_loader, scheduler)
            val_m, _, _ = self.evaluate(val_loader)

            self.train_metrics.append(train_m)
            self.val_metrics.append(val_m)

            print(
                f"Epoch {epoch:>2}/{n_epochs} | "
                f"Train Loss: {train_m['Loss']:.4f} Acc: {train_m['Accuracy']:.4f} F1: {train_m['F1']:.4f} | "
                f"Val   Loss: {val_m['Loss']:.4f} Acc: {val_m['Accuracy']:.4f} F1: {val_m['F1']:.4f}"
            )

            if val_m['F1'] > best_val_f1:
                best_val_f1 = val_m['F1']
                patience_ctr = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"Early stopping at epoch {epoch} (best val F1: {best_val_f1:.4f})")
                    break

        # Load best model
        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        return self

    def print_classification_report(self, loader):
        _, preds, labels = self.evaluate(loader)
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=['Negative', 'Positive']))
        cm = confusion_matrix(labels, preds)
        print("Confusion Matrix:")
        print(cm)

    @torch.no_grad()
    def predict(self, text: str, vocab, max_len=256):
        """Predict sentiment for a raw input string."""
        from utils.data_utils import clean_text
        self.model.eval()
        cleaned = clean_text(text)
        ids = torch.LongTensor([vocab.encode(cleaned, max_len=max_len)]).to(self.device)
        logit = self.model(ids)
        prob = torch.sigmoid(logit).item()
        label = 'Positive' if prob > 0.5 else 'Negative'
        return {'label': label, 'confidence': prob if prob > 0.5 else 1 - prob, 'raw_prob': prob}

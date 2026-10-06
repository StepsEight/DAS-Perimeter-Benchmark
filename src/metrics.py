"""Fixed-label multiclass metrics and publication-oriented confusion matrices."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

try:
    from .utils import dump_json
except ImportError:
    from utils import dump_json


def classification_metrics(y_true, y_pred, class_names):
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()),
            'macro_f1': float(f1.mean()),
            'per_class': [{'class_name': str(name), 'precision': float(p),
                           'recall': float(r), 'f1': float(f), 'support': int(n)}
                          for name, p, r, f, n in zip(class_names, precision, recall, f1, support)]}


def plot_confusion(matrix, class_names, path, normalized=False, title=None):
    values = matrix.astype(float)
    if normalized:
        values = np.divide(values, values.sum(1, keepdims=True),
                           out=np.zeros_like(values), where=values.sum(1, keepdims=True) > 0)
    fig, ax = plt.subplots(figsize=(6.8, 5.7), constrained_layout=True)
    display = ax.imshow(values, cmap='Blues', vmin=0,
                        vmax=1 if normalized else max(1, values.max()))
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel='Predicted class', ylabel='True class',
           title=title or ('Normalized confusion matrix' if normalized else 'Confusion matrix'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    for row in range(len(class_names)):
        for col in range(len(class_names)):
            label = f'{values[row,col]:.2f}' if normalized else str(int(values[row,col]))
            ax.text(col, row, label, ha='center', va='center',
                    color='white' if values[row,col] > values.max() * .55 else 'black', fontsize=10)
    fig.colorbar(display, ax=ax, fraction=.046, pad=.03)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return values


def save_evaluation(output_dir, split, y_true, y_pred, class_names, sample_ids, probabilities=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    true, predicted = np.asarray(y_true), np.asarray(y_pred)
    result = classification_metrics(true, predicted, class_names)
    result['n_samples'] = len(true)
    result['split'] = split
    dump_json(out / f'{split}_metrics.json', result)
    pd.DataFrame(result['per_class']).to_csv(out / f'{split}_per_class.csv', index=False)
    cm = confusion_matrix(true, predicted, labels=np.arange(len(class_names)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out / f'{split}_confusion_matrix.csv')
    normalized = plot_confusion(cm, class_names, out / f'{split}_confusion_matrix.png')
    norm = plot_confusion(cm, class_names, out / f'{split}_confusion_matrix_normalized.png', normalized=True)
    pd.DataFrame(norm, index=class_names, columns=class_names).to_csv(out / f'{split}_confusion_matrix_normalized.csv')
    frame = pd.DataFrame({'sample_id': sample_ids, 'true_class_index': true,
                          'predicted_class_index': predicted,
                          'true_class': [class_names[int(i)] for i in true],
                          'predicted_class': [class_names[int(i)] for i in predicted]})
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        for i, name in enumerate(class_names):
            frame[f'probability_{name}'] = probabilities[:, i]
        np.save(out / f'{split}_probabilities.npy', probabilities)
    frame.to_csv(out / f'{split}_predictions.csv', index=False)
    return result

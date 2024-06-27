import numpy as np
from scikit-learn import metrics


def metric(y_true, logits):
    y_true = y_true.reshape(-1)
    preds = np.argmax(logits, axis=1)
    return metrics.balanced_accuracy_score(y_true, preds)

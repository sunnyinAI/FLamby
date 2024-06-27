import numpy as np
from scikit-learn.metrics import accuracy_score


def metric(y_true, y_pred):
    y_true = y_true.astype("uint8")
    y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred)


# cv_learn/linear_classifiers/svm.py
import numpy as np

def svm_loss_vectorized(W, X, y, reg=0.0, delta=1.0):
    """
    W: (D, C) weights
    X: (N, D) data
    y: (N,) labels
    reg: L2 regularization strength
    delta: hinge margin (usually 1.0)

    Returns: loss (scalar), dW (D, C)
    """
    N = X.shape[0]
    scores = X @ W  # (N,C)
    correct = scores[np.arange(N), y][:, None]  # (N,1)
    margins = np.maximum(0.0, scores - correct + delta)  # (N,C)
    margins[np.arange(N), y] = 0.0
    loss = margins.sum() / N + reg * np.sum(W*W)

    # Gradient
    mask = (margins > 0).astype(np.float64)  # (N,C)
    row_sum = mask.sum(axis=1)               # (N,)
    mask[np.arange(N), y] = -row_sum
    dW = X.T @ mask / N + 2 * reg * W
    return loss, dW


# cv_learn/linear_classifiers/softmax.py
import numpy as np

def softmax_loss_vectorized(W, X, y, reg=0.0):
    """
    W: (D, C), X: (N, D), y: (N,)
    Returns loss, dW
    """
    N = X.shape[0]
    scores = X @ W
    scores -= scores.max(axis=1, keepdims=True)  # stability
    exp = np.exp(scores)
    probs = exp / exp.sum(axis=1, keepdims=True)  # (N,C)

    correct_logp = -np.log(probs[np.arange(N), y] + 1e-12)
    loss = correct_logp.mean() + reg * np.sum(W*W)

    # Gradient
    dscores = probs
    dscores[np.arange(N), y] -= 1.0
    dW = X.T @ dscores / N + 2 * reg * W
    return loss, dW


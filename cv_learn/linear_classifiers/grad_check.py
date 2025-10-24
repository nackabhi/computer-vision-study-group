# cv_learn/linear_classifiers/grad_check.py
import numpy as np

def eval_numerical_gradient(f, W, h=1e-5):
    """
    f: function mapping W -> scalar loss
    W: np.array weights
    Returns grad with same shape as W
    """
    grad = np.zeros_like(W)
    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = W[idx]
        W[idx] = old + h
        f_pos = f(W)
        W[idx] = old - h
        f_neg = f(W)
        W[idx] = old
        grad[idx] = (f_pos - f_neg) / (2*h)
        it.iternext()
    return grad

def relative_error(a, b, eps=1e-12):
    return np.max(np.abs(a-b) / np.maximum(1.0, np.abs(a)+np.abs(b)+eps))


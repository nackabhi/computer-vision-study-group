# cv_learn/linear_classifiers/optim.py
import numpy as np

class SGD:
    def __init__(self, lr=1e-3, weight_decay=0.0, momentum=0.0):
        self.lr = lr
        self.weight_decay = weight_decay  # not used (we add reg in loss already)
        self.momentum = momentum
        self.v = None

    def step(self, W, grad):
        if self.v is None and self.momentum > 0:
            self.v = np.zeros_like(W)
        if self.momentum > 0:
            self.v = self.momentum * self.v - self.lr * grad
            W += self.v
        else:
            W -= self.lr * grad
        return W


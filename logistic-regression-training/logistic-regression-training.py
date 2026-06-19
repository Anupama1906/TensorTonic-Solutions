import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    w = np.zeros_like(X[0])
    b = 0
    for _ in range(steps):
        p = _sigmoid(X@w + b)
        L_w = (X.T @ (p-y))/len(X)
        L_b = np.sum(p-y)/len(X)
        w = w - lr*L_w
        b = b - lr*L_b
    return (w,b)
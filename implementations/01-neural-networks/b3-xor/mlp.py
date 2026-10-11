"""Two-layer MLP for classification with manual backpropagation (milestone B3).

Model (docs/roadmap/01-neural-networks.md section 4.1):

    Z1 = X @ W1 + b1        (N, H)
    A1 = tanh(Z1)           (N, H)
    Z2 = A1 @ W2 + b2       (N, K)
    P  = softmax(Z2)        (N, K)   row-wise
    L  = -mean_i log P[i, y_i]       scalar
"""

import numpy as np


def init_params(D, H, K, rng):
    """Xavier/Glorot-normal weights, zero biases."""
    W1 = rng.normal(0.0, np.sqrt(2.0 / (D + H)), size=(D, H))
    b1 = np.zeros(H)
    W2 = rng.normal(0.0, np.sqrt(2.0 / (H + K)), size=(H, K))
    b2 = np.zeros(K)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def softmax(Z):
    # Subtracting the row max leaves softmax unchanged and prevents exp overflow.
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def forward(params, X):
    """Returns class probabilities P (N, K) and a cache of intermediates for backward()."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    N, D = X.shape
    H, K = W2.shape
    assert W1.shape == (D, H)

    Z1 = X @ W1 + b1  # (N, H)
    A1 = np.tanh(Z1)  # (N, H)
    Z2 = A1 @ W2 + b2  # (N, K)
    P = softmax(Z2)  # (N, K)

    assert P.shape == (N, K)
    cache = {"X": X, "Z1": Z1, "A1": A1, "P": P}
    return P, cache


def cross_entropy(P, y):
    """Mean negative log-likelihood. P: (N, K) probabilities, y: (N,) integer labels."""
    N = P.shape[0]
    assert y.shape == (N,)
    # Clipping guards against log(0) if a probability underflows to exactly zero.
    return -np.mean(np.log(np.clip(P[np.arange(N), y], 1e-12, None)))


def one_hot(y, K):
    Y = np.zeros((y.shape[0], K))
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def backward(params, cache, y):
    """Gradients of the mean cross-entropy loss with respect to every parameter.

    Returns a dict with the same keys and shapes as `params`:
        dW1 (D, H), db1 (H,), dW2 (H, K), db2 (K,)

    Available in `cache`: X (N, D), Z1 (N, H), A1 (N, H), P (N, K).
    Useful facts:
        - Y = one_hot(y, K) has shape (N, K)
        - d/dz tanh(z) = 1 - tanh(z)**2, and tanh(Z1) is already stored as A1
        - every gradient has the same shape as the tensor it differentiates
        - a bias broadcast over N rows in forward is summed over axis 0 in backward

    Run `python gradcheck.py` to verify your implementation.
    """
    # TODO(B3): implement using the equations in docs/roadmap/01-neural-networks.md section 4.2.
    raise NotImplementedError("backward() is yours to write; see the docstring")


def predict(params, X):
    P, _ = forward(params, X)
    return P.argmax(axis=1)

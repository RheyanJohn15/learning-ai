"""Compare backward() against central-difference numerical gradients in float64.

Usage: python gradcheck.py
Passes when every parameter's relative error is below 1e-6.
"""

import numpy as np

from mlp import backward, cross_entropy, forward, init_params


def loss_fn(params, X, y):
    P, _ = forward(params, X)
    return cross_entropy(P, y)


def numerical_grad(params, X, y, eps=1e-6):
    """dL/dtheta_j ~ (L(theta + eps*e_j) - L(theta - eps*e_j)) / (2*eps), one entry at a time."""
    grads = {}
    for name, p in params.items():
        g = np.zeros_like(p)
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            original = p[idx]
            p[idx] = original + eps
            loss_plus = loss_fn(params, X, y)
            p[idx] = original - eps
            loss_minus = loss_fn(params, X, y)
            p[idx] = original
            g[idx] = (loss_plus - loss_minus) / (2 * eps)
        grads[name] = g
    return grads


def relative_error(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b)))


def main():
    rng = np.random.default_rng(0)
    N, D, H, K = 5, 3, 4, 2
    X = rng.normal(size=(N, D))
    y = rng.integers(0, K, size=N)
    params = init_params(D, H, K, rng)

    _, cache = forward(params, X)
    analytic = backward(params, cache, y)
    numeric = numerical_grad(params, X, y)

    ok = True
    for name in params:
        assert analytic[name].shape == params[name].shape, (
            f"d{name} has shape {analytic[name].shape}, expected {params[name].shape}"
        )
        err = relative_error(analytic[name], numeric[name])
        passed = err < 1e-6
        ok &= passed
        print(f"{name:>3}: relative error {err:.2e}  {'PASS' if passed else 'FAIL'}")
    print("All gradients match." if ok else "Mismatch: re-check the derivation for the failing parameters.")


if __name__ == "__main__":
    main()

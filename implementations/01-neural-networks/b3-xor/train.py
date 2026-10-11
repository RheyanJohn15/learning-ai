"""Train the two-layer MLP on XOR with full-batch gradient descent.

Usage: python train.py [--hidden 4] [--lr 0.5] [--steps 2000] [--seed 0]
"""

import argparse

import numpy as np

from mlp import backward, cross_entropy, forward, init_params, predict

X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])  # (4, 2)
y = np.array([0, 1, 1, 0])  # (4,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(f"seed={args.seed} hidden={args.hidden} lr={args.lr} steps={args.steps}")

    rng = np.random.default_rng(args.seed)
    params = init_params(D=2, H=args.hidden, K=2, rng=rng)

    for step in range(args.steps + 1):
        P, cache = forward(params, X)
        loss = cross_entropy(P, y)
        if step % (args.steps // 10) == 0:
            acc = np.mean(P.argmax(axis=1) == y)
            print(f"step {step:5d}  loss {loss:.4f}  acc {acc:.2f}")
        if step == args.steps:
            break
        grads = backward(params, cache, y)
        for name in params:
            params[name] -= args.lr * grads[name]

    P, _ = forward(params, X)
    print("\ninput   label  pred  P(class 1)")
    for xi, yi, pi, pr in zip(X, y, predict(params, X), P[:, 1]):
        print(f"{xi}  {yi}      {pi}     {pr:.3f}")


if __name__ == "__main__":
    main()

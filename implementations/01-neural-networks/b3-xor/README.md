# B3 — XOR with manual backpropagation

Milestone [B3](../../../docs/roadmap/01-neural-networks.md#5-implementation): a two-layer MLP trained on the four XOR points with hand-derived gradients, NumPy only.

| File | Contents |
|---|---|
| `mlp.py` | Parameters, forward pass, softmax, cross-entropy, and `backward()` (**to be written by the learner**) |
| `gradcheck.py` | Compares `backward()` against central-difference numerical gradients in float64 |
| `train.py` | Full-batch gradient descent on XOR; prints loss, accuracy, and final predictions |

## Usage

From the repository root, with the [environment](../../README.md#setup) activated:

```bash
cd implementations/01-neural-networks/b3-xor
python gradcheck.py                  # must print PASS for W1, b1, W2, b2
python train.py                      # defaults: --hidden 4 --lr 0.5 --steps 2000 --seed 0
python train.py --hidden 2 --seed 3  # try the minimal 2-2-K network
```

Order of work: derive the backward pass on paper ([Phase B §4.2](../../../docs/roadmap/01-neural-networks.md#42-backpropagation-in-matrix-form)), implement `backward()`, make `gradcheck.py` pass, then train.

`gradcheck.py` is a minimal stand-in for the Phase A milestone A2 checker; replace it with your own A2 implementation once that exists.

## Related experiments

None yet. [Phase B experiment 1](../../../docs/roadmap/01-neural-networks.md#6-experiments) (XOR reliability across seeds and widths) builds directly on this code.

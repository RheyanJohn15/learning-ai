# Implementations

One self-contained directory per roadmap milestone, named after its ID and grouped by phase file number.

## Setup

Python 3.14 with a virtual environment at the repository root. Ubuntu's system Python lacks `ensurepip`, so pip is bootstrapped separately:

```bash
python3 -m venv --without-pip .venv
curl -sSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
source .venv/bin/activate
pip install -r requirements.txt
```

## Index

| Milestone | Directory | Status |
|---|---|---|
| B3 — XOR with manual backpropagation | [01-neural-networks/b3-xor/](01-neural-networks/b3-xor/README.md) | Scaffold: `backward()` not yet written |

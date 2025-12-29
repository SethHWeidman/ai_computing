# Multi-head attention comparison

This demo contrasts a batched multi-head attention implementation against a simple
“explainer” version that runs each head independently and concatenates the results. Both
share the utilities in `attention_helpers.py` and use the same weights for a fair check.

## How to run

From the repo root:
```bash
python 05_multihead_attention/compare_implementations.py
```
The script seeds deterministically, builds both models (bias disabled for Q/K/V), runs a
forward pass on random input, and reports output shapes plus whether
`torch.allclose(..., atol=1e-6)` succeeds.

## Files

- `compare_implementations.py` — defines the explainer and optimized MHA classes, copies
  weights between them, and prints the equality check.

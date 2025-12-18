## FlashAttention minimal demo

This folder now contains a single comparison:

- `compare_flash_v1.py` — builds `flash_attn_v1.cu` (float32, causal) as a lightweight
  extension, runs it against the Python helper (per-head output), and reports whether the
  results match.

Run it from the repo root:
```
python 04_flash_attention/compare_flash_v1.py
```
The script compiles in-place under `.torch_extensions`, seeds deterministically, and
expects a CUDA GPU. Use `--seq-len`, `--head-dim`, `--batch-size`, and `--num-heads` to
override defaults.

### Example output

Running 

```
python 04_flash_attention/compare_flash_v1.py --seq-len 16 --head-dim 16 --batch-size 1 --num-heads 2
```

for example, produces:

```
Shape: (1, 2, 16, 16)
Outputs match: yes
Max abs diff: 2.3842e-07
```

### Scheduling note: FA1 math, FA2-style partitioning

The `flash_attn_v1.cu` kernel uses the FlashAttention v1 online/streaming softmax
recurrence (exact attention, per-row running `m`/`l`, no NxN materialization) but is
scheduled in a v2-style way: the grid is `(B, H, ceil(N/Br))`, so each block owns one Q
tile and loops over K/V tiles. This improves occupancy relative to the paper’s “outer
over K/V, inner over Q” teaching layout. A precise label is “exact attention with online
softmax, FA2-style Q-tile-parallel scheduling.”

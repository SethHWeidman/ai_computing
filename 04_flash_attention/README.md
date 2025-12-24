## FlashAttention minimal demo

This folder contains a small comparison harness for a few teaching kernels:

- `compare_flash_v1.py` — builds two CUDA extensions (`flash_attn_v1.cu` and
  `flash_attn_v15.cu`), runs each against the Python helper (per-head output), and reports
  whether the results match.

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
python 04_flash_attention/compare_flash_v1.py \
  --seq-len 16 \
  --head-dim 16 \
  --batch-size 1 \
  --num-heads 2
```

for example, produces:

```
Shape: (1, 2, 16, 16)
flash_attn_v1.cu match: yes
flash_attn_v1.cu max abs diff: 2.3842e-07
flash_attn_v15.cu match: yes
flash_attn_v15.cu max abs diff: 2.3842e-07
```

### Kernel notes

- `flash_attn_v1.cu`: FlashAttention v1 online/streaming softmax recurrence (exact
  attention, per-row running `m`/`l`, no NxN materialization) with FA2-style Q-tile
  parallelism (grid `(B, H, ceil(N/Br))`, stream K/V tiles inside the block).
- `flash_attn_v15.cu`: same math, similar FA2-style tiling, but intentionally keeps the
  implementation simple (does not include many FA2 optimizations). Current version assumes
  no padding: `N % 16 == 0`.

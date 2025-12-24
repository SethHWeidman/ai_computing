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

- Naming: we call the second kernel “v1.5” because it adopts FlashAttention v2-style
  tiling (parallelize across Q tiles: grid includes the query-tile index), but it does
  not attempt to implement many of the other FA v2 optimizations (e.g., Tensor Core math,
  FP16/BF16 paths, fused/warp-level reductions, etc.). It’s “v1 math with v2-style
  tiling”.
- `flash_attn_v1.cu`: FlashAttention v1 online/streaming softmax recurrence (exact
  attention, per-row running `m`/`l`, no NxN materialization) with FA1-style scheduling
  (grid `(B, H)`, loop over Q tiles `tile_q` inside the kernel).
- `flash_attn_v15.cu`: same FA v1 recurrence, but with v2-style tiling (grid `(B, H,
  ceil(N/Br))`, one Q tile per block, stream K/V tiles inside the block). Current version
  assumes no padding: `N % 16 == 0`.

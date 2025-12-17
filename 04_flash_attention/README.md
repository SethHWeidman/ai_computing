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

Running `python 04_flash_attention/compare_flash_v1.py --seq-len 16
  --head-dim 16 --batch-size 1 --num-heads 2`, for example, produces:

```
Shape: (1, 2, 16, 16)
Outputs match: yes
Max abs diff: 2.3842e-07
```
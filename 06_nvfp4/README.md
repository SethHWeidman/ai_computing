# NVFP4 quantization

A manual implementation of NVIDIA's NVFP4 quantization format: FP4 E2M1 weights with
per-block FP8 E4M3 scales and a global FP32 tensor scale.

## How to run

```bash
cd 06_nvfp4
python nvfp4.py
```

Quantizes a hand-crafted 4×16 tensor whose four rows span four orders of magnitude, so
each block gets a visibly different E4M3 scale. Prints the tensor scale, per-block scales,
quantized payload values, reconstruction, and error.

```bash
python show_helpers.py
```

Walks through the three internal helper functions step by step, showing concrete inputs
and outputs for each.

## Files

- `nvfp4.py` — the quantizer. Key functions:
  - `_positive_e2m1_codebook` — the 8 positive FP4 E2M1 representable values and their 4-bit codes
  - `_positive_e4m3fn_codebook` — the 127 positive FP8 E4M3 representable values and their codes (used for block scales)
  - `_nearest_codebook_quantize_nonnegative` — snaps a nonneg tensor to the nearest codebook entry via `torch.searchsorted`
  - `pack_nibbles` — packs two FP4 codes per byte (first → low nibble, second → high nibble)
  - `quantize_nvfp4` — the full pipeline: tensor scale → block scales → FP4 quantization → nibble packing
- `show_helpers.py` — illustrated walkthroughs of the three helper functions above
- `pipeline.md` — explains the full quantization and dequantization pipeline, including how the three scaling levels (FP32, FP8, FP4) relate and how cuBLAS applies them during a matmul
- `pack_nibbles.md` — explains why nibble packing is necessary and what the term "nibble" means

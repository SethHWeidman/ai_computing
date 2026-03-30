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

```
tensor_scale_fp32 = 5.3760
  (= 6 × 448 / amax(x) = 2688 / 500.0)

block_scales_e4m3 (one per row):
  row 0: 0.0781   (row amax = 0.09)
  row 1: 2.7500   (row amax = 3.00)
  row 2: 44.0000   (row amax = 50.00)
  row 3: 448.0000   (row amax = 500.00)

payload_e2m1_values (quantized, shape 4×16):
  [0]   0.50  -1.50   3.00  -2.00   4.00  -6.00   3.00  -4.00
        6.00  -0.50   2.00  -4.00   1.50  -3.00   4.00  -3.00
  [1]   1.00  -2.00   3.00  -4.00   4.00  -6.00   2.00  -1.00
        4.00  -3.00   1.50  -4.00   2.00  -1.50   4.00  -2.00
  [2]   1.00  -2.00   4.00  -4.00   6.00  -1.00   2.00  -4.00
        4.00  -6.00   2.00  -3.00   4.00  -6.00   3.00  -4.00
  [3]   1.00  -2.00   4.00  -4.00   6.00  -1.00   2.00  -4.00
        4.00  -6.00   2.00  -3.00   4.00  -6.00   3.00  -4.00

original:
  [0]     0.010    -0.020     0.050    -0.030     0.070    -0.090     0.040    -0.060
          0.080    -0.010     0.030    -0.070     0.020    -0.050     0.060    -0.040
  [1]     0.500    -1.000     1.500    -2.000     2.500    -3.000     1.000    -0.500
          2.000    -1.500     0.750    -2.500     1.250    -0.750     2.250    -1.250
  [2]    10.000   -20.000    30.000   -40.000    50.000   -10.000    20.000   -30.000
         40.000   -50.000    15.000   -25.000    35.000   -45.000    25.000   -35.000
  [3]   100.000  -200.000   300.000  -400.000   500.000  -100.000   200.000  -300.000
        400.000  -500.000   150.000  -250.000   350.000  -450.000   250.000  -350.000

dequantized_fp32:
  [0]     0.007    -0.022     0.044    -0.029     0.058    -0.087     0.044    -0.058
          0.087    -0.007     0.029    -0.058     0.022    -0.044     0.058    -0.044
  [1]     0.512    -1.023     1.535    -2.046     2.046    -3.069     1.023    -0.512
          2.046    -1.535     0.767    -2.046     1.023    -0.767     2.046    -1.023
  [2]     8.185   -16.369    32.738   -32.738    49.107    -8.185    16.369   -32.738
         32.738   -49.107    16.369   -24.554    32.738   -49.107    24.554   -32.738
  [3]    83.333  -166.667   333.333  -333.333   500.000   -83.333   166.667  -333.333
        333.333  -500.000   166.667  -250.000   333.333  -500.000   250.000  -333.333

MSE = 271.987213
mean relative error = 0.0940
```

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

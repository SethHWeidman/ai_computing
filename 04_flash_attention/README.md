## FlashAttention vs Python baseline

This folder reuses the `flash-attn-101` submodule for the "fast" CUDA paths and benchmarks them
against a simple Python/torch attention implementation.

1. Build the `flash-attn-101` library once (from repo root):
   ```
   cmake -B flash-attn-101/build -S flash-attn-101
   cmake --build flash-attn-101/build
   ```
2. Run the comparison:
   ```
   python 04_flash_attention/compare.py
   ```

The script links directly against the built `flash-attn-101` library and re-runs all kernels
(Python, naive GPU, CUDA FA1/FA2, CuTe FA2) in-process. Defaults match the submodule settings:
batch=8, heads=16, seq_len=256, head_dim=64. Warm-ups are performed before timing to avoid cold
start skew, and results are reported as × factors versus the Python baseline.

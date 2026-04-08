# Separate kernels vs fused epilogue

A toy NumPy illustration of how GEMM output quantization changes from Hopper to
Blackwell.

## Why this matters for inference

In autoregressive decoding, every token generation involves many small GEMMs back to
back. On Hopper, quantizing the output of each GEMM requires separate kernel launches:
one for the matmul, then additional passes to read the full fp32 result back from global
memory, compute scales, and quantize. Each extra pass adds latency and memory bandwidth
cost. Blackwell's fused epilogue avoids those extra round-trips by quantizing each output
tile and emitting its scales as soon as that tile's accumulators are finished, inside the
GEMM kernel itself.

## How to run

```bash
cd 07_separate_vs_fused_epilogue
python separate_vs_fused.py
```

```
Separate pipeline fp32 shape (expect (8, 8)): (8, 8)
Separate pipeline output shape (expect (8, 8)): (8, 8)
Separate pipeline scales shape (expect (4,)): (4,)
Fused epilogue output shape (expect (8, 8)): (8, 8)
Number of fused tiles (expect 4): 4
```

## Files

- `separate_vs_fused.py`: Hopper-style two-step path (full GEMM, then a separate
  quantization pass) vs Blackwell-style path that tiles the GEMM and quantizes each tile
  immediately in the epilogue.

# NVFP4 Quantization & Dequantization Pipeline

## Data types

| Role | Format | Max magnitude |
|---|---|---|
| Weights / activations | FP4 E2M1 | 6.0 |
| Per-block scale | FP8 E4M3 (unsigned) | 448.0 |
| Tensor-level scale | FP32 | — |

Block size is always **16 elements** (along the K dimension for weights).

---

## Quantization (FP32 → NVFP4)

```
x  (FP32 tensor)
│
│  Step 1 — tensor-level scale
│
│    tensor_scale = (6.0 × 448.0) / amax(x)
│                 = 2688.0 / amax(x)
│
│    x_scaled = x × tensor_scale
│
│  After this, the full tensor fits within [-6×448, +6×448] = [-2688, +2688].
│  The ×448 headroom is reserved for the block scale to use.
│
├─ split into blocks of 16 elements ──────────────────────────┐
│                                                              │
│  Step 2 — per-block scale (one per block)                    │
│                                                              │
│    ideal_block_scale = amax(block) / 6.0                     │
│    block_scale = castToFP8_E4M3(ideal_block_scale)          │
│                                                              │
│  After this, block / block_scale fits within [-6, +6].       │
│                                                              │
│  Step 3 — quantize each element                              │
│                                                              │
│    x_q = castToFP4_E2M1(x_i / block_scale)                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
│
│  Step 4 — pack pairs of FP4 codes into bytes
│
│    packed = pack_nibbles(x_q)
│    (low nibble = first code, high nibble = second code)
│
▼
packed bytes  +  block_scales (FP8)  +  tensor_scale (FP32)
```

The three-level scaling hierarchy exists because:
- A single global scale can't handle local variation across blocks
- A per-element FP8 scale would cost more than the FP4 data itself (2× overhead)
- One FP8 scale per 16 elements adds 1 bit of overhead per stored value (8 bits / 16 =
  0.5 bits/value)

---

## Dequantization (NVFP4 → FP32)

```
packed bytes  +  block_scales (FP8)  +  tensor_scale (FP32)
│
│  Step 1 — unpack nibbles
│
│    x_q = unpack_nibbles(packed)       (FP4 E2M1 codes)
│
│  Step 2 — decode each element
│
│    x̂_i = FP32(x_q_i)
│          × FP32(block_scale)
│          × tensor_scale
│
▼
x̂  (FP32 reconstruction)
```

Written as one formula:

```
x̂_i = FP32(x_q_i) × FP32(block_scale_b) × tensor_scale
```

where `tensor_scale` here is `1 / encode_scale = amax(x) / 2688.0`.

Note: in the code, `tensor_scale_t` is the *encode* scale (`2688 / amax`), so
dequantization divides by it: `x̂ = payload_vals × block_scale / tensor_scale_t`.

**How the block scale is broadcast back over 16 elements** — after quantization,
`block_scale` has shape `(..., nblocks)`: one scalar per block. To multiply it against
`payload_vals` which has shape `(..., nblocks, 16)`, the code does `scale_expanded =
block_scale.unsqueeze(-1)` ([nvfp4.py:186](nvfp4.py#L186)), giving shape `(..., nblocks,
1)`. PyTorch then broadcasts that singleton dimension across all 16 positions in the last
axis. The two dequant lines ([nvfp4.py:199-200](nvfp4.py#L199)) are therefore:

```python
dequant_scaled = payload_vals * scale_expanded   # applies block scale to each element
dequant        = dequant_scaled / tensor_scale_t  # removes the global tensor scale
```

---

## In cuBLAS matmuls

When running a quantized matmul `D = A × B`:

- **Block scales** are applied *inside* the tensor core mainloop — each 16-element dot
  product is descaled before accumulation
- **Tensor-level scales** (one per A, one per B) are applied in the *epilogue* as a
  single multiply after accumulation

This means the hardware never accumulates large intermediate values; descaling happens at
the finest granularity the hardware supports.

---

## Sources

- [NVIDIA blog: Introducing
  NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVIDIA cuBLAS 12.9
  blog](https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9/)
- [TensorRT: Working with Quantized
  Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [cuDNN Frontend: Block
  Scaling](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/BlockScaling.html)
- [arXiv 2512.02010 — Four Over Six](https://arxiv.org/pdf/2512.02010)

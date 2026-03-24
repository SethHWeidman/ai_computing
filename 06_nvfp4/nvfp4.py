import warnings

import torch
from torch.nn import functional

BLOCK_SIZE: int = 16
FP4_MAX: float = 6.0
E4M3_MAX: float = 448.0


# ----------------------------
# Codebooks for the public formats
# ----------------------------


def _positive_e2m1_codebook() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Positive finite FP4 E2M1 values inferred from NVIDIA's public description:
    sign bit + 2 exponent bits + 1 mantissa bit, max magnitude 6.
    Positive values are:
      0, 0.5, 1, 1.5, 2, 3, 4, 6
    Codes returned here are the positive 4-bit payload codes without the sign bit.
    """
    vals = []
    codes = []

    for exp in range(4):  # 2 exponent bits
        for mant in range(2):  # 1 mantissa bit
            code = (exp << 1) | mant  # sign bit gets added later

            if exp == 0:
                # subnormal / zero, bias = 1
                val = mant / 2.0
            else:
                val = (1.0 + mant / 2.0) * (2.0 ** (exp - 1))

            vals.append(val)
            codes.append(code)

    return (
        torch.tensor(vals, dtype=torch.float32),
        torch.tensor(codes, dtype=torch.uint8),
    )


def _positive_e4m3fn_codebook() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Positive finite FP8 E4M3 values for the scale factors.

    Public NVIDIA docs say E4M3 stores finite values up to +/-448 and nan.
    This implementation uses the usual E4M3 finite-no-inf interpretation:
      - bias = 7
      - exp=0 => subnormals
      - exp=15, mant=7 reserved as NaN
    """
    vals = []
    codes = []

    for exp in range(16):  # 4 exponent bits
        for mant in range(8):  # 3 mantissa bits
            if exp == 15 and mant == 7:
                continue  # reserve NaN

            code = (exp << 3) | mant  # positive sign
            if exp == 0:
                val = (mant / 8.0) * (2.0 ** (1 - 7))
            else:
                val = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))

            vals.append(val)
            codes.append(code)

    vals = torch.tensor(vals, dtype=torch.float32)
    codes = torch.tensor(codes, dtype=torch.uint8)

    vals, order = torch.sort(vals)
    codes = codes[order]
    return vals, codes


def _nearest_codebook_quantize_nonnegative(
    x: torch.Tensor, codebook_vals: torch.Tensor, codebook_codes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize nonnegative x to nearest representable codebook value.
    Ties go to the lower-magnitude neighbor.
    """
    flat = x.reshape(-1)

    idx = torch.searchsorted(codebook_vals, flat)
    idx_lo = torch.clamp(idx - 1, 0, len(codebook_vals) - 1)
    idx_hi = torch.clamp(idx, 0, len(codebook_vals) - 1)

    v_lo = codebook_vals[idx_lo]
    v_hi = codebook_vals[idx_hi]

    use_hi = (flat - v_hi).abs() < (flat - v_lo).abs()
    best = torch.where(use_hi, idx_hi, idx_lo)

    return (codebook_vals[best].reshape_as(x), codebook_codes[best].reshape_as(x))


def pack_nibbles(payload_codes_fp4: torch.Tensor) -> torch.Tensor:
    """
    Packs 4-bit payload codes into bytes.
    Convention used here:
      - first FP4 value goes in low nibble
      - second FP4 value goes in high nibble
    This packing convention is a host-side convenience, not a claim about
    NVIDIA library internal storage layout.
    """
    flat = payload_codes_fp4.reshape(-1).to(torch.uint8)

    if flat.numel() % 2:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8)], dim=0)

    lo = flat[0::2] & 0x0F
    hi = (flat[1::2] & 0x0F) << 4
    return lo | hi


# ----------------------------
# Manual NVFP4 quantizer
# ----------------------------


def quantize_nvfp4(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Manual NVFP4 quantization along the last dimension.

    Args:
        x:
            FP32 tensor to quantize. Blocks of BLOCK_SIZE are formed along the last
            dimension.

    Returns:
        dict with:
          - tensor_scale_fp32      global FP32 scale (scalar)
          - block_scales_e4m3      per-block scale decoded to float32, shape (...,
            nblocks)
          - payload_e2m1_values    quantized values decoded to float32, shape same as x
          - dequantized_fp32       reconstruction: payload * block_scale / tensor_scale
    """
    if x.dtype != torch.float32:
        warnings.warn(
            f"quantize_nvfp4 expected float32 input, got {x.dtype}; casting to float32"
        )
        x = x.float()

    e2_vals, e2_codes = _positive_e2m1_codebook()
    e4_vals, e4_codes = _positive_e4m3fn_codebook()

    # Scale the whole tensor so its global max maps to FP4_MAX * E4M3_MAX.
    # That leaves room for the per-block E4M3 scale to absorb local variation.
    amax = float(x.abs().max().item())
    scale_val = 1.0 if amax == 0.0 else (FP4_MAX * E4M3_MAX) / amax
    tensor_scale_t = torch.tensor(scale_val, dtype=torch.float32)

    x_scaled = x * tensor_scale_t

    # Pad the last dimension to a multiple of BLOCK_SIZE so reshaping is clean.
    original_last = x_scaled.shape[-1]
    pad = (-original_last) % BLOCK_SIZE
    if pad:
        x_scaled = functional.pad(x_scaled, (0, pad))

    # Group into blocks: (..., nblocks, BLOCK_SIZE).
    xb = x_scaled.reshape(*x_scaled.shape[:-1], -1, BLOCK_SIZE)

    # Ideal block scale maps the block's max value to FP4_MAX, then we
    # snap it to the nearest representable E4M3 value.
    block_amax = xb.abs().amax(dim=-1)
    ideal_block_scale = block_amax / FP4_MAX
    block_scale, block_scale_codes = _nearest_codebook_quantize_nonnegative(
        ideal_block_scale, e4_vals, e4_codes
    )

    # All-zero blocks get a zero scale to avoid division by zero below.
    zero_blocks = block_amax == 0
    block_scale = torch.where(zero_blocks, torch.zeros_like(block_scale), block_scale)
    block_scale_codes = torch.where(
        zero_blocks, torch.zeros_like(block_scale_codes), block_scale_codes
    )

    # Divide each element by its block's scale; result should sit in [-FP4_MAX, FP4_MAX].
    scale_expanded = block_scale.unsqueeze(-1)
    normalized = torch.where(
        scale_expanded > 0, xb / scale_expanded, torch.zeros_like(xb)
    )

    # The E2M1 codebook is positive-only, so handle sign separately.
    signs = (normalized < 0).to(torch.uint8)
    q_abs_vals, _ = _nearest_codebook_quantize_nonnegative(
        normalized.abs(), e2_vals, e2_codes
    )
    payload_vals = torch.where(signs.bool(), -q_abs_vals, q_abs_vals)

    # Dequantize: reverse the two scaling steps.
    dequant_scaled = payload_vals * scale_expanded
    dequant = dequant_scaled / tensor_scale_t

    # Trim padding back to the original shape.
    payload_vals = payload_vals.reshape(*x_scaled.shape[:-1], -1)[..., :original_last]
    dequant = dequant.reshape(*x_scaled.shape[:-1], -1)[..., :original_last]

    return {
        "tensor_scale_fp32": tensor_scale_t,
        "block_scales_e4m3": block_scale,
        "payload_e2m1_values": payload_vals,
        "dequantized_fp32": dequant,
    }


# ----------------------------
# Example
# ----------------------------

if __name__ == "__main__":
    # 4 rows × 16 columns — each row is one block (axis=-1, block_size=16).
    # Rows have very different magnitudes so each block gets a distinct E4M3 scale.
    x = torch.tensor(
        [
            # row 0: tiny values  (amax ≈ 0.09)
            [
                0.01,
                -0.02,
                0.05,
                -0.03,
                0.07,
                -0.09,
                0.04,
                -0.06,
                0.08,
                -0.01,
                0.03,
                -0.07,
                0.02,
                -0.05,
                0.06,
                -0.04,
            ],
            # row 1: moderate values  (amax ≈ 3.0)
            [
                0.50,
                -1.00,
                1.50,
                -2.00,
                2.50,
                -3.00,
                1.00,
                -0.50,
                2.00,
                -1.50,
                0.75,
                -2.50,
                1.25,
                -0.75,
                2.25,
                -1.25,
            ],
            # row 2: large values  (amax ≈ 50.0)
            [
                10.0,
                -20.0,
                30.0,
                -40.0,
                50.0,
                -10.0,
                20.0,
                -30.0,
                40.0,
                -50.0,
                15.0,
                -25.0,
                35.0,
                -45.0,
                25.0,
                -35.0,
            ],
            # row 3: huge values  (amax ≈ 500.0) — drives tensor_scale
            [
                100.0,
                -200.0,
                300.0,
                -400.0,
                500.0,
                -100.0,
                200.0,
                -300.0,
                400.0,
                -500.0,
                150.0,
                -250.0,
                350.0,
                -450.0,
                250.0,
                -350.0,
            ],
        ],
        dtype=torch.float32,
    )

    out = quantize_nvfp4(x)

    print(f"tensor_scale_fp32 = {out['tensor_scale_fp32'].item():.6f}")
    print(f"  (= 6 × 448 / amax(x) = 2688 / {x.abs().max().item():.1f})")
    print()

    print("block_scales_e4m3 (one per row):")
    for i, s in enumerate(out["block_scales_e4m3"].squeeze().tolist()):
        amax_row = x[i].abs().max().item()
        print(f"  row {i}: {s:.4f}   (row amax = {amax_row:.2f})")
    print()

    print("payload_e2m1_values (quantized, shape 4×16):")
    print(out["payload_e2m1_values"])
    print()

    print("dequantized_fp32:")
    print(out["dequantized_fp32"])
    print()

    mse = torch.mean((x - out["dequantized_fp32"]) ** 2)
    rel_err = (x - out["dequantized_fp32"]).abs() / x.abs().clamp(min=1e-6)
    print(f"MSE = {mse.item():.6f}")
    print(f"mean relative error = {rel_err.mean().item():.4f}")

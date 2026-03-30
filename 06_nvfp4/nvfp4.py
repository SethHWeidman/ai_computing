import warnings

import torch
from torch.nn import functional

import codebook
import helpers

BLOCK_SIZE: int = 16
# largest finite E2M1 value: (1 + 1/2) * 2^(3-1) = 1.5 * 4
# see max_values.md
FP4_MAX: float = 6.0
# largest finite E4M3 value: (1 + 6/8) * 2^(15-7) = 1.75 * 256
# exp=1111/mant=111 reserved for NaN
E4M3_MAX: float = 448.0


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

    e2_vals, e2_codes = codebook.positive_e2m1_codebook()
    e4_vals, e4_codes = codebook.positive_e4m3fn_codebook()

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
    block_scale, block_scale_codes = codebook.nearest_codebook_quantize_nonnegative(
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
    q_abs_vals, _ = codebook.nearest_codebook_quantize_nonnegative(
        normalized.abs(), e2_vals, e2_codes
    )
    payload_vals = torch.where(signs.bool(), -q_abs_vals, q_abs_vals)

    # Dequantize: reverse the two scaling steps.
    dequant_scaled = payload_vals * scale_expanded  # block scale
    dequant = dequant_scaled / tensor_scale_t  # tensor scale

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

    print(f"tensor_scale_fp32 = {out['tensor_scale_fp32'].item():.4f}")
    print(f"  (= 6 × 448 / amax(x) = 2688 / {x.abs().max().item():.1f})")
    print()

    print("block_scales_e4m3 (one per row):")
    for i, s in enumerate(out["block_scales_e4m3"].squeeze().tolist()):
        amax_row = x[i].abs().max().item()
        print(f"  row {i}: {s:.4f}   (row amax = {amax_row:.2f})")
    print()

    helpers.print_tensor(
        "payload_e2m1_values (quantized, shape 4×16):",
        out["payload_e2m1_values"],
        fmt="{:5.2f}",
    )
    print()

    helpers.print_tensor("original:", x)
    print()
    helpers.print_tensor("dequantized_fp32:", out["dequantized_fp32"])
    print()

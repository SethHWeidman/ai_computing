import torch


def positive_e2m1_codebook() -> tuple[torch.Tensor, torch.Tensor]:
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


def positive_e4m3fn_codebook() -> tuple[torch.Tensor, torch.Tensor]:
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


def nearest_codebook_quantize_nonnegative(
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


if __name__ == "__main__":
    # ----------------------------
    # positive_e2m1_codebook
    # ----------------------------
    #
    # FP4 E2M1 format: [sign(1)] [exp(2)] [mant(1)]
    # This function returns POSITIVE values only (sign bit = 0).
    # The code is the 3-bit positive payload: [exp(2)][mant(1)]
    #
    # Exponent bias = 1, so:
    #   exp=0: subnormal => value = mant/2
    #   exp=1: normal    => value = (1 + mant/2) * 2^(1-1) = 1 + mant/2
    #   exp=2: normal    => value = (1 + mant/2) * 2^(2-1) = 2*(1 + mant/2)
    #   exp=3: normal    => value = (1 + mant/2) * 2^(3-1) = 4*(1 + mant/2)

    e2_vals, e2_codes = positive_e2m1_codebook()

    print("=" * 60)
    print("positive_e2m1_codebook  (FP4 E2M1, positive half)")
    print("=" * 60)
    print(f"{'code (bin)':>12}  {'exp':>4}  {'mant':>4}  {'value':>8}")
    print("-" * 60)
    for code, val in zip(e2_codes.tolist(), e2_vals.tolist()):
        exp = (code >> 1) & 0b11
        mant = code & 0b1
        print(f"  {code:03b} ({code:2d})    exp={exp}  mant={mant}  val={val:>6.2f}")

    print(f"\nAll values: {e2_vals.tolist()}")
    print(f"All codes:  {e2_codes.tolist()}")

    # ----------------------------
    # positive_e4m3fn_codebook
    # ----------------------------
    #
    # FP8 E4M3 format: [sign(1)] [exp(4)] [mant(3)]
    # Used for block scales in NVFP4.
    # "fn" = finite, no infinity; exp=15/mant=7 reserved as NaN.
    #
    # Exponent bias = 7, so:
    #   exp=0:  subnormal => value = (mant/8) * 2^(1-7) = mant/8 * 1/64
    #   exp>0:  normal    => value = (1 + mant/8) * 2^(exp-7)
    # Range: ~0 to 448 (= (1 + 7/8) * 2^(15-7) = 1.875 * 256)

    e4_vals, e4_codes = positive_e4m3fn_codebook()

    print()
    print("=" * 60)
    print("positive_e4m3fn_codebook  (FP8 E4M3fn, positive half)")
    print("=" * 60)
    print(f"  {len(e4_vals)} representable positive values (including 0), sorted")
    print()

    def _show_slice(label: str, vals: torch.Tensor, codes: torch.Tensor) -> None:
        print(f"  {label}:")
        print(f"    {'code (hex)':>10}  {'exp':>4}  {'mant':>4}  {'value':>12}")
        for code, val in zip(codes.tolist(), vals.tolist()):
            exp = (code >> 3) & 0xF
            mant = code & 0x7
            print(
                f"    0x{code:02x} ({code:3d})   exp={exp:2d}  mant={mant}  "
                f"val={val:>12.6f}"
            )

    _show_slice("first 8 (near zero)", e4_vals[:8], e4_codes[:8])
    print()
    _show_slice("last 8 (near max 448)", e4_vals[-8:], e4_codes[-8:])
    print()
    print(f"  Min value: {e4_vals[0].item():.2e}")
    print(f"  Max value: {e4_vals[-1].item():.1f}  (= 1.875 * 2^8 = 448)")
    print(f"  Total entries: {len(e4_vals)}")

    # ----------------------------
    # nearest_codebook_quantize_nonnegative
    # ----------------------------
    #
    # Given a nonnegative tensor x and a sorted codebook, snaps each value to
    # the nearest entry. Ties go to the lower-magnitude neighbor (i.e. strict
    # inequality is required to pick the higher one).
    #
    # Internally it uses torch.searchsorted to find the insertion point, then
    # compares the distance to the neighbors on each side.
    #
    # Below we use the E2M1 codebook [0, 0.5, 1, 1.5, 2, 3, 4, 6] and feed in
    # a hand-picked set of values that exercise the interesting cases:

    print()
    print("=" * 60)
    print("nearest_codebook_quantize_nonnegative  (using E2M1 codebook)")
    print("=" * 60)
    print(f"  Codebook values: {e2_vals.tolist()}")
    print()

    x = torch.tensor(
        [
            0.0,  # exact match at bottom
            0.25,  # midpoint 0↔0.5  → tie → rounds DOWN to 0.0
            0.4,  # closer to 0.5
            0.5,  # exact match
            1.3,  # closer to 1.5
            2.5,  # midpoint 2↔3    → tie → rounds DOWN to 2.0
            2.6,  # closer to 3.0
            5.9,  # closer to 6.0
            6.0,  # exact match at top
            7.0,  # beyond max       → clamps to 6.0
        ]
    )

    q_vals, q_codes = nearest_codebook_quantize_nonnegative(x, e2_vals, e2_codes)

    print(f"  {'input':>8}  {'quantized':>10}  {'code':>6}  note")
    print(f"  {'-'*8}  {'-'*10}  {'-'*6}  ----")
    notes = [
        "exact match",
        "tie → lower (0.0)",
        "closer to 0.5",
        "exact match",
        "closer to 1.5",
        "tie → lower (2.0)",
        "closer to 3.0",
        "closer to 6.0",
        "exact match",
        "beyond max → clamp to 6.0",
    ]
    for xi, qv, qc, note in zip(x.tolist(), q_vals.tolist(), q_codes.tolist(), notes):
        print(f"  {xi:>8.2f}  {qv:>10.2f}  {qc:>6}  {note}")

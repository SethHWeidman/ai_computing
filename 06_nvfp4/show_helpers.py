"""
Show the outputs of the three nvfp4 helper functions:
  _positive_e2m1_codebook
  _positive_e4m3fn_codebook
  _nearest_codebook_quantize_nonnegative
"""

import torch
import nvfp4


# ----------------------------
# _positive_e2m1_codebook
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

e2_vals, e2_codes = nvfp4._positive_e2m1_codebook()

print("=" * 60)
print("_positive_e2m1_codebook  (FP4 E2M1, positive half)")
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
# _positive_e4m3fn_codebook
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

e4_vals, e4_codes = nvfp4._positive_e4m3fn_codebook()

print()
print("=" * 60)
print("_positive_e4m3fn_codebook  (FP8 E4M3fn, positive half)")
print("=" * 60)
print(f"  {len(e4_vals)} representable positive values (including 0), sorted")
print()


# Show first few, last few, and a middle slice
def show_slice(label, vals, codes):
    print(f"  {label}:")
    print(f"    {'code (hex)':>10}  {'exp':>4}  {'mant':>4}  {'value':>12}")
    for code, val in zip(codes.tolist(), vals.tolist()):
        exp = (code >> 3) & 0xF
        mant = code & 0x7
        print(
            f"    0x{code:02x} ({code:3d})   exp={exp:2d}  mant={mant}  val={val:>12.6f}"
        )


show_slice("first 8 (near zero)", e4_vals[:8], e4_codes[:8])
print()
show_slice("last 8 (near max 448)", e4_vals[-8:], e4_codes[-8:])
print()
print(f"  Min value: {e4_vals[0].item():.2e}")
print(f"  Max value: {e4_vals[-1].item():.1f}  (= 1.875 * 2^8 = 448)")
print(f"  Total entries: {len(e4_vals)}")


# ----------------------------
# _nearest_codebook_quantize_nonnegative
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
print("_nearest_codebook_quantize_nonnegative  (using E2M1 codebook)")
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

q_vals, q_codes = nvfp4._nearest_codebook_quantize_nonnegative(x, e2_vals, e2_codes)

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


# ----------------------------
# pack_nibbles
# ----------------------------
#
# Packs pairs of 4-bit codes into single bytes.
# Convention: first code → low nibble (bits 3:0)
#             second code → high nibble (bits 7:4)
#
# Example with two codes A=0b0101 (5) and B=0b1010 (10):
#   byte = (B << 4) | A  =  0b10100101  =  0xA5
#
# Odd-length inputs get a zero padding nibble appended.

print()
print("=" * 60)
print("pack_nibbles")
print("=" * 60)

# Case 1: even number of codes — show bit layout for each pair
print()
print("  Case 1: four codes packed into two bytes")
codes_even = torch.tensor([5, 10, 3, 7], dtype=torch.uint8)
packed_even = nvfp4.pack_nibbles(codes_even)
print(
    f"  Input codes: {codes_even.tolist()}  (binary: {[f'{c:04b}' for c in codes_even.tolist()]})"
)
for i, byte in enumerate(packed_even.tolist()):
    lo = codes_even[i * 2].item()
    hi = codes_even[i * 2 + 1].item()
    print(
        f"  byte[{i}]: 0x{byte:02x} = {byte:08b}  ← hi={hi:04b} ({hi})  lo={lo:04b} ({lo})"
    )

# Case 2: odd number of codes — last nibble is zero-padded
print()
print("  Case 2: three codes (odd) → last byte is zero-padded in high nibble")
codes_odd = torch.tensor([5, 10, 3], dtype=torch.uint8)
packed_odd = nvfp4.pack_nibbles(codes_odd)
print(f"  Input codes: {codes_odd.tolist()}")
print(
    f"  byte[0]: 0x{packed_odd[0].item():02x} = {packed_odd[0].item():08b}  ← {codes_odd[1].item():04b} | {codes_odd[0].item():04b}"
)
print(
    f"  byte[1]: 0x{packed_odd[1].item():02x} = {packed_odd[1].item():08b}  ← 0000 (pad) | {codes_odd[2].item():04b}"
)

# Case 3: round-trip — pack then unpack manually to verify
print()
print("  Case 3: round-trip unpack (low nibble = & 0x0F, high nibble = >> 4)")
codes_rt = torch.tensor([0, 1, 5, 7, 15, 8], dtype=torch.uint8)
packed_rt = nvfp4.pack_nibbles(codes_rt)
unpacked = []
for byte in packed_rt.tolist():
    unpacked.append(byte & 0x0F)
    unpacked.append((byte >> 4) & 0x0F)
print(f"  Original: {codes_rt.tolist()}")
print(f"  Packed:   {[f'0x{b:02x}' for b in packed_rt.tolist()]}")
print(f"  Unpacked: {unpacked}")

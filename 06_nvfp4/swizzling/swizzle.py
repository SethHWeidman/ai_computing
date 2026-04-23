"""
Pedagogical implementation of cuBLASLt's FP4 scale-factor swizzle.

NVFP4 quantization produces a logical scale tensor: one FP8 E4M3 scale per block of 16
elements, stored in row-major order as scales[row, block]. But real GEMM kernels don't
read scales in that order. cuBLASLt documents a tiled, interleaved layout so the kernel
can fetch scales with coalesced memory accesses.

The swizzle does not change any scale values. It only changes where each byte lives in
memory.

For the VEC16_UE4M3 FP4 mode, one scale tile is 128 x 4 and covers a 128 x 64 data block
(128 rows, 64 columns, 4 blocks of 16 per row). The tile-local mapping from the cuBLASLt
docs is:

    offset = (outer % 32) * 16 + (outer // 32) * 4 + inner

where `outer` is the row index within the tile (0..127) and `inner` is the block index
within the tile (0..3).

In plain English: memory is not laid out row-by-row. Instead, it interleaves rows that
are 32 apart. The first 16 stored values are the 4 scales from row 0, then row 32, then
row 64, then row 96. After that come row 1, row 33, row 65, row 97, and so on.

Sources:
  - cuBLASLt scale-factor layout docs (NVIDIA cuBLAS 12.9+)
  - Jianyu Huang, "Quantized GEMM in cuBLAS" blog post
  - CUTLASS Blackwell FP4 examples (interleaved scale layout)
"""

import numpy as np


def swizzle_nvfp4_scales(logical_scales: np.ndarray) -> np.ndarray:
    """
    Reorder a logical scale tensor into cuBLASLt's tiled layout.

    Args:
        logical_scales: shape [outer_dim, inner_dim]
            For operand A: outer_dim = M, inner_dim = K // 16
            For operand B: outer_dim = N, inner_dim = K // 16

    Returns:
        1-D packed buffer in the 128x4 tiled layout.
    """
    outer_dim, inner_dim = logical_scales.shape

    # Pad to full tiles.
    padded_outer = ((outer_dim + 127) // 128) * 128
    padded_inner = ((inner_dim + 3) // 4) * 4

    num_tile_rows = padded_outer // 128
    num_tile_cols = padded_inner // 4

    # Each 128x4 tile holds 512 scale entries.
    tile_size = 128 * 4
    out = np.zeros(num_tile_rows * num_tile_cols * tile_size, dtype=logical_scales.dtype)

    for outer in range(outer_dim):
        for inner in range(inner_dim):
            # Which tile does this scale belong to?
            tile_row = outer // 128
            tile_col = inner // 4

            # Position within the tile.
            local_outer = outer % 128
            local_inner = inner % 4

            # Base offset of this tile in the global packed buffer.
            base = (tile_row * num_tile_cols + tile_col) * tile_size

            # cuBLASLt tile-local swizzle formula.
            offset_in_tile = (
                (local_outer % 32) * 16 + (local_outer // 32) * 4 + local_inner
            )

            out[base + offset_in_tile] = logical_scales[outer, inner]

    return out


def unswizzle_nvfp4_scales(
    packed: np.ndarray, outer_dim: int, inner_dim: int
) -> np.ndarray:
    """
    Inverse of swizzle_nvfp4_scales. Recovers the logical 2-D scale tensor
    from a packed buffer.
    """
    padded_outer = ((outer_dim + 127) // 128) * 128
    padded_inner = ((inner_dim + 3) // 4) * 4

    num_tile_rows = padded_outer // 128
    num_tile_cols = padded_inner // 4
    tile_size = 128 * 4

    logical = np.zeros((outer_dim, inner_dim), dtype=packed.dtype)

    for outer in range(outer_dim):
        for inner in range(inner_dim):
            tile_row = outer // 128
            tile_col = inner // 4

            local_outer = outer % 128
            local_inner = inner % 4

            base = (tile_row * num_tile_cols + tile_col) * tile_size
            offset_in_tile = (
                (local_outer % 32) * 16 + (local_outer // 32) * 4 + local_inner
            )

            logical[outer, inner] = packed[base + offset_in_tile]

    return logical


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Step 1: build a logical scale tensor ---
    #
    # A 128x64 data matrix with block_size=16 produces exactly one 128x4 scale
    # tile: 128 rows, each with 64/16 = 4 blocks. We fill it with sequential
    # integers so the reordering is easy to trace.

    M, K_blocks = 128, 4  # 128 rows, 4 scale-blocks per row
    logical = np.arange(M * K_blocks, dtype=np.int32).reshape(M, K_blocks)

    print("=" * 64)
    print("Logical scale tensor  (shape 128x4, row-major)")
    print("=" * 64)
    print()
    print("First 8 rows:")
    print(logical[:8])
    print()
    print("Rows 32-35:")
    print(logical[32:36])
    print()
    print("Rows 64-67:")
    print(logical[64:68])
    print()
    print("Rows 96-99:")
    print(logical[96:100])
    print()

    # --- Step 2: apply the swizzle ---

    packed = swizzle_nvfp4_scales(logical)

    print("=" * 64)
    print("Packed (swizzled) buffer  (first 32 entries)")
    print("=" * 64)
    print()
    print(packed[:32])
    print()

    # --- Step 3: explain the pattern ---
    #
    # The swizzle interleaves rows 32 apart. Within each group of 16 entries:
    #   entries  0.. 3  are logical[  0, 0:4]
    #   entries  4.. 7  are logical[ 32, 0:4]
    #   entries  8..11  are logical[ 64, 0:4]
    #   entries 12..15  are logical[ 96, 0:4]
    # Then the next group of 16:
    #   entries 16..19  are logical[  1, 0:4]
    #   entries 20..23  are logical[ 33, 0:4]
    #   entries 24..27  are logical[ 65, 0:4]
    #   entries 28..31  are logical[ 97, 0:4]

    print("Reading the pattern:")
    print()
    labels = [
        (0, 4, "logical[  0, 0:4]"),
        (4, 8, "logical[ 32, 0:4]"),
        (8, 12, "logical[ 64, 0:4]"),
        (12, 16, "logical[ 96, 0:4]"),
        (16, 20, "logical[  1, 0:4]"),
        (20, 24, "logical[ 33, 0:4]"),
        (24, 28, "logical[ 65, 0:4]"),
        (28, 32, "logical[ 97, 0:4]"),
    ]
    for lo, hi, desc in labels:
        print(f"  packed[{lo:2d}:{hi:2d}] = {packed[lo:hi]}  <- {desc}")
    print()
    print("The kernel reads 32 consecutive rows at a time (a warp's worth), so rows 0, ")
    print("32, 64, 96 are grouped together in memory.")
    print()

    # --- Step 4: round-trip check ---

    roundtrip = unswizzle_nvfp4_scales(packed, M, K_blocks)
    assert np.array_equal(logical, roundtrip), "round-trip failed!"
    print("Round-trip check: PASSED (unswizzle recovers the original tensor)")
    print()

    # --- Step 5: larger example with multiple tiles ---
    #
    # A 256x128 data matrix has a 256x8 logical scale tensor, which spans
    # 2 tile-rows x 2 tile-cols = 4 tiles of 128x4 each.

    M2, K_blocks2 = 256, 8
    logical2 = np.arange(M2 * K_blocks2, dtype=np.int32).reshape(M2, K_blocks2)
    packed2 = swizzle_nvfp4_scales(logical2)
    roundtrip2 = unswizzle_nvfp4_scales(packed2, M2, K_blocks2)
    assert np.array_equal(logical2, roundtrip2), "multi-tile round-trip failed!"

    print("=" * 64)
    print(
        f"Multi-tile example  ({M2}x{K_blocks2} logical -> {M2 // 128}x{K_blocks2 // 4} "
        "tiles)"
    )
    print("=" * 64)
    print()
    print(
        f"Packed buffer length: {packed2.size}  (= {M2 // 128} x {K_blocks2 // 4} x 512)"
    )
    print("Round-trip check: PASSED")
    print()

# Scale-factor swizzling for NVFP4 GEMM

NVFP4 quantization requires storing a tensor with the blocks scales in memory, with one
FP8 E4M3 scale per block of 16 elements. Naively, one might store them in row-major order
as `scales[row, block]`. That is the natural representation, and it is what `nvfp4.py` in
the parent directory computes.

GEMM kernels don't read memory row by row, so reading scales in row-major order during a
matmul causes many small, scattered memory fetches, which is slow. Swizzling fixes this
by storing the scales in memory in such a way that the kernel can read them in large,
contiguous chunks. The scale values themselves don't change, only where each byte lives
in memory.

## What swizzling is

Swizzling is a physical memory layout chosen specifically so the GEMM kernel can load its
scales with fewer, more efficient memory transactions. It compensates for the fact that
the "natural" row-major layout doesn't match how threads are actually going to consume
the data: without swizzling, each thread ends up reading way more data than it needs for
its GEMM work, because its scales are scattered across cache lines whose other bytes it
doesn't care about.

The swizzled layout rearranges the physical bytes so that each thread's scales land in
one contiguous chunk within a single cache line. One thread, one wide load, one cache
line. The total bytes in memory are the same; only the ordering changes, and both the
writer (quantization) and the reader (GEMM kernel) agree on that ordering via a shared
coordinate formula.

For the full story, read in this order:

1. [background.md](background.md) — general GPU concepts (cache lines, warps,
   coalescing). Skip if you already know them.
2. [coalescing.md](coalescing.md) — byte-level walkthrough of how the NVFP4 swizzle
   achieves coalesced reads for a 128 x 4 scale tile, plus a breakdown of the formula.

## The 128 x 4 scale tile

For the `VEC16_UE4M3` FP4 mode (16-element blocks, unsigned E4M3 scales), one scale tile
is **128 x 4**: it covers a 128 x 64 region of the data matrix (128 rows, 64 columns,
with 64 / 16 = 4 blocks per row).

The tile-local mapping from the cuBLASLt docs is:

```
offset = (outer % 32) * 16 + (outer // 32) * 4 + inner
```

where `outer` is the row index within the tile (0..127) and `inner` is the block index
within the tile (0..3).

### What the formula does

The formula interleaves rows that are 32 apart. Within the tile, memory is laid out in
groups of 16 consecutive values:

| Packed position | Source |
|---|---|
| 0..3 | `scales[0, 0:4]` |
| 4..7 | `scales[32, 0:4]` |
| 8..11 | `scales[64, 0:4]` |
| 12..15 | `scales[96, 0:4]` |
| 16..19 | `scales[1, 0:4]` |
| 20..23 | `scales[33, 0:4]` |
| 24..27 | `scales[65, 0:4]` |
| 28..31 | `scales[97, 0:4]` |
| ... | ... |

So the first 16 values are the 4 scales from row 0, then row 32, then row 64, then row
96. After that come row 1, row 33, row 65, row 97, and so on.

This pattern exists because the GEMM kernel processes 32 consecutive rows at a time (one
warp's worth of work). Grouping the scales for rows 0, 32, 64, 96 together means that
when the warp loads its scales, the reads are coalesced into contiguous memory accesses.

### Why the 4 x 16 toy is too small

The existing `nvfp4.py` example uses a 4 x 16 matrix. With 16-element blocks, that gives
only one scale per row, so `inner` is always 0 and the swizzle is trivial. To see the
interleaving, you need at least 128 rows and 4 blocks per row, which means a data matrix
of at least 128 x 64.

## How to run

```bash
cd 06_nvfp4/swizzling
python swizzle.py
```

```
================================================================
Logical scale tensor  (shape 128x4, row-major)
================================================================

First 8 rows:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]
 [24 25 26 27]
 [28 29 30 31]]

Rows 32-35:
[[128 129 130 131]
 [132 133 134 135]
 [136 137 138 139]
 [140 141 142 143]]

Rows 64-67:
[[256 257 258 259]
 [260 261 262 263]
 [264 265 266 267]
 [268 269 270 271]]

Rows 96-99:
[[384 385 386 387]
 [388 389 390 391]
 [392 393 394 395]
 [396 397 398 399]]

================================================================
Packed (swizzled) buffer  (first 32 entries)
================================================================

[  0   1   2   3 128 129 130 131 256 257 258 259 384 385 386 387   4   5
   6   7 132 133 134 135 260 261 262 263 388 389 390 391]

Reading the pattern:

  packed[ 0: 4] = [0 1 2 3]  <- logical[  0, 0:4]
  packed[ 4: 8] = [128 129 130 131]  <- logical[ 32, 0:4]
  packed[ 8:12] = [256 257 258 259]  <- logical[ 64, 0:4]
  packed[12:16] = [384 385 386 387]  <- logical[ 96, 0:4]
  packed[16:20] = [4 5 6 7]  <- logical[  1, 0:4]
  packed[20:24] = [132 133 134 135]  <- logical[ 33, 0:4]
  packed[24:28] = [260 261 262 263]  <- logical[ 65, 0:4]
  packed[28:32] = [388 389 390 391]  <- logical[ 97, 0:4]

The kernel reads 32 consecutive rows at a time (a warp's worth),
so rows 0, 32, 64, 96 are grouped together in memory.

Round-trip check: PASSED (unswizzle recovers the original tensor)

================================================================
Multi-tile example  (256x8 logical -> 2x2 tiles)
================================================================

Packed buffer length: 2048  (= 2 x 2 x 512)
Round-trip check: PASSED
```

### coalescing.py

Shows byte-by-byte what each warp thread reads from the scale tensor in both layouts, and
why the swizzled layout enables coalesced memory access.

```bash
python coalescing.py
```

```
========================================================================
Memory addresses each thread reads  (128x4 scale tile, 16 bytes/thread)
========================================================================

thread  0   (rows [0, 32, 64, 96])
  row-major : [0, 1, 2, 3, 128, 129, 130, 131, 256, 257, 258, 259, 384, 385, 386, 387]
  swizzled  : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

thread  1   (rows [1, 33, 65, 97])
  row-major : [4, 5, 6, 7, 132, 133, 134, 135, 260, 261, 262, 263, 388, 389, 390, 391]
  swizzled  : [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

thread  2   (rows [2, 34, 66, 98])
  row-major : [8, 9, 10, 11, 136, 137, 138, 139, 264, 265, 266, 267, 392, 393, 394, 395]
  swizzled  : [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

thread 31   (rows [31, 63, 95, 127])
  row-major : [124, 125, 126, 127, 252, 253, 254, 255, 380, 381, 382, 383, 508, 509, 510, 511]
  swizzled  : [496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511]

========================================================================
Coalescing analysis for the full warp (32 threads)
========================================================================

  Threads with contiguous 16-byte reads:
    row-major:  0 / 32
    swizzled:  32 / 32

  Full warp covers a contiguous 512-byte range?
    row-major:  True
    swizzled:  True

========================================================================
Visual: row-major layout (which thread reads each byte)
========================================================================

Each cell shows the thread ID (0-31) that reads that byte.
Contiguous thread IDs = coalesced. Gaps = scattered.

  Row-major (first 64 bytes):
  byte:    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
     0:    0  0  0  0  1  1  1  1  2  2  2  2  3  3  3  3
    16:    4  4  4  4  5  5  5  5  6  6  6  6  7  7  7  7
    32:    8  8  8  8  9  9  9  9 10 10 10 10 11 11 11 11
    48:   12 12 12 12 13 13 13 13 14 14 14 14 15 15 15 15

  Threads 0-3 read bytes 0-15, then jump to bytes 128-143.
  4 scattered 16-byte regions per thread = poor coalescing.

  Swizzled (first 64 bytes):
  byte:    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
     0:    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    16:    1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
    32:    2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2
    48:    3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3

  Thread 0 reads bytes 0-15, thread 1 reads 16-31, etc.
  32 threads x 16 contiguous bytes = one 512-byte coalesced load.
```

## Outer vs inner dimensions

For a GEMM input matrix, the scale tensor is indexed by "outer" and "inner" dimensions,
but the meaning depends on the operand:

| Operand | Outer dim | Inner dim | Blocks along |
|---|---|---|---|
| A (M x K) | M | K // 16 | K |
| B (N x K) | N | K // 16 | K |

In both cases, `inner` corresponds to the K (contraction) dimension, and `outer` is M for
A or N for B.

## Source

The authoritative specification is in the [cuBLAS
documentation](https://docs.nvidia.com/cuda/cublas/), section "1D Block Scaling Factors
Layout" (under "16/32-Element 1D Block Scaling for FP8 and FP4 Data Types"). It gives
both the tile layout and the swizzle formula verbatim, and specifies that for NVFP4
(`CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`) a single 128 x 4 scale tile is applied to a
128 x 64 block of source data. (The same tile shape applies to MXFP8 (`VEC32_UE8M0`),
where it covers a 128 x 128 block since blocks there are 32 elements wide.)

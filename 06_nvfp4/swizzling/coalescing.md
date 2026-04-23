# Why the swizzled layout enables coalesced memory access

This file walks through the byte-level picture of how the NVFP4 swizzle achieves
coalesced memory access for a 128 x 4 scale tile. It assumes you're familiar with cache
lines, warps, and coalescing; see [background.md](background.md) if you need a refresher
on those concepts. For the high-level "what is swizzling" intuition, see the top of
[README.md](README.md).

## How the GEMM kernel reads a 128 x 4 scale tile

A 128 x 4 scale tile covers 128 rows of the data matrix, with 4 block-scales per row.
cuBLASLt specifies the scale tile shape as 128 x 4 for the VEC16_UE4M3 FP4 mode. Since
each scale covers one 16-element block along K, each tile spans 4 x 16 = 64 elements
along K. (A real matrix may have K much larger than 64, but the scale tensor is tiled
into 128 x 4 chunks regardless. Multiple tiles cover the full K dimension.)

The GEMM kernel assigns each of the 32 warp threads 4 rows that are 32 apart:

```
thread  0  ->  rows   0,  32,  64,  96
thread  1  ->  rows   1,  33,  65,  97
thread  2  ->  rows   2,  34,  66,  98
...
thread 31  ->  rows  31,  63,  95, 127
```

Why strided rows rather than contiguous ones (e.g. thread 0 getting rows 0-3)? This comes
from how tensor core MMA (matrix multiply-accumulate) instructions distribute matrix
fragments across a warp. When a warp executes an MMA on a 128-row tile, thread `t`'s
output elements come from rows that are strided 32 apart, not from 4 contiguous rows.
**The scale-loading pattern has to match the compute pattern**, so each thread loads
scales for the same rows it will use during the MMA.

Each thread handles 4 rows, and for each row it needs all 4 block-scales (one per
16-element block along K). That is 4 rows x 4 scales = 16 scales. Each E4M3 scale is 1
byte, so **16 bytes per thread**. The entire warp needs 32 x 16 = 512 bytes, which is
exactly one tile.

## Scales and values live in separate memory

The NVFP4 format stores the FP4 _payload_ (the quantized values) and the E4M3 block
scales in **two completely separate buffers** in GPU memory. When the GEMM kernel loads
scales, it reads from the scale buffer only. The FP4 values are elsewhere.

So when we talk about cache lines below, we are talking about the **scale buffer** alone.
Every byte in the 512-byte tile is a scale. The 128-byte cache line is a hardware
property of the GPU memory system (how much DRAM the hardware fetches per transaction),
not a property of the data format.

The question is: where do each thread's 16 scale bytes live within that buffer?

## Row-major layout: scattered reads

In a plain row-major layout, row `r`, column `c` is at address `r * 4 + c` within the
scale buffer. The 512 bytes split across 4 cache lines:

```
cache line 0:  bytes   0 - 127   (scales for rows   0 -  31)
cache line 1:  bytes 128 - 255   (scales for rows  32 -  63)
cache line 2:  bytes 256 - 383   (scales for rows  64 -  95)
cache line 3:  bytes 384 - 511   (scales for rows  96 - 127)
```

Thread 0 needs scales for rows 0, 32, 64, and 96:

```
thread 0 addresses:  [0, 1, 2, 3,  128, 129, 130, 131,  256, 257, 258, 259,  384, 385, 386, 387]
                      ^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^
                      row 0            row 32                 row 64                row 96
                      cache line 0     cache line 1           cache line 2          cache line 3
```

Thread 0 alone touches all 4 cache lines, using only 4 bytes from each. Every other
thread has the same problem: its 16 bytes are scattered across all 4 cache lines.

## Swizzled layout: contiguous reads

The swizzle formula places row `outer`, column `inner` at:

```
offset = (outer % 32) * 16 + (outer // 32) * 4 + inner
```

This rearranges the bytes so each cache line is partitioned by **thread**, not by row:

```
cache line 0:  bytes   0 - 127   (threads  0 -  7, 16 contiguous bytes each)
cache line 1:  bytes 128 - 255   (threads  8 - 15)
cache line 2:  bytes 256 - 383   (threads 16 - 23)
cache line 3:  bytes 384 - 511   (threads 24 - 31)
```

Now thread 0's scales land at:

```
thread 0 addresses:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                      ^^^^^^^^^^  ^^^^^^^^^^  ^^^^^^^^^^^^  ^^^^^^^^^^^^^^
                        row 0        row 32      row 64         row 96
                      all within cache line 0
```

All 16 bytes are contiguous and within a single cache line. Thread 1 reads bytes 16-31
(also cache line 0), thread 8 reads bytes 128-143 (cache line 1), and so on. Each cache
line is read densely by 8 threads with no gaps, so the memory controller issues 4 clean
128-byte transactions and every fetched byte is consumed — and because the swizzle was
designed to match the MMA fragment layout, each thread's 16 bytes are exactly the scales
it needs for its assigned rows.

## Reading the formula

The swizzle formula looks opaque until you recognize that each term corresponds to a
level of the desired thread-major layout:

```
offset = (outer % 32) * 16   +   (outer // 32) * 4   +   inner
          \________/              \_________/            \___/
          which thread            which of the 4         which
          owns this row?          rows that thread       column
          (0..31)                 handles? (0..3)        (0..3)
```

Read left to right, the formula builds an address by nesting three levels:

1. **`(outer % 32) * 16`**: jump to the 16-byte slot owned by this thread. Since thread
   `t` handles rows `t, t+32, t+64, t+96`, all of which share the same `outer % 32 = t`,
   the entire thread's data lives in a single 16-byte region starting at `t * 16`.
2. **`(outer // 32) * 4`**: within the thread's slot, jump to the 4-byte sub-slot for the
   correct row. `outer // 32` is 0, 1, 2, or 3 depending on whether the row is in `[0,
   32)`, `[32, 64)`, `[64, 96)`, or `[96, 128)`. This orders the thread's 4 rows within
   its 16-byte slot.
3. **`inner`**: within the 4-byte sub-slot, pick the column (block index along K).

In short, the layout is "thread-major": the outermost grouping is by thread, the middle
grouping is by which of the thread's 4 rows, and the innermost grouping is by K block.
Row-major is the opposite nesting order (outermost is row, innermost is column), which is
why it scatters each thread's data across the buffer.

### Worked example

Row-major bytes 128–131 correspond to logical coordinates (row=32, col=0..3). Plugging
into the formula:

- `(32 % 32) * 16 = 0` — thread 0 owns row 32
- `(32 // 32) * 4 = 4` — this is the second of thread 0's four rows
- `inner = 0, 1, 2, 3`

So row-major bytes 128–131 map to swizzled offsets 4–7, landing immediately after bytes
0–3 (row 0's scales) in thread 0's contiguous 16-byte slot.

## Summary

|  | Row-major | Swizzled |
|---|---|---|
| Threads with contiguous 16-byte reads | 0 / 32 | 32 / 32 |
| Warp covers a contiguous 512-byte range | yes | yes |
| Cache lines touched per thread | 4 (scattered) | 1 (contiguous) |

Both layouts contain the same 512 bytes. Row-major spreads each thread's bytes across 4
distant regions. The swizzle gathers them into one contiguous chunk per thread, so the
warp loads the entire tile in a single coalesced memory transaction.

Run `python coalescing.py` to see the full byte-level access patterns for all 32 threads.

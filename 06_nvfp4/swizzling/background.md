# Background: cache lines, warps, and coalescing

This file covers the general GPU memory concepts you need to follow the NVFP4 swizzling
walkthrough in [coalescing.md](coalescing.md). None of this is NVFP4-specific.

## What is a cache line?

When a GPU thread asks for a single byte from global memory (DRAM), the hardware does
not fetch just that one byte. It fetches a fixed-size chunk called a **cache line**,
typically **128 bytes** on NVIDIA GPUs. That chunk is loaded into the L2 cache (and
possibly L1), so nearby bytes are immediately available without another DRAM round-trip.

This is the same principle as CPU caches, just tuned for different access patterns.

## Warps

GPU threads execute in groups of 32 called **warps**. All 32 threads in a warp run the
same instruction at the same time, in lockstep. When the instruction is a memory load,
all 32 threads issue their loads simultaneously, and the hardware tries to service them
together.

## Coalesced vs scattered access

The memory controller groups the warp's 32 simultaneous requests by which cache line
they fall into:

- If all 32 threads hit the **same** cache line (or a small number of adjacent lines),
  the hardware services them in one or a few transactions. This is **coalesced** access.
- If the 32 threads hit **many different** cache lines, each line must be fetched
  separately. The hardware does the same total work per line, but issues many more
  transactions. This is **scattered** access.

The penalty is wasted memory bandwidth. Each cache-line fetch pulls 128 bytes from DRAM
regardless of how many are actually used, so scattered reads waste most of that bandwidth
on bytes no thread asked for. The memory controller also has to issue many small
transactions instead of a few large ones, which adds overhead on top.

The goal of any GPU memory layout is therefore: **arrange data so that when a warp
issues 32 simultaneous loads, those loads fall into as few cache lines as possible, and
every byte fetched gets used.**

## Summary

| Term | Meaning |
|---|---|
| Cache line | Fixed-size chunk (128 bytes on NVIDIA GPUs) fetched as a unit from DRAM |
| Warp | Group of 32 threads that execute in lockstep, including memory loads |
| Coalesced access | Warp's loads fall into few cache lines, used densely |
| Scattered access | Warp's loads span many cache lines, wasting bandwidth |

With these concepts in hand, [coalescing.md](coalescing.md) walks through the specific
NVFP4 case: how the 128 x 4 scale tile is laid out so that one warp can load its scales
in a single coalesced transaction.

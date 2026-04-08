"""
Toy NumPy illustration of Hopper-style vs Blackwell-style GEMM quantization.

On Hopper, quantizing GEMM outputs typically requires separate kernel launches: one for
the matmul itself, then additional passes to compute amax / scales and quantize the full
output tensor. Each pass reads/writes the entire output from global memory.

On Blackwell, the GEMM epilogue can fuse quantization directly into the matmul kernel. As
soon as a tile's accumulators are finished, that tile is quantized and its per-block
scales are emitted immediately, avoiding extra global memory round-trips.

This file is pure Python/NumPy to show the structural difference. It does not use real
NVFP4, CUTLASS, or any GPU code.
"""

import numpy as np
from numpy import random


def _quantize_blockwise_symmetric(
    x: np.ndarray, block_size: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """
    Toy blockwise quantization:
    - stores quantized values in int8, pretending they are "int4-like" codes in [-7, 7]
    - returns:
        q: quantized tensor (int8)
        scales: one scale per contiguous block of `block_size` values
    """
    flat = x.reshape(-1)
    q = np.empty_like(flat, dtype=np.int8)
    scales = []

    num_levels = 7.0  # pretend signed 4-bit-ish range is [-7, 7]

    for start in range(0, flat.size, block_size):
        block = flat[start : start + block_size]
        amax = np.max(np.abs(block)) if block.size > 0 else 0.0
        scale = amax / num_levels if amax > 0 else 1.0
        q_block = np.round(block / scale).clip(-7, 7).astype(np.int8)

        q[start : start + block_size] = q_block
        scales.append(scale)

    return q.reshape(x.shape), np.array(scales, dtype=np.float32)


def separate_pipeline(
    A: np.ndarray, B: np.ndarray, block_size: int = 16
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hopper-style separate-kernel pipeline:
      1) full-precision GEMM, writing the entire fp32 result to global memory
      2) separate pass that reads it back to compute scales, quantize, and store
    """
    # Step 1: full-precision GEMM. The entire fp32 result is written to memory.
    C_fp32 = A @ B

    # Step 2: a separate pass reads back C_fp32 to compute scales and quantize.
    C_q, C_scales = _quantize_blockwise_symmetric(C_fp32, block_size=block_size)

    return C_fp32, C_q, C_scales


def fused_epilogue_pipeline(
    A: np.ndarray, B: np.ndarray, tile_m: int = 4, tile_n: int = 4, block_size: int = 16
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    'Blackwell-ish fused epilogue' conceptual pipeline:
      - do tiled GEMM
      - as soon as a tile's accumulators are finished,
        quantize that tile and emit its scales immediately

    This mimics the idea of 'fusing quantization into the GEMM epilogue'.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C_q = np.empty((M, N), dtype=np.int8)
    scale_list = []

    for m0 in range(0, M, tile_m):
        for n0 in range(0, N, tile_n):
            m1 = min(m0 + tile_m, M)
            n1 = min(n0 + tile_n, N)

            # Mainloop: accumulate this output tile
            acc = np.zeros((m1 - m0, n1 - n0), dtype=np.float32)
            for k0 in range(K):
                a_vec = A[m0:m1, k0 : k0 + 1]  # shape [tm, 1]
                b_vec = B[k0 : k0 + 1, n0:n1]  # shape [1, tn]
                acc += a_vec @ b_vec

            # Epilogue: quantize the finished tile immediately
            q_tile, scales_tile = _quantize_blockwise_symmetric(
                acc, block_size=block_size
            )
            C_q[m0:m1, n0:n1] = q_tile
            scale_list.append(scales_tile)

    return C_q, scale_list


if __name__ == "__main__":
    rng = random.default_rng(260408)
    A = rng.standard_normal((8, 16), dtype=np.float32)
    B = rng.standard_normal((16, 8), dtype=np.float32)

    C_fp32, C_q_sep, C_scales_sep = separate_pipeline(A, B, block_size=16)
    # C_fp32: the full (8, 8) fp32 result must be materialised before quantization.
    # This is the cost of separate kernels: an extra full-tensor read/write.
    #
    # C_scales_sep: one flat scale per block of 16 over all 64 output elements -> shape
    # (4,)
    C_q_fused, fused_scales = fused_epilogue_pipeline(
        A, B, tile_m=4, tile_n=4, block_size=16
    )

    print("Separate pipeline fp32 shape (expect (8, 8)):", C_fp32.shape)
    print("Separate pipeline output shape (expect (8, 8)):", C_q_sep.shape)
    print("Separate pipeline scales shape (expect (4,)):", C_scales_sep.shape)
    print("Fused epilogue output shape (expect (8, 8)):", C_q_fused.shape)
    # (8/tile_m) * (8/tile_n) = 2 * 2 = 4 tiles
    print("Number of fused tiles (expect 4):", len(fused_scales))

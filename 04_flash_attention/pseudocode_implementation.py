import math
import torch


def flash_attn_v1_kernel_py(
    Q: torch.Tensor,  # [N, D]
    K: torch.Tensor,  # [N, D]
    V: torch.Tensor,  # [N, D]
    Tr: int = 16,  # rows in a Q tile (outer loop tile size)
    Tc: int = 16,  # rows in a K/V tile (inner loop tile size)
) -> torch.Tensor:
    """
    “Whiteboard-aligned” Python reference that mirrors the CUDA teaching kernel’s control flow
    and state updates (float, causal). Single (batch, head) slice only: Q/K/V are [N, D].

    Assumptions (checked):
      - N is a multiple of Tr and Tc (so we do *not* need ragged tiles).
      - causal attention (lower-triangular).

    This mimics the CUDA kernel structure:
      - outer loop over q_tile_idx
      - per-row streaming state: row_max (m_i), row_sumexp (l_i)
      - inner loop over kv_tile_idx (up to causal boundary)
      - explicit loops for dot products / pv accumulation
      - masking handled like CUDA: `masked = (col > q_row)` with s=-inf and p=0
      - sh_S is overwritten in-place from scores -> exp(scores - row_max_new)
    """
    assert Q.dim() == 2 and K.dim() == 2 and V.dim() == 2, "Q, K, V must be 2D"
    N, D = Q.shape
    assert K.shape == (N, D) and V.shape == (N, D), "K, V must match Q shape"

    # Enforce exact tiling like the CUDA kernel (no ragged tiles).
    assert N % Tr == 0, f"N={N} must be a multiple of Tr={Tr}"
    assert N % Tc == 0, f"N={N} must be a multiple of Tc={Tc}"

    softmax_scale = 1 / math.sqrt(D)

    num_q_tiles = N // Tr
    num_kv_tiles = N // Tc

    O = torch.empty((N, D), dtype=Q.dtype)

    neg_inf = -float("inf")

    # Shared-memory analogs (allocated once; overwritten each tile).
    sh_Q = torch.empty((Tr, D), dtype=Q.dtype)
    sh_K = torch.empty((Tc, D), dtype=Q.dtype)
    sh_V = torch.empty((Tc, D), dtype=Q.dtype)
    sh_S = torch.empty((Tr, Tc), dtype=Q.dtype)  # scores/probs scratch
    # numerator accumulator for current Q tile
    sh_O_accum = torch.empty((Tr, D), dtype=Q.dtype)

    # Outer loop: iterate over Q tiles.
    for q_tile_idx in range(num_q_tiles):
        q_tile_row0 = q_tile_idx * Tr

        # Load Q tile and zero numerator accumulator (like CUDA: each thread loads one row).
        for row_in_q_tile in range(Tr):
            q_row = q_tile_row0 + row_in_q_tile  # global query row
            for d in range(D):
                sh_Q[row_in_q_tile, d] = Q[q_row, d]
                sh_O_accum[row_in_q_tile, d] = 0.0

        # Per-row streaming-softmax state (kept "in registers" conceptually).
        row_max = [neg_inf] * Tr
        row_sumexp = [0.0] * Tr

        # Causal pruning: skip KV tiles strictly in the future.
        kv_tile_max = q_tile_idx
        # if kv_tile_max > (num_kv_tiles - 1):
        #     kv_tile_max = num_kv_tiles - 1

        # Inner loop: stream over K/V tiles up to causal boundary.
        for kv_tile_idx in range(kv_tile_max + 1):
            kv_tile_row0 = kv_tile_idx * Tc

            # Step A: load K/V tile into "shared"
            for row_in_kv_tile in range(Tc):
                kv_row = kv_tile_row0 + row_in_kv_tile
                for d in range(D):
                    sh_K[row_in_kv_tile, d] = K[kv_row, d]
                    sh_V[row_in_kv_tile, d] = V[kv_row, d]

            # Step B: compute scores sh_S[row_in_q_tile, k] and per-row tile_row_max
            tile_row_max = [neg_inf] * Tr
            for row_in_q_tile in range(Tr):
                q_row = q_tile_row0 + row_in_q_tile
                rmax = neg_inf

                for k in range(Tc):
                    # global key position
                    k_col = kv_tile_row0 + k

                    s = neg_inf
                    masked = k_col > q_row  # causal mask
                    if not masked:
                        acc = 0.0
                        for d in range(D):
                            acc += sh_Q[row_in_q_tile, d] * sh_K[k, d]
                        s = acc * softmax_scale

                    sh_S[row_in_q_tile, k] = s
                    if s > rmax:
                        rmax = s

                tile_row_max[row_in_q_tile] = rmax

            # Step C + D: for each row, merge streaming max/sumexp and update numerator
            # accumulator.
            for row_in_q_tile in range(Tr):
                # Merge max frames
                rm = row_max[row_in_q_tile]
                tm = tile_row_max[row_in_q_tile]
                row_max_new = rm if rm > tm else tm

                # rescale old frame -> new frame
                rescale_old = math.exp(rm - row_max_new) if rm != neg_inf else 0.0

                # Convert scores -> probs in the new max frame (overwrite sh_S in place)
                tile_sumexp = 0.0
                for k in range(Tc):
                    s = sh_S[row_in_q_tile, k]
                    if s == neg_inf:  # s was masked out
                        p = 0.0
                    else:
                        p = math.exp(s - row_max_new)
                    sh_S[row_in_q_tile, k] = p
                    tile_sumexp += p

                row_sumexp_new = rescale_old * row_sumexp[row_in_q_tile] + tile_sumexp

                # Step D: dot product of sh_S with  pv[d] = sum_k P_{i,k} * V_k[d]
                # and merge numerator accumulator in the same frame:
                #   O_accum_new[d] = rescale_old * O_accum_old[d] + pv[d]
                for d in range(D):
                    pv = 0.0
                    for k in range(Tc):
                        pv += sh_S[row_in_q_tile, k] * sh_V[k, d]
                    sh_O_accum[row_in_q_tile, d] = (
                        rescale_old * sh_O_accum[row_in_q_tile, d] + pv
                    )

                # Commit streaming state
                row_max[row_in_q_tile] = row_max_new
                row_sumexp[row_in_q_tile] = row_sumexp_new

        # Final normalization: O = numerator / denominator
        for row_in_q_tile in range(Tr):
            q_row = q_tile_row0 + row_in_q_tile
            inv = 1.0 / row_sumexp[row_in_q_tile]
            for d in range(D):
                O[q_row, d] = sh_O_accum[row_in_q_tile, d] * inv

    return O


def naive_attention_py(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal=True
) -> torch.Tensor:
    N, D = Q.shape

    softmax_scale = 1 / math.sqrt(D)

    S = Q @ K.T * softmax_scale
    if causal:
        N = Q.shape[0]
        # https://docs.pytorch.org/docs/stable/generated/torch.triu.html
        mask = torch.triu(
            torch.ones((N, N), dtype=torch.bool, device=Q.device), diagonal=1
        )
        S = S.masked_fill(mask, -float("inf"))
    P = torch.softmax(S, dim=-1)
    return P @ V


if __name__ == "__main__":
    torch.manual_seed(260120)
    N, D = 64, 32
    Tr, Tc = 16, 16
    Q = torch.randn(N, D)
    K = torch.randn(N, D)
    V = torch.randn(N, D)
    scale = 1.0 / math.sqrt(D)

    O_ref = naive_attention_py(Q, K, V, causal=True)
    O_fa = flash_attn_v1_kernel_py(Q, K, V, Tr=Tr, Tc=Tc)

    diff = (O_ref - O_fa).abs().max().item()
    print("max abs diff:", diff)
    if diff < 1e-6:
        print(
            "Pseudocode FlashAttention implementation matches standard attention implementation"
        )

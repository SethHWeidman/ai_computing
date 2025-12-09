import math

import torch


def softmax_dot_full(q: torch.Tensor, v: torch.Tensor) -> float:
    """
    Compute (softmax(q) · v) in the usual "all-at-once" way,
    using a numerically stable softmax.
    """
    # Safe softmax over the whole vector
    max_q = torch.max(q)
    exp_q = torch.exp(q - max_q)
    weights = exp_q / torch.sum(exp_q)
    result = torch.sum(weights * v).item()
    return result


def softmax_dot_streaming(
    q: torch.Tensor, v: torch.Tensor, block_size: int = 2
) -> float:
    """
    Compute (softmax(q) · v) in a streaming way.

    We imagine that q and v arrive in small blocks instead of all at once. We keep track
    of:
      - running_max (m): max score seen so far
      - running_sum (l): denominator of softmax, in the shifted space
      - running_out (o): numerator of softmax_dot, in the shifted space

    At the end, result = running_out / running_sum.
    """
    n = q.numel()

    running_max = float("-inf")  # m
    running_out = 0.0  # top
    running_sum = 0.0  # bottom

    for start in range(0, n, block_size):
        q_block = q[start : start + block_size]
        v_block = v[start : start + block_size]

        # Local max within this block (for numerical stability)
        block_max = torch.max(q_block).item()

        # New global max across all blocks seen so far
        new_global_max = max(running_max, block_max)

        # Rescale old contributions so they share the same max
        scale_accumulation = math.exp(running_max - new_global_max)

        # Exponentials for this block, scaled by the new global max
        exp_scores = torch.exp(q_block - new_global_max)

        # Block contributions (already scaled by new_global_max)
        block_out = torch.sum(exp_scores * v_block).item()
        block_sum = torch.sum(exp_scores).item()

        rescaled_prior_output = running_out * scale_accumulation
        rescaled_prior_sum = running_sum * scale_accumulation

        running_out = rescaled_prior_output + block_out
        running_sum = rescaled_prior_sum + block_sum

        # update running max
        running_max = new_global_max

    result = running_out / running_sum
    return result


def run_demo() -> None:
    # Use a fixed seed for reproducibility.
    torch.manual_seed(251209)
    q = torch.rand(100) * 2
    v = torch.rand(100) * 2

    print(f"Vector length                        = {q.numel()}")
    print(f"Block size                           = 10")

    # 1. Standard "all-at-once" softmax-dot
    full_result = softmax_dot_full(q, v)
    print(f"Full softmax · V: {full_result:.6f}")

    # 2. Streaming version, processing 10 elements at a time
    streaming_result = softmax_dot_streaming(q, v, block_size=10)
    print(f"Streaming softmax · V: {streaming_result:.6f}")

    # 3. Verify they match
    diff = abs(full_result - streaming_result)
    print(f"Difference                           = {diff:.8f}")
    if diff < 1e-6:
        print("✅ SUCCESS: Streaming result matches full softmax-dot.")
    else:
        print("❌ MISMATCH: Online result does not match.")


if __name__ == "__main__":
    run_demo()

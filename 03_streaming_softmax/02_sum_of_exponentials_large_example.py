import math
import random


def sum_of_scaled_exponentials_full(values: list[float]) -> float:
    """
    "All at once" sum of scaled exponentials (softmax denominator).

    Computes sum_j exp(x_j - max_all) over all values.
    """
    max_all = max(values)
    exps = [math.exp(x - max_all) for x in values]
    return sum(exps)


def sum_of_scaled_exponentials_streaming(
    values: list[float], block_size: int = 50
) -> float:
    """
    Streaming sum of scaled exponentials (softmax denominator).

    Processes blocks one at a time, keeping track of:
      - running_max (m): max value seen so far
      - running_sum (l): sum of exp(x - running_max) in the shifted space
    """
    running_max = float("-inf")
    running_sum = 0.0

    n = len(values)
    for start in range(0, n, block_size):
        block = values[start : start + block_size]
        block_max = max(block)
        new_global_max = max(running_max, block_max)

        # Rescale old and new contributions so they share the same max
        scale_running_sum = math.exp(running_max - new_global_max)
        rescaled_prior_sum = running_sum * scale_running_sum

        # Exponentials for this block, scaled by the new global max
        block_exps = [math.exp(x - new_global_max) for x in block]
        block_sum = sum(block_exps)

        running_sum = rescaled_prior_sum + block_sum
        running_max = new_global_max

    return running_sum


def main() -> None:
    # Initialize a vector of length 1000 with random integers from 1 to 100, using a
    # fixed seed for reproducibility.
    random.seed(251209)
    values = [random.randint(1, 100) for _ in range(1_000)]

    full_sum = sum_of_scaled_exponentials_full(values)
    streaming_sum = sum_of_scaled_exponentials_streaming(values, block_size=50)
    diff = abs(full_sum - streaming_sum)

    print(f"Vector length                        = {len(values):,}")
    print(f"Block size                           = 50")
    print(f"Full sum of scaled exponentials      = {full_sum:.6f}")
    print(f"Streaming sum of scaled exponentials = {streaming_sum:.6f}")
    print(f"Difference                           = {diff:.8f}")
    if diff < 1e-9:
        print("✅ SUCCESS: Streaming sum of scaled exponentials matches full sum.")
    else:
        print(
            "❌ MISMATCH: Streaming sum of scaled exponentials does not match full sum."
        )


if __name__ == "__main__":
    main()

import math
import typing

VALUES = [0.5, 0.6, 0.0, 0.2, 0.8, 0.1]


def format_floats(values: typing.Iterable[float], decimals: int = 4) -> str:
    return "[" + ", ".join(format(x, f".{decimals}f") for x in values) + "]"


def sum_of_scaled_exponentials_full(values: list[float]) -> float:
    """
    "All at once" sum of scaled exponentials (softmax denominator).

    Computes sum_j exp(x_j - max_all) over all values.
    """
    max_all = max(values)
    return sum([math.exp(x - max_all) for x in values])


def sum_of_scaled_exponentials_streaming(
    values: list[float], block_size: int = 3, verbose: bool = False
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
    for i, start in enumerate(range(0, n, block_size)):
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

        if verbose:
            if i == 0:
                # First block
                print(f"First block max m_old = {block_max:.1f}")
                print(f"exp(x - m_old): {format_floats(block_exps, decimals=4)}")
                print(f"Running sum after first block (l_old) = {running_sum:.6f}")
                print()
            else:
                # Subsequent blocks
                label = "Second block" if i == 1 else f"Block {i}"
                print(f"{label} max m_block = {block_max:.1f}")
                print(f"exp(x - m_block): {format_floats(block_exps, decimals=4)}")
                print(f"Block sum (before rescaling) = {block_sum:.6f}")
                print()

                print("=== Rescaling step ===")
                print(f"New global max m_new = {new_global_max:.1f}")
                print(
                    "scale_running_sum = exp(m_old - m_new) = "
                    f"{scale_running_sum:.6f}"
                )
                print()

                print(f"Rescaled prior sum   = {rescaled_prior_sum:.6f}")
                print(f"New block sum        = {block_sum:.6f}")
                print(f"Running sum (online) = {running_sum:.6f}")

    return running_sum


def run_demo() -> None:
    print("All values:", format_floats(VALUES, decimals=1))
    print()

    # 1. Standard "all-at-once" sum of exponentials
    full_result = sum_of_scaled_exponentials_full(VALUES)

    # 2. Streaming version, processing 3 elements at a time
    print("Streaming computation (block_size=3):")
    print()
    streaming_result = sum_of_scaled_exponentials_streaming(
        VALUES, block_size=3, verbose=True
    )
    print(f"\nStreaming sum of scaled exponentials: {streaming_result:.6f}")

    print(f"Full sum of scaled exponentials: {full_result:.6f}")

    # 3. Verify they match
    diff = abs(full_result - streaming_result)
    print(f"\nDifference: {diff:.8f}")
    if diff < 1e-6:
        print("✅ SUCCESS: Streaming result matches full sum.")
    else:
        print("❌ MISMATCH: Online result does not match.")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

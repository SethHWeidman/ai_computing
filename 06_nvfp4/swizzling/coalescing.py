"""
Why the swizzled scale layout enables coalesced memory access.

This script shows, byte-by-byte, what each GPU thread in a warp needs to read from the
scale tensor, and how the swizzled layout turns scattered reads into one contiguous load.

Background
----------
A GPU warp has 32 threads that execute in lockstep. When all 32 threads issue memory
loads at the same time, the hardware can *coalesce* those loads into a small number of
cache-line transactions if the addresses are contiguous. If the addresses are scattered,
each thread's load may hit a different cache line, wasting memory bandwidth.

For a 128x4 scale tile (128 rows, 4 blocks per row), the GEMM kernel assigns work so that
each of the 32 threads in a warp handles 4 rows that are 32 rows apart:

    thread  0  ->  rows   0,  32,  64,  96
    thread  1  ->  rows   1,  33,  65,  97
    thread  2  ->  rows   2,  34,  66,  98
    ...
    thread 31  ->  rows  31,  63,  95, 127

Each thread needs all 4 block-scales for each of its 4 rows, so it needs 4 rows x 4
scales = 16 bytes total.

The question is: where do those 16 bytes live in memory?
"""


def row_major_addresses(thread_id: int, num_cols: int = 4) -> list[int]:
    """
    Byte addresses that `thread_id` must read from a plain row-major layout.

    The scale tensor is stored as a flat buffer: row r, column c is at address r *
    num_cols + c.

    Thread `thread_id` handles rows thread_id, thread_id+32, thread_id+64, thread_id+96.
    For each row it reads all `num_cols` scales.
    """
    addrs = []
    for group in range(4):
        row = thread_id + group * 32
        for col in range(num_cols):
            addrs.append(row * num_cols + col)
    return addrs


def swizzled_addresses(thread_id: int, num_cols: int = 4) -> list[int]:
    """
    Byte addresses that `thread_id` must read from the swizzled layout.

    The swizzle formula places row `outer`, column `inner` at:

        offset = (outer % 32) * 16 + (outer // 32) * 4 + inner

    Same rows and columns as row_major_addresses.
    """
    addrs = []
    for group in range(4):
        row = thread_id + group * 32
        for col in range(num_cols):
            offset = (row % 32) * 16 + (row // 32) * 4 + col
            addrs.append(offset)
    return addrs


def is_contiguous(addrs: list[int]) -> bool:
    """Check whether each thread's 16 bytes are contiguous"""
    return addrs == list(range(addrs[0], addrs[0] + len(addrs)))


def print_map(addr_map: dict[int, int], label: str) -> None:
    """# Print first 64 bytes of each layout as a 4x16 grid"""
    print(f"  {label}")
    print(f"  byte:  ", end="")
    for col in range(16):
        print(f"{col:3d}", end="")
    print()
    for row_start in range(0, 64, 16):
        print(f"  {row_start:4d}:  ", end="")
        for col in range(16):
            addr = row_start + col
            tid = addr_map.get(addr, -1)
            print(f"{tid:3d}", end="")
        print()
    print()


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1. Show the addresses for a few threads in both layouts
    # ------------------------------------------------------------------

    print("=" * 72)
    print("Memory addresses each thread reads  (128x4 scale tile, 16 bytes/thread)")
    print("=" * 72)

    for tid in [0, 1, 2, 31]:
        rm = row_major_addresses(tid)
        sw = swizzled_addresses(tid)

        rows = [tid + g * 32 for g in range(4)]

        print(f"\nthread {tid:2d}   (rows {rows})")
        print(f"  row-major : {rm}")
        print(f"  swizzled  : {sw}")

    # ------------------------------------------------------------------
    # 2. Highlight the coalescing difference
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("Coalescing analysis for the full warp (32 threads)")
    print("=" * 72)

    # Row-major: collect all addresses across the warp
    rm_all = []
    for tid in range(32):
        rm_all.extend(row_major_addresses(tid))

    # Swizzled: same
    sw_all = []
    for tid in range(32):
        sw_all.extend(swizzled_addresses(tid))

    rm_contig = sum(1 for t in range(32) if is_contiguous(row_major_addresses(t)))
    sw_contig = sum(1 for t in range(32) if is_contiguous(swizzled_addresses(t)))

    print(f"\n  Threads with contiguous 16-byte reads:")
    print(f"    row-major:  {rm_contig} / 32")
    print(f"    swizzled:  {sw_contig} / 32")

    # Check whether the entire warp's load covers a contiguous range
    rm_sorted = sorted(rm_all)
    sw_sorted = sorted(sw_all)

    rm_is_dense = rm_sorted == list(range(rm_sorted[0], rm_sorted[0] + len(rm_sorted)))
    sw_is_dense = sw_sorted == list(range(sw_sorted[0], sw_sorted[0] + len(sw_sorted)))

    print(f"\n  Full warp covers a contiguous 512-byte range?")
    print(f"    row-major:  {rm_is_dense}")
    print(f"    swizzled:  {sw_is_dense}")

    # ------------------------------------------------------------------
    # 3. Show the access pattern visually
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("Visual: row-major layout (which thread reads each byte)")
    print("=" * 72)
    print()
    print("Each cell shows the thread ID (0-31) that reads that byte.")
    print("Contiguous thread IDs = coalesced. Gaps = scattered.")
    print()

    # Build a map: address -> thread_id for row-major
    rm_map = {}
    for tid in range(32):
        for addr in row_major_addresses(tid):
            rm_map[addr] = tid

    sw_map = {}
    for tid in range(32):
        for addr in swizzled_addresses(tid):
            sw_map[addr] = tid

    print_map(rm_map, "Row-major (first 64 bytes):")
    print("  Threads 0-3 read bytes 0-15, then jump to bytes 128-143.")
    print("  4 scattered 16-byte regions per thread = poor coalescing.")
    print()

    print_map(sw_map, "Swizzled (first 64 bytes):")
    print("  Thread 0 reads bytes 0-15, thread 1 reads 16-31, etc.")
    print("  32 threads x 16 contiguous bytes = one 512-byte coalesced load.")

#!/usr/bin/env python3
import argparse
from os import environ
import time
import pathlib
import typing
import types  # Provides ModuleType, the runtime type of imported modules

import torch
from torch import cuda
from torch.utils import cpp_extension


def load_extension() -> types.ModuleType:
    here = pathlib.Path(__file__).resolve().parent
    environ.setdefault("TORCH_EXTENSIONS_DIR", str(here / ".torch_extensions"))
    # Assume NVIDIA L4 (sm_89); change if you're not on an L4.
    environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    src = here / "fast_matmul_extension.cu"
    return cpp_extension.load(
        name="fast_matmul_shared",
        sources=[str(src)],
        verbose=False,
    )


def benchmark(
    fn: typing.Callable[[], torch.Tensor], repeats: int
) -> tuple[torch.Tensor, float]:
    cuda.synchronize()
    start = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = fn()
    cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    return result, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare shared-memory CUDA matmul with torch.mm",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2048,
        help="Square matrix dimension to multiply (default: 2048)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs to average per implementation (default: 3)",
    )
    args = parser.parse_args()

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this comparison")

    torch.manual_seed(123)
    cuda.manual_seed_all(123)

    module = load_extension()
    device = torch.device("cuda")
    size = args.size

    with torch.no_grad():
        A = torch.randn(size, size, device=device, dtype=torch.float32)
        B = torch.randn(size, size, device=device, dtype=torch.float32)

        # Warm-up both kernels once so compilation/caching does not skew timings.
        module.fast_mm(A, B)
        torch.mm(A, B)
        cuda.synchronize()

        fast_result, fast_time = benchmark(lambda: module.fast_mm(A, B), args.repeats)
        torch_result, torch_time = benchmark(lambda: torch.mm(A, B), args.repeats)

        is_close = torch.allclose(fast_result, torch_result, rtol=1e-3, atol=1e-3)

    print(f"Matrix size: {size} x {size}")
    print(
        f"fast shared CUDA kernel: {fast_time * 1000:.2f} ms (avg over {args.repeats})"
    )
    print(
        f"torch.mm (cuBLAS):      {torch_time * 1000:.2f} ms (avg over {args.repeats})"
    )
    print(f"Relative speed (torch.mm / shared kernel): {torch_time / fast_time:.2f}x")
    print(f"Numerical match: {'yes' if is_close else 'NO!'}")


if __name__ == "__main__":
    main()

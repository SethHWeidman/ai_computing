#!/usr/bin/env python3

"""Minimal one-GPU Hopper GEMM runner.

This script stays close to the Hopper path used in `flash_attn/cute/flash_fwd_sm90.py`:
both rely on Hopper WGMMA kernels built with `sm90_utils.make_trivial_tiled_mma(...)`.

Instead of re-implementing the full SM90 GEMM kernel here, we load the existing CUTLASS
CuTe Hopper GEMM example from this repo and expose a much smaller command-line entry
point tuned for a single H100.
"""

import argparse
from importlib import util
import pathlib
import types

from torch import cuda

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HOPPER_GEMM_EXAMPLE = (
    REPO_ROOT
    / "csrc"
    / "cutlass"
    / "examples"
    / "python"
    / "CuTeDSL"
    / "hopper"
    / "dense_gemm.py"
)


def parse_csv_ints(value: str, expected_len: int) -> tuple[int, ...]:
    parts = tuple(int(x.strip()) for x in value.split(","))
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(
            f"expected {expected_len} comma-separated integers, got: {value!r}"
        )
    return parts


def load_hopper_dense_gemm_module() -> types.ModuleType:
    spec = util.spec_from_file_location("cute_hopper_dense_gemm", HOPPER_GEMM_EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {HOPPER_GEMM_EXAMPLE}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a large single-GPU FP16 GEMM on Hopper/H100 via CuTe DSL."
    )
    parser.add_argument(
        "--mnk",
        type=lambda s: parse_csv_ints(s, 3),
        default=(8192, 8192, 8192),
        help="Problem size as M,N,K. Default: 8192,8192,8192",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch count L in the underlying MxNxKxL GEMM. Default: 1",
    )
    parser.add_argument(
        "--tile-shape",
        type=lambda s: parse_csv_ints(s, 2),
        default=(128, 256),
        help="CTA tile shape M,N. Default: 128,256",
    )
    parser.add_argument(
        "--cluster-shape",
        type=lambda s: parse_csv_ints(s, 2),
        default=(1, 1),
        help="Cluster shape M,N. Default: 1,1",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="CUDA device index to run on. Default: 0"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations before benchmarking. Default: 3",
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Benchmark iterations. Default: 10"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run the reference check. This is much slower for large GEMMs.",
    )
    return parser.parse_args()


def main() -> None:
    if not cuda.is_available():
        raise RuntimeError("CUDA is required to run this example.")

    args = parse_args()
    cuda.set_device(args.device)
    device_name = cuda.get_device_name(args.device)
    cc = cuda.get_device_capability(args.device)
    if cc[0] != 9:
        raise RuntimeError(
            "This example is intended for Hopper SM90 GPUs; got compute capability "
            f"{cc}."
        )

    mod = load_hopper_dense_gemm_module()

    m, n, k = args.mnk
    l = args.batch
    exec_time_us = mod.run(
        (m, n, k, l),
        mod.cutlass.Float16,
        mod.cutlass.Float16,
        mod.cutlass.Float16,
        mod.cutlass.Float32,
        "k",
        "k",
        "n",
        args.tile_shape,
        args.cluster_shape,
        tolerance=1e-1,
        warmup_iterations=args.warmup,
        iterations=args.iterations,
        skip_ref_check=not args.check,
        use_cold_l2=False,
    )

    tflops = (2.0 * m * n * k * l) / (exec_time_us * 1e-6) / 1e12
    print()
    print(f"GPU: {device_name} (device {args.device}, cc {cc[0]}.{cc[1]})")
    print(
        f"Problem: M={m}, N={n}, K={k}, batch={l}, tile={args.tile_shape}, "
        f"cluster={args.cluster_shape}"
    )
    print(f"Average kernel time: {exec_time_us:.1f} us")
    print(f"Throughput: {tflops:.2f} TFLOP/s")


if __name__ == "__main__":
    main()

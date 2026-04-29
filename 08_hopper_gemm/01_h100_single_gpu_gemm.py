#!/usr/bin/env python3

"""Minimal one-GPU Hopper GEMM runner.

This script stays close to the Hopper path used in `flash_attn/cute/flash_fwd_sm90.py`:
both rely on Hopper WGMMA kernels built with `sm90_utils.make_trivial_tiled_mma(...)`.

Instead of re-implementing the full SM90 GEMM kernel here, we load the existing CUTLASS
CuTe Hopper GEMM example from this repo and expose a much smaller command-line entry
point tuned for a single H100.
"""

import argparse

import hopper_gemm_helpers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a large single-GPU FP16 GEMM on Hopper/H100 via CuTe DSL."
    )
    # One CTA computes one MxN output tile. The Hopper kernel supplies the K tile
    # internally; with the default this prints as tile_shape_mnk=(128, 256, 64).
    # cluster-shape groups neighboring CTAs in the M,N tile grid. For example,
    # tile=(128,256), cluster=(2,1) covers a 256 x 256 output region while each CTA
    # still owns one 128 x 256 tile. In this kernel, clustering along M can multicast B
    # tiles, and clustering along N can multicast A tiles.
    hopper_gemm_helpers.add_common_gemm_arguments(
        parser,
        tile_shape_help=(
            "CTA/thread-block output tile shape M,N in elements. Default: 128,256"
        ),
        cluster_shape_help="Thread-block cluster shape M,N in CTAs. Default: 1,1",
        check_help="Run the reference check. This is much slower for large GEMMs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_name, cc = hopper_gemm_helpers.require_hopper_device(args.device)

    mod = hopper_gemm_helpers.load_hopper_dense_gemm_module()
    mod_cutlass = mod.cutlass
    float16 = mod_cutlass.Float16
    float32 = mod_cutlass.Float32

    m, n, k = args.mnk
    l = args.batch
    exec_time_us = mod.run(
        (m, n, k, l),
        float16,
        float16,
        float16,
        float32,
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

    hopper_gemm_helpers.print_benchmark_summary(
        device_name=device_name,
        device=args.device,
        cc=cc,
        m=m,
        n=n,
        k=k,
        batch=l,
        tile_shape=args.tile_shape,
        cluster_shape=args.cluster_shape,
        exec_time_us=exec_time_us,
        checked=args.check,
    )


if __name__ == "__main__":
    main()

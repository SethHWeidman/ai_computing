#!/usr/bin/env python3

"""A Hopper-specific single-GPU GEMM walkthrough.

`02_h100_single_gpu_gemm.py` shows the host-side flow: make tensors, wrap them as CuTe
tensors, construct `HopperWgmmaGemmKernel`, compile, launch, and time.

This script goes one layer deeper without copying the large CUTLASS device kernel. After
JIT compilation, it prints the static Hopper plan derived by the same
`HopperWgmmaGemmKernel` class that launches the kernel:

1. WGMMA tiling and warp-group layout.
2. TMA global-to-shared/shared-to-global copy roles.
3. CTA cluster shape and which operands use multicast.
4. Mainloop/epilogue pipeline stage counts.
5. Shared-memory footprint and launch geometry.

The actual kernel implementation still comes from CUTLASS's CuTe DSL Hopper dense GEMM
example in this repo. The point here is to make the Hopper-specific decisions visible
while keeping the lesson file small enough to read.
"""

import argparse
import math

import cutlass
import cutlass.cute as cute
import torch

import hopper_gemm_helpers


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _format_count(value: object) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _format_bytes(value: object) -> str:
    if not isinstance(value, int):
        return str(value)
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value} B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Hopper/H100 GEMM and print WGMMA/TMA/cluster details."
    )
    hopper_gemm_helpers.add_common_gemm_arguments(
        parser,
        tile_shape_help=(
            "CTA tile shape M,N. Supported: 64/128 by 64/128/256. Default: 128,256"
        ),
        cluster_shape_help=(
            "CTA cluster shape M,N. Total cluster size must be <= 4. Default: 1,1"
        ),
        check_help="Run a Torch reference check. This is much slower for large GEMMs.",
    )
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="Compile and print the Hopper execution plan, then skip launch.",
    )
    args = parser.parse_args()

    tile_m, tile_n = args.tile_shape
    if tile_m not in (64, 128) or tile_n not in (64, 128, 256):
        parser.error("--tile-shape must be one of M in {64,128}, N in {64,128,256}")

    cluster_m, cluster_n = args.cluster_shape
    if (
        not _is_power_of_two(cluster_m)
        or not _is_power_of_two(cluster_n)
        or cluster_m * cluster_n > 4
    ):
        parser.error(
            "--cluster-shape M,N values must be positive powers of two with M*N <= 4"
        )

    if args.batch < 1:
        parser.error("--batch must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")

    return args


def print_hopper_plan(gemm: object, c: hopper_gemm_helpers.TensorPack) -> None:
    c_device = c.device_torch
    c_shape = c_device.shape
    m, n, batch = tuple(c_shape)

    # tile_shape_mnk is the CTA-level GEMM tile: the CTA computes tile_m x tile_n
    # output elements while the mainloop advances through K in tile_k-sized slices.
    tile_shape_mnk = gemm.tile_shape_mnk
    tile_m, tile_n, tile_k = tile_shape_mnk

    cluster_shape_mn = gemm.cluster_shape_mn
    cluster_m, cluster_n = cluster_shape_mn

    # WGMMA is organized at warp-group granularity. A warp group is 4 warps / 128
    # threads on Hopper. This atom layout is not the A/B/C tensor memory layout; it
    # describes how the CTA's MMA work is decomposed into WGMMA atoms.
    atom_layout_mnk = gemm.atom_layout_mnk
    mma_warp_groups = gemm.mma_warp_groups
    threads_per_cta = gemm.threads_per_cta
    acc_dtype = gemm.acc_dtype

    num_mcast_ctas_a = gemm.num_mcast_ctas_a
    num_mcast_ctas_b = gemm.num_mcast_ctas_b
    is_a_mcast = gemm.is_a_mcast
    is_b_mcast = gemm.is_b_mcast

    # The mainloop computes the full C tile, but the epilogue stores it through
    # narrower shared-memory/TMA stripes to keep the staging footprint regular.
    epi_tile = gemm.epi_tile
    epi_m, epi_n = epi_tile

    # These stage counts are static pipeline choices. More stages can overlap more
    # future K-slices, but every stage spends shared memory under the SM90 budget.
    ab_stage = gemm.ab_stage
    epi_stage = gemm.epi_stage
    smem_capacity = gemm.smem_capacity
    a_dtype = gemm.a_dtype
    b_dtype = gemm.b_dtype
    c_dtype = gemm.c_dtype
    a_dtype_width = a_dtype.width
    b_dtype_width = b_dtype.width
    c_dtype_width = c_dtype.width

    tiles_m = math.ceil(m / tile_m)
    tiles_n = math.ceil(n / tile_n)
    grid = (
        math.ceil(tiles_m / cluster_m) * cluster_m,
        math.ceil(tiles_n / cluster_n) * cluster_n,
        batch,
    )
    cluster = (*cluster_shape_mn, 1)
    ctas_per_cluster = math.prod(cluster)
    if ctas_per_cluster == 1:
        multicast_context = "single-CTA cluster: no inter-CTA multicast"
    else:
        multicast_context = "larger clusters may enable A and/or B multicast"

    threads_per_warp = 32
    warps_per_warp_group = 4
    warp_group_threads = warps_per_warp_group * threads_per_warp
    num_warps = threads_per_cta // threads_per_warp

    mcast_size = num_mcast_ctas_a + num_mcast_ctas_b - 1
    producer_warp_groups = 1
    consumer_warp_groups = mma_warp_groups - producer_warp_groups
    if consumer_warp_groups < 1:
        consumer_warp_groups = 1
    consumer_warps = consumer_warp_groups * warps_per_warp_group
    consumer_arrive_count_estimate = mcast_size * consumer_warps

    a_bytes_per_element = a_dtype_width // 8
    b_bytes_per_element = b_dtype_width // 8
    c_bytes_per_element = c_dtype_width // 8
    a_stage_elements = tile_m * tile_k
    b_stage_elements = tile_n * tile_k
    epi_stage_elements = epi_m * epi_n
    a_stage_bytes = a_stage_elements * a_bytes_per_element
    b_stage_bytes = b_stage_elements * b_bytes_per_element
    ab_stage_bytes = a_stage_bytes + b_stage_bytes
    a_total_bytes = a_stage_bytes * ab_stage
    b_total_bytes = b_stage_bytes * ab_stage
    ab_total_bytes = a_total_bytes + b_total_bytes
    epi_stage_bytes = epi_stage_elements * c_bytes_per_element
    epi_ring_bytes = epi_stage_bytes * epi_stage
    epi_stripes_n = math.ceil(tile_n / epi_n)
    full_c_tile = (tile_m, tile_n)

    print()
    print("Step 3: inspect the Hopper-specific static plan from CuTe compilation.")
    print("  WGMMA:")
    print(
        f"    CTA tile_shape_mnk={tile_shape_mnk}: C tile={full_c_tile}, "
        f"K stage={tile_k}"
    )
    print(
        f"    atom_layout_mnk={atom_layout_mnk}, warp_groups={mma_warp_groups}, "
        f"threads_per_cta={threads_per_cta}"
    )
    print(
        f"    warp-group math: {mma_warp_groups} groups * {warps_per_warp_group} "
        f"warps/group * {threads_per_warp} threads/warp = "
        f"{mma_warp_groups * warp_group_threads} threads"
    )
    print("    atom_layout_mnk is WGMMA atom layout, not A/B/C tensor layout")
    print(f"    accumulator_dtype={acc_dtype}")

    print("  TMA:")
    print(
        f"    A TMA G2S tile={(tile_m, tile_k)}, multicast_ctas={num_mcast_ctas_a}, "
        f"op={'G2SMulticast' if is_a_mcast else 'G2S'}"
    )
    print(
        f"    B TMA G2S tile={(tile_n, tile_k)}, multicast_ctas={num_mcast_ctas_b}, "
        f"op={'G2SMulticast' if is_b_mcast else 'G2S'}"
    )
    print(f"    C TMA S2G epilogue_tile={epi_tile}, epi_stage={epi_stage}")
    print("    G2S = global-to-shared mainloop load; S2G = shared-to-global store")
    print(
        f"    C full CTA tile={full_c_tile}, stored as {epi_stripes_n} "
        f"epilogue stripes of {epi_tile}"
    )

    print("  CTA cluster:")
    print(
        f"    cluster_shape={cluster}, CTAs_per_cluster={ctas_per_cluster}, grid={grid}"
    )
    print(
        f"    A multicast follows cluster-N, B multicast follows cluster-M, "
        f"pipeline_consumer_arrive_count_estimate={consumer_arrive_count_estimate}"
    )
    print(f"    {multicast_context}")
    print(
        f"    estimate uses {consumer_warps} consumer warps out of {num_warps} total "
        f"warps from the {mma_warp_groups}-warp-group producer/consumer split"
    )

    print("  Shared memory and pipeline:")
    print(
        f"    sm90_capacity={_format_bytes(smem_capacity)} user-addressable budget, "
        f"ab_stage={ab_stage}, epi_stage={epi_stage}"
    )
    print(
        f"    A per AB stage={_format_bytes(a_stage_bytes)}, "
        f"B per AB stage={_format_bytes(b_stage_bytes)}, "
        f"A+B per stage={_format_bytes(ab_stage_bytes)}"
    )
    print(
        f"    A staged={_format_bytes(a_total_bytes)}, B staged="
        f"{_format_bytes(b_total_bytes)}, "
        f"C epilogue per stage={_format_bytes(epi_stage_bytes)}"
    )
    print(
        f"    A+B staged total={_format_bytes(ab_total_bytes)}, "
        f"conceptual C epilogue ring={_format_bytes(epi_ring_bytes)}"
    )
    print("    note: epilogue staging may reuse mainloop SMEM after A/B are done")
    print(
        f"    nominal SMEM elements: A={_format_count(a_stage_elements * ab_stage)}, "
        f"B={_format_count(b_stage_elements * ab_stage)}, "
        f"C_epi={_format_count(epi_stage_elements * epi_stage)}"
    )


def main() -> None:
    args = parse_args()
    device_name, cc = hopper_gemm_helpers.require_hopper_device(args.device)

    mod = hopper_gemm_helpers.load_hopper_dense_gemm_module()
    kernel_cls = mod.HopperWgmmaGemmKernel
    m, n, k = args.mnk
    l = args.batch

    hopper_gemm_helpers.validate_fp16_hopper_gemm(kernel_cls, m, n, k, l)

    torch.manual_seed(260427)

    print("Step 1: create Hopper-friendly Torch tensors and wrap them as CuTe tensors.")
    a = hopper_gemm_helpers.make_tensor_pack(
        l, m, k, is_mode0_major=False, dtype=cutlass.Float16, major="k"
    )
    b = hopper_gemm_helpers.make_tensor_pack(
        l, n, k, is_mode0_major=False, dtype=cutlass.Float16, major="k"
    )
    c = hopper_gemm_helpers.make_tensor_pack(
        l, m, n, is_mode0_major=False, dtype=cutlass.Float16, major="n"
    )
    hopper_gemm_helpers.describe_tensor("A", a)
    hopper_gemm_helpers.describe_tensor("B", b)
    hopper_gemm_helpers.describe_tensor("C", c)

    print()
    print("Step 2: compile the Hopper kernel with CuTe.")
    gemm = kernel_cls(
        acc_dtype=cutlass.Float32,
        tile_shape_mn=args.tile_shape,
        cluster_shape_mn=args.cluster_shape,
    )
    torch_stream, cu_stream = hopper_gemm_helpers.make_cuda_stream_pair(args.device)
    a_cute = a.cute_tensor
    b_cute = b.cute_tensor
    c_cute = c.cute_tensor
    compiled_gemm = cute.compile(gemm, a_cute, b_cute, c_cute, cu_stream)
    print_hopper_plan(gemm, c)

    if args.describe_only:
        print()
        print("Describe-only mode: skipped kernel launch after compilation.")
        return

    print()
    print("Step 4: launch the compiled kernel on one CUDA stream.")
    exec_time_us = hopper_gemm_helpers.benchmark_kernel(
        compiled_gemm,
        a,
        b,
        c,
        torch_stream,
        cu_stream,
        warmup_iterations=args.warmup,
        iterations=args.iterations,
    )

    if args.check:
        print()
        print("Step 5: run a Torch reference check.")
        hopper_gemm_helpers.maybe_check_result(a, b, c)

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

#!/usr/bin/env python3

"""A slightly deeper single-GPU Hopper GEMM example.

Compared with `01_h100_single_gpu_gemm.py`, this version exposes more of the host-side
flow directly in this file:

1. Create Hopper-friendly Torch tensors for A, B, and C.
2. Wrap those buffers as CuTe tensors with `runtime.from_dlpack(...)`.
3. Instantiate `HopperWgmmaGemmKernel` and compile it with `cute.compile(...)`.
4. Launch the compiled kernel on one CUDA stream and report throughput.

The actual Hopper WGMMA kernel still comes from the CUTLASS example in this repo, so we
avoid copying a very large kernel into a tiny example file. This keeps the example small
while still showing the important pieces that `mod.run(...)` was hiding.
"""
import argparse

import cutlass
import cutlass.cute as cute
import torch

import hopper_gemm_helpers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deeper single-GPU FP16 GEMM example on Hopper/H100."
    )
    hopper_gemm_helpers.add_common_gemm_arguments(
        parser,
        tile_shape_help="CTA tile shape M,N. Default: 128,256",
        cluster_shape_help="Cluster shape M,N. Default: 1,1",
        check_help="Run a Torch reference check. This is much slower for large GEMMs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_name, cc = hopper_gemm_helpers.require_hopper_device(args.device)

    # `mod` is the loaded dense_gemm.py module; pull out its Hopper kernel class.
    mod = hopper_gemm_helpers.load_hopper_dense_gemm_module()
    kernel_cls = mod.HopperWgmmaGemmKernel
    m, n, k = args.mnk
    l = args.batch

    hopper_gemm_helpers.validate_fp16_hopper_gemm(kernel_cls, m, n, k, l)

    torch.manual_seed(260424)

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
    print("Step 2: instantiate HopperWgmmaGemmKernel and JIT-compile it with CuTe.")
    # tile_shape_mn is the per-CTA output tile; cluster_shape_mn groups CTAs in the M,N
    # tile grid. In this kernel, grouping along M can multicast B and grouping along N
    # can multicast A. Larger clusters save duplicated loads only when that benefit
    # beats the added scheduling and barrier coordination.
    gemm = kernel_cls(
        acc_dtype=cutlass.Float32,
        tile_shape_mn=args.tile_shape,
        cluster_shape_mn=args.cluster_shape,
    )

    # Create one CUDA stream for this example and wrap the same underlying stream in the
    # CUDA Driver API handle that CuTe's compiled launcher expects.
    torch_stream, cu_stream = hopper_gemm_helpers.make_cuda_stream_pair(args.device)

    # -------------------------------------------------------------------------
    # JIT COMPILATION BOUNDARY
    # -------------------------------------------------------------------------
    # cute.compile(...) is the key line in this example.
    #
    # It takes:
    #   1. gemm: the Python/CuTe DSL callable describing the Hopper GEMM kernel
    #   2. a/b/c.cute_tensor: example argument descriptors, not data to multiply
    #   3. cu_stream: the runtime stream argument type
    #
    # CuTe uses those inputs to specialize a compiled function for this signature: dtype,
    # rank, layout family, alignment, dynamic leading dimension, and kernel static
    # parameters such as tile shape and cluster shape.
    #
    # The result is a JIT executor. Calling compiled_gemm(...) later launches the
    # generated GPU kernel. It does not execute the Python kernel recipe again, and it
    # does not pay the JIT compilation cost again for this executor instance.
    a_cute = a.cute_tensor
    b_cute = b.cute_tensor
    c_cute = c.cute_tensor
    compiled_gemm = cute.compile(gemm, a_cute, b_cute, c_cute, cu_stream)

    atom_layout_mnk = gemm.atom_layout_mnk
    tile_shape_mnk = gemm.tile_shape_mnk
    threads_per_cta = gemm.threads_per_cta
    ab_stage = gemm.ab_stage
    epi_stage = gemm.epi_stage
    print(
        f"Kernel: atom_layout_mnk={atom_layout_mnk}, tile_shape_mnk={tile_shape_mnk}, "
        f"threads_per_cta={threads_per_cta}, ab_stage={ab_stage}, "
        f"epi_stage={epi_stage}"
    )

    print()
    print("Step 3: launch the compiled kernel on one CUDA stream.")
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
        print("Step 4: run a Torch reference check.")
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

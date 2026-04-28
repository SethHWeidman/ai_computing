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
import dataclasses
from importlib import util
import math
import pathlib
import types

from cuda.bindings import driver as cuda_driver
import torch
from torch import cuda, testing

import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime
import cutlass.torch as cutlass_torch
from cutlass.torch import TensorInitType


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


@dataclasses.dataclass(frozen=True)
class TensorPack:
    host_f32: torch.Tensor
    device_torch: torch.Tensor
    cute_tensor: object
    source_shape: tuple[int, int, int]
    permute_order: tuple[int, int, int]
    leading_dim: int
    major: str


def _parse_csv_ints(value: str, expected_len: int) -> tuple[int, ...]:
    parts = tuple(int(x.strip()) for x in value.split(","))
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(
            f"expected {expected_len} comma-separated integers, got: {value!r}"
        )
    return parts


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


def load_hopper_dense_gemm_module() -> types.ModuleType:
    # Load CUTLASS's dense_gemm.py as a normal Python module so this example can
    # reuse HopperWgmmaGemmKernel without copying the whole kernel implementation.
    spec = util.spec_from_file_location("cute_hopper_dense_gemm", HOPPER_GEMM_EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {HOPPER_GEMM_EXAMPLE}")
    loader = spec.loader
    module = util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Hopper/H100 GEMM and print WGMMA/TMA/cluster details."
    )
    parser.add_argument(
        "--mnk",
        type=lambda s: _parse_csv_ints(s, 3),
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
        type=lambda s: _parse_csv_ints(s, 2),
        default=(128, 256),
        help="CTA tile shape M,N. Supported: 64/128 by 64/128/256. Default: 128,256",
    )
    parser.add_argument(
        "--cluster-shape",
        type=lambda s: _parse_csv_ints(s, 2),
        default=(1, 1),
        help="CTA cluster shape M,N. Total cluster size must be <= 4. Default: 1,1",
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
        "--describe-only",
        action="store_true",
        help="Compile and print the Hopper execution plan, then skip launch.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a Torch reference check. This is much slower for large GEMMs.",
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


def make_tensor_pack(
    batch: int,
    mode0: int,
    mode1: int,
    *,
    is_mode0_major: bool,
    dtype: type[cutlass.Numeric],
    major: str,
) -> TensorPack:
    # If mode0 is major we construct storage as (L, mode1, mode0) and then permute to
    # the logical (mode0, mode1, L) view. Otherwise we start from (L, mode0, mode1).
    source_shape = (batch, mode1, mode0) if is_mode0_major else (batch, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    leading_dim = 0 if is_mode0_major else 1
    is_unsigned = dtype in {cutlass.Uint8}
    torch_dtype = (
        cutlass_torch.dtype(dtype)
        if dtype not in {cutlass.Float8E4M3FN, cutlass.Float8E5M2}
        else torch.uint8
    )

    host_tensor = cutlass_torch.create_and_permute_torch_tensor(
        source_shape,
        torch_dtype,
        permute_order=permute_order,
        init_type=TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(
            min_val=0 if is_unsigned else -2, max_val=4 if is_unsigned else 2
        ),
    )
    device_torch = host_tensor.cuda()
    host_f32 = host_tensor.to(dtype=torch.float32)

    cute_tensor = runtime.from_dlpack(device_torch, assumed_align=16)
    cute_tensor.element_type = dtype
    cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    cute_tensor = cutlass_torch.convert_cute_tensor(
        host_f32, cute_tensor, dtype, is_dynamic_layout=True
    )

    return TensorPack(
        host_f32=host_f32,
        device_torch=device_torch,
        cute_tensor=cute_tensor,
        source_shape=source_shape,
        permute_order=permute_order,
        leading_dim=leading_dim,
        major=major,
    )


def describe_tensor(name: str, tensor: TensorPack) -> None:
    # Torch strides are measured in elements. They show which logical axis is contiguous,
    # which is the layout signal that Hopper's WGMMA/TMA path cares about.
    device_tensor = tensor.device_torch
    logical_shape = tuple(device_tensor.shape)

    # A Torch stride is measured in elements, not bytes. Each entry says how far the
    # storage pointer moves when that logical index increases by 1. For this example, the
    # strides make the chosen k-major/n-major layout visible.
    strides = tuple(device_tensor.stride())
    print(
        f"{name}: major={tensor.major}, logical_shape={logical_shape}, "
        f"source_shape={tensor.source_shape}, permute={tensor.permute_order}, "
        f"strides={strides}, leading_dim={tensor.leading_dim}"
    )


def print_hopper_plan(gemm: object, c: TensorPack) -> None:
    c_device = c.device_torch
    c_shape = c_device.shape
    m, n, batch = tuple(c_shape)

    tile_shape_mnk = gemm.tile_shape_mnk
    tile_m, tile_n, tile_k = tile_shape_mnk
    cluster_shape_mn = gemm.cluster_shape_mn
    cluster_m, cluster_n = cluster_shape_mn
    atom_layout_mnk = gemm.atom_layout_mnk
    mma_warp_groups = gemm.mma_warp_groups
    threads_per_cta = gemm.threads_per_cta
    acc_dtype = gemm.acc_dtype
    num_mcast_ctas_a = gemm.num_mcast_ctas_a
    num_mcast_ctas_b = gemm.num_mcast_ctas_b
    is_a_mcast = gemm.is_a_mcast
    is_b_mcast = gemm.is_b_mcast
    epi_tile = gemm.epi_tile
    epi_m, epi_n = epi_tile
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
    num_warps = threads_per_cta // 32
    mcast_size = num_mcast_ctas_a + num_mcast_ctas_b - 1
    consumer_arrive_count = mcast_size * num_warps
    a_stage_bytes = tile_m * tile_k * a_dtype_width // 8
    b_stage_bytes = tile_n * tile_k * b_dtype_width // 8
    ab_stage_bytes = a_stage_bytes + b_stage_bytes
    a_total_bytes = a_stage_bytes * ab_stage
    b_total_bytes = b_stage_bytes * ab_stage
    epi_stage_bytes = epi_m * epi_n * c_dtype_width // 8

    print()
    print("Step 3: inspect the Hopper-specific static plan from CuTe compilation.")
    print("  WGMMA:")
    print(f"    CTA tile_shape_mnk={tile_shape_mnk}")
    print(
        f"    atom_layout_mnk={atom_layout_mnk}, "
        f"warp_groups={mma_warp_groups}, threads_per_cta={threads_per_cta}"
    )
    print(f"    accumulator_dtype={acc_dtype}")

    print("  TMA:")
    print(
        f"    A G2S tile={(tile_m, tile_k)}, "
        f"multicast_ctas={num_mcast_ctas_a}, "
        f"op={'G2SMulticast' if is_a_mcast else 'G2S'}"
    )
    print(
        f"    B G2S tile={(tile_n, tile_k)}, "
        f"multicast_ctas={num_mcast_ctas_b}, "
        f"op={'G2SMulticast' if is_b_mcast else 'G2S'}"
    )
    print(f"    C S2G epilogue_tile={epi_tile}, epi_stage={epi_stage}")

    print("  CTA cluster:")
    print(
        f"    cluster_shape={cluster}, CTAs_per_cluster={math.prod(cluster)}, "
        f"grid={grid}"
    )
    print(
        f"    A multicast follows cluster-N, B multicast follows cluster-M, "
        f"pipeline_consumer_arrive_count={consumer_arrive_count}"
    )

    print("  Shared memory and pipeline:")
    print(
        f"    sm90_capacity={_format_bytes(smem_capacity)}, "
        f"ab_stage={ab_stage}, epi_stage={epi_stage}"
    )
    print(
        f"    A per AB stage={_format_bytes(a_stage_bytes)}, "
        f"B per AB stage={_format_bytes(b_stage_bytes)}, "
        f"A+B per stage={_format_bytes(ab_stage_bytes)}"
    )
    print(
        f"    A staged={_format_bytes(a_total_bytes)}, "
        f"B staged={_format_bytes(b_total_bytes)}, "
        f"C epilogue per stage={_format_bytes(epi_stage_bytes)}"
    )
    print(
        f"    nominal SMEM elements: "
        f"A={_format_count(tile_m * tile_k * ab_stage)}, "
        f"B={_format_count(tile_n * tile_k * ab_stage)}, "
        f"C_epi={_format_count(epi_m * epi_n * epi_stage)}"
    )


def benchmark_kernel(
    compiled_gemm,
    a: TensorPack,
    b: TensorPack,
    c: TensorPack,
    torch_stream: cuda.Stream,
    cu_stream: cuda_driver.CUstream,
    warmup_iterations: int,
    iterations: int,
) -> float:
    a_cute = a.cute_tensor
    b_cute = b.cute_tensor
    c_cute = c.cute_tensor

    with cuda.stream(torch_stream):
        for _ in range(warmup_iterations):
            compiled_gemm(a_cute, b_cute, c_cute, cu_stream)
    torch_stream.synchronize()

    start = cuda.Event(enable_timing=True)
    end = cuda.Event(enable_timing=True)
    with cuda.stream(torch_stream):
        start.record(torch_stream)
        for _ in range(iterations):
            compiled_gemm(a_cute, b_cute, c_cute, cu_stream)
        end.record(torch_stream)
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def maybe_check_result(a: TensorPack, b: TensorPack, c: TensorPack) -> None:
    a_host = a.host_f32
    b_host = b.host_f32
    ref_f32 = torch.einsum("mkl,nkl->mnl", a_host, b_host)
    ref = ref_f32.to(dtype=torch.float16)

    c_device = c.device_torch
    c_host = c_device.cpu()
    testing.assert_close(c_host, ref, atol=1e-1, rtol=1e-3)


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
    kernel_cls = mod.HopperWgmmaGemmKernel
    m, n, k = args.mnk
    l = args.batch

    if not kernel_cls.is_valid_dtypes(
        cutlass.Float16, cutlass.Float16, cutlass.Float32, cutlass.Float16, "k", "k"
    ):
        raise TypeError("The fixed fp16/fp32/fp16 dtype combination is not valid.")

    if not kernel_cls.is_valid_tensor_alignment(
        m, n, k, l, cutlass.Float16, cutlass.Float16, "k", "k", "n"
    ):
        raise TypeError(
            "The contiguous dimension of A/B/C must be at least 16-byte aligned."
        )

    torch.manual_seed(260427)

    print("Step 1: create Hopper-friendly Torch tensors and wrap them as CuTe tensors.")
    a = make_tensor_pack(l, m, k, is_mode0_major=False, dtype=cutlass.Float16, major="k")
    b = make_tensor_pack(l, n, k, is_mode0_major=False, dtype=cutlass.Float16, major="k")
    c = make_tensor_pack(l, m, n, is_mode0_major=False, dtype=cutlass.Float16, major="n")
    describe_tensor("A", a)
    describe_tensor("B", b)
    describe_tensor("C", c)

    print()
    print("Step 2: compile the Hopper kernel with CuTe.")
    gemm = kernel_cls(
        acc_dtype=cutlass.Float32,
        tile_shape_mn=args.tile_shape,
        cluster_shape_mn=args.cluster_shape,
    )
    torch_stream = cuda.Stream(device=args.device)
    cu_stream = cuda_driver.CUstream(torch_stream.cuda_stream)
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
    exec_time_us = benchmark_kernel(
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
        maybe_check_result(a, b, c)

    tflops = (2.0 * m * n * k * l) / (exec_time_us * 1e-6) / 1e12
    print()
    print(f"GPU: {device_name} (device {args.device}, cc {cc[0]}.{cc[1]})")
    print(
        f"Problem: M={m}, N={n}, K={k}, batch={l}, tile={args.tile_shape}, "
        f"cluster={args.cluster_shape}"
    )
    print(f"Average kernel time: {exec_time_us:.1f} us")
    print(f"Throughput: {tflops:.2f} TFLOP/s")
    if args.check:
        print("Reference check: passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""A slightly deeper single-GPU Hopper GEMM example.

Compared with `01_h100_single_gpu_gemm.py`, this version exposes more of the host-side flow
directly in this file:

1. Create Hopper-friendly Torch tensors for A, B, and C.
2. Wrap those buffers as CuTe tensors with `runtime.from_dlpack(...)`.
3. Instantiate `HopperWgmmaGemmKernel` and compile it with `cute.compile(...)`.
4. Launch the compiled kernel on one CUDA stream and report throughput.

The actual Hopper WGMMA kernel still comes from the CUTLASS example in this repo, so we
avoid copying a very large kernel into a tiny example file. This keeps the example small
while still showing the important pieces that `mod.run(...)` was hiding.
"""
import argparse
import dataclasses
from importlib import util
import pathlib
import types

import cuda.bindings.driver as cuda_driver
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
        description="Run a deeper single-GPU FP16 GEMM example on Hopper/H100."
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
        help="Run a Torch reference check. This is much slower for large GEMMs.",
    )
    return parser.parse_args()


def make_tensor_pack(
    batch: int,
    mode0: int,
    mode1: int,
    *,
    is_mode0_major: bool,
    dtype: type[cutlass.Numeric],
    major: str,
) -> TensorPack:
    # If mode0 is major we construct storage as (L, mode1, mode0) and then permute to the
    # logical (mode0, mode1, L) view. Otherwise we start from (L, mode0, mode1).
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
    print(
        f"{name}: major={tensor.major}, "
        f"logical_shape={tuple(tensor.device_torch.shape)}, "
        f"source_shape={tensor.source_shape}, permute={tensor.permute_order}, "
        f"strides={tuple(tensor.device_torch.stride())}, "
        f"leading_dim={tensor.leading_dim}"
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
    with cuda.stream(torch_stream):
        for _ in range(warmup_iterations):
            compiled_gemm(a.cute_tensor, b.cute_tensor, c.cute_tensor, cu_stream)
    torch_stream.synchronize()

    start = cuda.Event(enable_timing=True)
    end = cuda.Event(enable_timing=True)
    with cuda.stream(torch_stream):
        start.record(torch_stream)
        for _ in range(iterations):
            compiled_gemm(a.cute_tensor, b.cute_tensor, c.cute_tensor, cu_stream)
        end.record(torch_stream)
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def maybe_check_result(a: TensorPack, b: TensorPack, c: TensorPack) -> None:
    ref = torch.einsum("mkl,nkl->mnl", a.host_f32, b.host_f32).to(dtype=torch.float16)
    testing.assert_close(c.device_torch.cpu(), ref, atol=1e-1, rtol=1e-3)


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

    torch.manual_seed(260424)

    print("Step 1: create Hopper-friendly Torch tensors and wrap them as CuTe tensors.")
    a = make_tensor_pack(l, m, k, is_mode0_major=False, dtype=cutlass.Float16, major="k")
    b = make_tensor_pack(l, n, k, is_mode0_major=False, dtype=cutlass.Float16, major="k")
    c = make_tensor_pack(l, m, n, is_mode0_major=False, dtype=cutlass.Float16, major="n")
    describe_tensor("A", a)
    describe_tensor("B", b)
    describe_tensor("C", c)

    print()
    print("Step 2: instantiate HopperWgmmaGemmKernel and JIT-compile it with CuTe.")
    gemm = kernel_cls(
        acc_dtype=cutlass.Float32,
        tile_shape_mn=args.tile_shape,
        cluster_shape_mn=args.cluster_shape,
    )
    torch_stream = cuda.Stream(device=args.device)
    cu_stream = cuda_driver.CUstream(torch_stream.cuda_stream)
    compiled_gemm = cute.compile(
        gemm, a.cute_tensor, b.cute_tensor, c.cute_tensor, cu_stream
    )
    print(
        f"Kernel: atom_layout_mnk={gemm.atom_layout_mnk}, "
        f"tile_shape_mnk={gemm.tile_shape_mnk}, threads_per_cta={gemm.threads_per_cta}, "
        f"ab_stage={gemm.ab_stage}, epi_stage={gemm.epi_stage}"
    )

    print()
    print("Step 3: launch the compiled kernel on one CUDA stream.")
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
        print("Step 4: run a Torch reference check.")
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

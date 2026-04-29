"""Shared helpers for the Hopper GEMM lesson scripts."""

import argparse
import dataclasses
from importlib import util
import pathlib
import types

from cuda.bindings import driver as cuda_driver
import cutlass
from cutlass.cute import runtime
import cutlass.torch as cutlass_torch
from cutlass.torch import TensorInitType
import torch
from torch import cuda as torch_cuda, testing


MODULE_PATH = pathlib.Path(__file__)
RESOLVED_MODULE_PATH = MODULE_PATH.resolve()
MODULE_PARENTS = RESOLVED_MODULE_PATH.parents
REPO_ROOT = MODULE_PARENTS[1]

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
    csv_parts = value.split(",")
    parts = tuple(int(part.strip()) for part in csv_parts)
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(
            f"expected {expected_len} comma-separated integers, got: {value!r}"
        )
    return parts


def add_common_gemm_arguments(
    parser: argparse.ArgumentParser,
    *,
    tile_shape_help: str,
    cluster_shape_help: str,
    check_help: str,
) -> None:
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
        help=tile_shape_help,
    )
    parser.add_argument(
        "--cluster-shape",
        type=lambda s: parse_csv_ints(s, 2),
        default=(1, 1),
        help=cluster_shape_help,
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
    parser.add_argument("--check", action="store_true", help=check_help)


def load_hopper_dense_gemm_module() -> types.ModuleType:
    """Load CUTLASS's Hopper dense GEMM example as a normal Python module."""
    spec = util.spec_from_file_location("cute_hopper_dense_gemm", HOPPER_GEMM_EXAMPLE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {HOPPER_GEMM_EXAMPLE}")
    loader = spec.loader
    module = util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def require_hopper_device(device: int) -> tuple[str, tuple[int, int]]:
    if not torch_cuda.is_available():
        raise RuntimeError("CUDA is required to run this example.")

    torch_cuda.set_device(device)
    device_name = torch_cuda.get_device_name(device)
    cc = torch_cuda.get_device_capability(device)
    cc_major = cc[0]
    if cc_major != 9:
        raise RuntimeError(
            "This example is intended for Hopper SM90 GPUs; got compute capability "
            f"{cc}."
        )
    return device_name, cc


def make_cuda_stream_pair(device: int) -> tuple[torch_cuda.Stream, cuda_driver.CUstream]:
    torch_stream = torch_cuda.Stream(device=device)
    torch_cuda_stream = torch_stream.cuda_stream
    cu_stream = cuda_driver.CUstream(torch_cuda_stream)
    return torch_stream, cu_stream


def validate_fp16_hopper_gemm(
    kernel_cls: object, m: int, n: int, k: int, batch: int
) -> None:
    a_dtype = cutlass.Float16
    b_dtype = cutlass.Float16
    acc_dtype = cutlass.Float32
    c_dtype = cutlass.Float16

    if not kernel_cls.is_valid_dtypes(a_dtype, b_dtype, acc_dtype, c_dtype, "k", "k"):
        raise TypeError("The fixed fp16/fp32/fp16 dtype combination is not valid.")

    if not kernel_cls.is_valid_tensor_alignment(
        m, n, k, batch, a_dtype, b_dtype, "k", "k", "n"
    ):
        raise TypeError(
            "The contiguous dimension of A/B/C must be at least 16-byte aligned."
        )


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
    float8_e4m3 = cutlass.Float8E4M3FN
    float8_e5m2 = cutlass.Float8E5M2
    torch_dtype = (
        cutlass_torch.dtype(dtype)
        if dtype not in {float8_e4m3, float8_e5m2}
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
    torch_float32 = torch.float32
    host_f32 = host_tensor.to(dtype=torch_float32)

    cute_tensor = runtime.from_dlpack(device_torch, assumed_align=16)
    cute_tensor.element_type = dtype
    dynamic_cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    converted_cute_tensor = cutlass_torch.convert_cute_tensor(
        host_f32, dynamic_cute_tensor, dtype, is_dynamic_layout=True
    )

    return TensorPack(
        host_f32=host_f32,
        device_torch=device_torch,
        cute_tensor=converted_cute_tensor,
        source_shape=source_shape,
        permute_order=permute_order,
        leading_dim=leading_dim,
        major=major,
    )


def describe_tensor(name: str, tensor: TensorPack) -> None:
    device_tensor = tensor.device_torch
    logical_shape = tuple(device_tensor.shape)
    source_shape = tensor.source_shape
    permute_order = tensor.permute_order
    leading_dim = tensor.leading_dim
    major = tensor.major

    # Torch strides are measured in elements. They show which logical axis is
    # contiguous, which is the layout signal Hopper's WGMMA/TMA path cares about.
    strides = tuple(device_tensor.stride())
    print(
        f"{name}: major={major}, logical_shape={logical_shape}, "
        f"source_shape={source_shape}, permute={permute_order}, "
        f"strides={strides}, leading_dim={leading_dim}"
    )


def benchmark_kernel(
    compiled_gemm,
    a: TensorPack,
    b: TensorPack,
    c: TensorPack,
    torch_stream: torch_cuda.Stream,
    cu_stream: cuda_driver.CUstream,
    warmup_iterations: int,
    iterations: int,
) -> float:
    a_cute = a.cute_tensor
    b_cute = b.cute_tensor
    c_cute = c.cute_tensor

    with torch_cuda.stream(torch_stream):
        for _ in range(warmup_iterations):
            compiled_gemm(a_cute, b_cute, c_cute, cu_stream)
    torch_stream.synchronize()

    start = torch_cuda.Event(enable_timing=True)
    end = torch_cuda.Event(enable_timing=True)
    with torch_cuda.stream(torch_stream):
        start.record(torch_stream)
        for _ in range(iterations):
            compiled_gemm(a_cute, b_cute, c_cute, cu_stream)
        end.record(torch_stream)
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def maybe_check_result(a: TensorPack, b: TensorPack, c: TensorPack) -> None:
    a_host = a.host_f32
    b_host = b.host_f32
    ref_f32 = torch.einsum("mkl,nkl->mnl", a_host, b_host)
    torch_float16 = torch.float16
    ref = ref_f32.to(dtype=torch_float16)

    # c.device_torch is the GEMM output buffer in GPU memory.
    c_device = c.device_torch

    # Copy it back to CPU memory so it lives on the same device as the reference.
    c_host = c_device.cpu()
    testing.assert_close(c_host, ref, atol=1e-1, rtol=1e-3)


def print_benchmark_summary(
    *,
    device_name: str,
    device: int,
    cc: tuple[int, int],
    m: int,
    n: int,
    k: int,
    batch: int,
    tile_shape: tuple[int, int],
    cluster_shape: tuple[int, int],
    exec_time_us: float,
    checked: bool,
) -> None:
    tflops = (2.0 * m * n * k * batch) / (exec_time_us * 1e-6) / 1e12
    cc_major, cc_minor = cc

    print()
    print(f"GPU: {device_name} (device {device}, cc {cc_major}.{cc_minor})")
    print(
        f"Problem: M={m}, N={n}, K={k}, batch={batch}, tile={tile_shape}, "
        f"cluster={cluster_shape}"
    )
    print(f"Average kernel time: {exec_time_us:.1f} us")
    print(f"Throughput: {tflops:.2f} TFLOP/s")
    if checked:
        print("Reference check: passed")

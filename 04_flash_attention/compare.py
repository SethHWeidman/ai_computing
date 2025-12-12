#!/usr/bin/env python3
import argparse
import pathlib
import sys
from os import environ
import typing

import torch
from torch import cuda, nn
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDA_HOME

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FLASH_SUBMODULE = REPO_ROOT / "flash-attn-101"
FLASH_BUILD = FLASH_SUBMODULE / "build" / "csrc"

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from attention_helpers import compute_context_vectors


def load_extension() -> nn.Module:
    here = pathlib.Path(__file__).resolve().parent
    environ.setdefault("TORCH_EXTENSIONS_DIR", str(here / ".torch_extensions"))
    environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA toolkit headers not found (CUDA_HOME is None). Install the CUDA toolkit or set "
            "CUDA_HOME."
        )

    # Ensure the submodule has been built so the shared library exists to link against.
    lib_path = FLASH_BUILD / "libflash-attn-101.so"
    if not lib_path.is_file():
        raise FileNotFoundError(
            f"Missing {lib_path}. Build the submodule first: "
            "cmake -B flash-attn-101/build && cmake --build flash-attn-101/build"
        )

    return cpp_extension.load(
        name="flash_attn_binding",
        sources=[str(here / "flash_attn_binding.cpp")],
        extra_include_paths=[
            str(FLASH_SUBMODULE / "include"),
            str(pathlib.Path(CUDA_HOME) / "include"),
        ],
        extra_ldflags=[
            f"-L{FLASH_BUILD}",
            "-lflash-attn-101",
            f"-Wl,-rpath,{FLASH_BUILD}",
        ],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def benchmark(
    fn: typing.Callable[[], torch.Tensor], repeats: int
) -> tuple[torch.Tensor, float]:
    start = cuda.Event(enable_timing=True)
    end = cuda.Event(enable_timing=True)
    outputs = None
    durations = []
    for _ in range(repeats):
        start.record()
        outputs = fn()
        end.record()
        cuda.synchronize()
        durations.append(start.elapsed_time(end))
    return outputs, sum(durations) / len(durations)


def standard_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    dropout: nn.Module,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Reference attention following attention_helpers.compute_context_vectors."""
    b, h, n, d = queries.shape
    context = compute_context_vectors(
        queries=queries,
        keys=keys,
        values=values,
        mask=mask,
        dropout=dropout,
        use_mask=True,
    )
    return context.view(b, n, h, d).transpose(1, 2).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Python attention against flash-attn-101 CUDA implementations."
    )
    parser.add_argument(
        "--seq-len", type=int, default=256, help="Sequence length (default: 256)"
    )
    parser.add_argument(
        "--head-dim", type=int, default=64, help="Per-head dimension (default: 64)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--num-heads", type=int, default=16, help="Number of heads (default: 16)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timed runs to average per implementation (default: 5)",
    )
    args = parser.parse_args()

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this comparison")

    torch.manual_seed(251212)
    cuda.manual_seed_all(251212)

    module = load_extension()
    device = torch.device("cuda")

    shape = (args.batch_size, args.num_heads, args.seq_len, args.head_dim)
    dropout = nn.Dropout(0.0)
    causal_mask = torch.triu(
        torch.ones(args.seq_len, args.seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )

    with torch.no_grad():
        q = torch.randn(shape, device=device, dtype=torch.float16)
        k = torch.randn(shape, device=device, dtype=torch.float16)
        v = torch.randn(shape, device=device, dtype=torch.float16)

        # Warm-up everything once so compilation/caching does not skew timings.
        standard_attention(q, k, v, dropout, causal_mask)
        module.naive_attention(q, k, v)
        module.flash_attention_01(q, k, v)
        module.flash_attention_02(q, k, v)
        module.cute_flash_attention_02(q, k, v)
        cuda.synchronize()

        python_out, python_ms = benchmark(
            lambda: standard_attention(q, k, v, dropout, causal_mask), args.repeats
        )
        naive_out, naive_ms = benchmark(
            lambda: module.naive_attention(q, k, v), args.repeats
        )
        fa1_out, fa1_ms = benchmark(
            lambda: module.flash_attention_01(q, k, v), args.repeats
        )
        fa2_out, fa2_ms = benchmark(
            lambda: module.flash_attention_02(q, k, v), args.repeats
        )
        cute_out, cute_ms = benchmark(
            lambda: module.cute_flash_attention_02(q, k, v), args.repeats
        )

        checks = {
            "naive vs python": torch.allclose(
                naive_out, python_out, rtol=1e-1, atol=2e-2
            ),
            "fa1 vs python": torch.allclose(fa1_out, python_out, rtol=1e-1, atol=2e-2),
            "fa2 vs python": torch.allclose(fa2_out, python_out, rtol=1e-1, atol=2e-2),
            "cute vs python": torch.allclose(
                cute_out, python_out, rtol=1e-1, atol=2e-2
            ),
        }

    def _fmt_row(items: typing.Iterable[str]) -> str:
        return " | ".join(items)

    headers = [
        "implementation",
        "latency ms",
        "speedup vs Python (x)",
        "matches Python",
    ]
    print(_fmt_row(headers))
    print(_fmt_row(["-" * len(h) for h in headers]))
    rows = [
        ("Python (torch)", f"{python_ms:.3f}", "1.0x", "—"),
        (
            "naive GPU",
            f"{naive_ms:.3f}",
            f"{python_ms / naive_ms:.1f}x",
            "yes" if checks["naive vs python"] else "NO",
        ),
        (
            "flash attn 01",
            f"{fa1_ms:.3f}",
            f"{python_ms / fa1_ms:.1f}x",
            "yes" if checks["fa1 vs python"] else "NO",
        ),
        (
            "flash attn 02",
            f"{fa2_ms:.3f}",
            f"{python_ms / fa2_ms:.1f}x",
            "yes" if checks["fa2 vs python"] else "NO",
        ),
        (
            "cute flash attn 02",
            f"{cute_ms:.3f}",
            f"{python_ms / cute_ms:.1f}x",
            "yes" if checks["cute vs python"] else "NO",
        ),
    ]
    for row in rows:
        print(_fmt_row(row))


if __name__ == "__main__":
    main()

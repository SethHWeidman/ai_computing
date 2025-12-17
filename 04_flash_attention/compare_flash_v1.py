#!/usr/bin/env python3
"""Compare the minimal FlashAttention v1 kernel against the Python helper implementation."""

import argparse
import pathlib
import sys
from os import environ

import torch
from torch import cuda, nn
from torch.utils import cpp_extension

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = pathlib.Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import attention_helpers


def load_extension() -> nn.Module:
    """Compile and load the flash_attn_v1 CUDA extension (float32 only)."""
    environ.setdefault("TORCH_EXTENSIONS_DIR", str(HERE / ".torch_extensions"))
    environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

    if cpp_extension.CUDA_HOME is None:
        raise RuntimeError(
            "CUDA toolkit headers not found (CUDA_HOME is None). Install the CUDA toolkit or set "
            "CUDA_HOME."
        )

    return cpp_extension.load(
        name="flash_attn_v1_ext",
        sources=[str(HERE / "flash_attn_v1.cu")],
        extra_cflags=["-std=c++17"],
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check outputs of flash_attn_v1.cu against the Python helper (float32, causal)."
    )
    parser.add_argument(
        "--seq-len", type=int, default=128, help="Sequence length (default: 128)"
    )
    parser.add_argument(
        "--head-dim", type=int, default=64, help="Per-head dimension (default: 64)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of heads (default: 8)"
    )
    args = parser.parse_args()

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this comparison")

    torch.manual_seed(202402)
    cuda.manual_seed_all(202402)

    module = load_extension()
    device = torch.device("cuda")
    dtype = torch.float32

    shape = (args.batch_size, args.num_heads, args.seq_len, args.head_dim)
    dropout = nn.Dropout(0.0)
    causal_mask = torch.triu(
        torch.ones(args.seq_len, args.seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )

    with torch.no_grad():
        q = torch.randn(shape, device=device, dtype=dtype)
        k = torch.randn(shape, device=device, dtype=dtype)
        v = torch.randn(shape, device=device, dtype=dtype)

        python_out = attention_helpers.compute_context_vectors_unflattened(
            q, k, v, causal_mask, dropout, use_mask=True
        )
        flash_out = module.flash_attn_v1(q, k, v)

        close = torch.allclose(flash_out, python_out, rtol=1e-4, atol=1e-4)
        max_diff = torch.max(torch.abs(flash_out - python_out)).item()

    print(f"Shape: {shape}")
    print(f"Outputs match: {'yes' if close else 'NO'}")
    print(f"Max abs diff: {max_diff:.4e}")


if __name__ == "__main__":
    main()

# CUDA vs PyTorch Matmul

This demo reuses the shared-memory tiled kernel from `intro_to_cuda/demo3_matmul` and compares it
against `torch.mm` for multiplying large square matrices (default 2048 × 2048).

## Requirements

- CUDA-capable GPU with a working CUDA toolkit (`nvcc` must be on PATH).
- PyTorch with CUDA support (`pip install torch --index-url https://download.pytorch.org/whl/cuXX`).

## Running the comparison

```bash
cd 02_cuda_vs_pytorch_matmul
python compare.py --size 2048 --repeats 3  # key benchmark: 2048 × 2048, averaged over 3 repeats
```

The script will JIT-build a PyTorch extension in `02_cuda_vs_pytorch_matmul/.torch_extensions`
the first time it runs. Subsequent runs reuse the compiled binary.

For each implementation you will see the averaged kernel time (milliseconds), the relative speed
ratio `torch.mm / shared kernel`, and whether the two results match numerically (`torch.allclose`
with `rtol=1e-3`, `atol=1e-3`).

Example output on an NVIDIA L4 with the default settings:

```text
Matrix size: 2048 x 2048
fast shared CUDA kernel: 8.32 ms (avg over 3)
torch.mm (cuBLAS):      1.11 ms (avg over 3)
Relative speed (torch.mm / shared kernel): 0.13x
Numerical match: yes
```

## Notes

- The custom kernel currently supports float32 inputs.
- `torch.mm` ultimately calls cuBLAS, so the relative speed effectively compares our tiled kernel
  to NVIDIA's library implementation.
- CUDA headers often use the `.cuh` extension. Functionally they are the same as `.h` files, but the
  CUDA naming convention makes it obvious that the header may declare `__global__`, `__device__`, or
  templated kernels you can include from both host C++ and device code. In this example,
  `fast_matmul_extension.cu` includes an externally defined tiled kernel from
  `../intro_to_cuda/demo3_matmul/matmul_kernels.cuh` and simply wires it up to PyTorch.

## Python ↔ CUDA bridge in this demo

This directory is also intended as a small, concrete reference for how to connect CUDA kernels to
Python via PyTorch:

- `../intro_to_cuda/demo3_matmul/matmul_kernels.cuh` defines the shared-memory tiled matmul kernel.
  We treat it as a reusable CUDA header and include it from `fast_matmul_extension.cu` instead of
  re-defining the kernel.
- `fast_matmul_extension.cu` defines a `fast_mm` wrapper that:
  - Uses PyTorch tensor types (`torch::Tensor`) and utilities like `TORCH_CHECK` for argument
    validation.
  - Ensures we are on the correct CUDA device with `at::cuda::CUDAGuard`.
  - Allocates an output tensor with `torch::empty` and launches the imported CUDA kernel on the
    underlying raw pointers.
  - Exposes `fast_mm` to Python using `PYBIND11_MODULE` and `torch/extension.h` (which sets up
    the pybind11 bindings and the `TORCH_EXTENSION_NAME` macro).
- `compare.py` shows how to:
  - JIT-build and load the C++/CUDA extension via `torch.utils.cpp_extension.load`.
  - Call `module.fast_mm(A, B)` alongside the standard `torch.mm(A, B)` in the same script.
  - Time both implementations under the same conditions and check numerical agreement with
    `torch.allclose`.

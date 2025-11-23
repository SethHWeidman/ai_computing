#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h> // Brings in TORCH_CHECK, Tensor, and pybind11 glue

#include "../cuda_matmul/matmul_kernels.cuh"

torch::Tensor fast_mm(torch::Tensor A, torch::Tensor B) {
  // TORCH_CHECK (from torch/extension.h) throws a Python-visible error if the condition fails
  TORCH_CHECK(A.is_cuda(), "Input A must be on CUDA");
  TORCH_CHECK(B.is_cuda(), "Input B must be on CUDA");
  TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
  TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");

  auto A_contig = A.contiguous();
  auto B_contig = B.contiguous();
  TORCH_CHECK(A_contig.size(1) == B_contig.size(0), "Inner matrix dims must match: got ",
              A_contig.sizes(), " and ", B_contig.sizes());

  // Temporarily set the current CUDA device to match A's device so kernel launches/allocations use
  // the right GPU; CUDAGuard saves whatever device was active before this call and restores it
  // when fast_mm exits, so other code sees the same device state it had going in. On a single-GPU
  // system this is effectively a no-op but keeps the code correct for multi-GPU use.
  at::cuda::CUDAGuard device_guard(A_contig.device());

  auto M = static_cast<size_t>(A_contig.size(0));
  auto K = static_cast<size_t>(A_contig.size(1));
  auto N = static_cast<size_t>(B_contig.size(1));

  auto C = torch::empty({static_cast<long>(M), static_cast<long>(N)}, A_contig.options());

  dim3 threads(BLOCK_DIM, BLOCK_DIM);
  dim3 blocks((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);

  mm_kernel_shared_memory<float><<<blocks, threads>>>(
      A_contig.data_ptr<float>(), B_contig.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return C;
}

// Define the Python module for this extension via pybind11 (a small header-only C++ library for
// creating Python bindings). TORCH_EXTENSION_NAME is a macro that torch.utils.cpp_extension.load
// defines to the name=... you pass from Python, so this C++ symbol is bound to the correct Python
// module name; we expose fast_mm as a Python-callable function.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_mm", &fast_mm, "Shared-memory CUDA matmul (float32)");
}

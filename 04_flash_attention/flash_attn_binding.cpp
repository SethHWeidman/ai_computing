#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <stdexcept>
#include <vector>

#include "cuda_attn.hpp"

namespace {

torch::Tensor check_and_contiguous(torch::Tensor t) {
  if (!t.is_cuda()) {
    throw std::runtime_error("Tensor must be on CUDA");
  }
  if (t.dtype() != torch::kFloat16) {
    throw std::runtime_error("Only float16 (__half) is supported for these kernels");
  }
  if (t.dim() != 4) {
    throw std::runtime_error("Expected tensor shape (batch, heads, seq_len, head_dim)");
  }
  return t.contiguous();
}

using LauncherFn = void (*)(const __half *, const __half *, const __half *, __half *, unsigned int,
                            unsigned int, unsigned int, unsigned int, cudaStream_t);

torch::Tensor run_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                            LauncherFn launcher) {
  auto q_c = check_and_contiguous(q);
  auto k_c = check_and_contiguous(k);
  auto v_c = check_and_contiguous(v);

  const auto batch_size = static_cast<unsigned int>(q_c.size(0));
  const auto num_heads = static_cast<unsigned int>(q_c.size(1));
  const auto seq_len = static_cast<unsigned int>(q_c.size(2));
  const auto head_dim = static_cast<unsigned int>(q_c.size(3));

  auto out = torch::zeros_like(q_c);

  auto stream = at::cuda::getDefaultCUDAStream();
  launcher(reinterpret_cast<const __half *>(q_c.data_ptr<at::Half>()),
           reinterpret_cast<const __half *>(k_c.data_ptr<at::Half>()),
           reinterpret_cast<const __half *>(v_c.data_ptr<at::Half>()),
           reinterpret_cast<__half *>(out.data_ptr<at::Half>()), batch_size, num_heads, seq_len,
           head_dim, stream);

  return out;
}

} // namespace

torch::Tensor naive_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  return run_attention(q, k, v, launch_naive_attention<__half>);
}

torch::Tensor flash_attention_01(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  return run_attention(q, k, v, launch_flash_attention_01<__half>);
}

torch::Tensor flash_attention_02(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  return run_attention(q, k, v, launch_flash_attention_02<__half>);
}

torch::Tensor cute_flash_attention_02(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  return run_attention(q, k, v, mha_fwd<__half>);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naive_attention", &naive_attention, "Naive attention (flash-attn-101)");
  m.def("flash_attention_01", &flash_attention_01, "FlashAttention v1 (flash-attn-101)");
  m.def("flash_attention_02", &flash_attention_02, "FlashAttention v2 (flash-attn-101)");
  m.def("cute_flash_attention_02", &cute_flash_attention_02,
        "CuTe FlashAttention v2 (flash-attn-101)");
}

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <limits>

static inline __host__ __device__ int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Minimal FlashAttention v1 teaching kernel (float32 only, causal).
// Layout: Q/K/V/O are [B, H, N, D] contiguous.
//
// Args:
// - Q: [B, H, N, D] queries
// - K: [B, H, N, D] keys
// - V: [B, H, N, D] values
// - O: [B, H, N, D] output
// - l: [B, H, N] running denominator
// - m: [B, H, N] running max
// FA1-style scheduling: grid is (B, H). Each block processes *one (batch, head)*,
// and loops over all query tiles tile_q = 0..Tr-1 inside the kernel.
__global__ void flash_attn_v1_kernel(const float *Q, const float *K, const float *V, float *O,
                                     float *l, float *m, int B, int H, int N, int D,
                                     float softmax_scale) {
  constexpr int Br = 16; // query rows per tile
  constexpr int Bc = 16; // key/value rows per tile (cols in attention matrix)

  int batch = blockIdx.x;
  int head = blockIdx.y;
  int tid = threadIdx.x; // 0..Br-1

  int bh_index = batch * H + head; // flattened (batch, head)

  // Shared memory layout (all float):
  // - sQ: [Br, D] current query tile
  // - sK: [Bc, D] current key tile
  // - sV: [Bc, D] current value tile
  // - sP: [Br, Bc] per-row attention scratch (scores -> exp(scores - mi_new))
  // - sY: [Br, D] running numerator for the current query tile (per row in the tile)
  extern __shared__ float smem[];
  float *sQ = smem;         // Br*D
  float *sK = sQ + Br * D;  // Bc*D
  float *sV = sK + Bc * D;  // Bc*D
  float *sP = sV + Bc * D;  // Br*Bc
  float *sY = sP + Br * Bc; // Br*D

  int Tc = ceil_div(N, Bc); // number of K/V tiles
  int Tr = ceil_div(N, Br); // number of Q tiles

  // Loop over ALL query tiles inside this single (batch, head) block (FA1-style).
  for (int tile_q = 0; tile_q < Tr; ++tile_q) {
    int token = tile_q * Br + tid;

    // Load Q row for this thread and init running numerator for this tile.

    int q_base = (bh_index * N + token) * D;
    for (int d = 0; d < D; ++d) {
      sQ[tid * D + d] = Q[q_base + d];
      sY[tid * D + d] = 0.0f;
    }

    // Per-row streaming softmax state for THIS token (registers).
    float mi = -CUDART_INF_F;
    float li = 0.0f;

    // Causal attention: K/V tiles strictly in the future are entirely masked.
    // Only need tj <= tile_q; the diagonal tile still needs elementwise mask (col > token).
    int j_max = (tile_q < (Tc - 1)) ? tile_q : (Tc - 1);

    for (int tj = 0; tj <= j_max; ++tj) {
      // Cooperative load K/V tile tj into shared: each thread loads one row k_row = tj*Bc + tid.
      int k_row = tj * Bc + tid;
      int kv_base = (bh_index * N + k_row) * D;

      for (int d = 0; d < D; ++d) {
        sK[tid * D + d] = K[kv_base + d];
        sV[tid * D + d] = V[kv_base + d];
      }

      __syncthreads(); // make sure sK/sV are ready for all threads

      // 1) Compute logits for this Q row against the Bc keys in this tile
      float block_max = -CUDART_INF_F;

      for (int k = 0; k < Bc; ++k) {
        int col = tj * Bc + k;

        float s = -CUDART_INF_F;
        if (col < N) {
          bool masked = (col > token); // causal mask within diagonal tile
          if (!masked) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) {
              acc += sQ[tid * D + d] * sK[k * D + d];
            }
            s = acc * softmax_scale;
          }
        }

        sP[tid * Bc + k] = s;
        block_max = fmaxf(block_max, s);
      }

      // 2/3) Online softmax merge
      float mi_new = fmaxf(mi, block_max);
      float alpha = __expf(mi - mi_new);

      float block_sum = 0.0f;
      for (int k = 0; k < Bc; ++k) {
        float s = sP[tid * Bc + k];
        float p = (s == -CUDART_INF_F) ? 0.0f : __expf(s - mi_new);
        sP[tid * Bc + k] = p; // overwrite with exp(logit - mi_new)
        block_sum += p;
      }

      float li_new = alpha * li + block_sum;

      // 4) Stream numerator y in the same mi_new frame: y_new = alpha*y_old + (p @ V_tile)
      for (int d = 0; d < D; ++d) {
        float pv = 0.0f;
        for (int k = 0; k < Bc; ++k) {
          pv += sP[tid * Bc + k] * sV[k * D + d];
        }
        sY[tid * D + d] = alpha * sY[tid * D + d] + pv;
      }

      mi = mi_new;
      li = li_new;
    }

    // Write normalized output O = y / l and save (m,l) for this token.
    int o_base = (bh_index * N + token) * D;
    int lm_idx = (bh_index * N + token);

    float inv_li = 1.0f / li;
    for (int d = 0; d < D; ++d) {
      O[o_base + d] = sY[tid * D + d] * inv_li;
    }

    m[lm_idx] = mi;
    l[lm_idx] = li;
  }
}

// No syncthreads needed between tile_q iterations: we overwrite per-tile shared rows anyway.

torch::Tensor flash_attn_v1(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, v must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4, "q must be 4D (B, H, N, D)");
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q, k, v shapes must match");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32, "flash_attn_v1 expects float32 inputs");

  auto q_contig = q.contiguous();
  auto k_contig = k.contiguous();
  auto v_contig = v.contiguous();

  c10::cuda::CUDAGuard device_guard(q_contig.device());

  int B = static_cast<int>(q_contig.size(0));
  int H = static_cast<int>(q_contig.size(1));
  int N = static_cast<int>(q_contig.size(2));
  int D = static_cast<int>(q_contig.size(3));

  auto o = torch::zeros_like(q_contig);
  auto l = torch::zeros({B, H, N}, q_contig.options());
  auto m = torch::full({B, H, N}, -std::numeric_limits<float>::infinity(), q_contig.options());

  constexpr int Br = 16;
  constexpr int Bc = 16;

  dim3 block(Br, 1, 1);
  // FA1-style: grid over (batch, head) only.
  dim3 grid(B, H, 1);

  float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
  size_t smem_bytes = (Br * D    // sQ
                       + Bc * D  // sK
                       + Bc * D  // sV
                       + Br * Bc // sP
                       + Br * D) // sY
                      * sizeof(float);

  flash_attn_v1_kernel<<<grid, block, smem_bytes, at::cuda::getDefaultCUDAStream()>>>(
      q_contig.data_ptr<float>(), k_contig.data_ptr<float>(), v_contig.data_ptr<float>(),
      o.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>(), B, H, N, D, softmax_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_attn_v1", &flash_attn_v1, "Minimal FlashAttention v1 (float32, causal)");
}

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
// Args:
// - Q: [B, H, N, D] queries
// - K: [B, H, N, D] keys
// - V: [B, H, N, D] values
// - O: [B, H, N, D] output
// - l: [B, H, N] running denominator
// - m: [B, H, N] running max
//   V2-style tiling (approx.): grid is (B, H, Tr). Each block processes *one query tile* (tile_q =
//   blockIdx.z) and loops over K/V tiles tj = 0..j_max inside the kernel.
__global__ void flash_attn_v1_5_kernel(const float *Q, const float *K, const float *V, float *O,
                                       float *l, float *m, int B, int H, int N, int D,
                                       float softmax_scale) {
  constexpr int Br = 16;
  constexpr int Bc = 16;
  // Each block is one query tile: Br query rows, iterating over K/V tiles of width Bc. Mapping to
  // the paper: the grid over query tiles corresponds to the paper’s inner loop over Q, while the
  // kernel’s loop over K/V tiles corresponds to the paper’s outer loop. This is the “FA2-style”
  // scheduling: parallelize across Q tiles (many thread blocks), then stream over K/V tiles within
  // each block.

  // grid: (B, H, Tr) where Tr = ceil(N / Br)
  int batch = blockIdx.x;
  int head = blockIdx.y;
  // query-tile index (which "block of 16" within the N dimension we're operating on)
  int tile_q = blockIdx.z;
  int tid = threadIdx.x; // 0..Br-1

  // row along the `N` dimension this thread owns (will be used to index into Q, O, l, and m)
  int token = tile_q * Br + tid;

  int bh_index = batch * H + head; // flattened batch/head index

  // Load this thread’s query vector Q[batch, head, token, :] into shared memory row sQ[tid, :] and
  // init running numerator for this tile.
  int q_base = (bh_index * N + token) * D;
  // l and m are [B, H, N] (one scalar per token position), so the offset is just: lm_idx = index of
  // (batch, head, token) in the flattened [B*H*N] array.
  int lm_idx = (bh_index * N + token);

  // Shared memory layout (all float):
  // - sQ: [Br, D] query tile
  // - sK: [Bc, D] key tile (current tj)
  // - sV: [Bc, D] value tile (current tj)
  // - sP: [Br, Bc] per-row attention scratch:
  //       first pre-softmax scores (QK^T * scale, with -inf for masked), then overwritten with
  //       exp(score - mi_new) (scaled, unnormalized weights).
  // - sY: [Br, D] running numerator y_i = sum_k p_{i,k} * v_k across all tiles
  extern __shared__ float smem[];
  float *sQ = smem;         // Br*D
  float *sK = sQ + Br * D;  // Bc*D
  float *sV = sK + Bc * D;  // Bc*D
  float *sP = sV + Bc * D;  // Br*Bc
  float *sY = sP + Br * Bc; // Br*D

  // Load this thread’s query vector into shared memory in sQ and init running numerator for this
  // tile.
  for (int d = 0; d < D; d++) {
    sQ[tid * D + d] = Q[q_base + d];
    sY[tid * D + d] = 0.0f; // y_i starts at 0 (unnormalized output numerator)
  }

  float mi = m[lm_idx];
  float li = l[lm_idx];

  // Number of K/V tiles (each tile covers Bc columns along N)
  int Tc = ceil_div(N, Bc);

  // Causal attention: K/V tiles strictly in the future are entirely masked. Only need tj <= tile_q;
  // the diagonal tile still needs elementwise mask (col > token).
  int j_max = min(tile_q, Tc - 1);
  // For non-causal: use `int j_max = Tc - 1;` and remove the `(col > token)` check below.
  for (int tj = 0; tj <= j_max; tj++) {
    // Cooperative load K/V tile tj into shared
    int k_row = tj * Bc + tid; // tid in [0,Br) and Br==Bc
    int kv_base = (bh_index * N + k_row) * D;
    for (int d = 0; d < D; d++) {
      sK[tid * D + d] = K[kv_base + d];
      sV[tid * D + d] = V[kv_base + d];
    }
    __syncthreads(); // make sure sK/sV are ready for all threads

    // 1) Compute logits for this Q row against the Bc keys in this tile
    float block_max = -CUDART_INF_F;
    for (int k = 0; k < Bc; k++) {
      int col = tj * Bc + k;

      float s = -CUDART_INF_F;
      bool masked = (col > token);
      if (!masked) {
        float acc = 0.0f;
        for (int d = 0; d < D; d++) {
          acc += sQ[tid * D + d] * sK[k * D + d];
        }
        s = acc * softmax_scale;
      }

      sP[tid * Bc + k] = s;
      block_max = fmaxf(block_max, s);
    }

    // 2/3) Online softmax merge
    float mi_new = fmaxf(mi, block_max);
    float alpha = __expf(mi - mi_new);

    float block_sum = 0.0f;
    for (int k = 0; k < Bc; k++) {
      float s = sP[tid * Bc + k];
      float p = (s == -CUDART_INF_F) ? 0.0f : __expf(s - mi_new);
      sP[tid * Bc + k] = p; // now sP holds exp(logit - mi_new)
      block_sum += p;
    }

    float li_new = alpha * li + block_sum;

    // 4) Stream numerator y in the same mi_new frame: y_new = alpha*y_old + (p @ V_tile)
    for (int d = 0; d < D; ++d) {
      float pv = 0.0f;
      // Dot product over the Bc columns in this tile:
      // (sP row for this token) · (column d of sV)
      for (int k = 0; k < Bc; ++k) {
        pv += sP[tid * Bc + k] * sV[k * D + d];
      }
      sY[tid * D + d] = alpha * sY[tid * D + d] + pv;
    }

    // Update running stats
    mi = mi_new;
    li = li_new;

    __syncthreads();
  }

  // Write normalized output O = y / l and save (m,l) for this token.
  int o_base = (bh_index * N + token) * D;
  float inv_li = 1.0f / li;
  for (int d = 0; d < D; ++d) {
    O[o_base + d] = sY[tid * D + d] * inv_li;
  }

  // Save per-row softmax stats (m, l). Not needed for forward output, but used by FlashAttention
  // backward to reconstruct P without storing NxN.
  m[lm_idx] = mi;
  l[lm_idx] = li;
}

torch::Tensor flash_attn_v1_5(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
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
  TORCH_CHECK(N % 16 == 0, "flash_attn_v1_5 requires seq len N % 16 == 0");

  auto o = torch::zeros_like(q_contig);
  auto l = torch::zeros({B, H, N}, q_contig.options());
  auto m = torch::full({B, H, N}, -std::numeric_limits<float>::infinity(), q_contig.options());

  constexpr int Br = 16;
  constexpr int Bc = 16;

  dim3 block(Br, 1, 1);
  // Grid over (batch, head, query-tile): the third dim breaks N (sequence length) into
  // chunks of Br rows of Q. Each block processes one chunk at a time (paper’s inner loop
  // over Q).
  dim3 grid(B, H, ceil_div(N, Br));

  float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
  size_t smem_bytes = (Br * D    // sQ
                       + Bc * D  // sK
                       + Bc * D  // sV
                       + Br * Bc // sP
                       + Br * D) // sY
                      * sizeof(float);

  flash_attn_v1_5_kernel<<<grid, block, smem_bytes, at::cuda::getDefaultCUDAStream()>>>(
      q_contig.data_ptr<float>(), k_contig.data_ptr<float>(), v_contig.data_ptr<float>(),
      o.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>(), B, H, N, D, softmax_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_attn_v1_5", &flash_attn_v1_5, "Minimal FlashAttention v1.5 (float32, causal)");
}

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
__global__ void flash_attn_v1_kernel(const float *Q, // [B,H,N,D]
                                     const float *K, // [B,H,N,D]
                                     const float *V, // [B,H,N,D]
                                     float *O,       // [B,H,N,D]
                                     float *l,       // [B,H,N]
                                     float *m,       // [B,H,N]
                                     int B, int H, int N, int D, float softmax_scale) {
  // We're in a 4D tensor [B, H, N, D]. B, H, N, and D represent the dimensions of the
  // Tensor. This kernel operates at index `blockIdx.x` B, `blockIdx.y` H, and on a 16
  // row slice within the 2D [N, D] array. Each thread handles a single row; each thread
  // block handles a tile of 16 rows.

  constexpr int Br = 16;
  constexpr int Bc = 16;
  // Each block is one query tile: Br query rows, iterating over K/V tiles of width Bc.
  // Mapping to the paper: the grid over query tiles corresponds to the paper’s inner
  // loop over Q, while the kernel’s loop over K/V tiles corresponds to the paper’s outer
  // loop. This is the “FA2-style” scheduling: parallelize across Q tiles (many thread
  // blocks), then stream over K/V tiles within each block.

  // grid: (B, H, Tr) where Tr = ceil(N / Br)
  int batch = blockIdx.x;
  int head = blockIdx.y;
  // query-tile index (which "block of 16" within the N dimension we're operating on)
  int tile_q = blockIdx.z;
  int tid = threadIdx.x; // 0..Br-1

  // row along the `N` dimension this thread owns (will be used to index into Q, O, l,
  // and m)
  int token = tile_q * Br + tid;
  if (token >= N)
    return;

  // base offsets
  // index for Q/K/V/O: (((b*H + h)*N + t)*D + d)
  int bh_index = batch * H + head; // flattened batch/head index
  // `token` is the position along the sequence-length dimension N that this thread owns.
  // Q is laid out as contiguous [B, H, N, D].
  //
  // Flatten (batch, head) -> bh_index, so each (batch, head) corresponds to one
  // contiguous [N, D] matrix.
  //
  // bh_index * N         = number of token-rows before this (batch, head) slice
  // + token              = selects the token-row within that slice
  // * D                  = converts token-row index into element offset (each row has D
  //                        floats)
  //
  // So `q_base` is the linear offset for Q[batch, head, token, 0].
  int q_base = (bh_index * N + token) * D;
  // l and m are [B, H, N] (one scalar per token position), so the offset is just: lm_idx
  // = index of (batch, head, token) in the flattened [B*H*N] array.
  int lm_idx = (bh_index * N + token);

  // Shared memory layout (all float):
  // sQ: [Br, D], sK: [Bc, D], sV: [Bc, D], sP: [Br, Bc]
  extern __shared__ float smem[];
  float *sQ = smem;        // Br*D
  float *sK = sQ + Br * D; // Bc*D
  float *sV = sK + Bc * D; // Bc*D
  float *sP = sV + Bc * D; // Br*Bc

  // Load this thread’s query vector Q[batch, head, token, :] into shared memory row
  // sQ[tid, :].
  for (int d = 0; d < D; d++) {
    sQ[tid * D + d] = Q[q_base + d];
  }

  float mi = m[lm_idx];
  float li = l[lm_idx];

  // Number of K/V tiles (each tile covers Bc columns along N)
  int Tc = ceil_div(N, Bc);
  // causal: restrict K/V tile index to the current query tile (tile_q) and clamp to the
  // last valid tile (Tc-1)
  int j_max = min(tile_q, Tc - 1);

  for (int tj = 0; tj <= j_max; tj++) {
    // Cooperative load K/V tile tj into shared
    int k_row = tj * Bc + tid; // tid in [0,Br) and Br==Bc
    // K/V are also laid out [B, H, N, D]; kv_base is the start of row k_row for this
    // (batch, head) slice
    int kv_base = (bh_index * N + k_row) * D;

    if (tid < Bc) {
      if (k_row < N) {
        for (int d = 0; d < D; d++) {
          sK[tid * D + d] = K[kv_base + d];
          sV[tid * D + d] = V[kv_base + d];
        }
      } else {
        for (int d = 0; d < D; d++) {
          sK[tid * D + d] = 0.0f;
          sV[tid * D + d] = 0.0f;
        }
      }
    }
    // Each thread writes one row of sK/sV, but the dot products below read every row.
    // Synchronize so all shared rows are fully written before any thread starts reading.
    __syncthreads();

    // 1) Compute logits for this Q row against the Bc keys in this tile
    float block_max = -CUDART_INF_F;
    for (int k = 0; k < Bc; k++) {
      int col = tj * Bc + k;

      float s = -CUDART_INF_F;
      if (col < N) {
        bool masked = (col > token);
        if (!masked) {
          float acc = 0.0f;
          // dot(Q[Q_row], K[col])
          for (int d = 0; d < D; d++) {
            acc += sQ[tid * D + d] * sK[k * D + d];
          }
          s = acc * softmax_scale;
        }
      }

      sP[tid * Bc + k] = s;
      block_max = fmaxf(block_max, s);
    }

    // 2/3) Streaming-softmax merge in the "global max" frame (mi_new)
    //
    // We already computed logits s = (Q·K)*scale and stored them in sP[tid, k].
    // block_max is max_k sP[tid,k] for this tile.
    //
    // We'll update the running max for this row:
    //   mi_new = max(mi, block_max)
    // and rescale the *previous* denominator sum into the new max frame:
    //   alpha = exp(mi - mi_new)   (== 1 if mi_new == mi)
    // Then compute this tile's contributions directly in the mi_new frame:
    //   p_k = exp(s_k - mi_new)
    //   block_sum = sum_k p_k
    // Finally:
    //   li_new = alpha * li + block_sum
    float mi_new = fmaxf(mi, block_max);
    float alpha = __expf(mi - mi_new);

    // Compute p_k in the mi_new frame (your "always scale by global max" method)
    float block_sum = 0.0f;
    for (int k = 0; k < Bc; k++) {
      float s = sP[tid * Bc + k];
      float p = (s == -CUDART_INF_F) ? 0.0f : __expf(s - mi_new);
      sP[tid * Bc + k] = p; // now sP holds exp(logit - mi_new)
      block_sum += p;
    }

    float li_new = alpha * li + block_sum;

    // 4) Update output row in the same mi_new frame.
    // Since sP now stores exp(logit - mi_new), we can form the numerator piece directly:
    //   pv[d] = sum_k exp(s_k - mi_new) * V[k,d]
    // and then the online-normalized update is:
    //   O = (alpha*li*O + pv) / li_new
    int o_base = (bh_index * N + token) * D;
    for (int d = 0; d < D; d++) {
      float pv = 0.0f;
      for (int k = 0; k < Bc; k++) {
        pv += sP[tid * Bc + k] * sV[k * D + d];
      }
      float old_o = O[o_base + d];
      float new_o = (alpha * li * old_o + pv) / li_new;
      O[o_base + d] = new_o;
    }

    // Update running stats
    mi = mi_new;
    li = li_new;

    __syncthreads();
  }

  m[lm_idx] = mi;
  l[lm_idx] = li;
}

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
  // Grid over (batch, head, query-tile): the third dim breaks N (sequence length) into
  // chunks of Br rows of Q. Each block processes one chunk at a time (paper’s inner loop
  // over Q).
  dim3 grid(B, H, ceil_div(N, Br));

  float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
  size_t smem_bytes = (Br * D + Bc * D + Bc * D + Br * Bc) * sizeof(float);

  flash_attn_v1_kernel<<<grid, block, smem_bytes, at::cuda::getDefaultCUDAStream()>>>(
      q_contig.data_ptr<float>(), k_contig.data_ptr<float>(), v_contig.data_ptr<float>(),
      o.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>(), B, H, N, D, softmax_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_attn_v1", &flash_attn_v1, "Minimal FlashAttention v1 (float32, causal)");
}

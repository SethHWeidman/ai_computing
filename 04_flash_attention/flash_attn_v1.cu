#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <limits>

static inline __host__ __device__ int ceil_div(int a, int b) { return (a + b - 1) / b; }

// -----------------------------------------------------------------------------
// Minimal FlashAttention v1 teaching kernel (float32 only, causal).
//
// Layout: Q/K/V/O are contiguous [B, H, N, D].
//
// FA1-style scheduling:
//   - grid = (B, H): one block owns ONE (batch, head) slice.
//   - within that slice, the block loops over query tiles (outer loop).
//   - within each query tile, it streams over key/value tiles (inner loop).
//
// Within a query tile:
//   - each thread owns one query row (one token position) in that tile.
//   - the thread maintains streaming-softmax state for that row:
//       row_max    = m_i  (running max logit)
//       row_sumexp = l_i  (running denominator = sum exp(logit - row_max))
//       O_accum    = \tilde{O}_i (running numerator accumulator, length D)
//   - at the end, it writes O = O_accum / row_sumexp.
//
// This is intentionally “whiteboard-aligned”, not performance-optimized.
// -----------------------------------------------------------------------------
__global__ void flash_attn_v1_kernel(const float *Q, const float *K, const float *V, float *O,
                                     int B, int H, int N, int D, float softmax_scale) {
  // Tile sizes (both 16 here to mirror the diagrams / simplify the kernel).

  // both rows in a Q tile (outer loop tile size) and rows in a K/V tile (inner loop tile size)
  constexpr int T = 16;

  // Which (batch, head) slice does this block own?
  int batch = blockIdx.x;
  int head = blockIdx.y;

  // Thread index within the block:
  //   one thread == one query row within the current Q tile.
  int row_in_q_tile = threadIdx.x; // 0..T-1

  int bh = batch * H + head; // flattened (batch, head)

  int bh_index = batch * H + head; // flattened (batch, head)

  // ---------------------------------------------------------------------------
  // Shared memory (float32):
  //   sh_Q       [T, D]    current query tile
  //   sh_K       [T, D]    current key tile
  //   sh_V       [T, D]    current value tile
  //   sh_S       [T, T]    score scratch for the current (Q_tile, K_tile) pair
  //                        (overwritten in-place with exp(score - row_max_new))
  //   sh_O_accum [T, D]    running numerator accumulator (\tilde{O}) for this Q tile
  // ---------------------------------------------------------------------------
  extern __shared__ float smem[];
  float *sh_Q = smem;               // T * D
  float *sh_K = sh_Q + T * D;       // T * D
  float *sh_V = sh_K + T * D;       // T * D
  float *sh_S = sh_V + T * D;       // T * T
  float *sh_O_accum = sh_S + T * T; // T * D

  int num_kv_tiles = ceil_div(N, T);
  int num_q_tiles = ceil_div(N, T);

  // Outer loop: iterate over Q tiles (and therefore output tiles).
  for (int q_tile_idx = 0; q_tile_idx < num_q_tiles; ++q_tile_idx) {
    // Global query row index (token position) owned by this thread in this tile.
    int q_row = q_tile_idx * T + row_in_q_tile;
    int q_base = (bh * N + q_row) * D;

    // Load this thread's Q row into shared (and zero its numerator accumulator row).
    // (Only this thread reads its own Q row / O_accum row; no sync needed.)
    for (int d = 0; d < D; ++d) {
      sh_Q[row_in_q_tile * D + d] = Q[q_base + d];
      sh_O_accum[row_in_q_tile * D + d] = 0.0f; // \tilde{O}_i starts at 0
    }

    // Streaming-softmax state for THIS query row (kept in registers).
    float row_max = -CUDART_INF_F;
    float row_sumexp = 0.0f;

    // Causal mask (lower-triangular):
    // - K/V tiles strictly in the future are fully masked and can be skipped.
    // - Only the diagonal tile needs per-element masking (col > q_row).
    int kv_tile_max = q_tile_idx;
    if (kv_tile_max > (num_kv_tiles - 1))
      kv_tile_max = num_kv_tiles - 1;

    // Inner loop: stream over K/V tiles up to the causal boundary.
    for (int kv_tile_idx = 0; kv_tile_idx <= kv_tile_max; ++kv_tile_idx) {
      // -----------------------------------------------------------------------
      // Step A: load K_tile and V_tile into shared (cooperative: one row per thread).
      // -----------------------------------------------------------------------
      int kv_row = kv_tile_idx * T + row_in_q_tile; // global row in K/V this thread loads
      int kv_base = (bh * N + kv_row) * D;

      for (int d = 0; d < D; ++d) {
        sh_K[row_in_q_tile * D + d] = K[kv_base + d];
        sh_V[row_in_q_tile * D + d] = V[kv_base + d];
      }
      // We will read ALL rows of sh_K/sh_V in the dot-products below.
      __syncthreads();

      // -----------------------------------------------------------------------
      // Step B: compute score row S_i for this query row vs the T keys in this tile.
      //
      //   S_{i,k} = (Q_i · K_k) * softmax_scale
      //
      // Store in sh_S (and track the max over k for this row).
      // -----------------------------------------------------------------------
      float tile_row_max = -CUDART_INF_F;

      for (int k = 0; k < T; ++k) {
        int col = kv_tile_idx * T + k; // global key position

        float s = -CUDART_INF_F;
        if (col < N) {
          // Only diagonal tile needs masking; this condition is safe everywhere.
          bool masked = (col > q_row);
          if (!masked) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) {
              acc += sh_Q[row_in_q_tile * D + d] * sh_K[k * D + d];
            }
            s = acc * softmax_scale;
          }
        }

        sh_S[row_in_q_tile * T + k] = s;
        tile_row_max = fmaxf(tile_row_max, s);
      }

      // -----------------------------------------------------------------------
      // Step C: streaming-softmax merge (this is the core FlashAttention trick).
      //
      // Maintain row-wise running max m_i and running sum-exp l_i across tiles:
      //
      //   row_max_new = max(row_max_old, max_k S_{i,k})
      //   rescale_old = exp(row_max_old - row_max_new)
      //
      // Then compute this tile’s contributions in the NEW max frame:
      //
      //   P_{i,k} = exp(S_{i,k} - row_max_new)   (0 if masked / -inf)
      //
      // Update denominator:
      //   row_sumexp_new = rescale_old * row_sumexp_old + sum_k P_{i,k}
      // -----------------------------------------------------------------------

      float row_max_new = fmaxf(row_max, tile_row_max);
      float rescale_old = __expf(row_max - row_max_new);

      float tile_row_sumexp = 0.0f;
      for (int k = 0; k < T; ++k) {
        float s = sh_S[row_in_q_tile * T + k];
        float p = (s == -CUDART_INF_F) ? 0.0f : __expf(s - row_max_new);
        sh_S[row_in_q_tile * T + k] = p; // overwrite scores with P in max-shifted frame
        tile_row_sumexp += p;
      }

      float row_sumexp_new = rescale_old * row_sumexp + tile_row_sumexp;

      // -----------------------------------------------------------------------
      // Step D: stream the numerator accumulator \tilde{O} in the same max frame.
      //
      // Tile contribution (for this query row) is:
      //   pv[d] = sum_k P_{i,k} * V_k[d]
      //
      // Merge with the running accumulator (rescaling the old frame -> new frame):
      //   O_accum_new[d] = rescale_old * O_accum_old[d] + pv[d]
      // -----------------------------------------------------------------------
      for (int d = 0; d < D; ++d) {
        float pv = 0.0f;
        for (int k = 0; k < T; ++k) {
          pv += sh_S[row_in_q_tile * T + k] * sh_V[k * D + d];
        }
        sh_O_accum[row_in_q_tile * D + d] = rescale_old * sh_O_accum[row_in_q_tile * D + d] + pv;
      }

      // Commit streaming state for next K/V tile.
      row_max = row_max_new;
      row_sumexp = row_sumexp_new;

      // No extra syncthreads needed here: next iteration overwrites sh_K/sh_V rows.
    }

    // -------------------------------------------------------------------------
    // Finalize this query row: O_i = O_accum_i / row_sumexp_i
    // -------------------------------------------------------------------------
    int o_base = (bh * N + q_row) * D;
    float inv_row_sumexp = 1.0f / row_sumexp;

    for (int d = 0; d < D; ++d) {
      O[o_base + d] = sh_O_accum[row_in_q_tile * D + d] * inv_row_sumexp;
    }

    // https://github.com/SethHWeidman/ai_computing/blob/master/03_streaming_softmax/02_sum_of_exponentials_large_example.py
  }

  // Write normalized output once: O = y / l. Follows the pattern of "dividing the accumulated
  // numerator and denominator at the end" from:
  // https://github.com/SethHWeidman/ai_computing/blob/master/03_streaming_softmax/03_softmax_dot_product_streaming_example.py
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
  TORCH_CHECK(N % 16 == 0, "flash_attn_v1 requires seq len N % 16 == 0");

  auto o = torch::zeros_like(q_contig);

  constexpr int T = 16;

  // One thread per row in the Q tile.
  dim3 block(T, 1, 1);

  // One block per (batch, head).
  dim3 grid(B, H, 1);

  float softmax_scale = 1.0f / std::sqrt(static_cast<float>(D));
  size_t smem_bytes = (T * D    // sh_Q
                       + T * D  // sh_K
                       + T * D  // sh_V
                       + T * T  // sh_S (scores/probs scratch)
                       + T * D) // sh_O_accum (numerator accumulator)
                      * sizeof(float);

  flash_attn_v1_kernel<<<grid, block, smem_bytes, at::cuda::getDefaultCUDAStream()>>>(
      q_contig.data_ptr<float>(), k_contig.data_ptr<float>(), v_contig.data_ptr<float>(),
      o.data_ptr<float>(), B, H, N, D, softmax_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_attn_v1", &flash_attn_v1, "Minimal FlashAttention v1 (float32, causal)");
}

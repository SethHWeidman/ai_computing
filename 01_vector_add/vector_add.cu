#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

// Helper macro to wrap CUDA API calls with error checking.
#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t err = (call);                                                                      \
    if (err != cudaSuccess) {                                                                      \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));      \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

// Device (GPU) kernel: each thread adds one element from the input vectors.
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // unique index per thread
  if (i < n)
    c[i] = a[i] + b[i]; // guard the boundary
}

int main() {
  const int N = 1 << 24; // ~17M elements
  const size_t bytes = N * sizeof(float);

  // Step 1: Allocate unified memory so both CPU and GPU can access it.
  float *a, *b, *c;
  CUDA_CHECK(cudaMallocManaged(&a, bytes));
  CUDA_CHECK(cudaMallocManaged(&b, bytes));
  CUDA_CHECK(cudaMallocManaged(&c, bytes));
  std::vector<float> c_cpu(N, 0.0f); // host (CPU)-side buffer for reference result

  // Step 2: Initialize input data on the CPU.
  for (int i = 0; i < N; ++i) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  // Step 3 : Prefetch unified memory to the GPU to minimize page migrations (moving pages of data
  // from CPU to GPU during computation).
  int dev = 0; // device 0
  CUDA_CHECK(cudaMemPrefetchAsync(a, bytes, dev, 0));
  CUDA_CHECK(cudaMemPrefetchAsync(b, bytes, dev, 0));
  CUDA_CHECK(cudaMemPrefetchAsync(c, bytes, dev, 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Step 4: Configure the kernel launch (256 threads/block is a common default).
  const int block = 256;
  const int grid = (N + block - 1) / block;

  // Step 5: Time the GPU kernel using CUDA events.
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  vector_add<<<grid, block>>>(a, b, c, N);
  CUDA_CHECK(cudaEventRecord(stop));

  // Step 6: Check for errors, synchronize, and extract the elapsed time.
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float gpu_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // Step 7: Ensure each GPU result equals the expected constant (3.0f).
  float max_err = 0.0f;
  for (int i = 0; i < N; ++i)
    max_err = fmaxf(max_err, fabsf(c[i] - 3.0f));
  printf("Max error vs 3.0f = %f\n", max_err);

  // Step 8: Prefetch the inputs back to the CPU to time the host (CPU) implementation.
  CUDA_CHECK(cudaMemPrefetchAsync(a, bytes, cudaCpuDeviceId, 0));
  CUDA_CHECK(cudaMemPrefetchAsync(b, bytes, cudaCpuDeviceId, 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Step 9: Compute the CPU reference result, timing it for a fair comparison.
  auto cpu_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; ++i)
    c_cpu[i] = a[i] + b[i];
  auto cpu_end = std::chrono::high_resolution_clock::now();
  double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

  float max_diff = 0.0f;
  for (int i = 0; i < N; ++i)
    max_diff = fmaxf(max_diff, fabsf(c[i] - c_cpu[i]));

  const double speedup = cpu_ms / gpu_ms;

  // Step 10: Report timing results.
  printf("Max difference GPU vs CPU = %f\n", max_diff);
  printf("GPU time: %.3f ms | CPU time: %.3f ms | Speedup: %.2fx\n", gpu_ms, cpu_ms, speedup);

  // Step 11: Free GPU allocations.
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
  return 0;
}

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err));                                        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// 1) A CUDA C++ kernel: runs on the GPU, launched from the CPU.
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // unique index for each thread
  if (i < n)
    c[i] = a[i] + b[i]; // bounds check (avoid OOB)
}

int main() {
  const int N = 1 << 20; // 1,048,576 elements
  const size_t bytes = N * sizeof(float);

  // 2) Allocate Unified Memory accessible by CPU and GPU
  float *a, *b, *c;
  CUDA_CHECK(cudaMallocManaged(&a, bytes));
  CUDA_CHECK(cudaMallocManaged(&b, bytes));
  CUDA_CHECK(cudaMallocManaged(&c, bytes));

  // 3) Initialize on the CPU
  for (int i = 0; i < N; ++i) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  // 4) Choose a launch configuration (256 threads per block is a good default)
  const int block = 256;
  const int grid = (N + block - 1) / block;

  // 5) Launch the kernel on the GPU
  vector_add<<<grid, block>>>(a, b, c, N);

  // 6) Always check for launch errors and synchronize before using results
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // 7) Validate on the CPU
  float max_err = 0.0f;
  for (int i = 0; i < N; ++i)
    max_err = fmaxf(max_err, fabsf(c[i] - 3.0f));
  printf("Max error = %f\n", max_err);

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
  return 0;
}

# ai_computing

## `01_vector_add`

View the implementation in [`01_vector_add/vector_add.cu`](01_vector_add/vector_add.cu).

This `01_vector_add` example was inspired by NVIDIA’s blog post “Even Easier Introduction to CUDA”:
https://developer.nvidia.com/blog/even-easier-introduction-cuda/

### Overview

Script containing a CUDA kernel that:

* Initializes two vectors of length 1M, one of which has values of 1.0, the other of which has
  values of 2.0.
* Adds them on the GPU, using 256 threads per block.
* Confirms that the resulting values are all 3.0.

### Instructions to run

To build, run:

```
cd 01_vector_add
nvcc -arch=compute_89 -code=sm_89 vector_add.cu -o vector_add
```

> **Note:** the `-arch=compute_89` and `-code=sm_89` arguments tell the code to generate PTX and
> machine code, respectively, for the Ada Lovelace / L4 GPUs which NVIDIA lists as "Compute
> Capability 8.9". See [NVIDIA's CUDA GPU list](https://developer.nvidia.com/cuda-gpus). Modify
> these flags if using a GPU other than an L4.

Then run `./vector_add` (from the `./01_vector_add` directory) to see the agreement between the
CPU/GPU results plus the measured performance, e.g.:

```
Max error vs 3.0f = 0.000000
Max difference GPU vs CPU = 0.000000
GPU time: 0.124 ms | CPU time: 4.902 ms | Speedup: 39.62x
```

## `02_cuda_vs_pytorch_matmul`

This example reuses the shared-memory tiled matmul kernel from `intro_to_cuda/demo3_matmul` and
wraps it in a small PyTorch C++/CUDA extension, allowing a custom CUDA kernel (`fast_mm`) and
standard `torch.mm` to be benchmarked side by side from Python. See
[`02_cuda_vs_pytorch_matmul/README.md`](02_cuda_vs_pytorch_matmul/README.md) for full details.

## `03_streaming_softmax`

This directory contains examples demonstrating the concept of streaming softmax, which is
crucial for memory-efficient attention mechanisms. It explores how to compute the sum of
scaled exponentials (the denominator of softmax) and the softmax dot product in a
streaming fashion, processing data in small blocks rather than all at once. This approach
is used to save memory in FlashAttention and similar algorithms. See
[`03_streaming_softmax/README.md`](03_streaming_softmax/README.md) for full details.

## KV cache reference

The file `LLMs-from-scratch/ch04/03_kv-cache/gpt_with_kv_cache_reference.py` contains a
reference implementation of a GPT-style key–value (KV) cache and a simple text generation
script that uses it.

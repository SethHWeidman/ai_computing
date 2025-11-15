# ai_computing

## `vector_add`

View the implementation in [vector_add.cu](vector_add.cu).

### Overview

Script containing a CUDA kernel that:

* Initializes two vectors of length 1M, one of which has values of 1.0, the other of which has values of 2.0.
* Adds them on the GPU, using 256 threads per block.
* Confirms that the resulting values are all 3.0.

### Instructions to run

To build, run:

```
nvcc -arch=compute_89 -code=sm_89 vector_add.cu -o vector_add
```

> **Note:** the `-arch=compute_89` and `-code=sm_89` arguments tell the code to generate PTX and
> machine code, respectively, for the Ada Lovelace / L4 GPUs which NVIDIA lists as "Compute
> Capability 8.9". See [NVIDIA's CUDA GPU list](https://developer.nvidia.com/cuda-gpus). Modify
> these flags if using a GPU other than an L4.

Then run `./vector_add` to see the agreement between the CPU/GPU results plus the measured
performance, e.g.:

```
Max error vs 3.0f = 0.000000
Max difference GPU vs CPU = 0.000000
GPU time: 0.124 ms | CPU time: 4.902 ms | Speedup: 39.62x
```

# Hopper GEMM

A minimal single-GPU FP16 GEMM runner for Hopper/H100 using CUTLASS CuTe DSL.

## How to run

```bash
cd 08_hopper_gemm
python h100_single_gpu_gemm.py
```

Example output:

```text
Running Hopper Dense GEMM with:
mnkl: (8192, 8192, 8192, 1)
A dtype: Float16, B dtype: Float16, C dtype: Float16, Acc dtype: Float32
Matrix majors - A: k, B: k, C: n
Tile Shape: (128, 256), Cluster Shape: (1, 1)
Tolerance: 0.1
Warmup iterations: 3
Iterations: 10
Skip reference checking: True
Use cold L2: False

GPU: NVIDIA H100 80GB HBM3 (device 0, cc 9.0)
Problem: M=8192, N=8192, K=8192, batch=1, tile=(128, 256), cluster=(1, 1)
Average kernel time: 1256.0 us
Throughput: 875.38 TFLOP/s
```

## Interpreting the throughput

This is a strong result for a dense FP16 GEMM on one H100. The fair comparison is the
dense FP16 Tensor Core peak. NVIDIA's
[H100 specs](https://www.nvidia.com/en-gb/data-center/h100/) list H100 SXM FP16 Tensor
Core peak as 1,979 TFLOP/s with sparsity, so the dense baseline is half of that: about
989.5 TFLOP/s.

For the example run above:

```text
875.38 / 989.5 ~= 88.5%
```

So this run is around 88.5% of the realistic dense FP16 Tensor Core peak. That is a good
sign that the Hopper WGMMA/Tensor Core path is being used effectively. Treat this as a
microbenchmark for one large, well-aligned GEMM shape rather than an end-to-end model
throughput number.

Use `--check` to run the slower reference check:

```bash
python h100_single_gpu_gemm.py --check
```

## Files

- `h100_single_gpu_gemm.py`: loads the CUTLASS CuTe DSL Hopper dense GEMM example from
  `csrc/cutlass`, runs it on one H100, and reports average kernel time plus throughput.

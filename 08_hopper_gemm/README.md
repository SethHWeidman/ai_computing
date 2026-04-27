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
dense FP16 Tensor Core peak. NVIDIA's [H100
specs](https://www.nvidia.com/en-gb/data-center/h100/) list H100 SXM FP16 Tensor Core
peak as 1,979 TFLOP/s with sparsity, so the dense baseline is half of that: about 989.5
TFLOP/s.

For the example run above:

```text
875.38 / 989.5 ~= 88.5%
```

So this run is around 88.5% of the realistic dense FP16 Tensor Core peak. That is a good
sign that the Hopper WGMMA/Tensor Core path is being used effectively for this large,
well-aligned GEMM shape.

## Interpreting the kernel time

`Average kernel time: 1256.0 us` is also excellent for this exact GEMM shape. The
operation count is:

```text
2 * 8192 * 8192 * 8192 = 1,099,511,627,776 FLOPs
```

Using the same dense FP16 H100 SXM baseline of about 989.5 TFLOP/s, the ideal peak time
would be:

```text
1.0995e12 FLOPs / 989.5e12 FLOP/s ~= 0.001111 s ~= 1111 us
```

The measured 1256 us is therefore only about `1256 / 1111 ~= 1.13x` above the dense
Tensor Core lower bound, or about 13% slower than theoretical peak. In practical terms,
this puts the run in the "excellent real kernel" range for a huge square GEMM with clean
alignment and a large K dimension:

```text
~1.1 ms      theoretical dense peak
~1.2-1.4 ms  excellent real kernel
~1.5-2.0 ms  good, but likely leaving more performance on the table
>2.0 ms      likely worth investigating layout, tiling, or occupancy
```

Use `--check` to run the slower reference check:

```bash
python h100_single_gpu_gemm.py --check
```

## Files

- `h100_single_gpu_gemm.py`: loads the CUTLASS CuTe DSL Hopper dense GEMM example from
  `csrc/cutlass`, runs it on one H100, and reports average kernel time plus throughput.

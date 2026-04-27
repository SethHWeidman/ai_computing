# Hopper GEMM

Single-GPU FP16 GEMM runners for Hopper/H100 using CUTLASS CuTe DSL.

## How to run

### `01_h100_single_gpu_gemm.py`

Run the minimal wrapper around the CUTLASS Hopper dense GEMM example:

```bash
cd 08_hopper_gemm
python 01_h100_single_gpu_gemm.py
```

### `02_h100_single_gpu_gemm.py`

Run the more explicit version that shows tensor creation, DLPack wrapping,
`HopperWgmmaGemmKernel` construction, `cute.compile(...)`, and launch timing:

```bash
cd 08_hopper_gemm
python 02_h100_single_gpu_gemm.py
```

## Tile Shape And Cluster Shape

### Execution Hierarchy

CUDA's execution hierarchy is roughly:

```text
thread -> warp -> CTA / thread block -> thread-block cluster -> grid
```

CUTLASS often uses "CTA" for the work assigned to one CUDA thread block. NVIDIA's [CUDA
programming
guide](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)
describes thread-block clusters as a Hopper-era hierarchy level: on compute capability
9.0 and newer GPUs, a cluster is a group of thread blocks that can be co-scheduled on one
GPU Processing Cluster.

### Tile Shape

`--tile-shape` is measured in matrix elements. With the default `--tile-shape 128,256`,
one CTA computes one `128 x 256` tile of the output matrix `C`. The K tile size is chosen
by the Hopper kernel setup; for the default configuration it becomes a CTA tile of `128 x
256 x 64`.

### Cluster Shape

`--cluster-shape` is measured in CTAs, not matrix elements. With `--tile-shape 128,256
--cluster-shape 2,1`, each CTA still owns one `128 x 256` tile of `C`, but Hopper groups
two neighboring CTAs along M into one thread-block cluster. That cluster covers a `256 x
256` region of `C`.

This is different from `--cluster-shape 1,1` because the CTAs are grouped into a Hopper
thread-block cluster instead of being launched only as standalone blocks. Each CTA still
computes its own `tile-shape` tile of `C`, but CTAs in the same cluster are co-scheduled
close enough together that Hopper can give them cluster-level coordination.

### TMA Multicast

For this GEMM, the easiest cluster feature to understand is **TMA multicast**. TMA is
Hopper's Tensor Memory Accelerator: a hardware path for copying tensor tiles between
global memory and shared memory. "Multicast" means one TMA load can deliver the same
input tile to multiple CTAs in the same cluster.

That helps because neighboring GEMM tiles reuse input data. CTAs grouped along M compute
different rows of `C` but the same columns, so they can share the same `B` tile. CTAs
grouped along N compute the same rows of `C` but different columns, so they can share the
same `A` tile.

For example, with `--tile-shape 128,256 --cluster-shape 2,1`, the cluster contains two
CTAs along M. Each CTA still owns one `128 x 256` output tile, but together the cluster
covers a `256 x 256` region of `C`. Those two CTAs need different `A` tiles but the same
`B` tile, so the kernel may use TMA multicast to load the `B` tile once and deliver it to
both CTAs' shared-memory buffers.

The table below is a quick map of what reuse each cluster shape can expose:

| Cluster shape | What is grouped? | Possible operand reuse |
|---|---:|---|
| `1,1` | One CTA | No inter-CTA multicast |
| `2,1` | Two CTAs along M | Share/multicast `B` tiles |
| `1,2` | Two CTAs along N | Share/multicast `A` tiles |
| `2,2` | Four CTAs in MxN | Share/multicast both `A` and `B` tiles |

Cluster shape can reduce duplicated global-memory traffic, but it is still a tuning knob
rather than an automatic win. Larger clusters also make multiple CTAs cooperate as one
cluster, which adds cluster-level coordination, scheduling constraints, and
pipeline/barrier bookkeeping. If the kernel already hides TMA load latency behind WGMMA
compute, or if nearby CTAs already get good L2 reuse, the extra coordination can outweigh
the saved input-tile traffic.

On the default `8192 x 8192 x 8192` problem with `--tile-shape 128,256`, this happened in
practice: `--cluster-shape 2,2` enabled multicast for both `A` and `B`, but it ran slower
than `--cluster-shape 1,1` on the measured H100 run. The `1,1` run took `1231.4 us`
(`892.91 TFLOP/s`), while the `2,2` run took `1377.3 us` (`798.29 TFLOP/s`). That is
about 12% longer kernel time, or about 11% lower throughput.

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

Use `--check` to run the slower reference check with either runner:

```bash
python 01_h100_single_gpu_gemm.py --check
python 02_h100_single_gpu_gemm.py --check
```

Successful checked runs print `Reference check: passed`.

## Files

- `01_h100_single_gpu_gemm.py`: loads the CUTLASS CuTe DSL Hopper dense GEMM example from
  `csrc/cutlass`, runs it on one H100, and reports average kernel time plus throughput.
- `02_h100_single_gpu_gemm.py`: exposes more of the host-side setup directly: tensor
  allocation/layout, CuTe tensor wrapping, kernel construction, JIT compilation, launch,
  optional Torch reference checking, and the same timing/throughput report.

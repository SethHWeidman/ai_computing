# Why `pack_nibbles` exists

## On the term "nibble"

A *nibble* (sometimes *nybble*) is a 4-bit unit of data, i.e. half a byte. Standard CS
terminology since the 1970s; NVIDIA uses it in the cuBLAS 12.9 release notes when
describing the FP4 packing layout.

## The problem

NVFP4 represents each value as a 4-bit code, but memory is byte-addressed — there is no
way to write a single nibble to RAM. Without packing, each 4-bit code would occupy its
own byte, wasting the upper 4 bits and defeating the point of FP4.

## What it does

`pack_nibbles` stuffs two FP4 codes into each byte:

```
byte = (code[1] << 4) | code[0]
        ^^^^^^^^^^^      ^^^^^^^
        high nibble      low nibble
        (bits 7:4)       (bits 3:0)
```

A tensor of 4096 FP4 codes becomes 2048 bytes — the 2× storage saving you'd expect from
halving the bit width.

## Where it sits in the quantization pipeline

```
x (FP32)
  → scale by tensor_scale          (FP32 scalar, 1 value)
  → split into blocks of 16
  → scale each block by block_scale (FP8 E4M3, 1 value per block)
  → quantize each element           (FP4 E2M1, 4 bits per value)
  → pack_nibbles                    (2 FP4 codes → 1 byte)
```

`pack_nibbles` is the last step: it converts the logical representation (one `uint8`
entry per FP4 code) into the packed byte stream you'd hand to a kernel or serialize to
disk.

## The caveat in the code

The docstring notes:

> "This packing convention is a host-side convenience, not a claim about NVIDIA library
> internal storage layout."

NVIDIA's cuBLAS/cuDNN kernels have their own internal layout for NVFP4 data. The
low-nibble-first convention here is a reasonable default for inspection and testing — if
you were feeding this into an actual CUDA kernel you'd need to match whatever convention
that kernel expects.

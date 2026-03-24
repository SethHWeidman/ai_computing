# Why `pack_nibbles` exists

## On the term "nibble"

Yes — standard CS terminology. A *nibble* (sometimes spelled *nybble*) is a 4-bit unit of
data, i.e. half a byte. The term has been in wide use since the 1970s, appears in
processor and networking documentation, and is used by NVIDIA in their cuBLAS 12.9
release notes when describing the FP4 packing layout.

## The problem

NVFP4 represents each value as a 4-bit code. But memory is byte-addressed — there is no
way to write a single nibble to RAM. Without packing, you'd have to store each 4-bit code
in its own byte, wasting the upper 4 bits and defeating the point of going to FP4.

## What it does

`pack_nibbles` stuffs two FP4 codes into each byte:

```
byte = (code[1] << 4) | code[0]
        ^^^^^^^^^^^      ^^^^^^^
        high nibble      low nibble
        (bits 7:4)       (bits 3:0)
```

A tensor that was 4096 FP4 codes becomes 2048 bytes — exactly the 2× storage saving you'd
expect from halving the bit width.

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
tensor entry per FP4 code) into the packed byte stream that you'd actually hand to a
kernel or serialize to disk.

## The caveat in the code

The docstring notes:

> "This packing convention is a host-side convenience, not a claim about NVIDIA library
> internal storage layout."

NVIDIA's cuBLAS/cuDNN kernels have their own internal layout for NVFP4 data. The
low-nibble-first convention here is just a reasonable default for inspection and testing
— if you were feeding this into an actual CUDA kernel you'd need to match whatever
convention that kernel expects.

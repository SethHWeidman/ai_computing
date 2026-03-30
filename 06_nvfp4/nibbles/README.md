# nibbles

A *nibble* (sometimes *nybble*) is a 4-bit unit of data — half a byte. NVFP4 stores each
value as a 4-bit code, but memory is byte-addressed, so two codes must be packed into
each byte:

```
byte = (code[1] << 4) | code[0]
        ^^^^^^^^^^^      ^^^^^^^
        high nibble      low nibble
        (bits 7:4)       (bits 3:0)
```

A tensor of 4096 FP4 codes becomes 2048 bytes — the 2× saving you'd expect from halving
the bit width. `pack_nibbles` is the last step of the quantization pipeline, converting
the logical representation into the packed byte stream.

One caveat: this low-nibble-first convention is a host-side choice, not a claim about
NVIDIA's internal kernel layout. Feeding packed data into a real CUDA kernel would
require matching whatever convention that kernel expects.

## How to run

```bash
python nibbles.py
```

Walks through three cases: even-length input, odd-length input (zero-padded), and a
round-trip encode/decode check.

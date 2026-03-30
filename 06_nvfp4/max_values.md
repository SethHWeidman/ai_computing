# Why FP4_MAX = 6 and E4M3_MAX = 448

These are the largest finite values the respective bit layouts can encode. For background
on the ExMy notation, bias, and how these formats compare to FP32/BF16/FP16, see
[number_formats.md](number_formats.md).

---

## FP4 E2M1 → max = 6

Format: `[sign: 1 bit] [exp: 2 bits] [mant: 1 bit]`, exponent bias = 1.

Normal value formula: `(1 + mant/2) * 2^(exp - 1)`

The largest positive pattern uses all available bits for magnitude:

```
(1 + 1/2) * 2^(3 - 1) = 1.5 * 4 = 6
```

All 8 positive representable values:

```
exp=00 mant=0 → 0.0   (subnormal)
exp=00 mant=1 → 0.5   (subnormal)
exp=01 mant=0 → 1.0
exp=01 mant=1 → 1.5
exp=10 mant=0 → 2.0
exp=10 mant=1 → 3.0
exp=11 mant=0 → 4.0
exp=11 mant=1 → 6.0   ← FP4_MAX
```

With a sign bit, the full range is ±{0, 0.5, 1, 1.5, 2, 3, 4, **6**}.

---

## FP8 E4M3 → max = 448

Format: `[sign: 1 bit] [exp: 4 bits] [mant: 3 bits]`, exponent bias = 7.

Normal value formula: `(1 + mant/8) * 2^(exp - 7)`

The naively largest pattern (`exp=1111, mant=111`) would give:

```
(1 + 7/8) * 2^(15 - 7) = 1.875 * 256 = 480
```

But NVIDIA's E4M3 reserves `exp=1111, mant=111` as NaN. The largest *finite* pattern uses
`mant=110` instead:

```
(1 + 6/8) * 2^(15 - 7) = 1.75 * 256 = 448
```

That single reserved NaN pattern is the only reason E4M3_MAX is 448 and not 480.

---

## Why 6 × 448 appears in the scaling formula

The tensor scale maps the global maximum of `x` to the product of both format maxima:

```
tensor_scale = (FP4_MAX * E4M3_MAX) / amax(x)
             = (6 * 448) / amax(x)
             = 2688 / amax(x)
```

After scaling, the tensor's largest value is 2688. The per-block E4M3 scale absorbs the
448× factor, leaving each element in `[-6, +6]` for FP4 quantization.

# Background: floating-point number formats

A reference for the `ExMy` notation and the choices behind it. Linked from
[max_values.md](max_values.md).

---

## The ExMy notation

Every IEEE-style float has three fields:

```
[sign: 1 bit] [exponent: x bits] [mantissa: y bits]
```

A normal value decodes as:

```
value = (-1)^sign * (1 + mantissa / 2^y) * 2^(stored_exp - bias)
```

The `1 +` is the "implicit leading bit" — it's not stored, just assumed for normal
numbers. Subnormals (stored_exp = 0) drop the `1 +` and use a fixed exponent of `1 -
bias` instead.

---

## Common formats

| Format   | Full name | Sign | Exp (x) | Mant (y) | Bias | Total bits |
|----------|-----------|------|---------|----------|------|------------|
| FP32     | E8M23     | 1    | 8       | 23       | 127  | 32         |
| BF16     | E8M7      | 1    | 8       | 7        | 127  | 16         |
| FP16     | E5M10     | 1    | 5       | 10       | 15   | 16         |
| FP8 E4M3 | E4M3fn    | 1    | 4       | 3        | 7    | 8          |
| FP4 E2M1 | E2M1      | 1    | 2       | 1        | 1    | 4          |

**BF16 vs FP16:** BF16 keeps FP32's exponent range (same 8 bits, same bias) but cuts the
mantissa from 23 to 7 bits. FP16 uses a narrower exponent (5 bits) but keeps more
mantissa precision (10 bits). BF16 is preferred for training because gradients need
dynamic range more than they need decimal precision.

---

## Why bias = 2^(k-1) - 1?

The exponent field is stored as an **unsigned** integer. Without a bias you could only
represent values ≥ 1 — no fractions. The bias is subtracted at decode time to produce a
signed effective exponent:

```
effective_exp = stored_exp - bias
```

With `k` exponent bits (stored values 0 to 2^k - 1), subtracting `2^(k-1) - 1` gives:

```
effective range: -(2^(k-1) - 1)  to  +2^(k-1)
```

For FP32 (k=8, bias=127): **-126 to +127** — nearly equal numbers of negative and
positive exponents, so roughly equal dynamic range below 1.0 (fractions) and above 1.0
(large numbers). Any other bias would skew that split.

---

## Sources

- [IEEE 754 — Wikipedia](https://en.wikipedia.org/wiki/IEEE_754): covers FP32, FP16, and
  the general structure of IEEE floating-point formats
- [Minifloat — Wikipedia](https://en.wikipedia.org/wiki/Minifloat): covers FP8 and FP4
  variants, the exponent bias formula, and ML-specific formats including E4M3 and E2M1

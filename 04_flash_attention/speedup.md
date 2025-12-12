# On the "cute flash attention" speedup numbers

The upstream `flash-attn-101` benchmark prints `speedup = naive_latency / custom_latency * 100`.
That `* 100` turns a reasonable ~217× ratio (11.7 ms → 0.054 ms) into `21722.9%`, which looks wildly
implausible. Two other details make the headline even noisier:

- Only one timed repeat is used in `profile.cu`, so CUDA events are measuring a single kernel launch
  on warm data. This favors the fast path and increases run-to-run variance.
- The baseline is already a GPU kernel ("naive attention"), not Python. It is intentionally
  unoptimized, so a fused tensor-core implementation can easily be two orders of magnitude faster.
- When the binary is run multiple times in one process, later implementations may benefit from being
  JIT/warm-cache hot, further inflating the apparent speedup of the last (CuTe) implementation.

The updated `compare.py` no longer parses the submodule's CLI output. It links directly against the
`flash-attn-101` library, warms up each kernel once (Python baseline plus all CUDA paths), and then
measures fresh timings in-process using CUDA events. Speedups are reported as `×` factors against
the Python baseline, which keeps the numbers interpretable and avoids the percent inflation.
Increase `--repeats` if you want steadier timings on your GPU.

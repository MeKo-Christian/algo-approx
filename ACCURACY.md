# Accuracy Report

This document summarizes the measured accuracy of the public API.

## Method

Metrics are produced by the helper in `internal/reference/accuracy.go`:

- **Reference** functions use Go's `math` package, with two exceptions.
  `LogCosh`'s reference is `|x| - ln2 + log1p(exp(-2|x|))`, because
  `math.Log(math.Cosh(x))` overflows to `+Inf` above `|x| ~ 710` and would be
  wrong exactly where the approximation matters. `Log`'s reference on subnormal
  inputs is built from `math.Frexp` (see the note at the end).
- **Approx** functions use the public API (`Fast*Prec`).
- **Metrics**:
  - `MaxAbsError`: $\max |approx-ref|$
  - `MaxRelError`: $\max \frac{|approx-ref|}{|ref|}$ (falls back to abs error when `ref == 0`)
  - `RMSError`: root mean square error over samples

Read the two error columns together. `MaxRelError` is the meaningful one for
`exp`, whose outputs span many decades; `MaxAbsError` is the meaningful one for
`log`, `tanh` and `logcosh`, whose outputs do not. Quoting the wrong one
produces misleading numbers: an absolute error read off the top of a
many-decade sweep says nothing about the digits that are actually correct.

### Sample sets

- **log**: 20001 log-spaced samples in $[10^{-12}, 10^{6}]$
- **exp**: 20001 linear samples in $[-10, 10]$
- **tanh**: 20001 linear samples in $[-20, 20]$
- **logcosh**: 20001 linear samples in $[-12, 12]$ (the consumer's domain)

## Results (2026-08-02)

- OS/Arch: Linux/amd64
- CPU: 12th Gen Intel(R) Core(TM) i7-1255U
- Go: 1.26.1

| Function       | Precision | MaxRelError | MaxAbsError |  RMSError |
| -------------- | --------: | ----------: | ----------: | --------: |
| `FastLog` (ln) |      Fast |   1.283e+00 |   1.789e-03 | 5.459e-04 |
| `FastLog` (ln) |  Balanced |   8.841e-03 |   1.242e-05 | 2.901e-06 |
| `FastLog` (ln) |      High |   7.544e-05 |   1.068e-07 | 2.099e-08 |
| `FastExp`      |      Fast |   7.943e-04 |   9.150e+00 | 5.847e-01 |
| `FastExp`      |  Balanced |   3.241e-06 |   3.723e-02 | 1.890e-03 |
| `FastExp`      |      High |   7.026e-09 |   8.048e-05 | 3.492e-06 |
| `FastTanh`     |      Fast |   2.007e-05 |   1.404e-05 | 5.589e-07 |
| `FastTanh`     |  Balanced |   2.515e-09 |   1.759e-09 | 5.644e-11 |
| `FastTanh`     |      High |   3.362e-12 |   2.351e-12 | 6.819e-14 |
| `FastLogCosh`  |      Fast |   9.445e-05 |   1.731e-05 | 1.277e-06 |
| `FastLogCosh`  |  Balanced |   3.138e-09 |   1.054e-09 | 4.196e-11 |
| `FastLogCosh`  |      High |   6.553e-10 |   1.201e-10 | 4.192e-12 |

`FastLog`'s `MaxRelError` looks alarming because the sample set straddles
`x = 1`, where `ln(x)` passes through zero and any nonzero absolute error is an
unbounded relative error. The absolute column is the one to read.

`FastLogCosh` at `PrecisionHigh` is bounded by its small-argument branch, not
by its exponential branch: the `log(cosh)` series in $z = x^2$ only gains a
factor ~7 per term at the branch point $z = 0.39$, so chasing it further costs
more than it is worth.

## float32 (2026-08-02)

The `*32` entry points are measured separately, in **ulps of float32** rather
than in absolute error. An absolute gate quietly changes meaning as the output
magnitude moves, and half an ulp is the floor no float32 kernel can go below,
so an ulp count is the only figure that says whether the remaining error is the
kernel's or the format's.

The reference is the float64 result **rounded to float32** — `float32(math.Exp(float64(x)))`,
not `math.Exp` itself. Comparing against the float64 value would fold float32's
own ~6e-8 representation gap into every sample and report the format's error
instead of the approximation's.

Sample sets mirror the float64 ones at 4001 points, with one exception noted
below.

| Function        |  Fast | Balanced | High |
| --------------- | ----: | -------: | ---: |
| `FastExp32`     |  9414 |       38 |    1 |
| `FastLog32`     | 14876 |      103 |    1 |
| `FastTanh32`    |   197 |        1 |    0 |
| `FastLogCosh32` |  1162 |        0 |    0 |

`FastTanh32` and `FastLogCosh32` are thin shims over float64 kernels that are
already accurate to 2.5e-9 and 3.1e-9, so at `PrecisionBalanced` the float32
result is the correctly rounded one — `FastLogCosh32` is bit-identical to the
rounded reference across the whole $[-12, 12]$ consumer domain. `FastExp32`'s
38 ulp is not a float32 artefact either: the balanced exp polynomial is good to
3.2e-6 relative in float64 too, which is ~27 float32 ulp.

This also settles a question the float64 table invites: the existing
`MaxAbsError <= 1e-7` gate on `tanh` is **not** too tight for float32. `tanh`
outputs live in $[-1, 1]$, where float32 spacing just below 1.0 is 5.96e-8, so
1e-7 is about 1.7 ulp — and the measured float32 `MaxAbsError` is exactly one
spacing, 5.96e-08.

### The batch float32 kernels are more accurate than the scalar ones

`FastExpBatch32` and `FastTanhLogCoshBatch32` are public behaviour, so their
numbers belong here rather than in an internal package comment. Measured against
float64 references rounded once to float32, over the **whole representable
domain** of each function — not a sweep, every one of the 2³² float32 bit
patterns:

| Batch function                        |                max ulp |
| ------------------------------------- | ---------------------: |
| `FastExpBatch32`                      |                      1 |
| `FastTanhLogCoshBatch32`, `tanh`      |                      1 |
| `FastTanhLogCoshBatch32`, `log(cosh)` | 2 (4 just above 0.625) |

The 4 ulp is confined to the seam just above the branch point at
$|x| = 0.625$; see `logCoshLarge32` in `internal/simd`.

**The batch and scalar float32 paths are not bit-identical, and which one is
more accurate depends on the function.** The scalar `*32` entry points are thin
shims over the float64 kernels: they widen, evaluate in float64, and round once.
The batch kernels are float32-native minimax polynomials with FMA-contracted
evaluation. Those are different trade-offs, and they do not all fall the same
way:

| Function          | scalar `*32`, Balanced | batch |            better |
| ----------------- | ---------------------: | ----: | ----------------: |
| `exp`             |                 38 ulp | 1 ulp | **batch**, by ~28x |
| `tanh`            |                  1 ulp | 1 ulp |            neither |
| `log(cosh)`       |                  0 ulp | 2 ulp (4 at the seam) | **scalar** |

So **do not reach for the batch path for accuracy alone.** For `exp` it is a
large win, and the reason is the basis rather than the width: the balanced
`expPoly` is a degree-5 Taylor whose truncation is $|r|^6/720 = 2.4$e-6 at
$|r| \le \ln 2/2$, i.e. ~20 float32 ulp before any rounding at all, so going
minimax at equal degree buys ~3 ulp for free. For `log(cosh)` it is a small loss:
the scalar shim inherits a float64 kernel accurate to 3.1e-9 and therefore
returns the correctly rounded float32 result, which a float32-native kernel
cannot beat and does not quite match at the branch seam. If you need
`log(cosh)` to the last ulp and do not need the throughput, `FastLogCosh32` is
the better call.

Read the two columns as indicative rather than as a like-for-like comparison:
the scalar figures come from the 4001-point sweeps described above, while the
batch figures are maxima over **every** float32 bit pattern. A sweep cannot find
a worst case it does not sample, so the scalar column is, if anything,
optimistic.

Code that mixes the two paths should therefore expect a disagreement, and the
derived agreement bound is the **sum** of the two accuracy bounds, not twice
either one: 38 + 1 = 39 ulp for `exp`, 1 + 1 = 2 for `tanh`, 0 + 4 = 4 for
`log(cosh)`. `approx_batch_test.go` pins exactly those bounds, and `exp`
measures 39 — right at the derived ceiling, which is what confirms the
derivation rather than what threatens it.

The AVX2 kernels are separately verified against their pure-Go twins over all
2³² bit patterns: `exp` within 1 ulp (99.95 % bit-identical), `tanh` 2 ulp
(99.98 %), `log(cosh)` 4 ulp (99.99 %). The residual is FMA contraction — the
assembly fuses its multiply-adds and Go's amd64 backend never does — and at the
seam where the worst cases live the assembly is fractionally _closer_ to the
truth than the pure-Go kernel on both outputs. So `FastExpBatch32` gives the
same answer to within 1 ulp whether or not the host has AVX2, and neither
answer is the worse one for having taken the vector path.

### arm64 / NEON

The NEON kernels agree with the pure-Go kernels far more closely than the AVX2
ones do, and the reason is the same mechanism read the other way round: Go's
arm64 backend **does** contract multiply-adds, so both sides evaluate the same
fused expressions instead of one side rounding twice where the other rounds
once.

Measured on an Apple M5:

| kernel                    |                 max drift vs Go |             bit-identical |
| ------------------------- | ------------------------------: | ------------------------: |
| `exp`, all 2³² patterns   |                           1 ulp |     all but 22 of 4.28e9  |
| `tanh`, 3.4M-point sweep  |                           0 ulp |                     100 % |
| `log(cosh)`, same sweep   |                           0 ulp |                     100 % |

The `exp` figure is a genuine full-domain sweep. The 22 disagreements are ties
in the range reduction: the Go kernel rounds `x*log2e` with the
add-a-magic-constant trick, which on arm64 becomes a fused
`fma(x, log2e, magic)` and so rounds the product exactly once, while the kernel
uses `FRINTN` on the already-rounded product. At a tie the two land on opposite
sides and the reduced argument differs by one, which costs the last bit.

The `tanh` / `log(cosh)` figures come from a 3.4M-point sweep that walks the
branch seam at |x| = 0.625 one float32 at a time — the region where every worst
case on amd64 lives — not from the full 2³² domain. The exhaustive test exists
(`TestHypNEONFullDomainDifferential`, opt-in via `ALGO_APPROX_EXHAUSTIVE=1`) but
has not been run to completion on arm64 hardware, so treat the 100 % as "nothing
found where the errors are" rather than as a proof over the whole domain.

Getting that agreement required one deliberate departure from the AVX2 kernel
rather than a faithful transliteration of it; see the note in
`internal/simd/hyperbolic32_arm64.s` about where contraction differs.

`FastLog32`'s sweep drops the band $|\ln x| < 1$. There the output passes
through zero, where any nonzero absolute error is an unbounded ulp count: the
full sweep reads 51138 ulp, all of it at $x \approx 1.0035$, and it measures
the crossing rather than the kernel. That band is gated on absolute error
instead, at a measured 1.242e-05 over $[10^{-1}, 10^{1}]$ — the same
`MaxRelError`-versus-`MaxAbsError` caveat that applies to `FastLog` in float64.

## The `FastTanh` / `FastLogCosh` guarantees

Beyond the error table these carry structural guarantees that hold at every
precision, and are covered by tests rather than by measurement:

- `FastTanh(-x)` is the **bit-for-bit** negation of `FastTanh(x)`.
- `FastTanh(x)` is **exactly** `±1` for `|x| >= 19.0625`, which is where
  `math.Tanh` saturates too.
- `FastLogCosh` is exactly even, and never returns an infinity for a finite
  input — `math.Log(math.Cosh(x))` overflows above `|x| ~ 710`.
- `d/dx FastLogCosh` agrees with `FastTanh` to **1.7e-7** over `|x| < 12`
  (measured by central difference at `h = 1e-3`, which contributes 1.3e-7 of
  that from its own `O(h^2)` truncation). The two share one branch point and
  one `exp(-2|x|)` evaluation precisely so this holds; a consumer that needs
  `tanh = d/dx logCosh` to stay true — a discrete-gradient energy scheme, for
  instance, where it is what makes the scheme passive — can rely on it to the
  approximation's own accuracy.

## Note: `math.Log` is not a usable reference for subnormals on amd64

Go's `math.Log` dispatches to `archLog` on amd64 (`src/math/log_amd64.s`),
which reconstructs the significand by masking the mantissa field and OR-ing in
an exponent of 0.5 — that is, it reproduces the implicit leading 1 that a
subnormal does not have. The result is wrong by up to ~36 nats below
`2.2e-308`:

```
math.Log(math.SmallestNonzeroFloat64) = -709.0895657128241   // wrong
correct                               = -744.4400719213812
```

`internal/approx.Log` had exactly the same bug, and it is now fixed. The
regression tests therefore build their reference from `math.Frexp` rather than
calling `math.Log`.

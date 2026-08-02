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

| Function        | Fast | Balanced | High |
| --------------- | ---: | -------: | ---: |
| `FastExp32`     | 9414 |       38 |    1 |
| `FastLog32`     | 14876 |     103 |    1 |
| `FastTanh32`    |  197 |        1 |    0 |
| `FastLogCosh32` | 1162 |        0 |    0 |

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

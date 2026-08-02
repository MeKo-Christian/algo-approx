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
`sqrt`, `invsqrt`, `exp` and `recip`, whose outputs span many decades;
`MaxAbsError` is the meaningful one for `log`, `tanh` and `logcosh`, whose
outputs do not. Quoting the wrong one produces misleading numbers: `FastSqrt`'s
1.1 absolute error is entirely the top of a $[10^{-12}, 10^{12}]$ sweep, and
`FastRecip`'s $10^{134}$ is the same artefact at 15 correct digits.

### Sample sets

- **sqrt / invsqrt**: 20001 log-spaced samples in $[10^{-12}, 10^{12}]$
- **log**: 20001 log-spaced samples in $[10^{-12}, 10^{6}]$
- **exp**: 20001 linear samples in $[-10, 10]$
- **tanh**: 20001 linear samples in $[-20, 20]$
- **logcosh**: 20001 linear samples in $[-12, 12]$ (the consumer's domain)
- **recip**: 20001 log-spaced samples in $[10^{-150}, 10^{150}]$

## Results (2026-08-02)

- OS/Arch: Linux/amd64
- CPU: 12th Gen Intel(R) Core(TM) i7-1255U
- Go: 1.26.1

| Function       | Precision | MaxRelError | MaxAbsError |  RMSError |
| -------------- | --------: | ----------: | ----------: | --------: |
| `FastSqrt`     |      Fast |   1.734e-03 |   1.277e+03 | 7.913e+01 |
| `FastSqrt`     |  Balanced |   1.501e-06 |   1.096e+00 | 4.984e-02 |
| `FastSqrt`     |      High |   1.127e-12 |   8.112e-07 | 2.686e-08 |
| `FastInvSqrt`  |      Fast |   1.751e-03 |   1.324e+03 | 1.416e+02 |
| `FastInvSqrt`  |  Balanced |   4.597e-06 |   3.179e+00 | 3.060e-01 |
| `FastInvSqrt`  |      High |   3.170e-11 |   2.072e-05 | 1.674e-06 |
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
| `FastRecip`    |      Fast |   1.661e-08 |           — |         — |
| `FastRecip`    |  Balanced |   4.804e-16 |           — |         — |
| `FastRecip`    |      High |   2.220e-16 |           — |         — |

`FastLog`'s `MaxRelError` looks alarming because the sample set straddles
`x = 1`, where `ln(x)` passes through zero and any nonzero absolute error is an
unbounded relative error. The absolute column is the one to read.

`FastRecip`'s absolute error column is omitted for the mirror-image reason:
over a $10^{300}$ dynamic range it measures nothing but the top of the sweep.
`PrecisionHigh` is one ulp.

`FastLogCosh` at `PrecisionHigh` is bounded by its small-argument branch, not
by its exponential branch: the `log(cosh)` series in $z = x^2$ only gains a
factor ~7 per term at the branch point $z = 0.39$, so chasing it further costs
more than it is worth.

## The `FastTanh` / `FastLogCosh` / `FastRecip` guarantees

Beyond the error table these carry structural guarantees that hold at every
precision, and are covered by tests rather than by measurement:

- `FastTanh(-x)` is the **bit-for-bit** negation of `FastTanh(x)`, and so is
  `FastRecip(-x)` of `FastRecip(x)`.
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

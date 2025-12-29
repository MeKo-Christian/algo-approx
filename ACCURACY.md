# Accuracy Report

This document summarizes the current accuracy of the Phase 1 MVP functions.

## Method

Metrics are produced by the helper in `internal/reference/accuracy.go`:

- **Reference** functions use Go's `math` package.
- **Approx** functions use the public API (`Fast*Prec`) with `PrecisionBalanced`.
- **Metrics**:
  - `MaxAbsError`: $\max |approx-ref|$
  - `MaxRelError`: $\max \frac{|approx-ref|}{|ref|}$ (falls back to abs error when `ref == 0`)
  - `MeanAbsError`: mean absolute error over samples
  - `RMSError`: root mean square error over samples
  - `DecimalDigits`: $-\log_{10}(\mathrm{MaxRelError})$

### Sample sets

- **sqrt / invsqrt**: 2001 log-spaced samples in $[10^{-12}, 10^{12}]$
- **log**: 2001 log-spaced samples in $[10^{-12}, 10^{6}]$
- **exp**: 2001 linear samples in $[-10, 10]$

## Results (2025-12-28)

Captured from `go test -run TestAccuracy_Balanced_MinimumDigits -v` on:

- OS/Arch: Linux/amd64
- CPU: 12th Gen Intel(R) Core(TM) i7-1255U

| Function       | Precision | DecimalDigits | MaxRelError | MaxAbsError | MeanAbsError |   RMSError |
| -------------- | --------: | ------------: | ----------: | ----------: | -----------: | ---------: |
| `FastSqrt`     |  Balanced |        5.8239 |  1.4999e-06 |  9.9335e-01 |   6.2998e-03 | 4.9601e-02 |
| `FastInvSqrt`  |  Balanced |        5.3375 |  4.5973e-06 |  3.1770e+00 |   6.3253e-02 | 3.0546e-01 |
| `FastLog` (ln) |  Balanced |        3.1206 |  7.5758e-04 |  1.2392e-05 |   1.2842e-06 | 2.9018e-06 |
| `FastExp`      |  Balanced |        5.4914 |  3.2258e-06 |  3.5970e-02 |   3.2779e-04 | 1.9328e-03 |

Notes:

- `DecimalDigits` is a conservative, worst-case summary (based on `MaxRelError`).
- The `MaxAbsError` for `FastSqrt` is dominated by the large-magnitude end of the test range.

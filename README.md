# algo-approx

Fast, allocation-free mathematical approximations for Go.

## Read this before reaching for a `Fast*` function

> **`FastSqrt` and `FastInvSqrt` are slower than `math.Sqrt` on every
> mainstream target, by a factor of 4–6.** This is not a tuning problem and it
> will not be fixed. Go lowers `math.Sqrt` to a single `SQRTSD` instruction on
> amd64 and `FSQRT` on arm64 — a hardware square root that no Newton iteration
> written in Go can beat. Use `math.Sqrt` and `1/math.Sqrt(x)`. The `Fast*`
> versions exist for targets without a hardware square root and for
> `PrecisionFast`, where a deliberately sloppy answer is acceptable.

> **`FastRecip` is slower than writing `1/x`, in both a dependent chain and an
> independent-throughput loop.** `DIVSD` has long latency but good throughput,
> and a bit-trick seed plus Newton-Raphson beats neither. It is published with
> its measurements so the question does not have to be asked again.

> **`FastTanh` is ~1.1× slower than `math.Tanh`.** It is worth using for its
> guarantees — bit-exact odd symmetry, exact saturation, and the derivative
> identity it holds with `FastLogCosh` — not for speed.

The functions that earn their place on speed are `FastLog` (1.3× faster) and
`FastLogCosh` (1.7× faster). `FastExp` is break-even at `PrecisionBalanced`
and marginal at `PrecisionFast`. See the tables below.

## Status

`sqrt`, `invsqrt`, `log` (ln), `exp`, the trig/arctrig family, `tanh`,
`logcosh` and `recip`, all with `float32`/`float64` generics and a `Precision`
knob. Every function is allocation-free (enforced by `approx_alloc_test.go`).

## Install

```bash
go get github.com/cwbudde/algo-approx
```

## Usage

```go
package main

import (
	"fmt"

	"github.com/cwbudde/algo-approx"
)

func main() {
	fmt.Println(approx.FastLog(10.0))
	fmt.Println(approx.FastExp(2.0))

	// tanh and log(cosh) are a consistent pair: tanh is exactly
	// d/dx logCosh, and the approximations preserve that.
	fmt.Println(approx.FastTanh(1.5), approx.FastLogCosh(1.5))

	// Precision control
	fmt.Println(approx.FastLogPrec(10.0, approx.PrecisionHigh))
}
```

## Benchmarks (2026-08-02)

### Method

Numbers below were measured on the CPU named in the header, with:

```bash
just bench-published   # GOMAXPROCS=1 go test -run=^$ -bench=. -benchtime=400ms -count=4 .
just bench-consumer    # the same, from consumerbench/ — a separate module
```

- **`GOMAXPROCS=1`, `-count >= 4`.** Both are part of the method, not
  decoration. The figures below are the _minimum_ over 24 samples per
  benchmark, taken while the machine was under load from other work;
  contention can only ever add time, so the minimum is the robust estimator
  here. Reproducing on an idle machine should give the same numbers or
  slightly better ones, and the _ratios_ are what the table is for.
- **Harness overhead is measured, not assumed.** Inputs are precomputed into a
  256-entry table and indexed with a mask;
  `BenchmarkHarnessOverhead_Float64` times the empty loop.
  It costs **0.47 ns**, and the per-call figures below have it subtracted.
  The previous harness recomputed its input inside the loop
  (`float64((i%1000)+1) * 1.001`) for about 1.2 ns per iteration, which was
  added to _both_ sides of every comparison and pulled every published ratio
  towards 1.0 — hardest on the cheapest operations. That is why the old table
  claimed `math.Sqrt` cost 1.94 ns when `SQRTSD` is about 0.3 ns amortised.

### Measured from inside the package

| Operation                 | approx ns | math ns | approx vs math |
| ------------------------- | --------: | ------: | -------------: |
| `Sqrt`                    |      7.20 |    0.85 |   8.48× slower |
| `InvSqrt`                 |      7.76 |    1.77 |   4.38× slower |
| `Log` (ln)                |      4.08 |    4.74 |   1.16× faster |
| `Exp`                     |      4.12 |    4.00 |   1.03× slower |
| `Tanh`                    |      6.99 |    6.08 |   1.15× slower |
| `LogCosh`                 |      9.95 |   17.42 |   1.75× faster |
| `Recip` (dependent chain) |     16.71 |    3.24 |   5.15× slower |
| `Recip` (throughput, 4×)  |      7.36 |    0.42 |  17.56× slower |

`math` here means `math.Sqrt`, `1/math.Sqrt(x)`, `math.Log`, `math.Exp`,
`math.Tanh`, `math.Log(math.Cosh(x))` and `1/x` respectively.

For comparison, the figures this table previously published were 11.09 / 12.83
/ 6.398 / 7.269 ns for approx and 1.942 / 5.887 / 10.83 / 13.56 ns for math —
about 1.2 ns of harness overhead on every entry, and a `math` column that does
not reproduce at all. The `InvSqrt`, `Log` and `Exp` ratios it quoted (2.18×
slower, 1.69× faster, 1.87× faster) were all wrong.

### Precision tiers

The `Fast*Prec` variants had no benchmarks at all until now. In-package, net of
harness:

| Call                                | ns/op | vs `math` |
| ----------------------------------- | ----: | --------: |
| `FastLogPrec(x, PrecisionFast)`     |  3.49 |     1.36× |
| `FastLogPrec(x, PrecisionBalanced)` |  3.74 |     1.27× |
| `FastLogPrec(x, PrecisionHigh)`     |  4.52 |     1.05× |
| `math.Log`                          |  4.74 |         — |
| `FastExpPrec(x, PrecisionFast)`     |  3.69 |     1.08× |
| `FastExpPrec(x, PrecisionBalanced)` |  4.19 |     0.96× |
| `FastExpPrec(x, PrecisionHigh)`     |  4.95 |     0.81× |
| `math.Exp`                          |  4.00 |         — |

**Dropping `FastLog` from four series terms to two buys 0.25 ns — about 7 % —
and costs 2.2 decimal digits** (max absolute error 1.24e-5 → 1.79e-3). Unless
the loop is very hot and the accuracy genuinely disposable, it is not worth it;
`PrecisionBalanced` is the right default and the tier exists for callers who
have measured that 7 % mattering.

### Measured from a consumer module — these are the numbers that matter

`consumerbench/` is a separate Go module that imports algo-approx by module
path. It exists because **a library that only wins when benchmarked from inside
itself does not win.** Same method, same CPU:

| Operation                      | approx ns | math ns | approx vs math |
| ------------------------------ | --------: | ------: | -------------: |
| `FastLog` (generic entry)      |      3.74 |    4.89 |   1.31× faster |
| `FastLog64` (concrete entry)   |      3.78 |    4.89 |   1.30× faster |
| `FastLogPrec`, Balanced        |      3.81 |    4.89 |   1.28× faster |
| `FastExp64`                    |      4.51 |    4.53 |     break-even |
| `FastTanh64`                   |      7.49 |    6.73 |   1.11× slower |
| `FastLogCosh64`                |     10.34 |   17.59 |   1.70× faster |
| `FastRecip64` (throughput, 4×) |      8.37 |    0.44 |  18.93× slower |

The generic and the concrete entry points now cost the same, which was not
true before: see the note below.

### Generic entry points used to cost a call frame

A field report had `approx.FastLog` running **2.4× slower** than `math.Log`
from a consumer module while the in-package benchmark showed it 1.3× faster,
with `FastLog64` landing in between at 1.3× slower. This harness did not
reproduce a regression that large — it measured `FastLog` at 5.43 ns against
`math.Log` at 5.72 ns from a consumer module, i.e. the win had merely
evaporated rather than inverted. But the _shape_ of the report reproduced
exactly: the generic entry point cost more than the concrete one, and both cost
more than the same call made from inside the package. That is what a
cross-package generic call looks like, and it was worth fixing at face value.

The cause was that the arithmetic itself lived in a generic function. Go
compiles a generic body once per _gcshape_ and calls it through a runtime
dictionary; the compiler will not inline such a call from another package, so
every call paid a real frame that no caller could remove. The fix is
structural: `internal/approx` now keeps each algorithm in a **non-generic
`float64` kernel**, with the generic function reduced to a shim
(`func Log[T Float](x T, prec Precision) T { return T(log64(float64(x), prec)) }`).
The shim is small enough to inline across packages, so a consumer lands on a
direct call to the kernel. After the change, the generic and concrete entry
points are indistinguishable (3.74 vs 3.78 ns) and both are 1.3× faster than
`math.Log` from outside the module — the same ratio the in-package benchmark
reports. `Log`, `Exp`, `Tanh`, `LogCosh` and `Recip` all use this structure;
`Sqrt` and `InvSqrt` are genuinely generic and were left alone.

For the record, the other suspect was ruled out by measurement, not argument.
The runtime `switch prec` inside each kernel costs nothing:

```
BenchmarkSwitchfulLog64   4.056 ns/op   # log64 with the precision switch
BenchmarkSwitchfreeLog64  4.115 ns/op   # the balanced path, switch removed
```

The branch is perfectly predicted and never on the floating-point critical
path.

### Why `Log` and `Exp` barely win

`math.Log` on amd64 is hand-written assembly (`math.archLog`,
`src/math/log_amd64.s`), and `math.Exp` is a tightly tuned Go routine. A
truncated series in portable Go has very little room against either. `FastExp`
is a real win only at `PrecisionFast`; at `PrecisionBalanced` it is roughly
break-even. If your hot loop needs `exp` and can tolerate 8e-4 relative error,
`FastExpPrec(x, PrecisionFast)` is worth measuring in place. Otherwise use
`math`.

`FastLogCosh` wins because it replaces _two_ chained libm calls, and because
it never forms `cosh` at all.

## Accuracy

See [ACCURACY.md](ACCURACY.md) for measured error metrics on representative
ranges, and for the guarantees `FastTanh`, `FastLogCosh` and `FastRecip` hold
exactly rather than approximately.

Two notes worth reading there:

- `FastTanh` and `FastLogCosh` are designed as a **consistent pair**: `tanh` is
  exactly `d/dx log(cosh(x))`, and the two share one branch point and one
  `exp(-2|x|)` evaluation so the identity survives the approximation. Consumers
  that depend on it — a discrete-gradient energy scheme, where the identity is
  what makes the scheme passive — can rely on it.
- Go's own `math.Log` is **wrong for subnormal inputs on amd64**, by up to ~36
  nats. This library had the same bug and it is now fixed.

## License

MIT

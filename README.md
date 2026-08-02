# algo-approx

Fast, allocation-free mathematical approximations for Go.

## Read this before reaching for a `Fast*` function

**The scalar functions are roughly break-even with `math`. The batch functions
are where this library wins, and they win by about 11×.**

If you have a slice of `float32` to transform, use `FastExpBatch32` or
`FastTanhLogCoshBatch32`. They run a hand-written AVX2+FMA kernel where the CPU
has it, and they are ~11× faster per element than a scalar loop. `FastExpBatch32`
is also far more accurate than `FastExp32` (1 ulp against 38) — but that is an
`exp` result and does not generalise: batch `tanh` ties the scalar path and batch
`log(cosh)` is slightly worse. Choose the batch path for throughput, and check
[ACCURACY.md](ACCURACY.md) before choosing it for accuracy. If you are calling a
`Fast*` function one value at a time, read the rest of this section first — the
case for the scalar API rests on its guarantees, not on its speed.

> **`FastSqrt`, `FastInvSqrt` and `FastRecip` have been removed.** They were
> 4–19× slower than the hardware they replaced: Go lowers `math.Sqrt` to a
> single `SQRTSD` on amd64 and `FSQRT` on arm64 — and intrinsifies it on every
> `GOARCH` it supports — while `1/x` lowers to `DIVSD`. No Newton iteration
> written in Go beats either, on any target. Use `math.Sqrt`, `1/math.Sqrt(x)`
> and `1/x`. See [docs/removed-kernels.md](docs/removed-kernels.md) for the
> measurements and the seed-convergence analysis that came out of the attempt.

> **`FastTanh` is ~1.1× slower than `math.Tanh`.** It is worth using for its
> guarantees — bit-exact odd symmetry, exact saturation, and the derivative
> identity it holds with `FastLogCosh` — not for speed.

The functions that earn their place on speed are `FastLog` (1.3× faster) and
`FastLogCosh` (1.7× faster). `FastExp` is break-even at `PrecisionBalanced`
and marginal at `PrecisionFast`. See the tables below.

## Status

`log` (ln), `exp`, the trig/arctrig family, `tanh` and `logcosh`, all with
`float32`/`float64` generics and a `Precision` knob. Every function is allocation-free (enforced by `approx_alloc_test.go`).

A **batch (slice) API** is public as of this release: `FastExpBatch32/64` and
`FastTanhLogCoshBatch32/64`. The float32 pair dispatches to an AVX2+FMA kernel
where the CPU has both features and to a float32-native pure-Go kernel
otherwise; the float64 pair is a scalar loop, bit-identical to the scalar
entry points, with the fused hyperbolic one sharing its `exp(-2|x|)` between
both outputs. Batch `log`, `sin` and the rest do not exist yet.

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

### Batch (slice) API

```go
src := make([]float32, 4096)
dst := make([]float32, 4096)

// SIMD where the CPU has AVX2+FMA, a float32-native Go kernel otherwise.
approx.FastExpBatch32(dst, src)

// In-place is supported.
approx.FastExpBatch32(src, src)

// tanh and log(cosh) in one fused pass: both outputs share the same
// exp(-2|x|), so the pair stays consistent exactly as the scalar pair does.
tanh := make([]float32, 4096)
logcosh := make([]float32, 4096)
approx.FastTanhLogCoshBatch32(tanh, logcosh, src)

// float64 variants exist too, as scalar loops over the same kernels the
// scalar API uses — bit-identical to FastExp64 / FastTanh64 / FastLogCosh64.
src64 := make([]float64, 4096)
tanh64 := make([]float64, 4096)
logcosh64 := make([]float64, 4096)
approx.FastTanhLogCoshBatch64(tanh64, logcosh64, src64)
```

Rules, identical for all four:

- A destination shorter than `src` **panics**. Exactly `len(src)` elements are
  written; a longer destination keeps its tail.
- Each destination must be either **identical to `src`** (in-place, supported
  and tested) or **non-overlapping**. Partial overlap is undefined — the SIMD
  kernels read a whole eight-element vector before writing any of it.
- The two destinations of a fused call must additionally **not overlap each
  other**, and in particular must not be the same slice. Both can be disjoint
  from `src` and still share storage, so the rule above does not cover it;
  passing one slice for both leaves it holding only `log(cosh)`.
- No `Precision` argument, and no `…BatchPrec` variants. Resolving a precision
  tier per element costs more than the polynomial it selects, so the tier is
  fixed and constant-folded.
- Allocation-free, like everything else public.
- There is no batch `tanh` or `log(cosh)` on its own. If you only need one,
  pass a reusable scratch buffer for the other destination.

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
  towards 1.0 — hardest on the cheapest operations. That is what made the
  hardware-backed `math` entries look several times more expensive than they
  are.

### Batch (slice) API — where the library actually wins

Reported in **ns per element**, so the sizes are comparable to each other and
to the scalar table below. "scalar API" is a plain `for` loop over the
corresponding `Fast*32` entry point; "pure-Go batch" is the float32-native
batch kernel with the assembly compiled out (`-tags purego`), which is the
honest denominator — a float64 kernel wrapped in float32 conversions would
flatter the assembly by ~2× before a single vector instruction ran.

`exp`, on an i7-1255U, `taskset -c 2` (P-core), `GOMAXPROCS=1`, medians of 10
separate runs:

|       N | scalar API | pure-Go batch | `FastExpBatch32` | vs pure-Go batch | vs scalar |
| ------: | ---------: | ------------: | ---------------: | ---------------: | --------: |
|      64 |      24.97 |          6.19 |        **0.536** |            11.5× |     46.6× |
|     256 |      24.62 |          6.56 |        **0.563** |            11.6× |     43.7× |
|    1024 |      25.38 |          6.35 |        **0.520** |            12.2× |     48.8× |
|    4096 |      24.41 |          6.07 |        **0.520** |            11.7× |     46.9× |
|   65536 |      24.40 |          6.00 |        **0.507** |            11.8× |     48.2× |
| 1048576 |      24.61 |          6.69 |        **0.545** |            12.3× |     45.1× |

Fused `tanh` + `log(cosh)`, on an **idle** Xeon Gold 5218 (2.30 GHz, Cascade
Lake), `GOMAXPROCS=1 taskset -c 0`, 10 separate `-count=1` runs through
`benchstat`, CV ≤ 5 %. Both outputs are produced per element, so these figures
cover two functions, not one:

|       N | scalar API | pure-Go batch | `FastTanhLogCoshBatch32` | vs pure-Go batch | vs scalar |
| ------: | ---------: | ------------: | -----------------------: | ---------------: | --------: |
|      64 |      67.80 |         32.16 |                **3.223** |            9.98× |     21.0× |
|     256 |      67.66 |         32.37 |                **2.910** |           11.12× |     23.3× |
|    1024 |      67.14 |         32.09 |                **2.832** |           11.33× |     23.7× |
|    4096 |      67.04 |         32.64 |                **2.827** |           11.55× |     23.7× |
|   65536 |      67.46 |         32.68 |                **2.829** |           11.55× |     23.8× |
| 1048576 |      66.98 |         33.12 |                **2.819** |           11.75× |     23.8× |

Three things are worth reading off these tables rather than skimming past:

- **The ratio does not decay with N.** No bandwidth collapse appears even at
  N = 1 M. At 2.82 ns/element the fused kernel moves ~4.3 GB/s counting all
  three slices, far under DRAM; it is compute-bound over the whole range.
- **11.7× exceeds the eight-lane theoretical ceiling**, so it is not all
  vectorization. The pure-Go baseline emits ~63 instructions/element where the
  assembly needs ~3.1. Against a hypothetically-tight scalar Go baseline the
  figure would be nearer 6×.
- **The fused kernel does not depend on a favourable branch mix.** `tanh` has a
  data-dependent branch at |x| = 0.625 and a vector kernel cannot branch per
  lane, so it evaluates both branches and blends. Benchmarked against an
  adversarial input where every element is past the branch point, N = 4096:
  11.47 µs against 11.58 µs for a ramp over [-10, 10]. Within 1 % either way.

The float64 batch functions are scalar loops and get none of this. `FastExpBatch64`
only amortises the call frame — do not expect a speedup. `FastTanhLogCoshBatch64`
is the exception that is worth using: sharing one `exp(-2|x|)` between the two
outputs measured **~1.7× faster** than two separate scalar loops at N = 4096
(10.7 vs 17.8 ns/element, minimum of 11 pinned runs on the i7-1255U — taken on a
shared machine, so treat it as an order-of-magnitude confirmation of the
argument rather than a published figure).

### Measured from inside the package

| Operation  | approx ns | math ns | approx vs math |
| ---------- | --------: | ------: | -------------: |
| `Log` (ln) |      4.08 |    4.74 |   1.16× faster |
| `Exp`      |      4.12 |    4.00 |   1.03× slower |
| `Tanh`     |      6.99 |    6.08 |   1.15× slower |
| `LogCosh`  |      9.95 |   17.42 |   1.75× faster |

`math` here means `math.Log`, `math.Exp`, `math.Tanh` and
`math.Log(math.Cosh(x))` respectively.

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

| Operation                    | approx ns | math ns | approx vs math |
| ---------------------------- | --------: | ------: | -------------: |
| `FastLog` (generic entry)    |      3.74 |    4.89 |   1.31× faster |
| `FastLog64` (concrete entry) |      3.78 |    4.89 |   1.30× faster |
| `FastLogPrec`, Balanced      |      3.81 |    4.89 |   1.28× faster |
| `FastExp64`                  |      4.51 |    4.53 |     break-even |
| `FastTanh64`                 |      7.49 |    6.73 |   1.11× slower |
| `FastLogCosh64`              |     10.34 |   17.59 |   1.70× faster |

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
reports. `Log`, `Exp`, `Tanh` and `LogCosh` all use this structure.

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
ranges, and for the guarantees `FastTanh` and `FastLogCosh` hold exactly rather
than approximately.

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

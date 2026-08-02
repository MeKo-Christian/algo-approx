package approx

import (
	"math"
	"testing"
)

// Benchmark harness.
//
// The published ratios in README.md are only as honest as the loop they are
// measured in. The previous harness recomputed its input every iteration
// (`float64((i%1000)+1) * 1.001`: an integer division, a conversion and a
// multiply), which cost roughly 1.2 ns and was added to both sides of every
// comparison. That dilutes every ratio towards 1.0, and it dilutes the cheap
// operations most: math.Sqrt is a single SQRTSD at ~0.3 ns amortised, so a
// 1.2 ns overhead made it look several times more expensive than it is.
//
// The inputs are therefore precomputed into a power-of-two table and indexed
// with a mask, which costs a load and an AND. BenchmarkHarnessOverhead_Float64
// measures what is left so the residue can be subtracted when quoting ratios.
const (
	benchTableSize = 256
	benchTableMask = benchTableSize - 1
)

var benchSink64 float64 //nolint:gochecknoglobals

// benchPositive spans (0, ~1000]: the domain shared by sqrt, invsqrt, log and
// recip.
var benchPositive = makeBenchTable(func(t float64) float64 { //nolint:gochecknoglobals
	return 1e-3 + t*1000
})

// benchExpArg spans [-10, 10].
var benchExpArg = makeBenchTable(func(t float64) float64 { //nolint:gochecknoglobals
	return -10 + 20*t
})

// benchTanhArg spans [-20, 20], so it exercises the rational core, the
// exponential branch and the saturated tail in proportion.
var benchTanhArg = makeBenchTable(func(t float64) float64 { //nolint:gochecknoglobals
	return -20 + 40*t
})

// benchLogCoshArg spans [-12, 12], the consumer's actual domain.
var benchLogCoshArg = makeBenchTable(func(t float64) float64 { //nolint:gochecknoglobals
	return -12 + 24*t
})

func makeBenchTable(gen func(t float64) float64) [benchTableSize]float64 {
	var table [benchTableSize]float64
	for i := range table {
		table[i] = gen(float64(i) / float64(benchTableSize-1))
	}

	return table
}

// BenchmarkHarnessOverhead_Float64 is the empty loop: table load, mask, add.
// Subtract it from every other result to recover the per-call cost.
func BenchmarkHarnessOverhead_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += benchPositive[i&benchTableMask]
	}

	benchSink64 = acc
}

func BenchmarkFastSqrt_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += FastSqrt(benchPositive[i&benchTableMask])
	}

	benchSink64 = acc
}

func BenchmarkMathSqrt_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += math.Sqrt(benchPositive[i&benchTableMask])
	}

	benchSink64 = acc
}

func BenchmarkFastInvSqrt_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += FastInvSqrt(benchPositive[i&benchTableMask])
	}

	benchSink64 = acc
}

func BenchmarkMathInvSqrt_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += 1.0 / math.Sqrt(benchPositive[i&benchTableMask])
	}

	benchSink64 = acc
}

func BenchmarkFastLog_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += FastLog(benchPositive[i&benchTableMask])
	}

	benchSink64 = acc
}

func BenchmarkFastLogPrec_Fast_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += FastLogPrec(benchPositive[i&benchTableMask], PrecisionFast)
	}

	benchSink64 = acc
}

func BenchmarkFastLogPrec_Balanced_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += FastLogPrec(benchPositive[i&benchTableMask], PrecisionBalanced)
	}

	benchSink64 = acc
}

func BenchmarkFastLogPrec_High_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += FastLogPrec(benchPositive[i&benchTableMask], PrecisionHigh)
	}

	benchSink64 = acc
}

func BenchmarkMathLog_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += math.Log(benchPositive[i&benchTableMask])
	}

	benchSink64 = acc
}

func BenchmarkFastExp_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += FastExp(benchExpArg[i&benchTableMask])
	}

	benchSink64 = acc
}

func BenchmarkFastExpPrec_Fast_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += FastExpPrec(benchExpArg[i&benchTableMask], PrecisionFast)
	}

	benchSink64 = acc
}

func BenchmarkFastExpPrec_Balanced_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += FastExpPrec(benchExpArg[i&benchTableMask], PrecisionBalanced)
	}

	benchSink64 = acc
}

func BenchmarkFastExpPrec_High_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += FastExpPrec(benchExpArg[i&benchTableMask], PrecisionHigh)
	}

	benchSink64 = acc
}

func BenchmarkMathExp_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += math.Exp(benchExpArg[i&benchTableMask])
	}

	benchSink64 = acc
}

func BenchmarkFastTanh_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += FastTanh(benchTanhArg[i&benchTableMask])
	}

	benchSink64 = acc
}

func BenchmarkMathTanh_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += math.Tanh(benchTanhArg[i&benchTableMask])
	}

	benchSink64 = acc
}

func BenchmarkFastLogCosh_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += FastLogCosh(benchLogCoshArg[i&benchTableMask])
	}

	benchSink64 = acc
}

// BenchmarkNaiveLogCosh_Float64 is the expression FastLogCosh has to beat to
// justify its existence. It is also the one that overflows above |x| ~ 710.
func BenchmarkNaiveLogCosh_Float64(b *testing.B) {
	b.ReportAllocs()

	var acc float64
	for i := range b.N {
		acc += math.Log(math.Cosh(benchLogCoshArg[i&benchTableMask]))
	}

	benchSink64 = acc
}

// Reciprocal: the dependent-chain and the independent-throughput case give
// different answers, because DIVSD has long latency but decent throughput.
// Both are published; a caller has to know which shape their loop has.

func BenchmarkFastRecip_Chain_Float64(b *testing.B) {
	b.ReportAllocs()

	acc := 1.0
	for i := range b.N {
		acc = FastRecip(acc + benchPositive[i&benchTableMask])
	}

	benchSink64 = acc
}

func BenchmarkDivRecip_Chain_Float64(b *testing.B) {
	b.ReportAllocs()

	acc := 1.0
	for i := range b.N {
		acc = 1.0 / (acc + benchPositive[i&benchTableMask])
	}

	benchSink64 = acc
}

func BenchmarkFastRecip_Throughput_Float64(b *testing.B) {
	b.ReportAllocs()

	var a0, a1, a2, a3 float64

	for i := 0; i < b.N; i += 4 {
		a0 += FastRecip(benchPositive[i&benchTableMask])
		a1 += FastRecip(benchPositive[(i+1)&benchTableMask])
		a2 += FastRecip(benchPositive[(i+2)&benchTableMask])
		a3 += FastRecip(benchPositive[(i+3)&benchTableMask])
	}

	benchSink64 = a0 + a1 + a2 + a3
}

func BenchmarkDivRecip_Throughput_Float64(b *testing.B) {
	b.ReportAllocs()

	var a0, a1, a2, a3 float64

	for i := 0; i < b.N; i += 4 {
		a0 += 1.0 / benchPositive[i&benchTableMask]
		a1 += 1.0 / benchPositive[(i+1)&benchTableMask]
		a2 += 1.0 / benchPositive[(i+2)&benchTableMask]
		a3 += 1.0 / benchPositive[(i+3)&benchTableMask]
	}

	benchSink64 = a0 + a1 + a2 + a3
}

func BenchmarkFastRecipPrec_Fast_Throughput_Float64(b *testing.B) {
	b.ReportAllocs()

	var a0, a1, a2, a3 float64

	for i := 0; i < b.N; i += 4 {
		a0 += FastRecipPrec(benchPositive[i&benchTableMask], PrecisionFast)
		a1 += FastRecipPrec(benchPositive[(i+1)&benchTableMask], PrecisionFast)
		a2 += FastRecipPrec(benchPositive[(i+2)&benchTableMask], PrecisionFast)
		a3 += FastRecipPrec(benchPositive[(i+3)&benchTableMask], PrecisionFast)
	}

	benchSink64 = a0 + a1 + a2 + a3
}

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
// operations most: a single hardware instruction costs a fraction of a
// nanosecond, so a 1.2 ns overhead made it look several times more expensive
// than it is.
//
// The inputs are therefore precomputed into a power-of-two table and indexed
// with a mask, which costs a load and an AND. BenchmarkHarnessOverhead_Float64
// measures what is left so the residue can be subtracted when quoting ratios.
const (
	benchTableSize = 256
	benchTableMask = benchTableSize - 1
)

var benchSink64 float64 //nolint:gochecknoglobals

// benchPositive spans (0, ~1000]: the domain of log.
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

package consumerbench

import (
	"math"
	"testing"

	approx "github.com/cwbudde/algo-approx"
)

const (
	tableSize = 256
	tableMask = tableSize - 1
)

var sink float64 //nolint:gochecknoglobals

var positive = table(func(t float64) float64 { return 1e-3 + t*1000 }) //nolint:gochecknoglobals

func table(gen func(float64) float64) [tableSize]float64 {
	var out [tableSize]float64
	for i := range out {
		out[i] = gen(float64(i) / float64(tableSize-1))
	}

	return out
}

func BenchmarkOverhead(b *testing.B) {
	var acc float64

	for i := range b.N {
		acc += positive[i&tableMask]
	}

	sink = acc
}

func BenchmarkFastLog_Generic(b *testing.B) {
	var acc float64

	for i := range b.N {
		acc += approx.FastLog(positive[i&tableMask])
	}

	sink = acc
}

func BenchmarkFastLog64_Concrete(b *testing.B) {
	var acc float64

	for i := range b.N {
		acc += approx.FastLog64(positive[i&tableMask])
	}

	sink = acc
}

func BenchmarkFastLogPrec_Balanced(b *testing.B) {
	var acc float64

	for i := range b.N {
		acc += approx.FastLogPrec(positive[i&tableMask], approx.PrecisionBalanced)
	}

	sink = acc
}

func BenchmarkMathLog(b *testing.B) {
	var acc float64

	for i := range b.N {
		acc += math.Log(positive[i&tableMask])
	}

	sink = acc
}

func BenchmarkFastExp64(b *testing.B) {
	var acc float64

	for i := range b.N {
		acc += approx.FastExp64(positive[i&tableMask]*0.01 - 5)
	}

	sink = acc
}

func BenchmarkMathExp(b *testing.B) {
	var acc float64

	for i := range b.N {
		acc += math.Exp(positive[i&tableMask]*0.01 - 5)
	}

	sink = acc
}

func BenchmarkFastTanh64(b *testing.B) {
	var acc float64

	for i := range b.N {
		acc += approx.FastTanh64(positive[i&tableMask]*0.04 - 20)
	}

	sink = acc
}

func BenchmarkMathTanh(b *testing.B) {
	var acc float64

	for i := range b.N {
		acc += math.Tanh(positive[i&tableMask]*0.04 - 20)
	}

	sink = acc
}

func BenchmarkFastLogCosh64(b *testing.B) {
	var acc float64

	for i := range b.N {
		acc += approx.FastLogCosh64(positive[i&tableMask]*0.024 - 12)
	}

	sink = acc
}

func BenchmarkNaiveLogCosh(b *testing.B) {
	var acc float64

	for i := range b.N {
		x := positive[i&tableMask]*0.024 - 12

		acc += math.Log(math.Cosh(x))
	}

	sink = acc
}

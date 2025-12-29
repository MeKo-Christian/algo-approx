package approx

import (
	"math"
	"testing"
)

var benchSink64 float64

func BenchmarkFastSqrt_Float64(b *testing.B) {
	b.ReportAllocs()
	var acc float64
	for i := 0; i < b.N; i++ {
		x := float64((i%1000)+1) * 1.001
		acc += float64(FastSqrt(x))
	}
	benchSink64 = acc
}

func BenchmarkMathSqrt_Float64(b *testing.B) {
	b.ReportAllocs()
	var acc float64
	for i := 0; i < b.N; i++ {
		x := float64((i%1000)+1) * 1.001
		acc += math.Sqrt(x)
	}
	benchSink64 = acc
}

func BenchmarkFastInvSqrt_Float64(b *testing.B) {
	b.ReportAllocs()
	var acc float64
	for i := 0; i < b.N; i++ {
		x := float64((i%1000)+1) * 1.001
		acc += float64(FastInvSqrt(x))
	}
	benchSink64 = acc
}

func BenchmarkMathInvSqrt_Float64(b *testing.B) {
	b.ReportAllocs()
	var acc float64
	for i := 0; i < b.N; i++ {
		x := float64((i%1000)+1) * 1.001
		acc += 1.0 / math.Sqrt(x)
	}
	benchSink64 = acc
}

func BenchmarkFastLog_Float64(b *testing.B) {
	b.ReportAllocs()
	var acc float64
	for i := 0; i < b.N; i++ {
		x := float64((i%1000)+1) * 1.001
		acc += float64(FastLog(x))
	}
	benchSink64 = acc
}

func BenchmarkMathLog_Float64(b *testing.B) {
	b.ReportAllocs()
	var acc float64
	for i := 0; i < b.N; i++ {
		x := float64((i%1000)+1) * 1.001
		acc += math.Log(x)
	}
	benchSink64 = acc
}

func BenchmarkFastExp_Float64(b *testing.B) {
	b.ReportAllocs()
	var acc float64
	for i := 0; i < b.N; i++ {
		x := -10.0 + 20.0*float64(i%1000)/999.0
		acc += float64(FastExp(x))
	}
	benchSink64 = acc
}

func BenchmarkMathExp_Float64(b *testing.B) {
	b.ReportAllocs()
	var acc float64
	for i := 0; i < b.N; i++ {
		x := -10.0 + 20.0*float64(i%1000)/999.0
		acc += math.Exp(x)
	}
	benchSink64 = acc
}

package simd

import (
	"fmt"
	"math"
	"testing"

	iapprox "github.com/cwbudde/algo-approx/internal/approx"
)

// Arms are benchmarked in one process over identical inputs so their ratios stay
// meaningful even when the machine's absolute timings drift.
//
// Everything reports via b.SetBytes(n*4) so throughput is comparable ACROSS
// sizes; raw ns/op is per batch call and is not. Read MB/s, or divide ns/op by n.
//
// Size sweep rationale: float32 in+out is 8 B/elem, so a fast kernel needs tens
// of GB/s and becomes DRAM-bound well before the top of this range. The decision
// gate is defined at n <= 4096 (16 KB in+out, inside this CPU's 48 KB L1d); the
// larger sizes are reported to show where the ratio decays toward 1 by
// construction, not as evidence that SIMD failed.
//
//nolint:gochecknoglobals // benchmark size sweep
var benchSizes = []int{64, 256, 1024, 4096, 65536, 1 << 20}

// benchInput fills a ramp over [-10, 10]: the realistic-distribution case. The
// adversarial all-exp-branch distribution is benchmarked separately below.
func benchInput(n int) []float32 {
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i)*(20.0/float32(n)) - 10
	}

	return src
}

// touch writes every page before the timer starts, so the first iterations do
// not measure page faults and first-touch zeroing (milliseconds at n = 1M).
func touch(s []float32) []float32 {
	for i := range s {
		s[i] = 0
	}

	return s
}

func benchEachSize(b *testing.B, run func(b *testing.B, n int)) {
	b.Helper()

	for _, n := range benchSizes {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 4)
			run(b, n)
		})
	}
}

// --- exp ------------------------------------------------------------------

func BenchmarkExp_ScalarAPI(b *testing.B) {
	benchEachSize(b, func(b *testing.B, n int) {
		b.Helper()

		src, dst := benchInput(n), touch(make([]float32, n))

		for b.Loop() {
			for i, v := range src {
				dst[i] = iapprox.Exp(v, iapprox.PrecisionBalanced)
			}
		}
	})
}

func BenchmarkExp_MathExp(b *testing.B) {
	benchEachSize(b, func(b *testing.B, n int) {
		b.Helper()

		src, dst := benchInput(n), touch(make([]float32, n))

		for b.Loop() {
			for i, v := range src {
				dst[i] = float32(math.Exp(float64(v)))
			}
		}
	})
}

// BenchmarkExp_BatchGo is the gate baseline: float32-native, branchless, unrolled.
func BenchmarkExp_BatchGo(b *testing.B) {
	benchEachSize(b, func(b *testing.B, n int) {
		b.Helper()

		src, dst := benchInput(n), touch(make([]float32, n))

		for b.Loop() {
			expBatch32Go(dst, src)
		}
	})
}

// BenchmarkExp_BatchDispatch goes through the public entry point, i.e. AVX2
// where available. Compare against BenchmarkExp_BatchGo for the gate.
func BenchmarkExp_BatchDispatch(b *testing.B) {
	benchEachSize(b, func(b *testing.B, n int) {
		b.Helper()

		src, dst := benchInput(n), touch(make([]float32, n))

		for b.Loop() {
			ExpFloat32(dst, src)
		}
	})
}

// --- fused tanh/logCosh (pure Go only; no asm kernel yet) -----------------

func BenchmarkTanhLogCosh_ScalarAPI(b *testing.B) {
	benchEachSize(b, func(b *testing.B, n int) {
		b.Helper()

		src := benchInput(n)
		dstTanh, dstLogCosh := touch(make([]float32, n)), touch(make([]float32, n))

		for b.Loop() {
			for i, v := range src {
				dstTanh[i] = iapprox.Tanh(v, iapprox.PrecisionBalanced)
				dstLogCosh[i] = iapprox.LogCosh(v, iapprox.PrecisionBalanced)
			}
		}
	})
}

func BenchmarkTanhLogCosh_BatchGo(b *testing.B) {
	benchEachSize(b, func(b *testing.B, n int) {
		b.Helper()

		src := benchInput(n)
		dstTanh, dstLogCosh := touch(make([]float32, n)), touch(make([]float32, n))

		for b.Loop() {
			TanhLogCoshFloat32(dstTanh, dstLogCosh, src)
		}
	})
}

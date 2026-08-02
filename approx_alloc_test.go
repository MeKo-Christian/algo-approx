package approx

import "testing"

// Not t.Parallel(): testing.AllocsPerRun panics when called from a parallel
// test, because a concurrently running test would pollute the measurement.
//
//nolint:paralleltest
func TestNoAllocs_PublicAPI_Float64(t *testing.T) {
	cases := []struct {
		name string
		run  func()
	}{
		{"FastLog", func() { _ = FastLog(2.0) }},
		{"FastExp", func() { _ = FastExp(2.0) }},
		{"FastLogPrec", func() { _ = FastLogPrec(2.0, PrecisionHigh) }},
		{"FastExpPrec", func() { _ = FastExpPrec(2.0, PrecisionHigh) }},
		{"FastTanh", func() { _ = FastTanh(2.0) }},
		{"FastTanhPrec", func() { _ = FastTanhPrec(2.0, PrecisionHigh) }},
		{"FastLogCosh", func() { _ = FastLogCosh(2.0) }},
		{"FastLogCoshPrec", func() { _ = FastLogCoshPrec(2.0, PrecisionHigh) }},
	}

	for _, tc := range cases {
		allocs := testing.AllocsPerRun(1000, tc.run)
		if allocs != 0 {
			t.Fatalf("%s allocated: %v", tc.name, allocs)
		}
	}
}

// Not t.Parallel(): see TestNoAllocs_PublicAPI_Float64.
//
// The buffers are deliberately allocated outside the measured closure. Making
// them inside it would measure make() — three guaranteed allocations — and say
// nothing about the batch call, which is the thing under test.
//
//nolint:paralleltest
func TestNoAllocs_PublicAPI_Batch(t *testing.T) {
	const n = 512

	src32 := make([]float32, n)
	dst32 := make([]float32, n)
	tanh32 := make([]float32, n)
	src64 := make([]float64, n)
	dst64 := make([]float64, n)
	tanh64 := make([]float64, n)

	for i := range n {
		x := float64(i)/float64(n)*8 - 4
		src32[i] = float32(x)
		src64[i] = x
	}

	cases := []struct {
		name string
		run  func()
	}{
		{"FastExpBatch32", func() { FastExpBatch32(dst32, src32) }},
		{"FastExpBatch64", func() { FastExpBatch64(dst64, src64) }},
		{"FastTanhLogCoshBatch32", func() { FastTanhLogCoshBatch32(tanh32, dst32, src32) }},
		{"FastTanhLogCoshBatch64", func() { FastTanhLogCoshBatch64(tanh64, dst64, src64) }},
	}

	for _, tc := range cases {
		allocs := testing.AllocsPerRun(100, tc.run)
		if allocs != 0 {
			t.Fatalf("%s allocated: %v", tc.name, allocs)
		}
	}
}

// Not t.Parallel(): see TestNoAllocs_PublicAPI_Float64.
//
//nolint:paralleltest
func TestNoAllocs_PublicAPI_Float32(t *testing.T) {
	cases := []struct {
		name string
		run  func()
	}{
		{"FastLog32", func() { _ = FastLog32(2) }},
		{"FastExp32", func() { _ = FastExp32(2) }},
		{"FastLogPrec32", func() { _ = FastLogPrec(float32(2), PrecisionHigh) }},
		{"FastExpPrec32", func() { _ = FastExpPrec(float32(2), PrecisionHigh) }},
		{"FastTanh32", func() { _ = FastTanh32(2) }},
		{"FastLogCosh32", func() { _ = FastLogCosh32(2) }},
		{"FastTanhPrec32", func() { _ = FastTanhPrec(float32(2), PrecisionHigh) }},
		{"FastLogCoshPrec32", func() { _ = FastLogCoshPrec(float32(2), PrecisionHigh) }},
	}

	for _, tc := range cases {
		allocs := testing.AllocsPerRun(1000, tc.run)
		if allocs != 0 {
			t.Fatalf("%s allocated: %v", tc.name, allocs)
		}
	}
}

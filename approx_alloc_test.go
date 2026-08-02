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
		{"FastSqrt", func() { _ = FastSqrt(2.0) }},
		{"FastInvSqrt", func() { _ = FastInvSqrt(2.0) }},
		{"FastLog", func() { _ = FastLog(2.0) }},
		{"FastExp", func() { _ = FastExp(2.0) }},
		{"FastSqrtPrec", func() { _ = FastSqrtPrec(2.0, PrecisionHigh) }},
		{"FastInvSqrtPrec", func() { _ = FastInvSqrtPrec(2.0, PrecisionHigh) }},
		{"FastLogPrec", func() { _ = FastLogPrec(2.0, PrecisionHigh) }},
		{"FastExpPrec", func() { _ = FastExpPrec(2.0, PrecisionHigh) }},
		{"FastTanh", func() { _ = FastTanh(2.0) }},
		{"FastTanhPrec", func() { _ = FastTanhPrec(2.0, PrecisionHigh) }},
		{"FastLogCosh", func() { _ = FastLogCosh(2.0) }},
		{"FastLogCoshPrec", func() { _ = FastLogCoshPrec(2.0, PrecisionHigh) }},
		{"FastRecip", func() { _ = FastRecip(2.0) }},
		{"FastRecipPrec", func() { _ = FastRecipPrec(2.0, PrecisionHigh) }},
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
//nolint:paralleltest
func TestNoAllocs_PublicAPI_Float32(t *testing.T) {
	cases := []struct {
		name string
		run  func()
	}{
		{"FastSqrt32", func() { _ = FastSqrt32(2) }},
		{"FastInvSqrt32", func() { _ = FastInvSqrt32(2) }},
		{"FastLog32", func() { _ = FastLog32(2) }},
		{"FastExp32", func() { _ = FastExp32(2) }},
		{"FastSqrtPrec32", func() { _ = FastSqrtPrec(float32(2), PrecisionHigh) }},
		{"FastInvSqrtPrec32", func() { _ = FastInvSqrtPrec(float32(2), PrecisionHigh) }},
		{"FastLogPrec32", func() { _ = FastLogPrec(float32(2), PrecisionHigh) }},
		{"FastExpPrec32", func() { _ = FastExpPrec(float32(2), PrecisionHigh) }},
		{"FastTanh32", func() { _ = FastTanh32(2) }},
		{"FastLogCosh32", func() { _ = FastLogCosh32(2) }},
		{"FastRecip32", func() { _ = FastRecip32(2) }},
		{"FastTanhPrec32", func() { _ = FastTanhPrec(float32(2), PrecisionHigh) }},
		{"FastLogCoshPrec32", func() { _ = FastLogCoshPrec(float32(2), PrecisionHigh) }},
		{"FastRecipPrec32", func() { _ = FastRecipPrec(float32(2), PrecisionHigh) }},
	}

	for _, tc := range cases {
		allocs := testing.AllocsPerRun(1000, tc.run)
		if allocs != 0 {
			t.Fatalf("%s allocated: %v", tc.name, allocs)
		}
	}
}

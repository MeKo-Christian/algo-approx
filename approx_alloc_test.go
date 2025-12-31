package approx

import "testing"

func TestNoAllocs_PublicAPI_Float64(t *testing.T) {
	t.Parallel()

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
	}

	for _, tc := range cases {
		allocs := testing.AllocsPerRun(1000, tc.run)
		if allocs != 0 {
			t.Fatalf("%s allocated: %v", tc.name, allocs)
		}
	}
}

func TestNoAllocs_PublicAPI_Float32(t *testing.T) {
	t.Parallel()

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
	}

	for _, tc := range cases {
		allocs := testing.AllocsPerRun(1000, tc.run)
		if allocs != 0 {
			t.Fatalf("%s allocated: %v", tc.name, allocs)
		}
	}
}

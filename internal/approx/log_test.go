package approx

import (
	"math"
	"testing"
)

func TestLogAgainstMath_Float64(t *testing.T) {
	t.Parallel()

	cases := []float64{0.125, 0.5, 1, 2, 10, 1e-12, 1e-6, 1e6}
	for _, x := range cases {
		got := Log[float64](x, PrecisionBalanced)

		ref := math.Log(x)
		if !closeRel(got, ref, 2e-3) {
			t.Fatalf("log(%g) got %g ref %g", x, got, ref)
		}
	}
}

func TestLogEdgeCases(t *testing.T) {
	t.Parallel()

	if !math.IsInf(float64(Log[float64](0, PrecisionBalanced)), -1) {
		t.Fatalf("expected -Inf for zero")
	}

	if !math.IsNaN(float64(Log[float64](-1, PrecisionBalanced))) {
		t.Fatalf("expected NaN for negative")
	}
}

// TestLogSubnormals is the regression test for the missing subnormal handling:
// a biased exponent of 0 carries no implicit leading 1, so reconstructing one
// made the result wrong by up to ~36 nats below 2.2e-308.
//
// The reference is Frexp-based rather than math.Log, because Go's amd64
// archLog has the same defect and returns -709.09 for the smallest subnormal.
func TestLogSubnormals(t *testing.T) {
	t.Parallel()

	refLog := func(x float64) float64 {
		frac, exp := math.Frexp(x)
		return math.Log(frac) + float64(exp)*math.Ln2
	}

	for _, x := range []float64{
		math.SmallestNonzeroFloat64,
		1e-322,
		1e-318,
		1e-310,
		2.2250738585072011e-308,
	} {
		want := refLog(x)

		got := Log(x, PrecisionBalanced)
		if !closeRel(got, want, 2e-3) {
			t.Fatalf("Log(%g) got %g want %g", x, got, want)
		}
	}
}

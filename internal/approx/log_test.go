package approx

import (
	"math"
	"testing"
)

func TestLogAgainstMath_Float64(t *testing.T) {
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
	if !math.IsInf(float64(Log[float64](0, PrecisionBalanced)), -1) {
		t.Fatalf("expected -Inf for zero")
	}
	if !math.IsNaN(float64(Log[float64](-1, PrecisionBalanced))) {
		t.Fatalf("expected NaN for negative")
	}
}

package approx

import (
	"math"
	"testing"
)

func TestInvSqrtAgainstMath_Float64(t *testing.T) {
	cases := []float64{1, 2, 4, 16, 1e-12, 1e-6, 1e6, 1e12}
	for _, x := range cases {
		got := InvSqrt[float64](x, PrecisionBalanced)
		ref := 1.0 / math.Sqrt(x)
		if !closeRel(got, ref, 8e-4) {
			t.Fatalf("invsqrt(%g) got %g ref %g", x, got, ref)
		}
	}
}

func TestInvSqrtEdgeCases(t *testing.T) {
	if !math.IsInf(float64(InvSqrt[float64](0, PrecisionBalanced)), 1) {
		t.Fatalf("expected +Inf for zero")
	}
	if !math.IsNaN(float64(InvSqrt[float64](-1, PrecisionBalanced))) {
		t.Fatalf("expected NaN for negative")
	}
}

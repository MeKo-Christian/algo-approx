package approx

import (
	"math"
	"testing"
)

func TestExpAgainstMath_Float64(t *testing.T) {
	t.Parallel()

	cases := []float64{-10, -2, -1, 0, 1, 2, 10}
	for _, x := range cases {
		got := Exp[float64](x, PrecisionBalanced)

		ref := math.Exp(x)
		if !closeRel(got, ref, 2e-3) {
			t.Fatalf("exp(%g) got %g ref %g", x, got, ref)
		}
	}
}

func TestExpEdgeCases(t *testing.T) {
	t.Parallel()

	if Exp[float64](math.Inf(-1), PrecisionBalanced) != 0 {
		t.Fatalf("expected 0 for -Inf")
	}

	if !math.IsInf(float64(Exp[float64](math.Inf(1), PrecisionBalanced)), 1) {
		t.Fatalf("expected +Inf for +Inf")
	}
}

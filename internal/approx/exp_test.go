package approx

import (
	"math"
	"testing"
)

// closeRel reports whether got is within tol relative error of ref, falling
// back to an absolute comparison when ref is zero.
func closeRel(got, ref, tol float64) bool {
	dval := math.Abs(got - ref)

	den := math.Abs(ref)
	if den == 0 {
		return dval <= tol
	}

	return dval/den <= tol
}

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

package approx

import (
	"math"
	"testing"
)

func TestSqrtAgainstMath_Float64(t *testing.T) {
	cases := []float64{0, 1, 2, 4, 16, 1e-12, 1e-6, 1e6, 1e12}
	for _, x := range cases {
		got := Sqrt[float64](x, PrecisionBalanced)
		ref := math.Sqrt(x)
		if !closeRel(got, ref, 5e-4) {
			t.Fatalf("sqrt(%g) got %g ref %g", x, got, ref)
		}
	}
}

func TestSqrtEdgeCases(t *testing.T) {
	if !math.IsNaN(float64(Sqrt[float64](-1, PrecisionBalanced))) {
		t.Fatalf("expected NaN for negative")
	}
	if Sqrt[float64](0, PrecisionBalanced) != 0 {
		t.Fatalf("expected 0 for zero")
	}
}

func closeRel(got, ref, tol float64) bool {
	d := math.Abs(got - ref)
	den := math.Abs(ref)
	if den == 0 {
		return d <= tol
	}
	return d/den <= tol
}

package reference

import (
	"math"
	"testing"
)

func TestSqrtMatchesMath(t *testing.T) {
	t.Parallel()

	for _, x := range []float64{0, 1, 2, 4, 16, 1e-9, 1e9} {
		got := Sqrt[float64](x)

		ref := math.Sqrt(x)
		if got != ref {
			t.Fatalf("sqrt(%g) got %g ref %g", x, got, ref)
		}
	}
}

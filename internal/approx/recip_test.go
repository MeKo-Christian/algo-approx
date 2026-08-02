package approx

import (
	"math"
	"testing"
)

// TestRecipMagicSeedError documents the seed the whole precision ladder is
// built on. If this constant ever changes, the step counts in Recip have to be
// revisited.
func TestRecipMagicSeedError(t *testing.T) {
	t.Parallel()

	var worst float64

	for i := range 100001 {
		m := 1 + float64(i)/100001.0

		y := math.Float64frombits(recipMagic - math.Float64bits(m))
		if err := math.Abs(y*m - 1); err > worst {
			worst = err
		}
	}

	t.Logf("magic seed max relative error: %.6g", worst)

	if worst > 5.06e-2 {
		t.Fatalf("magic seed relative error %g worse than documented 5.051e-2", worst)
	}
}

func TestRecipMantissaStaysInRange(t *testing.T) {
	t.Parallel()

	for i := range 10001 {
		m := 1 + float64(i)/10001.0

		y := recipMantissa(m, PrecisionHigh)
		if y <= 0.5-1e-12 || y > 1+1e-12 {
			t.Fatalf("recipMantissa(%g) = %g outside (0.5, 1]", m, y)
		}
	}
}

func TestRecipFloat32(t *testing.T) {
	t.Parallel()

	for _, x := range []float32{-1e30, -3, -0.25, 0.25, 3, 1e30} {
		got := Recip(x, PrecisionHigh)

		want := 1 / x
		if got != want {
			t.Fatalf("Recip(%v) = %v, want %v", x, got, want)
		}
	}
}

func TestScalePow2MatchesLdexp(t *testing.T) {
	t.Parallel()

	for _, y := range []float64{0.5000001, 0.75, 1} {
		for n := -1100; n <= 1023; n++ {
			if got, want := scalePow2(y, n), math.Ldexp(y, n); got != want {
				t.Fatalf("scalePow2(%g, %d) = %g, want %g", y, n, got, want)
			}
		}
	}
}

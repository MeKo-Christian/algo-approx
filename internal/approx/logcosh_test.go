package approx

import (
	"math"
	"testing"
)

func TestLogCoshBranchesAgreeAtTheSeam(t *testing.T) {
	t.Parallel()

	below := LogCosh(math.Nextafter(tanhBranch, 0), PrecisionBalanced)
	above := LogCosh(tanhBranch, PrecisionBalanced)

	if diff := math.Abs(above - below); diff > 1e-9 {
		t.Fatalf("logCosh seam jump %g at %g", diff, tanhBranch)
	}
}

func TestLog1pSmallMatchesMathLog1p(t *testing.T) {
	t.Parallel()

	// The range log1pSmall is only ever called on: u = exp(-2a) for
	// a >= tanhBranch, i.e. u in (0, exp(-1.25)].
	maxU := math.Exp(-2 * tanhBranch)

	limits := map[Precision]float64{
		PrecisionFast:     2e-5,
		PrecisionBalanced: 1e-10,
		PrecisionHigh:     1e-14,
	}

	for prec, limit := range limits {
		for i := range 5001 {
			u := maxU * float64(i) / 5000.0

			want := math.Log1p(u)
			if diff := math.Abs(log1pSmall(u, prec) - want); diff > limit {
				t.Fatalf("log1pSmall(%g, %v) abs error %g exceeds %g", u, prec, diff, limit)
			}
		}
	}
}

func TestLogCoshFloat32(t *testing.T) {
	t.Parallel()

	for _, x := range []float32{-10, -1, -0.25, 0, 0.25, 1, 10} {
		got := LogCosh(x, PrecisionBalanced)

		want := float32(math.Log(math.Cosh(float64(x))))
		if math.Abs(float64(got-want)) > 1e-5 {
			t.Fatalf("LogCosh(%v) = %v, want %v", x, got, want)
		}
	}
}

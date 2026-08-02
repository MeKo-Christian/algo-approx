package approx

import (
	"math"
	"testing"
)

func TestTanhBranchesAgreeAtTheSeam(t *testing.T) {
	t.Parallel()

	// Both sides of tanhBranch must meet to well inside the accuracy target,
	// otherwise the seam shows up as a spike in any numerical derivative.
	below := Tanh(math.Nextafter(tanhBranch, 0), PrecisionBalanced)
	above := Tanh(tanhBranch, PrecisionBalanced)

	if diff := math.Abs(above - below); diff > 1e-8 {
		t.Fatalf("tanh seam jump %g at %g", diff, tanhBranch)
	}
}

func TestExpNeg2MatchesMathExp(t *testing.T) {
	t.Parallel()

	limits := map[Precision]float64{
		PrecisionFast:     6e-5,
		PrecisionBalanced: 1e-8,
		PrecisionHigh:     1e-11,
	}

	for prec, limit := range limits {
		for i := range 5001 {
			a := tanhBranch + float64(i)/5000.0*370

			want := math.Exp(-2 * a)
			got := expNeg2(a, prec)

			if want == 0 {
				if got != 0 {
					t.Fatalf("expNeg2(%g) = %g, want 0", a, got)
				}

				continue
			}

			// A subnormal result has fewer significand bits than the
			// approximation has accuracy, so relative error is meaningless
			// there. Neither caller can observe it: Tanh saturates long
			// before, and LogCosh adds it to a term of order 350.
			if want < smallestNormalFloat64 {
				continue
			}

			if rel := math.Abs(got-want) / want; rel > limit {
				t.Fatalf("expNeg2(%g, %v) rel error %g exceeds %g", a, prec, rel, limit)
			}
		}
	}
}

func TestExpNeg2UnderflowsToZero(t *testing.T) {
	t.Parallel()

	for _, a := range []float64{expNeg2Zero, 1e6, 1e300, math.MaxFloat64} {
		if got := expNeg2(a, PrecisionBalanced); got != 0 {
			t.Fatalf("expNeg2(%g) = %g, want 0", a, got)
		}
	}
}

func TestTanhFloat32(t *testing.T) {
	t.Parallel()

	for _, x := range []float32{-10, -1, -0.25, 0, 0.25, 1, 10} {
		got := Tanh(x, PrecisionBalanced)

		want := float32(math.Tanh(float64(x)))
		if math.Abs(float64(got-want)) > 1e-7 {
			t.Fatalf("Tanh(%v) = %v, want %v", x, got, want)
		}
	}
}

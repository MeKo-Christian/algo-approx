package approx

import (
	"math"
	"testing"
)

func FuzzFastLog(f *testing.F) {
	seeds := []float64{-1, 0, 1e-12, 1e-6, 0.5, 1, 2, 10, 1e6, math.Inf(1), math.NaN()}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, x float64) {
		got := FastLog(x)
		_ = got

		if math.IsNaN(x) {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("log(NaN) expected NaN")
			}

			return
		}

		if x < 0 {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("log(negative) expected NaN")
			}

			return
		}

		if x == 0 {
			if !math.IsInf(float64(got), -1) {
				t.Fatalf("log(0) expected -Inf")
			}

			return
		}

		if math.IsInf(x, 1) {
			if !math.IsInf(float64(got), 1) {
				t.Fatalf("log(+Inf) expected +Inf")
			}

			return
		}
	})
}

func FuzzFastExp(f *testing.F) {
	seeds := []float64{-1000, -10, -1, 0, 1, 10, 1000, math.Inf(-1), math.Inf(1), math.NaN()}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, x float64) {
		got := FastExp(x)
		_ = got

		if math.IsNaN(x) {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("exp(NaN) expected NaN")
			}

			return
		}

		if math.IsInf(x, -1) {
			if got != 0 {
				t.Fatalf("exp(-Inf) expected 0")
			}

			return
		}

		if math.IsInf(x, 1) {
			if !math.IsInf(float64(got), 1) {
				t.Fatalf("exp(+Inf) expected +Inf")
			}

			return
		}

		if !math.IsNaN(float64(got)) && float64(got) < 0 {
			t.Fatalf("exp(x) should be >= 0")
		}
	})
}

func FuzzFastTanh(f *testing.F) {
	seeds := []float64{-25, -1, -0.5, 0, 0.5, 1, 19.0625, 25, math.Inf(1), math.Inf(-1), math.NaN()}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, x float64) {
		got := FastTanh(x)

		if math.IsNaN(x) {
			if !math.IsNaN(got) {
				t.Fatalf("tanh(NaN) expected NaN")
			}

			return
		}

		if got < -1 || got > 1 {
			t.Fatalf("tanh(%g) = %g outside [-1,1]", x, got)
		}

		if math.Float64bits(FastTanh(-x)) != math.Float64bits(-got) {
			t.Fatalf("tanh(%g) not exactly odd", x)
		}

		if math.Abs(got-math.Tanh(x)) > 1e-7 {
			t.Fatalf("tanh(%g) = %g, math.Tanh = %g", x, got, math.Tanh(x))
		}
	})
}

func FuzzFastLogCosh(f *testing.F) {
	seeds := []float64{-1e300, -800, -12, -0.5, 0, 0.5, 12, 800, 1e300, math.Inf(1), math.NaN()}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, x float64) {
		got := FastLogCosh(x)

		if math.IsNaN(x) {
			if !math.IsNaN(got) {
				t.Fatalf("logCosh(NaN) expected NaN")
			}

			return
		}

		if math.IsInf(x, 0) {
			if !math.IsInf(got, 1) {
				t.Fatalf("logCosh(%g) expected +Inf", x)
			}

			return
		}

		// The whole point: a finite input never produces an infinity.
		if math.IsInf(got, 0) || math.IsNaN(got) {
			t.Fatalf("logCosh(%g) = %g, want finite", x, got)
		}

		if got < 0 {
			t.Fatalf("logCosh(%g) = %g, want >= 0", x, got)
		}

		if got != FastLogCosh(-x) {
			t.Fatalf("logCosh(%g) not even", x)
		}
	})
}

package approx

import (
	"math"
	"testing"
)

func FuzzFastSqrt(f *testing.F) {
	seeds := []float64{-1, 0, 1, 2, 16, 1e-12, 1e12, math.Inf(1), math.NaN()}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, x float64) {
		got := FastSqrt(x)
		_ = got

		if math.IsNaN(x) {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("sqrt(NaN) expected NaN")
			}
			return
		}
		if x < 0 {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("sqrt(negative) expected NaN")
			}
			return
		}
		if x == 0 {
			if got != 0 {
				t.Fatalf("sqrt(0) expected 0")
			}
			return
		}
		if math.IsInf(x, 1) {
			if !math.IsInf(float64(got), 1) {
				t.Fatalf("sqrt(+Inf) expected +Inf")
			}
			return
		}
		if !math.IsNaN(float64(got)) && float64(got) < 0 {
			t.Fatalf("sqrt(x) should be non-negative")
		}
	})
}

func FuzzFastInvSqrt(f *testing.F) {
	seeds := []float64{-1, 0, 1, 2, 16, 1e-12, 1e12, math.Inf(1), math.NaN()}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, x float64) {
		got := FastInvSqrt(x)
		_ = got

		if math.IsNaN(x) {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("invsqrt(NaN) expected NaN")
			}
			return
		}
		if x < 0 {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("invsqrt(negative) expected NaN")
			}
			return
		}
		if x == 0 {
			if !math.IsInf(float64(got), 1) {
				t.Fatalf("invsqrt(0) expected +Inf")
			}
			return
		}
		if math.IsInf(x, 1) {
			if got != 0 {
				t.Fatalf("invsqrt(+Inf) expected 0")
			}
			return
		}
		if !math.IsNaN(float64(got)) && float64(got) <= 0 {
			t.Fatalf("invsqrt(x) should be positive for x>0")
		}
	})
}

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

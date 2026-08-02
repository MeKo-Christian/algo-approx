package approx

import (
	"math"
	"testing"
)

func TestPublicAPI_Sqrt(t *testing.T) {
	t.Parallel()

	got := FastSqrt(16.0)
	if math.Abs(got-4.0) > 1e-2 {
		t.Fatalf("FastSqrt(16) got %g", got)
	}
}

func TestPublicAPI_InvSqrt(t *testing.T) {
	t.Parallel()

	got := FastInvSqrt(4.0)
	if math.Abs(got-0.5) > 1e-2 {
		t.Fatalf("FastInvSqrt(4) got %g", got)
	}
}

func TestPublicAPI_LogExp(t *testing.T) {
	t.Parallel()

	x := 3.0
	if math.Abs(FastExp(FastLog(x))-x) > 5e-2 {
		t.Fatalf("exp(log(x)) composition too far")
	}
}

// TestFastSin tests the public FastSin API.
func TestFastSin(t *testing.T) {
	t.Parallel()

	x := math.Pi / 6.0
	got := FastSin(x)

	want := 0.5
	if math.Abs(got-want) > 0.01 { // Balanced precision
		t.Errorf("FastSin(%v) = %v, want ~%v", x, got, want)
	}
}

// TestFastSinPrec tests FastSin with explicit precision.
func TestFastSinPrec(t *testing.T) {
	t.Parallel()

	x := math.Pi / 6.0

	// Test each precision level
	precisions := []Precision{PrecisionFast, PrecisionBalanced, PrecisionHigh}
	for _, prec := range precisions {
		got := FastSinPrec(x, prec)
		want := 0.5
		// Higher precision should have smaller error
		maxError := 0.1 // Conservative for all precisions
		if math.Abs(got-want) > maxError {
			t.Errorf("FastSinPrec(%v, %v) = %v, want ~%v", x, prec, got, want)
		}
	}
}

// TestFastCos tests the public FastCos API.
func TestFastCos(t *testing.T) {
	t.Parallel()

	x := math.Pi / 3.0
	got := FastCos(x)

	want := 0.5
	if math.Abs(got-want) > 0.01 {
		t.Errorf("FastCos(%v) = %v, want ~%v", x, got, want)
	}
}

// TestFastCosPrec tests FastCos with explicit precision.
func TestFastCosPrec(t *testing.T) {
	t.Parallel()

	x := math.Pi / 3.0

	precisions := []Precision{PrecisionFast, PrecisionBalanced, PrecisionHigh}
	for _, prec := range precisions {
		got := FastCosPrec(x, prec)
		want := 0.5

		maxError := 0.1
		if math.Abs(got-want) > maxError {
			t.Errorf("FastCosPrec(%v, %v) = %v, want ~%v", x, prec, got, want)
		}
	}
}

// TestFastTan tests the public FastTan API.
func TestFastTan(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-10},
		{"π/6", math.Pi / 6, 0.01},
		{"π/4", math.Pi / 4, 0.02},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := FastTan(tt.input)
			want := math.Tan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastTan(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestFastTanPrec tests the public FastTanPrec API with different precision levels.
func TestFastTanPrec(t *testing.T) {
	t.Parallel()

	x := math.Pi / 6

	t.Run("PrecisionFast", func(t *testing.T) {
		t.Parallel()

		got := FastTanPrec(x, PrecisionFast)
		want := math.Tan(x)

		diff := math.Abs(got - want)
		if diff > 0.01 {
			t.Errorf("FastTanPrec(%v, PrecisionFast) diff too large: %v", x, diff)
		}
	})

	t.Run("PrecisionBalanced", func(t *testing.T) {
		t.Parallel()

		got := FastTanPrec(x, PrecisionBalanced)
		want := math.Tan(x)

		diff := math.Abs(got - want)
		if diff > 0.001 {
			t.Errorf("FastTanPrec(%v, PrecisionBalanced) diff too large: %v", x, diff)
		}
	})

	t.Run("PrecisionHigh", func(t *testing.T) {
		t.Parallel()

		got := FastTanPrec(x, PrecisionHigh)
		want := math.Tan(x)

		diff := math.Abs(got - want)
		if diff > 0.000001 {
			t.Errorf("FastTanPrec(%v, PrecisionHigh) diff too large: %v", x, diff)
		}
	})
}

// TestFastCotan tests the public FastCotan API.
func TestFastCotan(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"π/12", math.Pi / 12, 0.01},
		{"π/6", math.Pi / 6, 0.01},
		{"π/4", math.Pi / 4, 0.02},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := FastCotan(tt.input)
			want := 1.0 / math.Tan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastCotan(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestFastArctan tests the public FastArctan API.
func TestFastArctan(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-10},
		{"small positive", 0.1, 1e-5},
		{"π/12", math.Pi / 12, 2e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := FastArctan(tt.input)
			want := math.Atan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastArctan(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestFastArctanPrec tests the public FastArctanPrec API with different precision levels.
//
//nolint:dupl
func TestFastArctanPrec(t *testing.T) {
	t.Parallel()

	x := 0.1

	t.Run("PrecisionFast", func(t *testing.T) {
		t.Parallel()

		got := FastArctanPrec(x, PrecisionFast)
		want := math.Atan(x)

		diff := math.Abs(got - want)
		if diff > 1e-5 {
			t.Errorf("FastArctanPrec(%v, PrecisionFast) diff too large: %v", x, diff)
		}
	})

	t.Run("PrecisionBalanced", func(t *testing.T) {
		t.Parallel()

		got := FastArctanPrec(x, PrecisionBalanced)
		want := math.Atan(x)

		diff := math.Abs(got - want)
		if diff > 1e-5 {
			t.Errorf("FastArctanPrec(%v, PrecisionBalanced) diff too large: %v", x, diff)
		}
	})

	t.Run("PrecisionHigh", func(t *testing.T) {
		t.Parallel()

		got := FastArctanPrec(x, PrecisionHigh)
		want := math.Atan(x)

		diff := math.Abs(got - want)
		if diff > 1e-10 {
			t.Errorf("FastArctanPrec(%v, PrecisionHigh) diff too large: %v", x, diff)
		}
	})
}

// TestFastArccotan tests the public FastArccotan API.
func TestFastArccotan(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"small positive", 0.1, 1e-5},
		{"π/12", math.Pi / 12, 2e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := FastArccotan(tt.input)
			want := math.Pi/2 - math.Atan(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastArccotan(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestFastArccos tests the public FastArccos API.
func TestFastArccos(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		input     float64
		tolerance float64
	}{
		{"zero", 0.0, 1e-5},
		{"half", 0.5, 1e-3},
		{"sqrt(2)/2", math.Sqrt(2) / 2, 2e-4},
		{"one", 1.0, 1e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := FastArccos(tt.input)
			want := math.Acos(tt.input)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastArccos(%v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.input, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestFastArccosPrec tests the public FastArccosPrec API with different precision levels.
//
//nolint:dupl
func TestFastArccosPrec(t *testing.T) {
	t.Parallel()

	x := 0.5

	t.Run("PrecisionFast", func(t *testing.T) {
		t.Parallel()

		got := FastArccosPrec(x, PrecisionFast)
		want := math.Acos(x)

		diff := math.Abs(got - want)
		if diff > 1e-3 {
			t.Errorf("FastArccosPrec(%v, PrecisionFast) diff too large: %v", x, diff)
		}
	})

	t.Run("PrecisionBalanced", func(t *testing.T) {
		t.Parallel()

		got := FastArccosPrec(x, PrecisionBalanced)
		want := math.Acos(x)

		diff := math.Abs(got - want)
		if diff > 1e-3 {
			t.Errorf("FastArccosPrec(%v, PrecisionBalanced) diff too large: %v", x, diff)
		}
	})

	t.Run("PrecisionHigh", func(t *testing.T) {
		t.Parallel()

		got := FastArccosPrec(x, PrecisionHigh)
		want := math.Acos(x)

		diff := math.Abs(got - want)
		if diff > 1e-5 {
			t.Errorf("FastArccosPrec(%v, PrecisionHigh) diff too large: %v", x, diff)
		}
	})
}

// TestFastPower tests the public FastPower API.
func TestFastPower(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		base      float64
		exponent  float64
		tolerance float64
	}{
		{"2^3", 2.0, 3.0, 1e-3},
		{"3^2", 3.0, 2.0, 1e-4},
		{"10^0.5", 10.0, 0.5, 1e-4},
		{"2^-2", 2.0, -2.0, 1e-4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := FastPower(tt.base, tt.exponent)
			want := math.Pow(tt.base, tt.exponent)
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastPower(%v, %v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.base, tt.exponent, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestFastRoot tests the public FastRoot API.
func TestFastRoot(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		value     float64
		n         int
		tolerance float64
	}{
		{"sqrt(4)", 4.0, 2, 1e-5},
		{"cbrt(8)", 8.0, 3, 1e-4},
		{"cbrt(27)", 27.0, 3, 1e-4},
		{"4th root(16)", 16.0, 4, 1e-4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := FastRoot(tt.value, tt.n)
			want := math.Pow(tt.value, 1.0/float64(tt.n))
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastRoot(%v, %v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.value, tt.n, got, want, diff, tt.tolerance)
			}
		})
	}
}

// TestFastIntPower tests the public FastIntPower API.
func TestFastIntPower(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		base      float64
		exponent  int
		tolerance float64
	}{
		{"2^0", 2.0, 0, 1e-15},
		{"2^1", 2.0, 1, 1e-15},
		{"2^3", 2.0, 3, 1e-15},
		{"2^10", 2.0, 10, 1e-12},
		{"2^-2", 2.0, -2, 1e-15},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := FastIntPower(tt.base, tt.exponent)
			want := math.Pow(tt.base, float64(tt.exponent))
			diff := math.Abs(got - want)

			if diff > tt.tolerance {
				t.Errorf("FastIntPower(%v, %v) = %v, want %v (diff: %v, tolerance: %v)",
					tt.base, tt.exponent, got, want, diff, tt.tolerance)
			}
		})
	}
}

// --- FastTanh -------------------------------------------------------------

// TestFastTanh_OddSymmetryIsBitExact checks the guarantee that costs nothing
// to keep and everything to lose: FastTanh(-x) must be the bit-for-bit
// negation of FastTanh(x), not merely close to it.
func TestFastTanh_OddSymmetryIsBitExact(t *testing.T) {
	t.Parallel()

	for _, prec := range []Precision{PrecisionFast, PrecisionBalanced, PrecisionHigh} {
		for i := range 20001 {
			x := -20 + 40*float64(i)/20000.0

			pos := math.Float64bits(FastTanhPrec(x, prec))
			neg := math.Float64bits(FastTanhPrec(-x, prec))

			if pos != math.Float64bits(-math.Float64frombits(neg)) {
				t.Fatalf("prec=%v x=%g: tanh(x)=%#016x tanh(-x)=%#016x not an exact negation",
					prec, x, pos, neg)
			}
		}
	}
}

// TestFastTanh_SaturatesExactly checks that beyond the point where float64
// tanh is indistinguishable from 1, FastTanh returns exactly 1 (and matches
// math.Tanh at the crossover).
func TestFastTanh_SaturatesExactly(t *testing.T) {
	t.Parallel()

	for _, x := range []float64{19.0625, 19.5, 20, 100, 1e300, math.MaxFloat64} {
		if got := FastTanh(x); got != 1 {
			t.Fatalf("FastTanh(%g) = %v, want exactly 1", x, got)
		}

		if got := FastTanh(-x); got != -1 {
			t.Fatalf("FastTanh(%g) = %v, want exactly -1", -x, got)
		}

		if math.Tanh(x) != 1 {
			t.Fatalf("saturation constant is too small: math.Tanh(%g) != 1", x)
		}
	}

	// Just below the constant, the result must not be clamped: the true
	// crossover is 19.06154746..., so 19.0 is genuinely below 1.
	if FastTanh(19.0) >= 1 {
		t.Fatalf("FastTanh(19) = %v, want < 1", FastTanh(19.0))
	}
}

func TestFastTanh_EdgeCases(t *testing.T) {
	t.Parallel()

	if got := FastTanh(math.NaN()); !math.IsNaN(got) {
		t.Fatalf("FastTanh(NaN) = %v, want NaN", got)
	}

	if got := FastTanh(math.Inf(1)); got != 1 {
		t.Fatalf("FastTanh(+Inf) = %v, want 1", got)
	}

	if got := FastTanh(math.Inf(-1)); got != -1 {
		t.Fatalf("FastTanh(-Inf) = %v, want -1", got)
	}

	if got := FastTanh(0.0); got != 0 || math.Signbit(got) {
		t.Fatalf("FastTanh(+0) = %v, want +0", got)
	}

	negZero := math.Copysign(0, -1)
	if got := FastTanh(negZero); got != 0 || !math.Signbit(got) {
		t.Fatalf("FastTanh(-0) = %v, want -0", got)
	}
}

func TestFastTanh_MaxAbsErrorBalanced(t *testing.T) {
	t.Parallel()

	const tolerance = 1e-7

	var worst, worstAt float64

	for i := range 400001 {
		x := -20 + 40*float64(i)/400000.0

		diff := math.Abs(FastTanhPrec(x, PrecisionBalanced) - math.Tanh(x))
		if diff > worst {
			worst, worstAt = diff, x
		}
	}

	t.Logf("FastTanh balanced: max abs error %.4g at x=%g", worst, worstAt)

	if worst > tolerance {
		t.Fatalf("max abs error %g at x=%g exceeds %g", worst, worstAt, tolerance)
	}
}

// --- FastLogCosh ----------------------------------------------------------

// TestFastLogCosh_DerivativeMatchesTanh is the consistency requirement: tanh
// is exactly d/dx log(cosh(x)), and the pair is designed so that the identity
// survives the approximation. A central difference is used, so the tolerance
// has to cover its own O(h^2) truncation (h^2*max|f”'|/6 = 1.3e-7 for
// h = 1e-3) on top of the branch seam.
func TestFastLogCosh_DerivativeMatchesTanh(t *testing.T) {
	t.Parallel()

	const (
		step      = 1e-3
		tolerance = 5e-6
	)

	var worst, worstAt float64

	for i := range 48001 {
		x := -12 + 24*float64(i)/48000.0

		derivative := (FastLogCoshPrec(x+step, PrecisionBalanced) -
			FastLogCoshPrec(x-step, PrecisionBalanced)) / (2 * step)

		diff := math.Abs(derivative - FastTanhPrec(x, PrecisionBalanced))
		if diff > worst {
			worst, worstAt = diff, x
		}
	}

	t.Logf("d/dx FastLogCosh vs FastTanh: max deviation %.4g at x=%g", worst, worstAt)

	if worst > tolerance {
		t.Fatalf("derivative of FastLogCosh deviates from FastTanh by %g at x=%g (limit %g)",
			worst, worstAt, tolerance)
	}
}

// TestFastLogCosh_NoOverflow covers the reason this function exists at all:
// math.Log(math.Cosh(x)) returns +Inf above |x| ~ 710 because cosh overflows
// before the log can pull it back.
func TestFastLogCosh_NoOverflow(t *testing.T) {
	t.Parallel()

	for _, x := range []float64{700, 710, 745, 800, 1e6, 1e300, math.MaxFloat64} {
		got := FastLogCosh(x)
		if math.IsInf(got, 0) || math.IsNaN(got) {
			t.Fatalf("FastLogCosh(%g) = %v, want finite", x, got)
		}

		// Above the underflow point of exp(-2x) the answer is exactly
		// x - ln2 to within rounding.
		want := x - math.Ln2
		if math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
			t.Fatalf("FastLogCosh(%g) = %v, want ~%v", x, got, want)
		}

		if FastLogCosh(-x) != got {
			t.Fatalf("FastLogCosh is not even at %g", x)
		}
	}

	// The naive form really does overflow, which is what is being avoided.
	if !math.IsInf(math.Log(math.Cosh(800)), 1) {
		t.Fatal("expected math.Log(math.Cosh(800)) to overflow")
	}
}

func TestFastLogCosh_MaxAbsErrorBalanced(t *testing.T) {
	t.Parallel()

	const tolerance = 1e-7

	var worst, worstAt float64

	for i := range 240001 {
		x := -12 + 24*float64(i)/240000.0

		diff := math.Abs(FastLogCoshPrec(x, PrecisionBalanced) - math.Log(math.Cosh(x)))
		if diff > worst {
			worst, worstAt = diff, x
		}
	}

	t.Logf("FastLogCosh balanced: max abs error %.4g at x=%g", worst, worstAt)

	if worst > tolerance {
		t.Fatalf("max abs error %g at x=%g exceeds %g", worst, worstAt, tolerance)
	}
}

func TestFastLogCosh_EdgeCases(t *testing.T) {
	t.Parallel()

	if got := FastLogCosh(math.NaN()); !math.IsNaN(got) {
		t.Fatalf("FastLogCosh(NaN) = %v, want NaN", got)
	}

	if got := FastLogCosh(math.Inf(1)); !math.IsInf(got, 1) {
		t.Fatalf("FastLogCosh(+Inf) = %v, want +Inf", got)
	}

	if got := FastLogCosh(math.Inf(-1)); !math.IsInf(got, 1) {
		t.Fatalf("FastLogCosh(-Inf) = %v, want +Inf", got)
	}

	for _, x := range []float64{0, math.Copysign(0, -1)} {
		if got := FastLogCosh(x); got != 0 || math.Signbit(got) {
			t.Fatalf("FastLogCosh(%v) = %v, want +0", x, got)
		}
	}
}

// --- FastRecip ------------------------------------------------------------

func TestFastRecip_EdgeCases(t *testing.T) {
	t.Parallel()

	if got := FastRecip(math.NaN()); !math.IsNaN(got) {
		t.Fatalf("FastRecip(NaN) = %v, want NaN", got)
	}

	if got := FastRecip(0.0); !math.IsInf(got, 1) {
		t.Fatalf("FastRecip(+0) = %v, want +Inf", got)
	}

	if got := FastRecip(math.Copysign(0, -1)); !math.IsInf(got, -1) {
		t.Fatalf("FastRecip(-0) = %v, want -Inf", got)
	}

	if got := FastRecip(math.Inf(1)); got != 0 || math.Signbit(got) {
		t.Fatalf("FastRecip(+Inf) = %v, want +0", got)
	}

	if got := FastRecip(math.Inf(-1)); got != 0 || !math.Signbit(got) {
		t.Fatalf("FastRecip(-Inf) = %v, want -0", got)
	}
}

// TestFastRecip_Subnormals covers the inputs whose biased exponent is zero,
// where the mantissa carries no implicit leading 1.
func TestFastRecip_Subnormals(t *testing.T) {
	t.Parallel()

	cases := []float64{
		math.SmallestNonzeroFloat64,
		1e-320,
		1e-315,
		5e-310,
		2.2250738585072011e-308, // largest subnormal
	}

	for _, x := range cases {
		// 1/x overflows for the smallest of these, exactly as a divide does.
		want := 1 / x

		got := FastRecipPrec(x, PrecisionHigh)
		if math.IsInf(want, 0) {
			if !math.IsInf(got, 1) {
				t.Fatalf("FastRecip(%g) = %v, want +Inf", x, got)
			}

			continue
		}

		if rel := math.Abs(got-want) / want; rel > 1e-15 {
			t.Fatalf("FastRecip(%g) = %v, want %v (rel %g)", x, got, want, rel)
		}

		if FastRecipPrec(-x, PrecisionHigh) != -got {
			t.Fatalf("FastRecip is not exactly odd at %g", x)
		}
	}
}

func TestFastRecip_RelativeErrorByPrecision(t *testing.T) {
	t.Parallel()

	limits := map[Precision]float64{
		PrecisionFast:     3e-8,
		PrecisionBalanced: 1e-15,
		PrecisionHigh:     2.3e-16, // one ulp
	}

	for prec, limit := range limits {
		var worst, worstAt float64

		for i := range 200001 {
			// Log-spaced over the whole normal range, both signs.
			x := math.Pow(10, -300+600*float64(i)/200000.0)
			if i%2 == 1 {
				x = -x
			}

			want := 1 / x

			rel := math.Abs(FastRecipPrec(x, prec)-want) / math.Abs(want)
			if rel > worst {
				worst, worstAt = rel, x
			}
		}

		t.Logf("FastRecip %v: max rel error %.4g at x=%g", prec, worst, worstAt)

		if worst > limit {
			t.Fatalf("FastRecip %v: max rel error %g at x=%g exceeds %g",
				prec, worst, worstAt, limit)
		}
	}
}

// --- FastLog subnormal regression ----------------------------------------

// TestFastLog_Subnormals is the regression test for the bug where the biased
// exponent field was read as -1023 and an implicit leading 1 was reconstructed
// for inputs that do not have one, making the result wrong by up to ~36 nats
// below 2.2e-308.
//
// Note the reference: math.Log cannot be used here. Go's amd64 archLog
// (src/math/log_amd64.s) makes exactly the same mistake, so on this platform
// math.Log(math.SmallestNonzeroFloat64) returns -709.09 instead of -744.44.
// The reference below normalizes the input first and is correct on every
// platform.
func TestFastLog_Subnormals(t *testing.T) {
	t.Parallel()

	refLog := func(x float64) float64 {
		frac, exp := math.Frexp(x)
		return math.Log(frac) + float64(exp)*math.Ln2
	}

	cases := []float64{
		math.SmallestNonzeroFloat64,
		math.Float64frombits(2),
		1e-323,
		1e-320,
		1e-315,
		5e-310,
		2.2250738585072011e-308, // largest subnormal
		2.2250738585072014e-308, // smallest normal
		1e-307,
	}

	for _, x := range cases {
		want := refLog(x)

		got := FastLogPrec(x, PrecisionBalanced)
		if rel := math.Abs(got-want) / math.Abs(want); rel > 1e-3 {
			t.Fatalf("FastLog(%g) = %v, want %v (rel %g)", x, got, want, rel)
		}
	}

	// Sweep the whole subnormal range on a log grid.
	var worst, worstAt float64

	for i := range 20001 {
		x := math.Ldexp(1, -1074+i%52) * (1 + float64(i%1000)/1000)
		if x <= 0 || x >= 2.2250738585072014e-308 {
			continue
		}

		want := refLog(x)

		rel := math.Abs(FastLogPrec(x, PrecisionBalanced)-want) / math.Abs(want)
		if rel > worst {
			worst, worstAt = rel, x
		}
	}

	t.Logf("FastLog over subnormals: max rel error %.4g at x=%g", worst, worstAt)

	if worst > 1e-3 {
		t.Fatalf("FastLog subnormal max rel error %g at x=%g", worst, worstAt)
	}
}

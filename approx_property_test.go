package approx

import (
	"math"
	"testing"
)

func TestProperty_ExpLog_RoundTrip_Float64(t *testing.T) {
	t.Parallel()
	// For x>0: exp(log(x)) ≈ x
	for _, x := range []float64{1e-6, 1e-3, 0.1, 0.5, 1, 2, 10, 1e3, 1e6} {
		got := FastExp(FastLog(x))
		if !closeRel(float64(got), x, 5e-2) {
			t.Fatalf("exp(log(%g)) got %g", x, got)
		}
	}
}

func TestProperty_Sqrt_Square_Float64(t *testing.T) {
	t.Parallel()
	// For x>=0: sqrt(x)^2 ≈ x
	for _, x := range []float64{0, 1e-12, 1e-6, 0.1, 1, 2, 10, 1e6, 1e12} {
		y := FastSqrt(x)

		got := float64(y * y)
		if !closeRel(got, x, 2e-2) {
			t.Fatalf("sqrt(%g)^2 got %g", x, got)
		}
	}
}

func TestProperty_InvSqrt_Sqrt_Product_Float64(t *testing.T) {
	t.Parallel()
	// For x>0: invsqrt(x)*sqrt(x) ≈ 1
	for _, x := range []float64{1e-12, 1e-6, 0.1, 1, 2, 10, 1e6, 1e12} {
		p := float64(FastInvSqrt(x) * FastSqrt(x))
		if math.Abs(p-1) > 2e-2 {
			t.Fatalf("invsqrt(x)*sqrt(x) for x=%g got %g", x, p)
		}
	}
}

func TestProperty_Monotonicity_Sqrt_Float64(t *testing.T) {
	t.Parallel()

	prev := FastSqrt(0.0)
	for _, x := range []float64{1e-12, 1e-6, 1e-3, 0.1, 1, 2, 10, 1e3, 1e6} {
		cur := FastSqrt(x)
		if cur < prev {
			t.Fatalf("sqrt not monotone: sqrt(%g)=%g < prev=%g", x, cur, prev)
		}

		prev = cur
	}
}

func closeRel(got, ref, tol float64) bool {
	dval := math.Abs(got - ref)

	den := math.Abs(ref)
	if den == 0 {
		return dval <= tol
	}

	return dval/den <= tol
}

// TestTrigIdentity_SinSquaredPlusCosSquared tests sin²(x) + cos²(x) ≈ 1.
func TestTrigIdentity_SinSquaredPlusCosSquared(t *testing.T) {
	t.Parallel()

	testValues := []float64{
		0, math.Pi / 6, math.Pi / 4, math.Pi / 3, math.Pi / 2,
		2 * math.Pi / 3, 3 * math.Pi / 4, 5 * math.Pi / 6,
	}

	for _, x := range testValues {
		sinVal := FastSin(x)
		cosVal := FastCos(x)
		result := sinVal*sinVal + cosVal*cosVal

		// Should be very close to 1.0
		// Tolerance adjusted for approximation errors
		if math.Abs(result-1.0) > 0.01 {
			t.Errorf("sin²(%v) + cos²(%v) = %v, want 1.0 (±0.01)", x, x, result)
		}
	}
}

// TestTrigIdentity_SinSquaredPlusCosSquared_NearPi tests the identity near π
// where approximation errors can accumulate.
func TestTrigIdentity_SinSquaredPlusCosSquared_NearPi(t *testing.T) {
	t.Parallel()

	x := math.Pi
	sinVal := FastSin(x)
	cosVal := FastCos(x)
	result := sinVal*sinVal + cosVal*cosVal

	// Larger tolerance near π due to accumulated approximation errors
	if math.Abs(result-1.0) > 0.05 {
		t.Errorf("sin²(%v) + cos²(%v) = %v, want 1.0 (±0.05)", x, x, result)
	}
}

// TestTrigIdentity_SecIsCosReciprocal tests sec(x) ≈ 1/cos(x).
func TestTrigIdentity_SecIsCosReciprocal(t *testing.T) {
	t.Parallel()

	testValues := []float64{0, math.Pi / 6, math.Pi / 4, math.Pi / 3}

	for _, x := range testValues {
		secVal := FastSec(x)
		cosVal := FastCos(x)
		expected := 1.0 / cosVal

		if math.Abs(secVal-expected) > 0.01 {
			t.Errorf("sec(%v) = %v, want 1/cos(%v) = %v", x, secVal, x, expected)
		}
	}
}

// TestTrigIdentity_CscIsSinReciprocal tests csc(x) ≈ 1/sin(x).
func TestTrigIdentity_CscIsSinReciprocal(t *testing.T) {
	t.Parallel()

	testValues := []float64{math.Pi / 6, math.Pi / 4, math.Pi / 3, math.Pi / 2}

	for _, x := range testValues {
		cscVal := FastCsc(x)
		sinVal := FastSin(x)
		expected := 1.0 / sinVal

		if math.Abs(cscVal-expected) > 0.01 {
			t.Errorf("csc(%v) = %v, want 1/sin(%v) = %v", x, cscVal, x, expected)
		}
	}
}

// TestTrigSymmetry_SinIsOdd tests sin(-x) ≈ -sin(x).
func TestTrigSymmetry_SinIsOdd(t *testing.T) {
	t.Parallel()

	testValues := []float64{math.Pi / 6, math.Pi / 4, math.Pi / 3, math.Pi / 2}

	for _, x := range testValues {
		sinPos := FastSin(x)
		sinNeg := FastSin(-x)

		if math.Abs(sinNeg+sinPos) > 0.01 {
			t.Errorf("sin(-%v) = %v, want -sin(%v) = %v", x, sinNeg, x, -sinPos)
		}
	}
}

// TestTrigSymmetry_CosIsEven tests cos(-x) ≈ cos(x).
func TestTrigSymmetry_CosIsEven(t *testing.T) {
	t.Parallel()

	testValues := []float64{math.Pi / 6, math.Pi / 4, math.Pi / 3, math.Pi / 2}

	for _, x := range testValues {
		cosPos := FastCos(x)
		cosNeg := FastCos(-x)

		if math.Abs(cosNeg-cosPos) > 0.01 {
			t.Errorf("cos(-%v) = %v, want cos(%v) = %v", x, cosNeg, x, cosPos)
		}
	}
}

// TestTangentIdentity tests the identity: tan(x) * cotan(x) ≈ 1.
func TestTangentIdentity(t *testing.T) {
	t.Parallel()
	// Test in range where both tan and cotan are well-behaved
	for i := range 20 {
		x := float64(i) * math.Pi / 24 // 0 to 5π/6 in steps of π/24
		if x == 0 {
			continue // Skip zero (cotan undefined)
		}

		tanVal := FastTan(x)
		cotanVal := FastCotan(x)
		product := tanVal * cotanVal

		diff := math.Abs(product - 1.0)
		if diff > 0.05 {
			t.Errorf("tan(%v) * cotan(%v) = %v, want 1.0 (diff: %v)",
				x, x, product, diff)
		}
	}
}

// TestTangentReciprocal tests: cotan(x) ≈ 1/tan(x).
func TestTangentReciprocal(t *testing.T) {
	t.Parallel()

	for i := 1; i < 20; i++ {
		x := float64(i) * math.Pi / 24

		cotanVal := FastCotan(x)
		recipTan := 1.0 / FastTan(x)

		diff := math.Abs(cotanVal - recipTan)
		if diff > 1e-10 {
			t.Errorf("cotan(%v) = %v, 1/tan(%v) = %v (diff: %v)",
				x, cotanVal, x, recipTan, diff)
		}
	}
}

// TestTangentPeriodicityProperty tests: tan(x + π) ≈ tan(x).
func TestTangentPeriodicityProperty(t *testing.T) {
	t.Parallel()

	for i := range 10 {
		x := float64(i) * math.Pi / 12

		tanX := FastTan(x)
		tanXPlusPi := FastTan(x + math.Pi)

		diff := math.Abs(tanX - tanXPlusPi)
		if diff > 0.05 {
			t.Errorf("tan(%v) = %v, tan(%v + π) = %v (diff: %v)",
				x, tanX, x, tanXPlusPi, diff)
		}
	}
}

// TestArctanArccotanReciprocal verifies that arctan(x) + arccotan(x) ≈ π/2.
func TestArctanArccotanReciprocal(t *testing.T) {
	t.Parallel()

	tolerance := 0.0001

	tests := []float64{0.1, 0.2, 0.25}

	for _, x := range tests {
		arctan := FastArctan(x)
		arccotan := FastArccotan(x)
		sum := arctan + arccotan

		expected := math.Pi / 2
		diff := math.Abs(sum - expected)

		if diff > tolerance {
			t.Errorf("arctan(%v) + arccotan(%v) = %v, want ~π/2=%v (diff: %v)",
				x, x, sum, expected, diff)
		}
	}
}

// TestArccosComplementarity verifies that arccos(x) + arcsin(x) ≈ π/2 for small x.
func TestArccosComplementarity(t *testing.T) {
	t.Parallel()

	tolerance := 0.002

	tests := []float64{0.0, 0.5, math.Sqrt(2) / 2}

	for _, x := range tests {
		arccos := FastArccos(x)
		// arcsin(x) ≈ π/2 - arccos(x)
		arcsin := math.Pi/2 - arccos

		expected := math.Asin(x)
		diff := math.Abs(arcsin - expected)

		if diff > tolerance {
			t.Errorf("arcsin(%v) via arccos = %v, want %v (diff: %v)",
				x, arcsin, expected, diff)
		}
	}
}

// TestInverseTrigRoundTrip tests that tan(arctan(x)) ≈ x for small x.
func TestInverseTrigRoundTrip(t *testing.T) {
	t.Parallel()

	tolerance := 0.0001

	tests := []float64{0.0, 0.1, 0.2}

	for _, x := range tests {
		arctan := FastArctan(x)
		result := math.Tan(arctan)

		diff := math.Abs(result - x)

		if diff > tolerance {
			t.Errorf("tan(arctan(%v)) = %v, want %v (diff: %v)",
				x, result, x, diff)
		}
	}
}

// TestPowerRootIdentity tests the identity: root(x^n, n) ≈ x.
func TestPowerRootIdentity(t *testing.T) {
	t.Parallel()

	tolerance := 1e-4

	tests := []struct {
		x float64
		n int
	}{
		{2.0, 2},
		{2.0, 3},
		{3.0, 2},
		{5.0, 3},
		{10.0, 2},
	}

	for _, tt := range tests {
		// x^n
		powered := FastIntPower(tt.x, tt.n)
		// root(x^n, n) should give back x
		result := FastRoot(powered, tt.n)

		diff := math.Abs(result - tt.x)
		if diff > tolerance {
			t.Errorf("root(%v^%v, %v) = %v, want %v (diff: %v)",
				tt.x, tt.n, tt.n, result, tt.x, diff)
		}
	}
}

// TestIntPowerVsPower tests that IntPower and Power give similar results for integer exponents.
func TestIntPowerVsPower(t *testing.T) {
	t.Parallel()

	tests := []struct {
		base      float64
		exponent  int
		tolerance float64
	}{
		{2.0, 3, 1e-3},
		{3.0, 2, 1e-3},
		{2.0, 10, 0.2}, // Higher exponents accumulate more error
		{1.5, 5, 0.01},
	}

	for _, tt := range tests {
		intPowResult := FastIntPower(tt.base, tt.exponent)
		powResult := FastPower(tt.base, float64(tt.exponent))

		diff := math.Abs(intPowResult - powResult)
		if diff > tt.tolerance {
			t.Errorf("IntPower(%v, %v) = %v, Power(%v, %v) = %v (diff: %v, tolerance: %v)",
				tt.base, tt.exponent, intPowResult, tt.base, tt.exponent, powResult, diff, tt.tolerance)
		}
	}
}

// TestPowerExponentLaws tests power laws: (a^b)^c = a^(b*c).
func TestPowerExponentLaws(t *testing.T) {
	t.Parallel()

	tolerance := 1e-2

	tests := []struct {
		a float64
		b float64
		c float64
	}{
		{2.0, 2.0, 3.0},
		{3.0, 1.5, 2.0},
		{4.0, 0.5, 2.0},
	}

	for _, tt := range tests {
		// (a^b)^c
		ab := FastPower(tt.a, tt.b)
		abc := FastPower(ab, tt.c)

		// a^(b*c)
		bc := tt.b * tt.c
		expected := FastPower(tt.a, bc)

		diff := math.Abs(abc - expected)
		relDiff := diff / math.Abs(expected)

		if relDiff > tolerance {
			t.Errorf("(%v^%v)^%v = %v, want %v^(%v*%v) = %v (rel diff: %v)",
				tt.a, tt.b, tt.c, abc, tt.a, tt.b, tt.c, expected, relDiff)
		}
	}
}

// TestRootSquareIdentity tests: sqrt(x) = root(x, 2).
func TestRootSquareIdentity(t *testing.T) {
	t.Parallel()

	tolerance := 1e-10

	tests := []float64{1.0, 2.0, 4.0, 9.0, 16.0, 100.0}

	for _, x := range tests {
		sqrtVal := FastSqrt(x)
		rootVal := FastRoot(x, 2)

		diff := math.Abs(sqrtVal - rootVal)
		if diff > tolerance {
			t.Errorf("sqrt(%v) = %v, root(%v, 2) = %v (diff: %v)",
				x, sqrtVal, x, rootVal, diff)
		}
	}
}

// TestIntPowerNegativeExponent tests: x^(-n) = 1/(x^n).
func TestIntPowerNegativeExponent(t *testing.T) {
	t.Parallel()

	tolerance := 1e-14

	tests := []struct {
		base float64
		exp  int
	}{
		{2.0, 2},
		{3.0, 3},
		{10.0, 2},
	}

	for _, tt := range tests {
		negPow := FastIntPower(tt.base, -tt.exp)
		posPow := FastIntPower(tt.base, tt.exp)
		expected := 1.0 / posPow

		diff := math.Abs(negPow - expected)
		if diff > tolerance {
			t.Errorf("IntPower(%v, -%v) = %v, want 1/IntPower(%v, %v) = %v (diff: %v)",
				tt.base, tt.exp, negPow, tt.base, tt.exp, expected, diff)
		}
	}
}

package approx

import (
	"math"
	"testing"
)

func TestProperty_ExpLog_RoundTrip_Float64(t *testing.T) {
	// For x>0: exp(log(x)) ≈ x
	for _, x := range []float64{1e-6, 1e-3, 0.1, 0.5, 1, 2, 10, 1e3, 1e6} {
		got := FastExp(FastLog(x))
		if !closeRel(float64(got), x, 5e-2) {
			t.Fatalf("exp(log(%g)) got %g", x, got)
		}
	}
}

func TestProperty_Sqrt_Square_Float64(t *testing.T) {
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
	// For x>0: invsqrt(x)*sqrt(x) ≈ 1
	for _, x := range []float64{1e-12, 1e-6, 0.1, 1, 2, 10, 1e6, 1e12} {
		p := float64(FastInvSqrt(x) * FastSqrt(x))
		if math.Abs(p-1) > 2e-2 {
			t.Fatalf("invsqrt(x)*sqrt(x) for x=%g got %g", x, p)
		}
	}
}

func TestProperty_Monotonicity_Sqrt_Float64(t *testing.T) {
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
	d := math.Abs(got - ref)
	den := math.Abs(ref)
	if den == 0 {
		return d <= tol
	}
	return d/den <= tol
}

// TestTrigIdentity_SinSquaredPlusCosSquared tests sin²(x) + cos²(x) ≈ 1
func TestTrigIdentity_SinSquaredPlusCosSquared(t *testing.T) {
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
// where approximation errors can accumulate
func TestTrigIdentity_SinSquaredPlusCosSquared_NearPi(t *testing.T) {
	x := math.Pi
	sinVal := FastSin(x)
	cosVal := FastCos(x)
	result := sinVal*sinVal + cosVal*cosVal

	// Larger tolerance near π due to accumulated approximation errors
	if math.Abs(result-1.0) > 0.05 {
		t.Errorf("sin²(%v) + cos²(%v) = %v, want 1.0 (±0.05)", x, x, result)
	}
}

// TestTrigIdentity_SecIsCosReciprocal tests sec(x) ≈ 1/cos(x)
func TestTrigIdentity_SecIsCosReciprocal(t *testing.T) {
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

// TestTrigIdentity_CscIsSinReciprocal tests csc(x) ≈ 1/sin(x)
func TestTrigIdentity_CscIsSinReciprocal(t *testing.T) {
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

// TestTrigSymmetry_SinIsOdd tests sin(-x) ≈ -sin(x)
func TestTrigSymmetry_SinIsOdd(t *testing.T) {
	testValues := []float64{math.Pi / 6, math.Pi / 4, math.Pi / 3, math.Pi / 2}

	for _, x := range testValues {
		sinPos := FastSin(x)
		sinNeg := FastSin(-x)

		if math.Abs(sinNeg+sinPos) > 0.01 {
			t.Errorf("sin(-%v) = %v, want -sin(%v) = %v", x, sinNeg, x, -sinPos)
		}
	}
}

// TestTrigSymmetry_CosIsEven tests cos(-x) ≈ cos(x)
func TestTrigSymmetry_CosIsEven(t *testing.T) {
	testValues := []float64{math.Pi / 6, math.Pi / 4, math.Pi / 3, math.Pi / 2}

	for _, x := range testValues {
		cosPos := FastCos(x)
		cosNeg := FastCos(-x)

		if math.Abs(cosNeg-cosPos) > 0.01 {
			t.Errorf("cos(-%v) = %v, want cos(%v) = %v", x, cosNeg, x, cosPos)
		}
	}
}

// TestTangentIdentity tests the identity: tan(x) * cotan(x) ≈ 1
func TestTangentIdentity(t *testing.T) {
	// Test in range where both tan and cotan are well-behaved
	for i := 0; i < 20; i++ {
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

// TestTangentReciprocal tests: cotan(x) ≈ 1/tan(x)
func TestTangentReciprocal(t *testing.T) {
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

// TestTangentPeriodicityProperty tests: tan(x + π) ≈ tan(x)
func TestTangentPeriodicityProperty(t *testing.T) {
	for i := 0; i < 10; i++ {
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
